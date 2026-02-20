# %% [markdown]
# fix redownloading gdrive data
# download only from thunder

# %%
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from learning_metrics import get_grad_metrics
from learning_metrics import get_weight_metrics
from settings import ModelSettings
import gdown

# %% [markdown]
# Mode settings
# 

# %%
minified = False
colab = False
thunder = True
checkpoint: int | None = None
compile = True

# %% [markdown]
# Paths

# %%
if colab:
    data_dir = "/content/drive/MyDrive"
    checkpoint_dir = "/content/drive/MyDrive/pre_checkpoints"
elif thunder:
    os.makedirs("output/pre_checkpoints", exist_ok=True)
    if not os.path.exists("tokenized_data/train.bin"):
        gdown.download(id="15t3259RbsF772b35aaZGouGwQopFAX96",output="tokenized_data/train.bin")
    if not os.path.exists("tokenized_data/test.bin"):
        gdown.download(id="1rE_MOBhBPQGUuhYmevNZOFj-LMBWkLFD",output="tokenized_data/test.bin")
    data_dir = "tokenized_data"
    checkpoint_dir = "output/pre_checkpoints"
else:
    data_dir = "tokenized_data"
    checkpoint_dir = "pre_checkpoints"
info_dir = checkpoint_dir + "/info"
state_dir = checkpoint_dir + "/state"

# %% [markdown]
# General settings

# %%
if not minified:
    # Training data
    block_size = ModelSettings.max_context_length
    batch_size = 32

    # Learning
    max_iters = 200_000  # total number of training iterations
    learning_rate = 6e-4
    min_lr = 6e-5
    lr_decay_steps = max_iters  # should be ~= max_iters per Chinchilla
    warmup_steps = 4000
    eval_iters = 100
    eval_interval = 2000
    grad_clip = 1.0
    log_metrics_interval = 100
    log_text=100
else:
    # Training data
    block_size = 64
    batch_size = 8

    # Learning
    max_iters = 600  # total number of training iterations
    learning_rate = 6e-3
    min_lr = 6e-4
    lr_decay_steps = max_iters  # should be ~= max_iters per Chinchilla
    warmup_steps = 60
    eval_iters = 2
    eval_interval = 10
    grad_clip = 1.0
    log_metrics_interval = 2
    log_text=50

# %%
max_iters * batch_size * block_size / 1_500_000_000

# %% [markdown]
# Hardware settings

# %%
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_enabled = device_type == "cuda"
print(device)

# %% [markdown]
# Training data stream

# %%
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin' if not minified else "test.bin"), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    if not minified:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        ix = torch.randint(5000 - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# %%
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# %% [markdown]
# Scaler for FP16

# %%
scaler = torch.amp.GradScaler(device_type)

# %% [markdown]
# Model settings

# %%
from model import ChatModel
from settings import ModelSettings

if not minified:
    model = ChatModel(
        vocabulary_size=ModelSettings.vocabulary_size,
        embedding_size=ModelSettings.embedding_size,
        max_context_length=block_size,
        ff_size_multiplier=ModelSettings.ff_size_multiplier,
        transformer_blocks=ModelSettings.transformer_blocks,
        attention_heads=ModelSettings.attention_heads,
        dropout=0.0,
        bias=ModelSettings.bias,
        device=device,
    )
else:
    model = ChatModel(
        vocabulary_size=ModelSettings.vocabulary_size,
        embedding_size=64,
        max_context_length=block_size,
        ff_size_multiplier=2,
        transformer_blocks=4,
        attention_heads=4,
        dropout=0.0,
        bias=ModelSettings.bias,
        device=device,
    )

model = model.to(device)

if compile:
    model = torch.compile(model)

# %% [markdown]
# Optimizer

# %%
from optimizer import get_optim_groups

optim_groups = get_optim_groups(model)

# apply dynamic learning rate to the optimizer
optimizer = torch.optim.AdamW(
    optim_groups,
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8
)

# %% [markdown]
# Generate

# %%
from tokenizers.tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")


@torch.no_grad()
def generate(model, start, max_new_tokens=50):
    idx = torch.tensor([tokenizer.encode(start).ids], device=device, dtype=torch.long)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ModelSettings.max_context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())

# %% [markdown]
# Checkpointer

# %%
os.makedirs(info_dir, exist_ok=True)
os.makedirs(state_dir, exist_ok=True)


def save_checkpoint(
        step,
        model,
        optimizer,
        scaler,
        train_loss,
        val_loss,
        learning_rate,
        metric_logs
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
    }

    info = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "time": datetime.now().isoformat(),
        "block_size": block_size,
        "batch_size": batch_size,
        "eval_interval": eval_interval,
        "step": step,
        "learning_rate": learning_rate,
        "text": generate(model, "Once upon a time", log_text),
        "metrics": json.dumps(metric_logs)
    }

    state_path = f"{state_dir}/{step:05d}.pt"
    info_path = f"{info_dir}/{step:05d}.pt"

    torch.save(state, state_path)
    torch.save(info, info_path)

    return state_path,info_path

# %% [markdown]
# Load training state

# %%
def load_checkpoint(step: int):
    state = torch.load(f"{state_dir}/{step:05d}.pt")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    print(f"Loaded checkpoint {step}")


if checkpoint is not None:
    load_checkpoint(checkpoint)

# %% [markdown]
# Learning scheduler

# %%
import math


def get_lr(step):
    # 1) linear warmup for warmup_steps steps
    if step < warmup_steps:
        return learning_rate * (step + 1) / (warmup_steps + 1)
    # 2) if it > lr_decay_steps, return min learning rate
    if step > lr_decay_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coefficient ranges 0..1
    return min_lr + coefficient * (learning_rate - min_lr)

# %% [markdown]
# Clean up old checkpoints

# %%
from checkpoint_cleaner import CheckpointCleaner

checkpoint_cleaner=CheckpointCleaner(3)

# %% [markdown]
# Training loop

# %%
from system_metrics import get_system_metrics

metric_logs = []

for step in range(checkpoint or 0, max_iters):
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad()

    xb, yb = get_batch("train")

    if autocast_enabled:
        with torch.amp.autocast(dtype=torch.float16, device_type=device_type):
            logits, loss = model(xb, yb)
    else:
        logits, loss = model(xb, yb)

    # exit if the loss is invalid
    if not torch.isfinite(loss):
        raise Exception("Non-finite loss detected.")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    if step % log_metrics_interval == 0:
        total_norm, max_grad = get_grad_metrics(model)
        max_weight, total_weight_norm = get_weight_metrics(model)
        metric_logs.append({
            "gradient": {
                "total_norm": total_norm,
                "max_grad": max_grad,
            },
            "weight": {
                "max_weight": max_weight,
                "total_weight_norm": total_weight_norm,
            },
            "system": get_system_metrics(),
            "current_loss": loss.item()
        })

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0 or step == max_iters-1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        state_path,info_path= save_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loss=losses["train"],
            val_loss=losses["val"],
            learning_rate=lr,
            metric_logs=metric_logs
        )
        checkpoint_cleaner.step(state_path)
        metric_logs = []


# %% [markdown]
# Test the model

# %%
start_token_id = get_batch("test")[0][0][0].item()
start_text = tokenizer.decode([start_token_id])
print(generate(model, start_text))


