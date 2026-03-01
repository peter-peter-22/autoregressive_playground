# %%
import json
import os
from datetime import datetime

import gdown
import torch
import torch.nn as nn
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from millify import millify

from chat_template import chat_template
from learning_metrics import get_grad_metrics
from learning_metrics import get_weight_metrics
from settings import ModelSettings
from special_tokens import special_tokens

# %% [markdown]
# Mode settings

# %%
minified = False
colab = False
thunder = True
checkpoint: int | None = None
compile = True
upload = True
cpu_only = False

# %% [markdown]
# Paths

# %%
load_dotenv()
if upload:
    if os.getenv("MEGA_EMAIL") is None or os.getenv("MEGA_PASSWORD") is None:
        raise Exception("Missing env values")

if colab:
    data_dir = "/content/drive/MyDrive/tokenized_data"
    checkpoint_dir = "/content/drive/MyDrive/instruction_checkpoints"
    pre_training_file = "not specified"
elif thunder:
    data_dir = "tokenized_data"
    os.makedirs("output/instruction_checkpoints", exist_ok=True)
    if not os.path.exists(data_dir):
        gdown.download_folder(id="15x5BNdwty_4y5ezFoIVS47i1-H2dCjLv", output=data_dir)
    checkpoint_dir = "output/instruction_checkpoints"
    if not minified:
        pre_training_file = "pre_training/weights.pt"
    else:
        pre_training_file = "pre_checkpoints/state/00599.pt"
else:
    data_dir = "tokenized_data"
    checkpoint_dir = "instruction_checkpoints"
    if not minified:
        pre_training_file = "pre_training/weights.pt"
    else:
        pre_training_file = "pre_checkpoints/state/00599.pt"
info_dir = checkpoint_dir + "/info"
state_dir = checkpoint_dir + "/state"
train_ds_name = data_dir + "/train_chats"
test_ds_name = data_dir + "/test_chats"

# %% [markdown]
# General settings

# %%
if not minified:
    if colab:
        # Training data
        block_size = ModelSettings.max_context_length
        batch_size = 8

        # Learning
        max_iters = 6_000
        learning_rate = 1e-6
        eval_iters = 30
        eval_interval = 300
        grad_clip = 1.0
        log_metrics_interval = 30
        log_text = 100
    else:
        # Training data
        block_size = ModelSettings.max_context_length
        batch_size = 32

        # Learning
        max_iters = 12_000
        learning_rate = 1e-5
        eval_iters = 30
        eval_interval = 300
        grad_clip = 1.0
        log_metrics_interval = 30
        log_text = 100
else:
    # Training data
    block_size = 64
    batch_size = 8

    # Learning
    max_iters = 5000
    learning_rate = 1e-4
    eval_iters = 10
    eval_interval = 20
    grad_clip = 1.0
    log_metrics_interval = 2
    log_text = 50

print("epochs", max_iters * batch_size / 210499)
print("chats", max_iters * batch_size)
print("max tokens", millify(max_iters * batch_size * block_size))

# %%
from tokenizers.tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
pad_id = tokenizer.token_to_id(special_tokens["pad"])

# %% [markdown]
# Hardware settings

# %%
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device = "cuda" if torch.cuda.is_available() and not cpu_only else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_enabled = device_type == "cuda"
print(device)

# %% [markdown]
# Training data stream

# %%
def infinite_iterator(ds: Dataset):
    n = 0
    while True:
        iterator = iter(ds.shuffle(n))
        n += 1
        for el in iterator:
            yield el

# %%
ignore_index = -100
if not minified:
    ds_test = infinite_iterator(load_from_disk(test_ds_name))
    ds_train = infinite_iterator(load_from_disk(train_ds_name))
else:
    ds_test = infinite_iterator(load_from_disk(test_ds_name).take(100))
    ds_train = infinite_iterator(load_from_disk(test_ds_name).take(10))


def prepare_chat(chat, target_size, pad_element):
    token_ids = chat["tokens"]
    assistant_mask = chat["assistant_mask"]
    length = len(token_ids)
    # truncate to target size
    if length > target_size:
        trim = length - target_size
        return token_ids[trim:], assistant_mask[trim:]
    # pad to target size
    if length < target_size:
        padding = target_size - length
        return token_ids + [pad_element] * padding, assistant_mask + [False] * padding
    # unchanged
    return token_ids, assistant_mask


def apply_mask(tokens, assistant_mask):
    return [
        t if assistant_mask[i] else ignore_index
        for i, t in enumerate(tokens)
    ]


def get_batch(split):
    iterator = ds_train if split == 'train' else ds_test
    batch = [next(iterator) for _ in range(batch_size)]
    longest_chat = max([len(row["tokens"]) for row in batch])
    target_length = min(longest_chat, block_size + 1)
    chats = [prepare_chat(chat, target_length, pad_id) for chat in batch]
    x = torch.stack([torch.tensor(tokens[0:target_length - 1], dtype=torch.long) for tokens, mask in chats])
    y = torch.stack(
        [torch.tensor(apply_mask(tokens[1:target_length], mask[1:target_length]), dtype=torch.long) for tokens, mask in
         chats])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# %%
get_batch(split="test")[0].shape

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

# %%
def freeze_lower_layers(num_freeze):
    # embeddings
    model.emb.requires_grad_(False)

    # transformer blocks
    for i in range(num_freeze):
        model.transformer[i].requires_grad_(False)


freeze_lower_layers(2 if minified else 6)

print("trainable", len([n for n, p in model.named_parameters() if p.requires_grad]))
print("un-trainable", len([n for n, p in model.named_parameters() if not p.requires_grad]))

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
# Optimizer

# %%
from optimizer import get_optim_groups

optim_groups = get_optim_groups(model)

# apply dynamic learning rate to the optimizer
optimizer = torch.optim.AdamW(
    optim_groups,
    lr=learning_rate,
    betas=(0.9, 0.95),
    eps=1e-8
)

# %% [markdown]
# Generate

# %%
@torch.no_grad()
def generate(model, start, max_new_tokens=50):
    idx = torch.tensor([tokenizer.encode(start, add_special_tokens=False).ids], device=device, dtype=torch.long)

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

if not minified:
    messages = [{
        "role": "user",
        "content": "I'm trying to create a menu with different kinds of pasta. Help me come up with different types of pasta and what they are best used for."
    }]
else:
    messages = [{
        "role": "user",
        "content": "Hello."
    }]
test_text = chat_template(messages, add_generation_token=True)


def save_checkpoint(
        step,
        model,
        optimizer,
        scaler,
        train_loss,
        val_loss,
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
        "text": generate(model, test_text, log_text),
        "metrics": json.dumps(metric_logs)
    }

    state_path = f"{state_dir}/{step:05d}.pt"
    info_path = f"{info_dir}/{step:05d}.pt"

    torch.save(state, state_path)
    torch.save(info, info_path)

    return state_path, info_path

# %% [markdown]
# Load pre-training

# %%
if checkpoint is None:
    state = torch.load(pre_training_file)

    state_dict = state["model"]
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    print("Loaded pre-training")

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
# Clean up old checkpoints

# %%
from checkpoint_cleaner import CheckpointCleaner

keep_progress = [0.5, 0.7, 0.8, 0.9]
preserve_checkpoints = []
i = 0
last_keep = 0
while True:
    i += eval_interval
    progress = i / max_iters
    target_progress = keep_progress[last_keep]
    if progress > target_progress:
        state_path = f"{state_dir}/{i:05d}.pt"
        preserve_checkpoints.append(state_path)
        last_keep += 1
        if last_keep == len(keep_progress):
            break
    if i >= max_iters:
        break

checkpoint_cleaner = CheckpointCleaner(3, preserve_checkpoints)
preserve_checkpoints

# %% [markdown]
# Training loop

# %%
from system_metrics import get_system_metrics

metric_logs = []

best_val_loss = float("inf")
patience = 5
min_delta = 0.01
patience_counter = 0

for step in range(checkpoint or 0, max_iters):
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

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        state_path, info_path = save_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loss=losses["train"],
            val_loss=losses["val"],
            metric_logs=metric_logs
        )
        checkpoint_cleaner.step(state_path)
        metric_logs = []

        val_loss = losses['val']
        improvement = best_val_loss - val_loss
        if improvement >= min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Early stopping")
                break

# %% [markdown]
# Test the model

# %%
print(generate(model, test_text, 20 if minified else 200))

# %% [markdown]
# Export results if needed

# %%
from zip import zip_directory
from mega_upload import mega_upload
import asyncio

if upload:
    output_path = "output.zip"
    if os.path.exists(output_path):
        os.remove(output_path)
    zip_directory(checkpoint_dir, output_path)
    asyncio.run(mega_upload(output_path))


