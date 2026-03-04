import torch
from datasets import Dataset

from checkpoints import Checkpointer, CheckpointCleaner
from generation import ChatCompletion
from learning_schedule import LearningScheduler
from load_pre_trained import model, tokenizer
from optimizer import get_optimizer
from training_data_reader import TrainingDataReader
from training_loop import AutoCastConfig, SimpleInference
from training_loop import TrainingLoop
from zip import zip_directory


def sft_training(
        epochs: int,
        batch_size: int,
        eval_steps: int,
        test_dataset: Dataset,
        train_dataset: Dataset,
        max_steps: int,
        checkpoint_interval: int,
        log_interval: int,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        preserve_checkpoints: list[float] | None,
        eval_text_tokens: int,
        context_length: int,
        eos_id: int,
        inference:SimpleInference,
        checkpoint_dir:str
):
    # Cuda settings
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # Optimizer
    optimizer = get_optimizer(
        model=model,
        beta_1=0.9,
        beta_2=0.95,
        learning_rate=0,
        eps=1e-8
    )

    # Scaler
    scaler = torch.amp.GradScaler(device_type)

    # Cleaner
    cleaner = CheckpointCleaner(3)

    # Checkpoints
    checkpointer = Checkpointer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        cleaner=cleaner,
        checkpoint_dir=checkpoint_dir,
        starting_info={
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_steps": eval_steps,
            "max_steps": max_steps,
            "checkpoint_interval": checkpoint_interval,
            "log_interval": log_interval,
            "peak_lr": peak_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "context_length": context_length,
            "eos_id": eos_id
        }
    )
    checkpointer.preserve_progress(
        preserve_checkpoints,
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps
    )
    starting_step = checkpointer.auto_load()

    # Data reader
    data_reader = TrainingDataReader(
        context_length=context_length,
        padding_token_id=eos_id,
        batch_size=batch_size,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        ignore_index=eos_id,
        device=device,
    )

    # Learning scheduler
    scheduler = LearningScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        lr_decay_steps=max_steps - warmup_steps,
        min_lr=min_lr,
        max_lr=peak_lr
    )

    # Generate text
    chat = ChatCompletion(
        tokenizer=tokenizer,
        inference=inference,
        device=device,
        stop_token_ids=[eos_id],
        max_context_length=context_length,
        top_p=0.9,
        top_k=None,
        temperature=0.7,
        max_new_tokens=eval_text_tokens
    )

    def generate():
        return chat.generate(instruction="Hello")

    # Training loop
    training_loop = TrainingLoop(
        inference=inference,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        checkpointer=checkpointer,
        early_stopping=None,
        data_reader=data_reader,
        autocast=AutoCastConfig(dtype=torch.float16, device_type=device_type),
        gradient_clip=1,
        learning_schedule=scheduler,
        eval_steps=eval_steps,
        generate=generate
    )
    training_loop.train(max_steps=max_steps, starting_step=starting_step)

    # Save
    zip_directory()
