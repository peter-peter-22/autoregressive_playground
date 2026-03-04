import math

from datasets import load_from_disk

from instruction_following_v2.sft_training import sft_training
from load_pre_trained import model

if __name__ == "__main__":
    # Config
    epochs = 5
    batch_size = 8
    eval_steps = math.ceil(800 / batch_size)
    test_dataset = load_from_disk("tokenized_data/test")
    train_dataset = load_from_disk("tokenized_data/train")
    max_steps = math.ceil(epochs * train_dataset.num_rows / batch_size)
    checkpoint_interval = math.ceil(max_steps / 20)
    log_interval = math.ceil(max_steps / 200)
    peak_lr = 5e-5
    min_lr = 5e-6
    warmup_steps = math.ceil(max_steps * 0.05)
    preserve_checkpoints = [0.5, 0.75]
    eval_text_tokens = 200

    # Extract model config
    context_length = model.config.n_positions
    eos_id = model.config.eos_token_id

    # Standard inference function
    def inference(x: torch.Tensor):
        return model(x).logits

    # Run
    sft_training(
        epochs=epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        max_steps=max_steps,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        peak_lr=peak_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        preserve_checkpoints=preserve_checkpoints,
        eval_text_tokens=eval_text_tokens,
        context_length=context_length,
        eos_id=eos_id
    )
