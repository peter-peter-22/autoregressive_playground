import torch.nn as nn

from instruction_following.model import ChatModel


def get_optim_groups(model: nn.Module):
    """Apply weight decay to one dimensional to linear weights, exclude bias and layer norm."""
    decay = set()
    no_decay = set()

    for name, param in model.named_parameters():
        if name.endswith("bias"):
            no_decay.add(name)
        elif "ln" in name or "norm" in name:
            no_decay.add(name)
        else:
            decay.add(name)

    optim_groups = [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return optim_groups

if __name__ == "__main__":
    model = ChatModel(
        vocabulary_size=24_000,
        embedding_size=256,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        max_context_length=1_000,
        ff_size_multiplier=4,
        ff_dropout=0.0,
        transformer_blocks=6,
        attention_heads=8
    )

    groups=get_optim_groups(model)
    print(groups[0])
    print(groups[1])