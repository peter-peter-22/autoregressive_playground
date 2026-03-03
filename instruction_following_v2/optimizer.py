import torch
import torch.nn as nn

def get_optim_groups(model: nn.Module):
    """Apply weight decay to one dimensional to linear weights, exclude bias and layer norm."""
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    optim_groups = [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return optim_groups

def get_optimizer(model: nn.Module, beta_1, beta_2, learning_rate, eps):
    return torch.optim.AdamW(
        get_optim_groups(model),
        lr=learning_rate,
        betas=(beta_1, beta_2),
        eps=eps
    )
