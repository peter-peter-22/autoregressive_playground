import torch.nn as nn


def get_optim_groups(model: nn.Module):
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
