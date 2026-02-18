import torch
from torch import nn


def get_grad_metrics(model:nn.Module):
    total_norm = 0.0
    max_grad = 0.0
    num_grads = 0

    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            total_norm += grad.norm(2).item() ** 2
            max_grad = max(max_grad, grad.abs().max().item())
            num_grads += 1

    total_norm = total_norm ** 0.5

    return total_norm, max_grad

def get_weight_metrics(model:nn.Module):
    total_weight_norm = 0.0
    max_weight = 0.0

    for p in model.parameters():
        w = p.detach()
        total_weight_norm += w.norm(2).item() ** 2
        max_weight = max(max_weight, w.abs().max().item())

    total_weight_norm = total_weight_norm ** 0.5

    return max_weight, total_weight_norm