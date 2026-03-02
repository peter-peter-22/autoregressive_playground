import torch

import math

import torch.nn as nn




class LearningScheduler:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: int,
            lr_decay_steps: int,
            min_lr: float,
            max_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.max_lr = max_lr

    def get_lr(self,step):
        # 1) linear warmup for warmup_steps steps
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / (self.warmup_steps + 1)
        # 2) if it > lr_decay_steps, return min learning rate
        if step > self.lr_decay_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (self.lr_decay_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coefficient ranges 0..1
        return self.min_lr + coefficient * (self.max_lr - self.min_lr)

    def update(self,step):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

