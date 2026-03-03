from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from checkpoints import Checkpointer
from instruction_following_v2.learning_schedule import LearningScheduler
from loss import calculate_loss
from training_data_reader import TrainingDataReader


class EarlyStopping:
    def __init__(
            self,
            patience: int,
            min_delta: float = 0.01
    ):
        self.patience = patience
        self.best_loss = float('inf')
        self.min_delta = min_delta
        self.patience_counter = 0

    def step(self, loss: float):
        improvement = self.best_loss - loss
        if improvement >= self.min_delta:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"Not enough improvement, patience: {self.patience - self.patience_counter}")
            if self.patience_counter == self.patience:
                print("Early stopping")
                return True
        return False


@dataclass
class AutoCastConfig:
    dtype: torch.dtype
    device_type: str


class TrainingLoop:
    def __init__(
            self,
            inference: Callable[[torch.Tensor], torch.Tensor],
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scaler: torch.amp.GradScaler,
            eval_interval: int,
            log_interval: int,
            checkpointer: Checkpointer,
            early_stopping: EarlyStopping,
            data_reader: TrainingDataReader,
            autocast: AutoCastConfig | None,
            gradient_clip: float | None,
            learning_schedule: LearningScheduler
    ):
        self.step = 0
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.eval_interval = eval_interval
        self.checkpointer = checkpointer
        self.early_stopping = early_stopping
        self.data_reader = data_reader
        self.autocast = autocast
        self.log_interval = log_interval
        self.gradient_clip = gradient_clip
        self.inference = inference
        self.learning_schedule = learning_schedule

    def train(self, max_steps: int, starting_step: int = 0):
        for step in range(starting_step, max_steps):
            lr = self.learning_schedule.update(step)
            self.optimizer.zero_grad()

            xb, yb = self.data_reader.get_batch(False)

            if self.autocast:
                with torch.amp.autocast(dtype=self.autocast.dtype, device_type=self.autocast.device_type):
                    logits = self.inference(xb)
            else:
                logits = self.inference(xb)
            loss = calculate_loss(logits, yb)

            # exit if the loss is invalid
            if not torch.isfinite(loss):
                raise Exception("Non-finite loss detected.")

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if step % self.log_interval == 0:
                self.checkpointer.create_log(loss.item())

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if step % self.eval_interval == 0 or step == max_steps - 1:
                losses = estimate_loss()
                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.checkpointer.save(
                    step=step,
                    train_loss=losses["train"],
                    val_loss=losses["val"],
                    learning_rate=lr,
                    eval_text="coming soon"
                )

                if self.early_stopping and self.early_stopping.step(val_loss):
                    break
