import json
import os
from datetime import datetime

import torch
import torch.nn as nn

from learning_metrics import get_grad_metrics, get_weight_metrics
from system_metrics import get_system_metrics


class CheckpointCleaner:
    def __init__(
            self,
            max_checkpoints: int,
            preserve_checkpoints: list | None = None,
    ):
        self.max_checkpoints = max_checkpoints
        self.history: list[str] = []
        self.preserve_checkpoints = preserve_checkpoints

    def step(self, new_checkpoint_path: str):
        self.history.append(new_checkpoint_path)
        count = len(self.history)
        if count > self.max_checkpoints:
            remove = count - self.max_checkpoints
            for i in range(remove):
                path = self.history[i]
                if not self.preserve_checkpoints or path not in self.preserve_checkpoints:
                    try:
                        os.remove(path)
                        print(f"Removed checkpoint {path}")
                    except FileNotFoundError:
                        print(f"The removed checkpoint {path} does not exist")
                else:
                    print(f"Preserving checkpoint {path}")
            self.history = self.history[remove:]


class Checkpointer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scaler: torch.amp.GradScaler | None,
            checkpoint_dir: str,
            cleaner: CheckpointCleaner,
            max_context_length: int,
            eval_interval: int,
            batch_size: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.state_dir = os.path.join(checkpoint_dir, "state")
        self.info_dir = os.path.join(checkpoint_dir, "info")
        self.cleaner = cleaner
        self.pending_metric_logs: list = []

        # Create missing directories
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.info_dir, exist_ok=True)

        # Write starting info
        starting_info_path = os.path.join(checkpoint_dir, "start.pt")
        torch.save({
            "time": datetime.now().isoformat(),
            "block_size": max_context_length,
            "batch_size": batch_size,
            "eval_interval": eval_interval,
        }, starting_info_path)

    def get_info_checkpoint_path(self, step):
        return os.path.join(self.info_dir, f"{step:06d}.pt")

    def get_state_checkpoint_path(self, step):
        return os.path.join(self.state_dir, f"{step:06d}.pt")

    def save(
            self,
            step: int,
            train_loss: float,
            val_loss: float,
            learning_rate: float,
            eval_text: str
    ):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

        info = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": datetime.now().isoformat(),
            "learning_rate": learning_rate,
            "step": step,
            "text": eval_text,
            "metrics": json.dumps(self.pending_metric_logs)
        }

        state_path = self.get_state_checkpoint_path(step)
        info_path = self.get_info_checkpoint_path(step)

        torch.save(state, state_path)
        torch.save(info, info_path)

        self.cleaner.step(state_path)
        self.pending_metric_logs = []

    def load(self, path: str, remove_prefix: bool = True):
        state = torch.load(path)

        state_dict = state["model"]
        if remove_prefix:
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler:
            self.scaler.load_state_dict(state["scaler"])
        print(f"Loaded checkpoint {path}")

    def load_num(self, step: int):
        self.load(self.get_state_checkpoint_path(step))

    def preserve_progress(self, keep_progress: list[float], checkpoint_interval: int, max_steps: int):
        preserve_checkpoints = []
        i = 0
        last_keep = 0
        while True:
            i += checkpoint_interval
            progress = i / max_steps
            target_progress = keep_progress[last_keep]
            if progress >= target_progress:
                preserve_checkpoints.append(i)
                last_keep += 1
                if last_keep == len(keep_progress):
                    break
            if i >= max_steps:
                break

        print(f"Preserving checkpoints {preserve_checkpoints}")
        paths = [
            self.get_state_checkpoint_path(step)
            for step in preserve_checkpoints
        ]
        self.cleaner.preserve_checkpoints = paths

    def auto_load(self):
        files = [f for f in os.listdir(self.state_dir) if os.path.isfile(os.path.join(self.state_dir, f))]
        print(f"Found {len(files)} checkpoints")

        files = sorted(files)
        self.cleaner.history = files
        last_step = files[-1]
        self.load(last_step)

        step_num = int(last_step.split("/")[-2]) + 1
        print(f"Starting from step {step_num}")
        return step_num

    def create_log(self, current_loss: float):
        total_norm, max_grad = get_grad_metrics(self.model)
        max_weight, total_weight_norm = get_weight_metrics(self.model)
        self.pending_metric_logs.append({
            "gradient": {
                "total_norm": total_norm,
                "max_grad": max_grad,
            },
            "weight": {
                "max_weight": max_weight,
                "total_weight_norm": total_weight_norm,
            },
            "system": get_system_metrics(),
            "current_loss": current_loss
        })
        print(f"Current loss: {current_loss}")
