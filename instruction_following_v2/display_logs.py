import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import torch


@dataclass
class Step:
    train_loss: float
    val_loss: float
    time: datetime
    learning_rate: float
    step: int
    text: str
    metrics: dict[str, Any]


@dataclass
class Start:
    epochs: int
    batch_size: int
    eval_steps: int
    max_steps: int
    checkpoint_interval: int
    log_interval: int
    peak_lr: float
    min_lr: float
    warmup_steps: int
    context_length: int
    eos_id: int
    time: datetime


@dataclass
class GradientMetrics:
    total_norm: float
    max_grad: float


@dataclass
class WeightMetrics:
    max_weight: float
    total_weight_norm: float


@dataclass
class CudaMetrics:
    mem_allocated: float
    mem_reserved: float
    max_mem: float


@dataclass
class SystemMetrics:
    cpu_percent: float
    ram_gb: float
    cuda: CudaMetrics | None = None


@dataclass
class Metrics:
    gradient: GradientMetrics
    weight: WeightMetrics
    system: SystemMetrics
    current_loss: float
    time: datetime
    step: int


def read_logs(info_dir: str):
    # Read starting info
    start_info_path = os.path.join(info_dir, "start.pt")
    data = torch.load(start_info_path)
    start = Start(
        epochs=data["epochs"],
        batch_size=data["batch_size"],
        eval_steps=data["eval_steps"],
        max_steps=data["max_steps"],
        checkpoint_interval=data["checkpoint_interval"],
        log_interval=data["log_interval"],
        peak_lr=data["peak_lr"],
        min_lr=data["min_lr"],
        warmup_steps=data["warmup_steps"],
        context_length=data["context_length"],
        eos_id=data["eos_id"],
        time=datetime.fromisoformat(data["time"])
    )

    # Read steps
    info_dir = os.path.join(info_dir, "info")
    files = [f for f in os.listdir(info_dir) if os.path.isfile(os.path.join(info_dir, f))]
    print(f"Found {len(files)} files")
    steps: list[Step] = []
    for p in files:
        data = torch.load(os.path.join(info_dir, p))
        steps.append(Step(
            step=data["step"],
            train_loss=data["train_loss"],
            val_loss=data["val_loss"],
            time=datetime.fromisoformat(data["time"]),
            learning_rate=data["learning_rate"],
            metrics=json.loads(data["metrics"]),
            text=data["text"]
        ))
    steps = sorted(steps, key=lambda x: x.step)

    # Extract metrics
    metrics: list[Metrics] = []
    for step in steps:
        for data in step.metrics:
            data: dict[str, Any]
            system_metrics = data["system"]
            cuda_metrics = system_metrics["cuda"]
            metrics.append(Metrics(
                gradient=GradientMetrics(**data["gradient"]),
                weight=WeightMetrics(**data["weight"]),
                system=SystemMetrics(
                    cpu_percent=system_metrics["cpu_percent"],
                    ram_gb=system_metrics["ram_gb"],
                    cuda=CudaMetrics(**cuda_metrics) if cuda_metrics else None
                ),
                current_loss=data["current_loss"],
                time=datetime.fromisoformat(data["time"]),
                step=step.step
            ))

    print(f"Found {len(files)} metric logs")

    return start, steps, metrics


def display_loss(x: list[int], steps: list[Step]):
    train_loss = [step.train_loss for step in steps]
    test_loss = [step.val_loss for step in steps]

    plt.plot(x, train_loss, label="train", color="cyan")
    plt.plot(x, test_loss, label="test", color="red")
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Losses')
    plt.grid(True)
    plt.show()


def display_perplexity(x: list[int], steps: list[Step]):
    train_p = [math.exp(step.train_loss) for step in steps]
    test_p = [math.exp(step.val_loss) for step in steps]

    plt.plot(x, train_p, label="train", color="cyan")
    plt.plot(x, test_p, label="test", color="red")
    plt.xlabel('step')
    plt.title('Perplexity')
    plt.grid(True)
    plt.show()


def display_tokens_per_second(x: list[int], steps: list[Step], start: Start):
    last_step_time: datetime | None = None
    values = []

    for step in steps:

        time = step.time
        if last_step_time is None:
            last_step_time = time
            values.append(0.0)
            continue
        time_passed = time - last_step_time
        last_step_time = time

        block_size = start.context_length
        batch_size = start.batch_size
        eval_interval = start.checkpoint_interval

        tokens = block_size * batch_size * eval_interval
        tokens_per_second = tokens / time_passed.total_seconds()

        values.append(tokens_per_second)

    plt.plot(x, values, color="red")
    plt.xlabel('step')
    plt.ylabel('tokens')
    plt.title('Tokens per second')
    plt.grid(True)
    plt.show()


def display_learning_rate(x: list[int], steps: list[Step]):
    rate = [step.learning_rate for step in steps]

    plt.plot(x, rate, color="red")
    plt.xlabel('step')
    plt.ylabel('rate')
    plt.title('Learning rate')
    plt.grid(True)
    plt.show()


def display_step_durations(x: list[int], steps: list[Step]):
    last_step_time: datetime | None = None

    durations = []

    for step in steps:
        time = step.time
        if last_step_time is None:
            last_step_time = time
            durations.append(0.0)
            continue
        time_passed = time - last_step_time
        last_step_time = time
        durations.append(time_passed.total_seconds())

    plt.plot(x, durations, color="red")
    plt.xlabel('step')
    plt.ylabel('seconds')
    plt.title('Step durations')
    plt.grid(True)
    plt.show()


def display_cpu_metrics(x: list[int], metrics: list[Metrics]):
    cpu = [info.system.cpu_percent for info in metrics]

    plt.plot(x, cpu, color="red")
    plt.xlabel('step')
    plt.ylabel('%')
    plt.title('CPU usage')
    plt.grid(True)
    plt.show()


def display_current_loss(x: list[int], metrics: list[Metrics]):
    values = [info.current_loss for info in metrics]

    plt.plot(x, values, color="red")
    plt.xlabel('step')
    plt.title('Processed train loss')
    plt.grid(True)
    plt.show()


def display_ram_usage(x: list[int], metrics: list[Metrics]):
    values = [info.system.ram_gb for info in metrics]

    plt.plot(x, values, color="red")
    plt.xlabel('step')
    plt.ylabel('GB')
    plt.title('RAM usage')
    plt.grid(True)
    plt.show()


def display_total_gradient(x: list[int], metrics: list[Metrics]):
    values = [info.gradient.total_norm for info in metrics]

    plt.plot(x, values, color="red")
    plt.xlabel('step')
    plt.title('Gradient total normalized')
    plt.grid(True)
    plt.show()


def display_max_gradient(x: list[int], metrics: list[Metrics]):
    values = [info.gradient.max_grad for info in metrics]

    plt.plot(x, values, color="red")
    plt.xlabel('step')
    plt.title('Max gradient')
    plt.grid(True)
    plt.show()


def display_max_weight(x: list[int], metrics: list[Metrics]):
    values = [info.weight.max_weight for info in metrics]

    plt.plot(x, values, color="red")
    plt.xlabel('step')
    plt.title('Max weight')
    plt.grid(True)
    plt.show()


def display_total_weight(x: list[int], metrics: list[Metrics]):
    values = [info.weight.total_weight_norm for info in metrics]

    plt.plot(x, values, color="red")
    plt.xlabel('step')
    plt.title('Total weight normalized')
    plt.grid(True)
    plt.show()


def display_cuda_ram(x: list[int], metrics: list[Metrics]):
    mem_allocated = [info.system.cuda.mem_allocated for info in metrics]
    mem_max = [info.system.cuda.max_mem for info in metrics]
    mem_reserved = [info.system.cuda.mem_reserved for info in metrics]

    plt.plot(x, mem_allocated, color="red", label="allocated")
    plt.plot(x, mem_max, color="green", label="max")
    plt.plot(x, mem_reserved, color="cyan", label="reserved")
    plt.xlabel('step')
    plt.title('Cuda ram usage')
    plt.grid(True)
    plt.show()


def display_logs(info_dir: str = "output"):
    start, steps, metrics = read_logs(info_dir)
    x = list(range(len(steps)))
    x_metrics = list(range(len(metrics)))
    total_time = (steps[-1].time - start.time)
    print(f"Total training time: {total_time.total_seconds() / 60 / 60:.2f} hours")

    display_loss(x, steps)
    display_perplexity(x, steps)
    display_tokens_per_second(x, steps, start)
    display_learning_rate(x, steps)
    display_step_durations(x, steps)

    display_cpu_metrics(x_metrics, metrics)
    display_current_loss(x_metrics, metrics)
    display_ram_usage(x_metrics, metrics)
    display_total_gradient(x_metrics, metrics)
    display_max_gradient(x_metrics, metrics)
    display_max_weight(x_metrics, metrics)
    display_total_weight(x_metrics, metrics)
    if metrics[0].system.cuda:
        display_cuda_ram(x_metrics, metrics)


if __name__ == "__main__":
    display_logs()
