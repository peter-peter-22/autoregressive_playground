import torch
import psutil
import os


def cuda_metrics():
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    max_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"GPU mem allocated: {mem_alloc:.2f} GB")
    print(f"GPU mem reserved : {mem_reserved:.2f} GB")
    print(f"GPU max allocated: {max_mem:.2f} GB")

    return {
        "mem_allocated": mem_alloc,
        "mem_reserved": mem_reserved,
        "max_mem": max_mem,
    }


def get_system_metrics():
    process = psutil.Process(os.getpid())

    cpu_percent = psutil.cpu_percent()
    ram_gb = process.memory_info().rss / 1e9

    print(f"CPU usage: {cpu_percent:.1f}%")
    print(f"RAM usage: {ram_gb:.2f} GB")

    result= {
        "cpu_percent": cpu_percent,
        "ram_gb": ram_gb,
    }

    if torch.cuda.is_available():
        result["cuda"]=cuda_metrics()

    return result

if __name__ == "__main__":
    print(get_system_metrics())
