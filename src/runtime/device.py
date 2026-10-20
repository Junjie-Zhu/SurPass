from __future__ import annotations

from contextlib import nullcontext

import torch


def to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    return value


def resolve_cuda_device(local_rank: int, device_count: int) -> torch.device:
    if int(device_count) <= 0:
        return torch.device("cpu")
    index = int(local_rank) % int(device_count)
    return torch.device("cuda", index)


def resolve_dist_backend(use_cuda: bool) -> str:
    return "nccl" if use_cuda else "gloo"


def normalize_precision(precision: str, device: torch.device) -> str:
    precision = str(precision).lower()
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be 'float32' or 'bfloat16'")
    if device.type != "cuda":
        return "float32"
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 was requested but this CUDA device does not support it")
    return precision


def autocast_context(device: torch.device, precision: str):
    precision = normalize_precision(precision, device)
    if device.type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
