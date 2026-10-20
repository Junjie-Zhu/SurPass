from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture or getattr(model, "architecture", None),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def save_checkpoint(path: str | Path, model: torch.nn.Module, **kwargs: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, **kwargs), path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    map_location: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    load_training_state: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    architecture = payload.get("architecture") if isinstance(payload, dict) else None
    expected = getattr(model, "architecture", None)
    if architecture and expected and architecture != expected:
        raise ValueError(
            f"Checkpoint architecture {architecture!r} does not match {expected!r}"
        )
    model.load_state_dict(state_dict, strict=strict)
    if not isinstance(payload, dict):
        return {"model_state_dict": state_dict}
    if load_training_state:
        if optimizer is None or scheduler is None:
            raise ValueError("optimizer and scheduler are required to resume training")
        for key in ("optimizer_state_dict", "scheduler_state_dict", "epoch"):
            if key not in payload:
                raise KeyError(f"Checkpoint is missing {key!r}")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
