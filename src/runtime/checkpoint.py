"""Versioned live/EMA checkpoints and explicit weight selection."""
from pathlib import Path
from typing import Any
import warnings
import torch


def model_contract(model):
    return {"bins": getattr(model, "bin_spec", None),
            "contact_bins": getattr(model, "contact_bins", None),
            "recycle_passes": getattr(model, "recycle_passes", None),
            "protein_pooling": getattr(model, "protein_pooling", None)}


def checkpoint_payload(model, *, epoch=None, optimizer=None, scheduler=None,
                       architecture=None, ema=None, training_state=None, config=None):
    payload = {"format_version": 2, "model_state_dict": model.state_dict(),
               "architecture": architecture or getattr(model, "architecture", None),
               "model_contract": model_contract(model)}
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if ema is not None:
        if ema.backup:
            raise ValueError("Save live weights before saving a complete EMA checkpoint")
        payload["ema_state_dict"] = ema.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    if config is not None:
        payload["config"] = config
    return payload


def save_checkpoint(path, model, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(model, **kwargs), temporary)
    temporary.replace(path)


def load_checkpoint(path, model, *, map_location="cpu", optimizer=None, scheduler=None,
                    load_training_state=False, strict=True, weights="live", ema=None):
    if weights not in {"live", "ema", "legacy"}:
        raise ValueError("checkpoint weights must be live, ema, or legacy")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    wrapped = isinstance(payload, dict) and "model_state_dict" in payload
    if not wrapped:
        payload = {"model_state_dict": payload}
    architecture = payload.get("architecture")
    expected = getattr(model, "architecture", None)
    if architecture and expected and architecture != expected:
        raise ValueError(f"Checkpoint architecture {architecture!r} does not match {expected!r}")
    contract = payload.get("model_contract")
    if contract is not None and contract != model_contract(model):
        raise ValueError("Checkpoint model/bin contract differs from the configured model")
    if contract is None and getattr(model, "bin_spec", {}).get("convention") == "left_closed":
        raise ValueError("Legacy checkpoint has no exact-bin contract; use its legacy configuration")
    state = payload["model_state_dict"]
    if weights == "ema":
        if "ema_state_dict" not in payload:
            raise ValueError("Checkpoint has no EMA state; select weights=legacy explicitly for an old single-model checkpoint")
        state = payload["ema_state_dict"]["shadow"]
    if weights == "legacy":
        warnings.warn("Using a legacy single-model weight interpretation", UserWarning)
    if load_training_state:
        if weights != "live" or optimizer is None or scheduler is None:
            raise ValueError("Resume requires live weights, optimizer, and scheduler")
        required = ["optimizer_state_dict", "scheduler_state_dict", "epoch"]
        if ema is not None:
            required.append("ema_state_dict")
        for key in required:
            if key not in payload:
                raise KeyError(f"Checkpoint is missing {key!r}; use model-only loading for a warm start")
        saved_scheduler = payload["scheduler_state_dict"]
        # A changed horizon must be a deliberate new experiment, not a silent resume.
        for key in ("decay_steps", "warmup_steps", "lr", "min_lr"):
            if key in saved_scheduler and hasattr(scheduler, key) and saved_scheduler[key] != getattr(scheduler, key):
                raise ValueError(f"Scheduler {key} changed on resume")
    model.load_state_dict(state, strict=strict)
    if load_training_state:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        if ema is not None:
            ema.load_state_dict(payload["ema_state_dict"])
    return payload
