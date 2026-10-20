from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def cfg_get(cfg: DictConfig, path: str, default=None):
    value = OmegaConf.select(cfg, path, default=None)
    return default if value is None else value


def build_model_kwargs(cfg: DictConfig) -> dict:
    model = cfg_get(cfg, "model", {})
    if isinstance(model, DictConfig):
        model = OmegaConf.to_container(model, resolve=True)
    kwargs = dict(model or {})
    for name in (
        "distance_bin_start",
        "distance_bin_end",
        "distance_bin_count",
        "contact_threshold",
    ):
        kwargs[name] = cfg_get(cfg, f"data.{name}", kwargs.get(name))
    return kwargs
