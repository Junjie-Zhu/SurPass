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
    kwargs.pop("name", None)
    for name in (
        "distance_bin_start",
        "distance_bin_end",
        "distance_bin_count",
        "contact_threshold",
        "distance_bin_convention",
    ):
        value = cfg_get(cfg, f"data.{name}", kwargs.get(name))
        if value is not None:
            kwargs[name] = value
    return kwargs


def create_model(cfg: DictConfig):
    """Select an architecture explicitly; old configs continue to use ResOnly."""
    from src.model.surpass import ResOnly
    from src.model.res_pair import ResPair

    name = str(cfg_get(cfg, "model.name", "ResOnly"))
    classes = {"ResOnly": ResOnly, "ResPair": ResPair}
    if name not in classes:
        raise ValueError(f"Unknown model.name {name!r}; choose ResOnly or ResPair")
    return classes[name](**build_model_kwargs(cfg))
