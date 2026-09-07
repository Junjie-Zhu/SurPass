from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.ddp_utils import DIST_WRAPPER

INPUT_KEYS = (
    "plm_emb",
    "residue_position",
    "mask",
    "residue_type",
    "residue_index",
    "chain_index",
)
LABEL_KEYS = ("label_2d_bins", "label_2d_mask", "is_positive")
WATCH_ATTRS = ("residue_embedder", "pair_out_layernorm", "pair_out_linear")
WATCH_LISTS = (
    "residue_blocks",
    "outer_product_mean",
    "triangle_multiplication_outgoing",
    "triangle_multiplication_incoming",
    "pair_blocks",
)


class NonFiniteTrainingError(RuntimeError):
    """Raised on the first non-finite loss/grad so Adam cannot poison weights."""


def _as_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for item in value.values():
            tensors.extend(_as_tensors(item))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_as_tensors(item))
        return tensors
    return []


def finite_stats(name: str, value: Any) -> dict[str, Any]:
    tensors = [item for item in _as_tensors(value) if item.numel()]
    if not tensors:
        return {
            "name": name,
            "finite": True,
            "nan": 0,
            "inf": 0,
            "abs_max": 0.0,
            "numel": 0,
        }
    nan = 0
    inf = 0
    abs_max = 0.0
    numel = 0
    for tensor in tensors:
        flat = tensor.detach()
        numel += int(flat.numel())
        if not (flat.is_floating_point() or flat.is_complex()):
            continue
        nan += int(torch.isnan(flat).sum().item())
        inf += int(torch.isinf(flat).sum().item())
        finite = flat[torch.isfinite(flat)]
        if finite.numel():
            abs_max = max(abs_max, float(finite.abs().max().item()))
    return {
        "name": name,
        "finite": nan == 0 and inf == 0,
        "nan": nan,
        "inf": inf,
        "abs_max": abs_max,
        "numel": numel,
    }


def _nonfinite_batch_indices(tensor: torch.Tensor) -> list[int]:
    if tensor.ndim == 0 or tensor.shape[0] == 0:
        return []
    flat = tensor.reshape(tensor.shape[0], -1)
    bad = ~torch.isfinite(flat).all(dim=1)
    return torch.nonzero(bad, as_tuple=False).flatten().tolist()


def _empty_mask_indices(mask: torch.Tensor) -> list[int]:
    counts = mask.to(dtype=torch.bool).reshape(mask.shape[0], -1).sum(dim=1)
    return torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()


def _unwrap_model(model: nn.Module | None) -> nn.Module | None:
    if model is None:
        return None
    return model.module if hasattr(model, "module") else model


def _modules_to_watch(model: nn.Module) -> list[tuple[str, nn.Module]]:
    root = _unwrap_model(model)
    if root is None:
        return []
    watched: list[tuple[str, nn.Module]] = []
    for attr in WATCH_ATTRS:
        module = getattr(root, attr, None)
        if isinstance(module, nn.Module):
            watched.append((attr, module))
    for list_name in WATCH_LISTS:
        modules = getattr(root, list_name, None)
        if modules is None:
            continue
        for index, module in enumerate(modules):
            watched.append((f"{list_name}.{index}", module))
    if watched:
        return watched
    return [(name, child) for name, child in root.named_children()]


def probe_first_nonfinite_module(
    model: nn.Module,
    residue_batch: dict[str, torch.Tensor],
    recycle_rounds: int = 1,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def _hook(name: str):
        def hook(module, inputs, output) -> None:
            in_stats = finite_stats("in", inputs)
            out_stats = finite_stats("out", output)
            if out_stats["finite"]:
                return
            findings.append(
                {
                    "name": name,
                    "module": type(module).__name__,
                    "true_risk": bool(in_stats["finite"]),
                    "out_inf": out_stats["inf"] > 0,
                    "out_nan": out_stats["nan"] > 0,
                    "in_finite": bool(in_stats["finite"]),
                    "out_abs_max": out_stats["abs_max"],
                }
            )

        return hook

    hooks = []
    for name, module in _modules_to_watch(model):
        hooks.append(module.register_forward_hook(_hook(name)))
    try:
        with torch.no_grad():
            model(residue_batch, recycle_rounds=recycle_rounds)
    except Exception as exc:
        findings.append(
            {
                "name": "forward",
                "module": type(_unwrap_model(model)).__name__,
                "true_risk": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        for hook in hooks:
            hook.remove()
    return findings


def classify_nan_cause(report: dict[str, Any]) -> str:
    dirty = [
        name
        for name, stats in report.get("inputs", {}).items()
        if not stats.get("finite", True)
    ]
    if dirty:
        return f"dirty_input:{','.join(dirty)}"

    empty = report.get("empty_mask_samples") or []
    if empty:
        return f"empty_residue_mask:samples={empty}"

    logits = report.get("logits") or {}
    if not logits.get("finite", True):
        true_risks = [
            item for item in report.get("module_findings") or [] if item.get("true_risk")
        ]
        samples = report.get("nonfinite_logit_samples") or []
        suffix = f" samples={samples}" if samples else ""
        if true_risks:
            first = true_risks[0]
            kind = "inf" if first.get("out_inf") else "nan"
            return f"forward:{first['name']} produced {kind} from finite inputs{suffix}"
        return f"forward:logits nonfinite{suffix}"

    bad_terms = [
        name
        for name, stats in report.get("terms", {}).items()
        if not stats.get("finite", True)
    ]
    if bad_terms:
        return f"loss:{','.join(bad_terms)} from finite logits"

    bad_grads = [
        name
        for name, stats in report.get("grads", {}).items()
        if not stats.get("finite", True)
    ]
    if bad_grads:
        preview = ",".join(bad_grads[:8])
        return f"backward:nonfinite grads ({preview})"

    bad_params = [
        name
        for name, stats in report.get("params", {}).items()
        if not stats.get("finite", True)
    ]
    if bad_params:
        preview = ",".join(bad_params[:8])
        return f"parameters already nonfinite ({preview})"

    return "unknown:loss was nonfinite but no source was isolated"


def collect_training_nan_report(
    *,
    residue_batch: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    intra_mask: torch.Tensor,
    inter_mask: torch.Tensor,
    terms: dict[str, torch.Tensor],
    loss: torch.Tensor,
    model: nn.Module | None = None,
    recycle_rounds: int = 1,
    stage: str = "loss",
    epoch: int | None = None,
    step: int | None = None,
    lr: float | None = None,
    probe_modules: bool = False,
) -> dict[str, Any]:
    inputs = {
        key: finite_stats(key, residue_batch[key])
        for key in INPUT_KEYS
        if key in residue_batch
    }
    for key in LABEL_KEYS:
        if key in labels:
            inputs[key] = finite_stats(key, labels[key])

    mask = residue_batch.get("mask")
    empty_mask_samples = _empty_mask_indices(mask) if torch.is_tensor(mask) else []
    report: dict[str, Any] = {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "lr": lr,
        "recycle_rounds": recycle_rounds,
        "rank": int(DIST_WRAPPER.rank),
        "inputs": inputs,
        "logits": finite_stats("logits", logits),
        "nonfinite_logit_samples": _nonfinite_batch_indices(logits),
        "empty_mask_samples": empty_mask_samples,
        "empty_intra_samples": _empty_mask_indices(intra_mask),
        "empty_inter_samples": _empty_mask_indices(inter_mask),
        "pair_mask": finite_stats("pair_mask", pair_mask),
        "terms": {name: finite_stats(name, value) for name, value in terms.items()},
        "loss": finite_stats("loss", loss),
        "module_findings": [],
        "grads": {},
        "params": {},
        "batch_shapes": {
            "p1_length": (
                residue_batch["p1_length"].detach().cpu().tolist()
                if "p1_length" in residue_batch
                else None
            ),
            "p2_length": (
                residue_batch["p2_length"].detach().cpu().tolist()
                if "p2_length" in residue_batch
                else None
            ),
            "mask_counts": (
                mask.to(dtype=torch.long).sum(dim=-1).detach().cpu().tolist()
                if torch.is_tensor(mask)
                else None
            ),
            "is_positive": (
                labels["is_positive"].detach().cpu().tolist()
                if "is_positive" in labels
                else None
            ),
        },
    }
    if model is not None:
        report["grads"] = {
            name: finite_stats(name, param.grad)
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        report["params"] = {
            name: finite_stats(name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        inputs_finite = all(stats.get("finite", True) for stats in inputs.values())
        if probe_modules and inputs_finite:
            report["module_findings"] = probe_first_nonfinite_module(
                model, residue_batch, recycle_rounds=recycle_rounds
            )
    report["cause"] = classify_nan_cause(report)
    return report


def format_nan_report(report: dict[str, Any]) -> str:
    lines = [
        "=== SurPass non-finite training step ===",
        f"cause: {report.get('cause')}",
        (
            f"stage={report.get('stage')} epoch={report.get('epoch')} "
            f"step={report.get('step')} rank={report.get('rank')} "
            f"lr={report.get('lr')} recycle={report.get('recycle_rounds')}"
        ),
        f"batch: {report.get('batch_shapes')}",
        "",
        "inputs:",
    ]
    for name, stats in report.get("inputs", {}).items():
        lines.append(
            f"  {name}: finite={stats['finite']} nan={stats['nan']} "
            f"inf={stats['inf']} abs_max={stats['abs_max']:.4g}"
        )
    logits = report.get("logits") or {}
    lines.extend(
        [
            "",
            (
                f"logits: finite={logits.get('finite')} nan={logits.get('nan')} "
                f"inf={logits.get('inf')} abs_max={logits.get('abs_max')}"
            ),
            f"nonfinite_logit_samples: {report.get('nonfinite_logit_samples')}",
            f"empty_residue_mask_samples: {report.get('empty_mask_samples')}",
            f"empty_intra_samples: {report.get('empty_intra_samples')}",
            f"empty_inter_samples: {report.get('empty_inter_samples')}",
            "",
            "loss terms:",
        ]
    )
    for name, stats in report.get("terms", {}).items():
        lines.append(
            f"  {name}: finite={stats['finite']} nan={stats['nan']} inf={stats['inf']}"
        )
    findings = report.get("module_findings") or []
    if findings:
        lines.extend(["", "module probe (first finite→Inf/NaN is the forward source):"])
        for item in findings:
            extra = f" error={item['error']}" if "error" in item else ""
            lines.append(
                f"  {item.get('name')} ({item.get('module')}): "
                f"true_risk={item.get('true_risk')} inf={item.get('out_inf')} "
                f"nan={item.get('out_nan')}{extra}"
            )
    bad_grads = [
        name
        for name, stats in (report.get("grads") or {}).items()
        if not stats.get("finite", True)
    ]
    if bad_grads:
        lines.extend(["", f"nonfinite grads ({len(bad_grads)}): {bad_grads[:12]}"])
    bad_params = [
        name
        for name, stats in (report.get("params") or {}).items()
        if not stats.get("finite", True)
    ]
    if bad_params:
        lines.extend(["", f"nonfinite params ({len(bad_params)}): {bad_params[:12]}"])
    lines.append("")
    return "\n".join(lines)


def _cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def dump_nan_report(
    report: dict[str, Any],
    residue_batch: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    dump_dir: str | Path | None,
    logits: torch.Tensor | None = None,
) -> tuple[Path | None, Path | None]:
    if dump_dir is None:
        return None, None
    directory = Path(dump_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(report.get("rank", DIST_WRAPPER.rank))
    epoch = int(report.get("epoch") or 0)
    step = int(report.get("step") or 0)
    stem = f"nan_rank{rank}_epoch{epoch}_step{step}"
    text_path = directory / f"{stem}.txt"
    dump_path = directory / f"{stem}.pt"
    text_path.write_text(format_nan_report(report), encoding="utf-8")
    torch.save(
        {
            "report": report,
            "residue_batch": _cpu_copy(residue_batch),
            "labels": _cpu_copy(labels),
            "logits": _cpu_copy(logits) if logits is not None else None,
        },
        dump_path,
    )
    return text_path, dump_path


def diagnose_nonfinite_step(
    *,
    residue_batch: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    intra_mask: torch.Tensor,
    inter_mask: torch.Tensor,
    terms: dict[str, torch.Tensor],
    loss: torch.Tensor,
    model: nn.Module | None = None,
    recycle_rounds: int = 1,
    stage: str = "loss",
    epoch: int | None = None,
    step: int | None = None,
    lr: float | None = None,
    dump_dir: str | Path | None = None,
    probe_modules: bool | None = None,
) -> dict[str, Any]:
    inputs_look_finite = all(
        finite_stats(key, residue_batch[key])["finite"]
        for key in INPUT_KEYS
        if key in residue_batch
    ) and all(
        finite_stats(key, labels[key])["finite"] for key in LABEL_KEYS if key in labels
    )
    if probe_modules is None:
        probe_modules = bool(model is not None and inputs_look_finite)
    report = collect_training_nan_report(
        residue_batch=residue_batch,
        labels=labels,
        logits=logits,
        pair_mask=pair_mask,
        intra_mask=intra_mask,
        inter_mask=inter_mask,
        terms=terms,
        loss=loss,
        model=model,
        recycle_rounds=recycle_rounds,
        stage=stage,
        epoch=epoch,
        step=step,
        lr=lr,
        probe_modules=probe_modules,
    )
    text = format_nan_report(report)
    print(text, flush=True)
    text_path, dump_path = dump_nan_report(
        report, residue_batch, labels, dump_dir, logits=logits
    )
    if text_path is not None:
        print(f"NaN diagnosis written to {text_path} and {dump_path}", flush=True)
    return report


def abort_if_nonfinite_loss(
    *,
    loss: torch.Tensor,
    residue_batch: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    intra_mask: torch.Tensor,
    inter_mask: torch.Tensor,
    terms: dict[str, torch.Tensor],
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    recycle_rounds: int = 1,
    epoch: int | None = None,
    step: int | None = None,
    dump_dir: str | Path | None = None,
) -> None:
    if torch.isfinite(loss).all().item():
        return
    lr = None
    if optimizer is not None and optimizer.param_groups:
        lr = float(optimizer.param_groups[0].get("lr", float("nan")))
    report = diagnose_nonfinite_step(
        residue_batch=residue_batch,
        labels=labels,
        logits=logits,
        pair_mask=pair_mask,
        intra_mask=intra_mask,
        inter_mask=inter_mask,
        terms=terms,
        loss=loss,
        model=model,
        recycle_rounds=recycle_rounds,
        stage="loss",
        epoch=epoch,
        step=step,
        lr=lr,
        dump_dir=dump_dir,
    )
    raise NonFiniteTrainingError(
        f"{report['cause']}. See dump under {dump_dir!s} for the replay batch."
    )
