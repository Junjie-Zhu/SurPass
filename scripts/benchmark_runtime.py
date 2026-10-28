#!/usr/bin/env python3
"""Reproducible, model-only SurPass length and memory benchmark.

Synthetic PLM/residue features are constructed before timing. By default,
measure the configured ResPair model at 2,048 total residues, batch size one.
Training mode also measures loss, backward, clipping, and an optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import statistics
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LENGTHS = (2048,)
CSV_FIELDS = (
    "model_name",
    "architecture",
    "mode",
    "n_layers",
    "pair_update_every",
    "pair_update_layers",
    "recycle_passes",
    "combined_length",
    "p1_length",
    "p2_length",
    "batch_size",
    "precision",
    "device",
    "repeats",
    "warmup",
    "parameter_count",
    "trainable_parameter_count",
    "success",
    "error",
    "error_stage",
    "wall_time_seconds",
    "total_wall_time_seconds",
    "median_wall_time_seconds",
    "min_wall_time_seconds",
    "max_wall_time_seconds",
    "seconds_per_sample",
    "samples_per_second",
    "iteration_times_seconds",
    "warmup_times_seconds",
    "peak_allocated_cuda_bytes",
    "peak_reserved_cuda_bytes",
    "output_shapes",
    "last_loss",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check ResPair time and memory at 2,048 total residues by default. "
            "Synthetic PLM inputs exclude embedding generation and data loading."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "inference.yaml",
        help="Hydra YAML config (default: configs/inference.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint. Without it, seeded random weights are benchmarked.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Combined chain lengths (default: 2048 = 1024 + 1024).",
    )
    parser.add_argument("--p1-length", type=int, help="First-chain length; default: half of each total.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("inference", "training"), default="inference",
        help="Forward only, or forward/loss/backward/clipping/optimizer (default: inference).",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Repeat for Hydra overrides, e.g. --override model.triangle_chunk_size=32.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="Auto follows the config on CUDA and uses FP32 on CPU. BF16 requires CUDA.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms.",
    )
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help=(
            "Use CPU, FP32, a tiny model, and length 16 (unless --lengths is set) "
            "for a quick installation smoke test."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_runtime.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_runtime.json"),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    lengths = args.lengths if args.lengths is not None else ([16] if args.cpu_smoke else DEFAULT_LENGTHS)
    if any(length < 2 for length in lengths):
        raise SystemExit("Every combined length must be at least 2.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive.")
    if args.gpu_id < 0:
        raise SystemExit("--gpu-id must be non-negative.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    if args.p1_length is not None and any(not 0 < args.p1_length < length for length in lengths):
        raise SystemExit("--p1-length must leave at least one residue in each chain.")


def _load_model_api():
    """Import change-prone SurPass APIs only when a benchmark is executed."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.runtime.config import build_model_kwargs, create_model

    return create_model, build_model_kwargs


def _load_config(args: argparse.Namespace):
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit(
            "OmegaConf is required. Install the SurPass environment before benchmarking."
        ) from exc

    if not args.config.is_file():
        raise SystemExit(f"Missing SurPass config: {args.config}")
    cfg = OmegaConf.load(args.config)
    if "defaults" in cfg:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(args.config.resolve().parent), version_base="1.3"):
            cfg = compose(config_name=args.config.stem, overrides=args.override)
    elif args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    if args.cpu_smoke:
        cfg = OmegaConf.merge(
            cfg,
            {
                "model": {
                    "n_layers": 2,
                    "dim_token": 32,
                    "dim_pair": 16,
                    "n_heads": 4,
                    "plm_in_dim": 32,
                    "plm_out_dim": 16,
                    "index_emb_dim": 8,
                    "allow_missing_physics": True,
                }
            },
        )
        if str(cfg.model.get("name", "ResOnly")) == "ResPair":
            cfg.model.triangle_hidden_dim = 8
            cfg.model.triangle_chunk_size = 8
    return cfg


def _resolve_device(args: argparse.Namespace, torch):
    requested = "cpu" if args.cpu_smoke else args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
        if args.gpu_id >= torch.cuda.device_count():
            raise SystemExit(
                f"--gpu-id {args.gpu_id} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)."
            )
        torch.cuda.set_device(args.gpu_id)
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cpu")


def _checkpoint_state(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a state dict or mapping containing one")
    for key in ("model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return payload


def _load_checkpoint(model, checkpoint: Path, device, torch) -> None:
    if not checkpoint.is_file():
        raise SystemExit(f"Missing SurPass checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture") if isinstance(payload, Mapping) else None
    if architecture and architecture != getattr(model, "architecture", None):
        raise ValueError(f"Checkpoint architecture {architecture!r} does not match {model.architecture!r}")
    state = _checkpoint_state(payload)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def _synthetic_batch(
    combined_length: int,
    plm_in_dim: int,
    seed: int,
    device,
    torch,
    *,
    batch_size: int = 1,
    p1_length: int | None = None,
) -> dict[str, Any]:
    p1_length = combined_length // 2 if p1_length is None else p1_length
    p2_length = combined_length - p1_length
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(combined_length))
    batch = {
        "plm_emb": torch.randn(
            batch_size,
            combined_length,
            plm_in_dim,
            generator=generator,
            dtype=torch.float32,
        ),
        "residue_type": torch.randint(
            0,
            20,
            (batch_size, combined_length),
            generator=generator,
            dtype=torch.long,
        ),
        "residue_index": torch.arange(combined_length, dtype=torch.long)[None].expand(batch_size, -1),
        "chain_index": torch.cat(
            [
                torch.zeros(p1_length, dtype=torch.long),
                torch.ones(p2_length, dtype=torch.long),
            ]
        )[None].expand(batch_size, -1),
        "mask": torch.ones(batch_size, combined_length, dtype=torch.bool),
        "padding_mask": torch.ones(batch_size, combined_length, dtype=torch.bool),
        "p1_length": torch.full((batch_size,), p1_length, dtype=torch.long),
        "p2_length": torch.full((batch_size,), p2_length, dtype=torch.long),
    }
    return {key: value.to(device) for key, value in batch.items()}


def _output_shapes(value: Any, torch, seen: set[int] | None = None) -> Any:
    """Describe tensors in dict/tuple/dataclass outputs without copying them."""
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if value is None or isinstance(value, (str, int, float, bool)):
        return None
    if seen is None:
        seen = set()
    if id(value) in seen:
        return "<recursive>"
    seen.add(id(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _output_shapes(getattr(value, field.name), torch, seen)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _output_shapes(item, torch, seen)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_output_shapes(item, torch, seen) for item in value]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _output_shapes(item, torch, seen)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return type(value).__name__


def _write_results(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    csv_path: Path,
    json_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key, value in csv_row.items():
                if isinstance(value, (dict, list, tuple)):
                    csv_row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            writer.writerow(csv_row)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "results": rows},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _autocast_context(precision: str, device, torch):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _training_components(model, cfg, device, torch):
    from src.model.loss import SurpassLoss
    from src.model.optimizer import get_adamw, get_optimizer
    from src.runtime.config import cfg_get

    weights = {name: float(cfg_get(cfg, f"loss.{name}.weight", default)) for name, default in (
        ("protein", 1.0), ("contact", 1.0), ("distance", 0.1), ("aux_protein", 0.1),
    )}
    criterion = SurpassLoss(
        model.contact_bins, weights,
        focal_gamma=float(cfg_get(cfg, "loss.contact.gamma", 1.5)),
        contact_positive_weight=float(cfg_get(cfg, "loss.contact.positive_weight", 4.0)),
        negatives_per_positive=float(cfg_get(cfg, "loss.contact.negatives_per_positive", 8.0)),
    ).to(device)
    lr = float(cfg_get(cfg, "optimizer.lr", 0.0005))
    weight_decay = float(cfg_get(cfg, "optimizer.weight_decay", 1e-6))
    betas = (float(cfg_get(cfg, "optimizer.beta1", 0.9)), float(cfg_get(cfg, "optimizer.beta2", 0.999)))
    if bool(cfg_get(cfg, "optimizer.use_adamw", True)):
        optimizer = get_adamw(model, weight_decay, lr, betas, device.type)
    else:
        optimizer = get_optimizer(model, lr, weight_decay, betas, use_adamw=False)
    return criterion, optimizer, float(cfg_get(cfg, "optimizer.max_grad_norm", 1.0))


def _synthetic_labels(batch, bin_count, torch):
    batch_size = batch["plm_emb"].shape[0]
    first, second = int(batch["p1_length"][0]), int(batch["p2_length"][0])
    device = batch["plm_emb"].device
    positive = torch.arange(batch_size, device=device) % 2 == 0
    bins = torch.full((batch_size, first, second), bin_count - 1, dtype=torch.long, device=device)
    bins[positive, 0, 0] = 0
    return {
        "label_inter_bins": bins,
        "label_inter_mask": torch.ones_like(bins, dtype=torch.bool),
        "crop_is_positive": positive, "source_is_positive": positive,
        "sample_confidence": torch.ones(batch_size, device=device),
    }


def _run_iteration(model, batch, labels, criterion, optimizer, max_grad_norm, precision, device, torch):
    if optimizer is None:
        with torch.inference_mode(), _autocast_context(precision, device, torch):
            return model(batch), None
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(precision, device, torch):
        output = model(batch)
        loss, _ = criterion(output, labels)
    loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return output, loss.detach()


def _synchronize(device, torch):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _capture_memory(row, device, torch):
    if device.type == "cuda":
        row["peak_allocated_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
        row["peak_reserved_cuda_bytes"] = torch.cuda.max_memory_reserved(device)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required. Install the SurPass environment before benchmarking.") from exc

    cfg = _load_config(args)
    device = _resolve_device(args, torch)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.cpu_smoke:
        precision = "fp32"
    elif args.precision == "auto":
        configured = str(cfg.get("runtime", {}).get("precision", "float32"))
        precision = "bf16" if device.type == "cuda" and configured == "bfloat16" else "fp32"
    else:
        precision = args.precision
    if precision == "bf16":
        if device.type != "cuda":
            raise SystemExit("BF16 benchmarking is supported only with --device cuda.")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("The selected CUDA device does not support BF16.")

    lengths = list(args.lengths) if args.lengths is not None else ([16] if args.cpu_smoke else list(DEFAULT_LENGTHS))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    create_model, build_model_kwargs = _load_model_api()
    model_kwargs = build_model_kwargs(cfg)
    print(f"Creating {cfg.model.get('name', 'ResOnly')} from {args.config} on {device} ({precision})...", flush=True)
    model = create_model(cfg).to(device)
    if args.checkpoint is not None:
        _load_checkpoint(model, args.checkpoint, device, torch)
    model.train(args.mode == "training")
    criterion = optimizer = None
    max_grad_norm = 0.0
    if args.mode == "training":
        criterion, optimizer, max_grad_norm = _training_components(model, cfg, device, torch)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    pair_update_layers = [int(key) + 1 for key in model.pair_blocks] if hasattr(model.pair_blocks, "keys") else list(range(1, model.n_layers + 1))
    model_info = {
        "model_name": type(model).__name__, "architecture": model.architecture,
        "mode": args.mode, "n_layers": model.n_layers,
        "pair_update_every": getattr(model, "pair_update_every", 1),
        "pair_update_layers": pair_update_layers,
        "recycle_passes": getattr(model, "recycle_passes", 1),
    }
    print(f"Model ready: parameters={parameter_count:,}, layers={model.n_layers}, "
          f"pair updates={pair_update_layers}, passes={model_info['recycle_passes']}, "
          f"mode={args.mode}, batch={args.batch_size}", flush=True)
    metadata = {
        **model_info,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "seed": args.seed, "deterministic": bool(args.deterministic),
        "cpu_smoke": bool(args.cpu_smoke), "overrides": args.override,
        "batch_size": args.batch_size, "lengths": lengths, "precision": precision,
        "device": str(device), "parameter_count": parameter_count,
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        "measurement_scope": (
            "forward + loss + backward + gradient clipping + optimizer step; excludes EMA, DDP, scheduler and data preparation"
            if args.mode == "training" else "model forward only; excludes PLM generation and data preparation"
        ),
        "memory_scope": "CUDA process peaks include model, inputs and warmup; CPU peaks are unavailable",
        "model_kwargs": {str(key): value for key, value in model_kwargs.items()
                         if isinstance(value, (str, int, float, bool, type(None)))},
    }

    rows = []
    for combined_length in lengths:
        p1_length = args.p1_length if args.p1_length is not None else combined_length // 2
        row = {key: None for key in CSV_FIELDS}
        row.update({
            **model_info,
            "combined_length": combined_length, "p1_length": p1_length,
            "p2_length": combined_length - p1_length, "batch_size": args.batch_size,
            "precision": precision, "device": str(device), "repeats": args.repeats,
            "warmup": args.warmup, "parameter_count": parameter_count,
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "success": False, "error": "", "error_stage": "",
            "iteration_times_seconds": [], "warmup_times_seconds": [],
        })
        batch = labels = output = loss = None
        stage = "input allocation"
        print(f"Testing total length={combined_length} ({p1_length} + {combined_length - p1_length})...", flush=True)
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            batch = _synthetic_batch(
                combined_length, int(model_kwargs.get("plm_in_dim", 2560)), args.seed, device, torch,
                batch_size=args.batch_size, p1_length=p1_length,
            )
            if args.mode == "training":
                labels = _synthetic_labels(batch, model.distance_bin_count, torch)
            for phase, count in (("warmup", args.warmup), ("timed", args.repeats)):
                for index in range(count):
                    stage = f"{phase} iteration {index + 1}/{count}"
                    print(f"  {stage} starting...", flush=True)
                    # Release the previous outputs before measuring another pass.
                    output = loss = None
                    _synchronize(device, torch)
                    start = time.perf_counter()
                    output, loss = _run_iteration(
                        model, batch, labels, criterion, optimizer, max_grad_norm, precision, device, torch,
                    )
                    _synchronize(device, torch)
                    elapsed = time.perf_counter() - start
                    key = "warmup_times_seconds" if phase == "warmup" else "iteration_times_seconds"
                    row[key].append(elapsed)
                    print(f"  {stage}: {elapsed:.4f} s", flush=True)
            _capture_memory(row, device, torch)
            stage = "output validation"
            if not bool(torch.isfinite(output.protein_logit).all()) or not bool(torch.isfinite(output.aux_protein_logit).all()):
                raise FloatingPointError("Model produced non-finite protein logits")
            if loss is not None:
                row["last_loss"] = float(loss)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Training loss is non-finite")
            times = row["iteration_times_seconds"]
            mean = statistics.mean(times)
            row.update({
                "success": True, "wall_time_seconds": mean, "total_wall_time_seconds": sum(times),
                "median_wall_time_seconds": statistics.median(times),
                "min_wall_time_seconds": min(times), "max_wall_time_seconds": max(times),
                "seconds_per_sample": mean / args.batch_size, "samples_per_second": args.batch_size / mean,
                "output_shapes": _output_shapes(output, torch),
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["error_stage"] = stage
            _capture_memory(row, device, torch)
        finally:
            output = loss = batch = labels = None
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(row)
        _write_results(rows, metadata, args.output_csv, args.output_json)
        if row["success"]:
            print(f"PASS length={combined_length}: mean={row['wall_time_seconds']:.4f} s/batch, "
                  f"median={row['median_wall_time_seconds']:.4f} s, "
                  f"per sample={row['seconds_per_sample']:.4f} s", flush=True)
        else:
            print(f"FAIL length={combined_length} during {row['error_stage']}: {row['error']}", flush=True)
        if device.type == "cuda":
            print(f"  Peak CUDA allocated={row['peak_allocated_cuda_bytes'] / 2**30:.3f} GiB, "
                  f"reserved={row['peak_reserved_cuda_bytes'] / 2**30:.3f} GiB", flush=True)

    print(f"Wrote {args.output_csv}", flush=True)
    print(f"Wrote {args.output_json}", flush=True)
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
