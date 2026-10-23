#!/usr/bin/env python3
"""Reproducible, model-only SurPass length and memory benchmark.

Synthetic PLM/residue features are constructed before timing. The measured wall
time therefore covers only ``ResOnly.forward`` for batch size one.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import platform
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LENGTHS = (256, 512, 1024, 2048)
CSV_FIELDS = (
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
    "wall_time_seconds",
    "total_wall_time_seconds",
    "peak_allocated_cuda_bytes",
    "peak_reserved_cuda_bytes",
    "output_shapes",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark model-only SurPass forwards at combined protein lengths. "
            "Inputs are prepared outside the timed region; batch size is fixed at 1."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "inference.yaml",
        help="SurPass YAML config (default: configs/inference.yaml).",
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
        help="Combined lengths (default: 256 512 1024 2048).",
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
        choices=("fp32", "bf16"),
        default="fp32",
        help="Forward precision. BF16 requires CUDA.",
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
    lengths = args.lengths if args.lengths is not None else DEFAULT_LENGTHS
    if any(length < 2 for length in lengths):
        raise SystemExit("Every combined length must be at least 2.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive.")
    if args.gpu_id < 0:
        raise SystemExit("--gpu-id must be non-negative.")


def _load_model_api():
    """Import change-prone SurPass APIs only when a benchmark is executed."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.model.surpass import ResOnly
    from src.runtime.config import build_model_kwargs

    return ResOnly, build_model_kwargs


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
    if args.cpu_smoke:
        cfg = OmegaConf.merge(
            cfg,
            {
                "model": {
                    "n_layers": 1,
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
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
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
) -> dict[str, Any]:
    p1_length = combined_length // 2
    p2_length = combined_length - p1_length
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(combined_length))
    batch = {
        "plm_emb": torch.randn(
            1,
            combined_length,
            plm_in_dim,
            generator=generator,
            dtype=torch.float32,
        ),
        "residue_type": torch.randint(
            0,
            20,
            (1, combined_length),
            generator=generator,
            dtype=torch.long,
        ),
        "residue_index": torch.arange(combined_length, dtype=torch.long).unsqueeze(0),
        "residue_position": torch.zeros(
            1, combined_length, 3, dtype=torch.float32
        ),
        "chain_index": torch.cat(
            [
                torch.zeros(p1_length, dtype=torch.long),
                torch.ones(p2_length, dtype=torch.long),
            ]
        ).unsqueeze(0),
        "mask": torch.ones(1, combined_length, dtype=torch.bool),
        "p1_length": torch.tensor([p1_length], dtype=torch.long),
        "p2_length": torch.tensor([p2_length], dtype=torch.long),
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
            csv_row["output_shapes"] = json.dumps(
                row["output_shapes"], sort_keys=True, separators=(",", ":")
            )
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


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Install the SurPass environment before benchmarking."
        ) from exc

    if args.cpu_smoke:
        args.precision = "fp32"
    device = _resolve_device(args, torch)
    if args.precision == "bf16":
        if device.type != "cuda":
            raise SystemExit("BF16 benchmarking is supported only with --device cuda.")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise SystemExit("The selected CUDA device does not support BF16.")

    lengths = (
        list(args.lengths)
        if args.lengths is not None
        else ([16] if args.cpu_smoke else list(DEFAULT_LENGTHS))
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    cfg = _load_config(args)
    ResOnly, build_model_kwargs = _load_model_api()
    model_kwargs = build_model_kwargs(cfg)
    model = ResOnly(**model_kwargs).to(device)
    if args.checkpoint is not None:
        _load_checkpoint(model, args.checkpoint, device, torch)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    precision = "fp32" if args.cpu_smoke else args.precision
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        "config": str(args.config.resolve()),
        "checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "batch_size": 1,
        "lengths": lengths,
        "precision": precision,
        "device": str(device),
        "parameter_count": parameter_count,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "model_kwargs": {
            str(key): value
            for key, value in model_kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }

    rows: list[dict[str, Any]] = []
    plm_in_dim = int(model_kwargs.get("plm_in_dim", 1280))
    for combined_length in lengths:
        p1_length = combined_length // 2
        row: dict[str, Any] = {
            "combined_length": combined_length,
            "p1_length": p1_length,
            "p2_length": combined_length - p1_length,
            "batch_size": 1,
            "precision": precision,
            "device": str(device),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "success": False,
            "error": "",
            "wall_time_seconds": None,
            "total_wall_time_seconds": None,
            "peak_allocated_cuda_bytes": None,
            "peak_reserved_cuda_bytes": None,
            "output_shapes": None,
        }
        batch = None
        output = None
        try:
            batch = _synthetic_batch(
                combined_length,
                plm_in_dim,
                args.seed,
                device,
                torch,
            )
            with torch.inference_mode(), _autocast_context(precision, device, torch):
                for _ in range(args.warmup):
                    output = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                for _ in range(args.repeats):
                    output = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start

            row["success"] = True
            row["wall_time_seconds"] = elapsed / args.repeats
            row["total_wall_time_seconds"] = elapsed
            row["output_shapes"] = _output_shapes(output, torch)
            if device.type == "cuda":
                row["peak_allocated_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
                row["peak_reserved_cuda_bytes"] = torch.cuda.max_memory_reserved(device)
        except Exception as exc:  # Keep later, potentially smaller/diagnostic rows.
            row["error"] = f"{type(exc).__name__}: {exc}"
            if device.type == "cuda":
                row["peak_allocated_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
                row["peak_reserved_cuda_bytes"] = torch.cuda.max_memory_reserved(device)
        finally:
            del output
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(row)
        _write_results(rows, metadata, args.output_csv, args.output_json)
        status = "ok" if row["success"] else row["error"]
        print(f"length={combined_length} status={status}")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
