"""Extract ESMC residue embeddings from a FASTA file for SurPass inference.

Saved files live in ``<output-dir>/<protein_id>.pt`` and contain ``plm_emb``
of shape ``[length, hidden]`` plus the source ``sequence``.

``--num-gpus`` or ``--devices`` splits the pending proteins into contiguous
shards and embeds one shard per GPU, each in its own process.
"""

import argparse
import multiprocessing as mp
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_plm_embeddings import (
    DEFAULT_MODEL,
    _configure_worker_threads,
    _contiguous_shards,
    _cuda_index,
    _empty_stats,
    _exclusive_lock,
    _failed_shard,
    _merge_stats,
    _resolve_inference_devices,
    embed_sequences,
    load_esmc,
)


def parse_fasta(fasta_path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    protein_id: str | None = None
    chunks: list[str] = []

    def _flush() -> None:
        if protein_id is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"FASTA record {protein_id} has an empty sequence.")
        records.append((protein_id, sequence))

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header = line[1:].strip()
                if not header:
                    raise ValueError("FASTA header is missing a sequence ID.")
                protein_id = header.split()[0]
                chunks = []
                continue
            if protein_id is None:
                raise ValueError("FASTA sequence found before a header.")
            chunks.append(line)

    _flush()
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}.")
    return records


def resolve_output_dir(
    root_dir: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(root_dir) / "embeddings"


def save_embedding(
    output_dir: str | Path,
    protein_id: str,
    embedding: torch.Tensor,
    sequence: str,
) -> Path:
    output_path = Path(output_dir) / f"{protein_id}.pt"
    payload = {
        "plm_emb": embedding.to(dtype=torch.float32).cpu(),
        "sequence": sequence,
    }
    torch.save(payload, output_path)
    return output_path


def _select_pending(
    records: list[tuple[str, str]],
    output_dir: str | Path,
    skip_existing: bool,
) -> tuple[list[tuple[str, str]], int]:
    pending: list[tuple[str, str]] = []
    skipped = 0
    output_path = Path(output_dir)
    for protein_id, sequence in records:
        dest = output_path / f"{protein_id}.pt"
        if skip_existing and dest.exists():
            skipped += 1
            continue
        pending.append((protein_id, sequence))
    return pending, skipped


def _embed_records(
    records: list[tuple[str, str]],
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    batch_size: int,
    progress_desc: str,
    progress_position: int | None = None,
    error_prefix: str = "",
) -> tuple[dict[str, Any], list[str]]:
    stats = _empty_stats()
    errors: list[str] = []
    batches = [
        records[start : start + batch_size] for start in range(0, len(records), batch_size)
    ]
    progress_kwargs: dict[str, Any] = {}
    if progress_position is not None:
        progress_kwargs["position"] = progress_position
    for batch in tqdm(batches, desc=progress_desc, **progress_kwargs):
        protein_ids = [protein_id for protein_id, _ in batch]
        sequences = [sequence for _, sequence in batch]
        try:
            embeddings = embed_sequences(model, tokenizer, sequences)
        except Exception as exc:
            stats["failed"] += len(batch)
            errors.append(
                f"{error_prefix}{', '.join(protein_ids)}: {type(exc).__name__}: {exc}"
            )
            continue
        for protein_id, sequence, embedding in zip(protein_ids, sequences, embeddings):
            try:
                if int(embedding.shape[0]) != len(sequence):
                    raise ValueError(
                        f"embedding length {int(embedding.shape[0])} does not match "
                        f"sequence length {len(sequence)}"
                    )
                save_embedding(output_dir, protein_id, embedding, sequence)
            except Exception as exc:
                stats["failed"] += 1
                errors.append(f"{error_prefix}{protein_id}: {type(exc).__name__}: {exc}")
                continue
            stats["saved"] += 1
            stats["residues"] += int(embedding.shape[0])
            stats["embedding_dim"] = int(embedding.shape[-1])
    return stats, errors


def _embed_shard(
    rank: int,
    device: str,
    records: list[tuple[str, str]],
    batch_size: int,
    model_name: str,
    output_dir: str,
    num_devices: int,
    load_lock_path: str,
) -> tuple[dict[str, Any], list[str]]:
    """Load one ESMC copy onto ``device`` and embed this shard."""
    if not records:
        return _empty_stats(), []
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _configure_worker_threads(num_devices)
        if _cuda_index(device) is not None:
            torch.cuda.set_device(device)
        print(
            f"[shard {rank}] waiting to load {model_name} on {device} ({len(records)} proteins)",
            flush=True,
        )
        with _exclusive_lock(Path(load_lock_path)):
            print(f"[shard {rank}] loading {model_name} on {device}", flush=True)
            model, tokenizer = load_esmc(model_name, device)
        print(f"[shard {rank}] model ready on {device}", flush=True)
    except Exception as exc:
        return _failed_shard(device, len(records), f"failed to start: {type(exc).__name__}: {exc}")
    return _embed_records(
        records,
        model,
        tokenizer,
        output_dir,
        batch_size,
        progress_desc=f"ESMC {device}",
        progress_position=rank if sys.stderr.isatty() else None,
        error_prefix=f"[{device}] ",
    )


def _embed_on_devices(
    devices: list[str],
    records: list[tuple[str, str]],
    batch_size: int,
    model_name: str,
    output_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    shards = _contiguous_shards(records, len(devices))
    active = []
    for rank, (device, shard) in enumerate(zip(devices, shards)):
        if shard:
            print(
                f"shard {rank}: {device}, {len(shard)} proteins "
                f"({shard[0][0]} .. {shard[-1][0]})"
            )
            active.append((rank, device, shard))
        else:
            print(f"shard {rank}: {device}, 0 proteins")
    if not active:
        return _empty_stats(), []

    # Spawn so each GPU gets a fresh interpreter. Forking after CUDA init is unsafe.
    ctx = mp.get_context("spawn")
    lock_path = Path(tempfile.gettempdir()) / f"surpass_esmc_load_{os.getpid()}.lock"
    lock_path.touch()
    parts: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with ProcessPoolExecutor(max_workers=len(active), mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    _embed_shard,
                    rank,
                    device,
                    shard,
                    batch_size,
                    model_name,
                    str(output_dir),
                    len(active),
                    str(lock_path),
                )
                for rank, device, shard in active
            ]
            for (rank, device, shard), future in zip(active, futures):
                try:
                    stats, shard_errors = future.result()
                except Exception as exc:
                    stats, shard_errors = _failed_shard(
                        device,
                        len(shard),
                        f"worker failed: {type(exc).__name__}: {exc}",
                    )
                    print(f"[shard {rank}] {shard_errors[0]}", flush=True)
                parts.append(stats)
                errors.extend(shard_errors)
    finally:
        lock_path.unlink(missing_ok=True)
    return _merge_stats(parts), errors


def extract_embeddings(
    fasta_path: str | Path,
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    skip_existing: bool = False,
    batch_size: int = 1,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records = parse_fasta(fasta_path)
    pending, skipped = _select_pending(records, output_path, skip_existing)
    stats = _summary_stats(len(records), skipped)
    if not pending:
        return stats
    embed_stats, errors = _embed_records(
        pending,
        model,
        tokenizer,
        output_path,
        batch_size,
        progress_desc="Extracting inference embeddings",
    )
    _apply_embed_stats(stats, embed_stats)
    stats["errors"] = errors
    return stats


def _summary_stats(records: int, skipped: int) -> dict[str, Any]:
    stats = _empty_stats()
    stats["records"] = records
    stats["skipped"] = skipped
    stats["errors"] = []
    return stats


def _apply_embed_stats(stats: dict[str, Any], embed_stats: dict[str, Any]) -> None:
    for key in ("saved", "failed", "residues", "embedding_dim"):
        stats[key] = embed_stats[key]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ESMC embeddings from a FASTA file for SurPass inference. "
            "Use --num-gpus or --devices to run one contiguous shard per GPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # single visible GPU (default)\n"
            "  python prepare_inference_embeddings.py --fasta proteins.fasta --skip-existing\n"
            "\n"
            "  # every GPU visible to this process\n"
            "  python prepare_inference_embeddings.py --fasta proteins.fasta "
            "--num-gpus -1 --skip-existing\n"
            "\n"
            "  # the first 4 GPUs, or an explicit subset\n"
            "  python prepare_inference_embeddings.py --fasta proteins.fasta "
            "--num-gpus 4 --batch-size 2\n"
            "  python prepare_inference_embeddings.py --fasta proteins.fasta "
            "--devices cuda:0,cuda:1,cuda:3\n"
        ),
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file.")
    parser.add_argument("--root-dir", default="./pdb_metadata", help="Dataset root directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Embedding output directory. Defaults to <root-dir>/embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Sequences per forward pass on each GPU.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Single device. Defaults to cuda when CUDA is visible, otherwise cpu.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Split pending proteins across this many GPUs (cuda:0 .. cuda:N-1), "
            "one process per card. Use -1 for every visible GPU. "
            "Omit to keep single-device inference."
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices, one shard each, for example cuda:0,cuda:1,cuda:3.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip proteins whose .pt file already exists.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")

    devices = _resolve_inference_devices(args)
    output_dir = resolve_output_dir(args.root_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = parse_fasta(args.fasta)
    pending, skipped = _select_pending(records, output_dir, args.skip_existing)

    print(f"fasta records: {len(records)}")
    print(f"pending: {len(pending)}")
    print(f"skipped existing: {skipped}")

    stats = _summary_stats(len(records), skipped)
    errors: list[str] = []
    if pending and len(devices) == 1:
        device = devices[0]
        print(f"Loading {DEFAULT_MODEL} on {device}")
        model, tokenizer = load_esmc(DEFAULT_MODEL, device)
        embed_stats, errors = _embed_records(
            pending,
            model,
            tokenizer,
            output_dir,
            args.batch_size,
            progress_desc="Extracting inference embeddings",
        )
        _apply_embed_stats(stats, embed_stats)
    elif pending:
        print(
            f"Splitting {len(pending)} pending proteins across {len(devices)} GPUs: "
            + ", ".join(devices)
        )
        embed_stats, errors = _embed_on_devices(
            devices,
            pending,
            args.batch_size,
            DEFAULT_MODEL,
            output_dir,
        )
        _apply_embed_stats(stats, embed_stats)

    print("\nEmbedding extraction summary")
    print(f"fasta records: {stats['records']}")
    print(f"saved chains: {stats['saved']}")
    print(f"skipped existing: {stats['skipped']}")
    print(f"failed chains: {stats['failed']}")
    print(f"total embedded residues: {stats['residues']}")
    print(f"embedding dim: {stats['embedding_dim']}")
    print(f"output directory: {output_dir}")
    if errors:
        print("example errors:")
        for error in errors[:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
