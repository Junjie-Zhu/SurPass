"""Attach ESMC residue embeddings to prepared PDB or AFDB chain features.

PDB chain ids come from the heterodimer table, and each sequence is read from
that chain's processed pickle. AFDB sequences are read from the FASTA written
during domain preparation, so those pickles are opened only for coordinates.

Saved files live in ``<source>/embedded/<chain_id>.pt`` and keep the structure
tensors used by training, plus ``plm_emb`` of shape ``[length, hidden]``.

PDB files are plain ``torch.save`` archives with float32 embeddings. AFDB
files are gzip-compressed and store ``plm_emb`` as float16. Training reloads
that embedding as float32.

``--num-gpus`` or ``--devices`` splits the pending chains into contiguous shards
and embeds one shard per GPU, each in its own process.
"""

import argparse
import fcntl
import gzip
import io
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

DEFAULT_ROOT = Path("/groups/sbinlab/clschf/datasets")
DEFAULT_MODEL = "biohub/ESMC-6B"
SOURCE_SUBDIRS = {
    "pdb": "PDB_PPI",
    "afdb": "AFDB_DDI",
}
FEATURE_DTYPES = {
    "atom14_positions": torch.float32,
    "atom14_mask": torch.bool,
    "cb_positions": torch.float32,
    "cb_mask": torch.bool,
    "residue_type": torch.long,
    "residue_index": torch.long,
}
# Level 1 is enough: the payload is mostly float16 embeddings, which barely
# shrink further, and higher levels stall the embedding workers.
AFDB_GZIP_LEVEL = 1


def read_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    header: Optional[str] = None
    chunks: list[str] = []

    def _flush() -> None:
        if header is None:
            return
        sequence = _clean_sequence("".join(chunks))
        if not sequence:
            raise ValueError(f"FASTA record {header!r} has an empty sequence.")
        if header in sequences:
            raise ValueError(f"Duplicate FASTA id {header!r} in {fasta_path}.")
        sequences[header] = sequence

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header_text = line[1:].strip()
                if not header_text:
                    raise ValueError(f"FASTA header in {fasta_path} is missing an id.")
                header = header_text.split()[0]
                chunks = []
                continue
            if header is None:
                raise ValueError(f"FASTA sequence found before a header in {fasta_path}.")
            chunks.append(line)
    _flush()
    if not sequences:
        raise ValueError(f"No FASTA records found in {fasta_path}.")
    return sequences


def load_esmc(model_name: str, device: str):
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

    model = EsmcForMaskedLM.from_pretrained(model_name, device=device).eval()
    tokenizer = EsmcTokenizer()
    return model, tokenizer


def embed_sequences(model, tokenizer, sequences: list[str]) -> list[torch.Tensor]:
    """Residue embeddings for one batch, with BOS, EOS, and padding removed."""
    if not sequences:
        return []
    inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    attention_mask = inputs["attention_mask"]
    with torch.inference_mode():
        output = model(**inputs)
    hidden = _hidden_states(output)
    residues = []
    for index, sequence in enumerate(sequences):
        real = attention_mask[index].bool()
        tokens = hidden[index][real]
        if tokens.shape[0] < 2:
            raise ValueError(f"Sequence of length {len(sequence)} returned no BOS/EOS span.")
        matrix = tokens[1:-1].to(dtype=torch.float32, device="cpu").contiguous()
        if matrix.shape[0] != len(sequence):
            raise ValueError(
                f"ESMC residue rows ({matrix.shape[0]}) do not match sequence length "
                f"({len(sequence)})."
            )
        residues.append(matrix)
    return residues


def tensorize_feature_dict(feature_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    features = {}
    for key, value in feature_dict.items():
        tensor = _to_feature_tensor(key, value)
        if tensor is not None:
            features[key] = tensor
    return features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ESMC embeddings for prepared PDB or AFDB chains. "
            "Use --num-gpus or --devices to run one contiguous shard per GPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # single visible GPU (default)\n"
            "  python prepare_plm_embeddings.py --source pdb --skip-existing\n"
            "\n"
            "  # every GPU visible to this process\n"
            "  python prepare_plm_embeddings.py --source pdb --num-gpus -1 --skip-existing\n"
            "\n"
            "  # the first 4 GPUs, or an explicit subset\n"
            "  python prepare_plm_embeddings.py --source afdb --num-gpus 4 --batch-size 2\n"
            "  python prepare_plm_embeddings.py --source afdb --devices cuda:0,cuda:1,cuda:3\n"
        ),
    )
    parser.add_argument(
        "--source",
        choices=tuple(SOURCE_SUBDIRS),
        default="pdb",
        help="PDB reads sequences from processed pickles. AFDB reads the prepared FASTA.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory that contains the PDB_PPI and AFDB_DDI datasets.",
    )
    parser.add_argument("--pdb-subdir", default=SOURCE_SUBDIRS["pdb"])
    parser.add_argument("--afdb-subdir", default=SOURCE_SUBDIRS["afdb"])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Dataset directory. Defaults to <root-dir>/<pdb-subdir or afdb-subdir>.",
    )
    parser.add_argument(
        "--pairs-tsv",
        default="verified_heterodimers.tsv",
        help="PDB pair table, relative to the PDB directory, used to collect chain ids.",
    )
    parser.add_argument(
        "--fasta",
        default="processed_domains.fasta",
        help="AFDB sequence FASTA, relative to the AFDB directory or absolute.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Sequences per forward pass on each GPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Processes used to read PDB chain sequences from pickles.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
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
            "Split pending chains across this many GPUs (cuda:0 .. cuda:N-1), "
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
        help="Skip chains whose embedded .pt file already exists.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.workers < 1:
        raise ValueError("workers must be positive.")

    source_dir = _source_dir(args)
    processed_dir = source_dir / "processed"
    embedding_dir = source_dir / "embedded"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "pdb":
        pending, missing, skipped = _pdb_entries(
            source_dir,
            processed_dir,
            embedding_dir,
            args.pairs_tsv,
            skip_existing=args.skip_existing,
            workers=args.workers,
        )
    else:
        pending, missing, skipped = _afdb_entries(
            source_dir,
            embedding_dir,
            args.fasta,
            skip_existing=args.skip_existing,
        )
    entries_count = len(pending) + skipped + len(missing)

    compress = args.source == "afdb"
    print(f"source: {args.source}")
    print(f"source directory: {source_dir}")
    if compress:
        print("save format: gzip-compressed .pt, plm_emb float16")
    print(f"chains: {entries_count}")
    print(f"pending: {len(pending)}")
    print(f"skipped existing: {skipped}")
    print(f"missing inputs: {len(missing)}")

    devices = _resolve_inference_devices(args)
    errors = list(missing)
    if not pending:
        stats = _empty_stats()
    elif len(devices) == 1:
        device = devices[0]
        print(f"Loading {args.model_name} on {device}")
        model, tokenizer = load_esmc(args.model_name, device)
        stats, embed_errors = _embed_pending(
            pending,
            model,
            tokenizer,
            args.batch_size,
            processed_dir,
            embedding_dir,
            progress_desc="Extracting ESMC embeddings",
            compress=compress,
        )
        errors.extend(embed_errors)
    else:
        print(
            f"Splitting {len(pending)} pending chains across {len(devices)} GPUs: "
            + ", ".join(devices)
        )
        stats, embed_errors = _embed_on_devices(
            devices,
            pending,
            args.batch_size,
            args.model_name,
            processed_dir,
            embedding_dir,
            compress=compress,
        )
        errors.extend(embed_errors)

    print("\nEmbedding extraction summary")
    print(f"saved chains: {stats['saved']}")
    print(f"skipped existing: {skipped}")
    print(f"missing inputs: {len(missing)}")
    print(f"failed chains: {stats['failed']}")
    print(f"total embedded residues: {stats['residues']}")
    print(f"embedding dim: {stats['embedding_dim']}")
    print(f"output directory: {embedding_dir}")
    if errors:
        print("example errors:")
        for error in errors[:10]:
            print(f"  - {error}")


def _source_dir(args: argparse.Namespace) -> Path:
    if args.source_dir is not None:
        return Path(args.source_dir)
    subdir = args.pdb_subdir if args.source == "pdb" else args.afdb_subdir
    return Path(args.root_dir) / subdir


def _empty_stats() -> dict[str, Any]:
    return {
        "saved": 0,
        "failed": 0,
        "residues": 0,
        "embedding_dim": None,
    }


def _merge_stats(parts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged = _empty_stats()
    for part in parts:
        merged["saved"] += part["saved"]
        merged["failed"] += part["failed"]
        merged["residues"] += part["residues"]
        if part["embedding_dim"] is not None:
            merged["embedding_dim"] = part["embedding_dim"]
    return merged


def _contiguous_shards(items: list, num_shards: int) -> list[list]:
    """Split items into num_shards contiguous slices that differ in length by at most one."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive.")
    count = len(items)
    return [
        items[(count * index) // num_shards : (count * (index + 1)) // num_shards]
        for index in range(num_shards)
    ]


def _cuda_index(device: str) -> Optional[int]:
    if device == "cuda":
        return 0
    prefix = "cuda:"
    if not device.startswith(prefix):
        return None
    suffix = device[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError(
            f"Unrecognized CUDA device {device!r}. Expected 'cuda' or 'cuda:<index>'."
        )
    return int(suffix)


def _parse_device_list(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices did not contain any device names.")
    if len(devices) != len(set(devices)):
        raise ValueError("--devices contains a repeated device.")
    return devices


def _validate_devices(devices: list[str]) -> None:
    indexed = []
    for device in devices:
        index = _cuda_index(device)
        if index is not None:
            indexed.append((device, index))
    if not indexed:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        names = ", ".join(device for device, _ in indexed)
        raise RuntimeError(
            f"CUDA device(s) requested ({names}), but no GPU is visible. "
            "Check CUDA_VISIBLE_DEVICES and nvidia-smi."
        )
    available = torch.cuda.device_count()
    for device, index in indexed:
        if index >= available:
            raise RuntimeError(
                f"Device {device} is outside the {available} visible GPU(s). "
                "Check CUDA_VISIBLE_DEVICES and nvidia-smi."
            )


def _resolve_inference_devices(args: argparse.Namespace) -> list[str]:
    multi = args.devices or (args.num_gpus is not None and args.num_gpus != 1)
    if args.device and multi:
        raise ValueError(
            "--device selects one card. Omit it when using --num-gpus or --devices."
        )
    if args.devices and args.num_gpus is not None:
        raise ValueError("Pass only one of --devices and --num-gpus.")
    if args.devices:
        devices = _parse_device_list(args.devices)
    elif args.num_gpus is None or args.num_gpus == 1:
        devices = [args.device or ("cuda" if torch.cuda.is_available() else "cpu")]
    else:
        if args.num_gpus == 0 or args.num_gpus < -1:
            raise ValueError("--num-gpus must be positive, or -1 to use every visible GPU.")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError(
                "Multi-GPU embedding requires CUDA, but no GPU is visible. "
                "Check CUDA_VISIBLE_DEVICES and nvidia-smi."
            )
        available = torch.cuda.device_count()
        requested = available if args.num_gpus < 0 else args.num_gpus
        if requested > available:
            raise RuntimeError(
                f"Requested {requested} GPUs, but only {available} are visible. "
                "Check CUDA_VISIBLE_DEVICES and nvidia-smi."
            )
        devices = [f"cuda:{index}" for index in range(requested)]
    _validate_devices(devices)
    return devices


@contextmanager
def _exclusive_lock(path: Path):
    """Serialize ESMC loads on this node so host memory is not hit by every card at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _configure_worker_threads(num_devices: int) -> None:
    cpu_count = os.cpu_count() or 1
    threads = max(1, cpu_count // max(num_devices, 1))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _failed_shard(device: str, count: int, message: str) -> tuple[dict[str, Any], list[str]]:
    return (
        {"saved": 0, "failed": count, "residues": 0, "embedding_dim": None},
        [f"[{device}] {message}"],
    )


def _embed_pending(
    pending: list[tuple[str, str]],
    model,
    tokenizer,
    batch_size: int,
    processed_dir: Path,
    embedding_dir: Path,
    progress_desc: str,
    progress_position: Optional[int] = None,
    error_prefix: str = "",
    compress: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    stats = _empty_stats()
    errors: list[str] = []
    batches = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]
    progress_kwargs = {}
    if progress_position is not None:
        progress_kwargs["position"] = progress_position
    for batch in tqdm(batches, desc=progress_desc, **progress_kwargs):
        chain_ids = [chain_id for chain_id, _ in batch]
        sequences = [sequence for _, sequence in batch]
        try:
            embeddings = embed_sequences(model, tokenizer, sequences)
        except Exception as exc:
            stats["failed"] += len(batch)
            errors.append(f"{error_prefix}{', '.join(chain_ids)}: {type(exc).__name__}: {exc}")
            continue
        for chain_id, sequence, embedding in zip(chain_ids, sequences, embeddings):
            try:
                _save_chain_embedding(
                    chain_id,
                    sequence,
                    embedding,
                    processed_dir,
                    embedding_dir,
                    compress=compress,
                )
            except Exception as exc:
                stats["failed"] += 1
                errors.append(f"{error_prefix}{chain_id}: {type(exc).__name__}: {exc}")
                continue
            stats["saved"] += 1
            stats["residues"] += int(embedding.shape[0])
            stats["embedding_dim"] = int(embedding.shape[-1])
    return stats, errors


def _embed_shard(
    rank: int,
    device: str,
    pending: list[tuple[str, str]],
    batch_size: int,
    model_name: str,
    processed_dir: str,
    embedding_dir: str,
    num_devices: int,
    load_lock_path: str,
    compress: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Load one ESMC copy onto ``device`` and embed this shard."""
    if not pending:
        return _empty_stats(), []
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _configure_worker_threads(num_devices)
        if _cuda_index(device) is not None:
            torch.cuda.set_device(device)
        print(
            f"[shard {rank}] waiting to load {model_name} on {device} ({len(pending)} chains)",
            flush=True,
        )
        with _exclusive_lock(Path(load_lock_path)):
            print(f"[shard {rank}] loading {model_name} on {device}", flush=True)
            model, tokenizer = load_esmc(model_name, device)
        print(f"[shard {rank}] model ready on {device}", flush=True)
    except Exception as exc:
        return _failed_shard(device, len(pending), f"failed to start: {type(exc).__name__}: {exc}")
    return _embed_pending(
        pending,
        model,
        tokenizer,
        batch_size,
        Path(processed_dir),
        Path(embedding_dir),
        progress_desc=f"ESMC {device}",
        progress_position=rank if sys.stderr.isatty() else None,
        error_prefix=f"[{device}] ",
        compress=compress,
    )


def _embed_on_devices(
    devices: list[str],
    pending: list[tuple[str, str]],
    batch_size: int,
    model_name: str,
    processed_dir: Path,
    embedding_dir: Path,
    compress: bool,
) -> tuple[dict[str, Any], list[str]]:
    shards = _contiguous_shards(pending, len(devices))
    active = []
    for rank, (device, shard) in enumerate(zip(devices, shards)):
        if shard:
            print(
                f"shard {rank}: {device}, {len(shard)} chains "
                f"({shard[0][0]} .. {shard[-1][0]})"
            )
            active.append((rank, device, shard))
        else:
            print(f"shard {rank}: {device}, 0 chains")
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
                    str(processed_dir),
                    str(embedding_dir),
                    len(active),
                    str(lock_path),
                    compress,
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


def _read_pdb_sequence(
    task: tuple[str, str, str, bool],
) -> tuple[str, str | None, str | None, bool]:
    chain_id, processed_dir, embedding_dir, skip_existing = task
    if skip_existing and (Path(embedding_dir) / f"{chain_id}.pt").is_file():
        return chain_id, None, None, True
    feature_path = Path(processed_dir) / f"{chain_id}.pkl"
    if not feature_path.is_file():
        return chain_id, None, f"{chain_id}: missing feature file {feature_path}", False
    try:
        with feature_path.open("rb") as handle:
            feature_dict = pickle.load(handle)
        sequence = _clean_sequence(feature_dict["resseq"])
        if not sequence:
            raise ValueError("resseq is empty")
    except Exception as exc:
        return chain_id, None, f"{chain_id}: {type(exc).__name__}: {exc}", False
    return chain_id, sequence, None, False


def _pdb_entries(
    source_dir: Path,
    processed_dir: Path,
    embedding_dir: Path,
    pairs_tsv: str,
    skip_existing: bool,
    workers: int,
) -> tuple[list[tuple[str, str]], list[str], int]:
    pairs_path = _relative_or_absolute(source_dir, pairs_tsv)
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Cannot find PDB pair table: {pairs_path}")
    pairs = pd.read_csv(pairs_path, sep="\t")
    missing_columns = {"chain1", "chain2"}.difference(pairs.columns)
    if missing_columns:
        raise ValueError(f"{pairs_path} is missing columns: {sorted(missing_columns)}")
    chain_ids = sorted(
        set(pairs["chain1"].dropna().astype(str)) | set(pairs["chain2"].dropna().astype(str))
    )
    tasks = [
        (chain_id, str(processed_dir), str(embedding_dir), skip_existing)
        for chain_id in chain_ids
    ]
    if workers > 1 and len(tasks) > 1:
        chunksize = max(1, len(tasks) // (workers * 4))
        with mp.Pool(workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_read_pdb_sequence, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc="Reading PDB sequences",
                )
            )
    else:
        results = [
            _read_pdb_sequence(task) for task in tqdm(tasks, desc="Reading PDB sequences")
        ]

    entries = []
    errors = []
    skipped = 0
    for chain_id, sequence, error, was_skipped in results:
        if was_skipped:
            skipped += 1
            continue
        if error is not None:
            errors.append(error)
            continue
        entries.append((chain_id, sequence))
    return entries, errors, skipped


def _afdb_entries(
    source_dir: Path,
    embedding_dir: Path,
    fasta: str,
    skip_existing: bool,
) -> tuple[list[tuple[str, str]], list[str], int]:
    fasta_path = _relative_or_absolute(source_dir, fasta)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Cannot find AFDB FASTA: {fasta_path}")
    sequences = read_fasta_sequences(fasta_path)
    print(f"AFDB sequences: {fasta_path} ({len(sequences)} records)")
    entries = []
    skipped = 0
    for chain_id, sequence in sorted(sequences.items()):
        if skip_existing and (embedding_dir / f"{chain_id}.pt").is_file():
            skipped += 1
            continue
        entries.append((chain_id, sequence))
    return entries, [], skipped


def _save_chain_embedding(
    chain_id: str,
    sequence: str,
    embedding: torch.Tensor,
    processed_dir: Path,
    embedding_dir: Path,
    compress: bool = False,
) -> None:
    feature_path = processed_dir / f"{chain_id}.pkl"
    if not feature_path.is_file():
        raise FileNotFoundError(f"missing feature file {feature_path}")
    with feature_path.open("rb") as handle:
        feature_dict = pickle.load(handle)
    features = tensorize_feature_dict(feature_dict)
    feature_len = _infer_feature_length(features)
    if feature_len is None:
        raise ValueError("processed features have no residue axis")
    if feature_len != len(sequence) or feature_len != int(embedding.shape[0]):
        raise ValueError(
            f"feature length {feature_len}, sequence length {len(sequence)}, "
            f"and embedding length {int(embedding.shape[0])} do not match"
        )
    if compress:
        embedding = embedding.to(dtype=torch.float16).contiguous()
    features["plm_emb"] = embedding
    _write_embedded_file(features, embedding_dir / f"{chain_id}.pt", compress=compress)


def _write_embedded_file(
    features: dict[str, torch.Tensor],
    path: Path,
    compress: bool,
) -> None:
    """Write one chain file, replacing it only after the payload is complete."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if compress:
            buffer = io.BytesIO()
            torch.save(features, buffer)
            tmp_path.write_bytes(gzip.compress(buffer.getvalue(), compresslevel=AFDB_GZIP_LEVEL))
        else:
            torch.save(features, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _relative_or_absolute(source_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return source_dir / path


def _clean_sequence(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray)):
        text = value.decode()
    elif isinstance(value, np.ndarray) and value.shape == ():
        text = str(value.item())
    else:
        text = "".join(str(item) for item in list(value))
    return "".join(text.split()).upper()


def _hidden_states(output: object) -> torch.Tensor:
    for name in ("embeddings", "last_hidden_state"):
        value = output[name] if isinstance(output, dict) else getattr(output, name, None)
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            return value
    hidden_states = output.get("hidden_states") if isinstance(output, dict) else getattr(
        output, "hidden_states", None
    )
    if isinstance(hidden_states, (tuple, list)) and hidden_states:
        last = hidden_states[-1]
        if isinstance(last, torch.Tensor) and last.ndim == 3:
            return last
    raise RuntimeError("ESMC forward returned no residue hidden states.")


def _to_feature_tensor(key: str, value: Any) -> Optional[torch.Tensor]:
    if key == "chain_index":
        return _encode_chain_index(value)
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        return None
    dtype = FEATURE_DTYPES.get(key)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _encode_chain_index(chain_index: Any) -> torch.Tensor:
    if isinstance(chain_index, torch.Tensor):
        return chain_index.long()
    values = chain_index.tolist() if isinstance(chain_index, np.ndarray) else list(chain_index)
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return torch.tensor([mapping[value] for value in values], dtype=torch.long)


def _infer_feature_length(features: dict[str, torch.Tensor]) -> Optional[int]:
    for key in (
        "residue_type",
        "residue_index",
        "atom14_positions",
        "cb_positions",
        "cb_mask",
        "atom14_mask",
    ):
        value = features.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


if __name__ == "__main__":
    main()
