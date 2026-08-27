import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_plm_embeddings import generate_embeddings_from_request


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


def extract_embeddings(
    fasta_path: str | Path,
    output_dir: str | Path,
    model: Any,
    device: Any,
    skip_existing: bool = False,
    batch_preparer: Any = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records = parse_fasta(fasta_path)
    stats = {
        "records": len(records),
        "saved": 0,
        "skipped": 0,
        "failed": 0,
        "residues": 0,
        "embedding_dim": None,
    }
    errors: list[str] = []

    for protein_id, sequence in tqdm(records, desc="Extracting inference embeddings"):
        dest = output_path / f"{protein_id}.pt"
        if skip_existing and dest.exists():
            stats["skipped"] += 1
            continue
        try:
            embedding = generate_embeddings_from_request(
                request_payload={"chain_id": protein_id, "sequence": sequence},
                model=model,
                device=device,
                msa_directory="",
                embed_mode="single",
                batch_preparer=batch_preparer,
            )
            if int(embedding.shape[0]) != len(sequence):
                raise ValueError(
                    f"embedding length {int(embedding.shape[0])} does not match "
                    f"sequence length {len(sequence)}"
                )
            save_embedding(output_path, protein_id, embedding, sequence)
            stats["saved"] += 1
            stats["residues"] += int(embedding.shape[0])
            stats["embedding_dim"] = int(embedding.shape[-1])
        except Exception as exc:
            stats["failed"] += 1
            errors.append(f"{protein_id}: {type(exc).__name__}: {exc}")

    stats["errors"] = errors
    return stats


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract E1 embeddings from a FASTA file for SurPass inference."
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file.")
    parser.add_argument("--root-dir", default="./pdb_metadata", help="Dataset root directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Embedding output directory. Defaults to <root-dir>/embeddings.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip proteins whose .pt file already exists.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    args = build_arg_parser().parse_args(argv)

    from E1.batch_preparer import E1BatchPreparer
    from E1.modeling import E1ForMaskedLM

    device = _resolve_device(args.device)
    output_dir = resolve_output_dir(args.root_dir, args.output_dir)
    model = E1ForMaskedLM.from_pretrained("Profluent-Bio/E1-600m").to(device)
    model.eval()
    batch_preparer = E1BatchPreparer()

    stats = extract_embeddings(
        fasta_path=args.fasta,
        output_dir=output_dir,
        model=model,
        device=device,
        skip_existing=args.skip_existing,
        batch_preparer=batch_preparer,
    )

    print("\nEmbedding extraction summary")
    print(f"fasta records: {stats['records']}")
    print(f"saved chains: {stats['saved']}")
    print(f"skipped existing: {stats['skipped']}")
    print(f"failed chains: {stats['failed']}")
    print(f"total embedded residues: {stats['residues']}")
    print(f"embedding dim: {stats['embedding_dim']}")
    print(f"output directory: {output_dir}")
    if stats["errors"]:
        print("example errors:")
        for error in stats["errors"][:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
