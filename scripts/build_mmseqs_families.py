#!/usr/bin/env python3
"""Jointly cluster every training and benchmark sequence with MMseqs2."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


def read_fasta(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing FASTA: {path}")
    records: dict[str, str] = {}
    identifier: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if identifier is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            raise ValueError(f"Empty sequence for {identifier!r} in {path}")
        records[identifier] = sequence

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                identifier = line[1:].split()[0]
                chunks = []
            else:
                if identifier is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                chunks.append(line)
    flush()
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine PDB, AFDB, and benchmark FASTAs, cluster them jointly, "
            "and write entity_id/family_id/length assignments."
        )
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        action="append",
        required=True,
        help="Training-source FASTA; repeat for PDB and AFDB sources.",
    )
    parser.add_argument(
        "--benchmark-fasta",
        type=Path,
        action="append",
        default=[],
        help="Benchmark FASTA; included in clustering and written as excluded IDs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    executable = shutil.which("mmseqs")
    if executable is None:
        raise SystemExit("MMseqs2 is required but was not found on PATH.")
    if not 0.0 < args.min_seq_id <= 1.0:
        raise SystemExit("--min-seq-id must be in (0, 1].")
    if not 0.0 < args.coverage <= 1.0:
        raise SystemExit("--coverage must be in (0, 1].")
    if args.threads < 1:
        raise SystemExit("--threads must be positive.")

    sequences: dict[str, str] = {}
    benchmark_ids: set[str] = set()
    for fasta in [*args.fasta, *args.benchmark_fasta]:
        for identifier, sequence in read_fasta(fasta).items():
            previous = sequences.get(identifier)
            if previous is not None and previous != sequence:
                raise ValueError(
                    f"ID {identifier!r} has different sequences across FASTAs."
                )
            sequences[identifier] = sequence
            if fasta in args.benchmark_fasta:
                benchmark_ids.add(identifier)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / "all_sequences.fasta"
    with combined.open("w", encoding="utf-8") as handle:
        for identifier, sequence in sorted(sequences.items()):
            handle.write(f">{identifier}\n{sequence}\n")

    prefix = args.output_dir / "families"
    cluster_tsv = args.output_dir / "families_cluster.tsv"
    if args.overwrite or not cluster_tsv.is_file():
        subprocess.run(
            [
                executable,
                "easy-cluster",
                str(combined),
                str(prefix),
                str(args.output_dir / "mmseqs_tmp"),
                "--min-seq-id",
                str(args.min_seq_id),
                "-c",
                str(args.coverage),
                "--cov-mode",
                "0",
                "--threads",
                str(args.threads),
            ],
            check=True,
        )
    if not cluster_tsv.is_file():
        raise RuntimeError(f"MMseqs2 did not produce {cluster_tsv}")

    entity_to_family: dict[str, str] = {}
    with cluster_tsv.open("r", encoding="utf-8") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) < 2:
                continue
            representative, member = fields[0], fields[1]
            entity_to_family[representative] = representative
            entity_to_family[member] = representative

    output = args.output_dir / "family_mapping.tsv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("entity_id", "family_id", "length"),
            delimiter="\t",
        )
        writer.writeheader()
        for identifier, sequence in sorted(sequences.items()):
            writer.writerow(
                {
                    "entity_id": identifier,
                    "family_id": entity_to_family.get(identifier, identifier),
                    "length": len(sequence),
                }
            )
    benchmark_output = args.output_dir / "benchmark_entities.tsv"
    with benchmark_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("entity_id",),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            {"entity_id": identifier} for identifier in sorted(benchmark_ids)
        )
    print(f"Wrote {output} ({len(sequences)} sequences)")
    print(f"Wrote {benchmark_output} ({len(benchmark_ids)} benchmark sequences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
