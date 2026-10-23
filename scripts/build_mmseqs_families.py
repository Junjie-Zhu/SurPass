#!/usr/bin/env python3
"""Search training sequences against benchmarks, then cluster training sequences.

Benchmark exclusions are direct alignment hits, never interaction components
or transitive membership in a joint training/benchmark cluster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

IDENTITY_THRESHOLD = 0.3


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
        if identifier in records and records[identifier] != sequence:
            raise ValueError(f"Conflicting duplicate ID {identifier!r} in {path}")
        records[identifier] = sequence

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                if not line[1:].strip():
                    raise ValueError(f"Empty FASTA header in {path}")
                identifier = line[1:].split()[0]
                chunks = []
            else:
                if identifier is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                chunks.append(line)
    flush()
    if not records:
        raise ValueError(f"Empty FASTA: {path}")
    return records


def merge_fastas(paths: list[Path]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for path in paths:
        for identifier, sequence in read_fasta(path).items():
            if identifier in sequences and sequences[identifier] != sequence:
                raise ValueError(f"ID {identifier!r} has different sequences across FASTAs")
            sequences[identifier] = sequence
    return sequences


def write_fasta(path: Path, sequences: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for identifier, sequence in sorted(sequences.items()):
            handle.write(f">{identifier}\n{sequence}\n")


def read_benchmark_hits(path: Path, training_ids: set[str], benchmark_ids: set[str], coverage: float) -> set[str]:
    """Read explicit fractional fident/qcov/tcov, including exact 0.3 hits."""
    excluded = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected query,target,fident,qcov,tcov")
            query, target = fields[:2]
            if query not in training_ids or target not in benchmark_ids:
                raise ValueError(f"{path}:{line_number}: unexpected sequence ID")
            identity, qcov, tcov = map(float, fields[2:])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (identity, qcov, tcov)):
                raise ValueError(f"{path}:{line_number}: expected fractional identity and coverage in [0, 1]")
            if identity >= IDENTITY_THRESHOLD and min(qcov, tcov) >= coverage:
                excluded.add(query)
    return excluded


def count_negative_pairs(paths: list[Path], training_ids: set[str], excluded: set[str]) -> dict[str, dict[str, int]]:
    counts = {}
    for path in paths:
        total = kept = 0
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"chain1", "chain2"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: negative pair table requires chain1 and chain2")
            for row in reader:
                endpoints = {row["chain1"].strip(), row["chain2"].strip()}
                if not endpoints.issubset(training_ids):
                    raise ValueError(f"{path}: negative pair contains an ID absent from training FASTAs")
                total += 1
                kept += not endpoints.intersection(excluded)
        if not total:
            raise ValueError(f"{path}: negative pair table is empty")
        counts[str(path)] = {"input_pairs": total, "kept_after_sequence_filter": kept}
    return counts


def read_clusters(path: Path, sequences: dict[str, str]) -> dict[str, str]:
    assignments = {}
    with path.open(encoding="utf-8") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) != 2:
                raise ValueError(f"Invalid cluster row in {path}: {fields}")
            representative, member = fields
            if representative not in sequences or member not in sequences:
                raise ValueError(f"Unknown sequence in {path}: {fields}")
            if member in assignments and assignments[member] != representative:
                raise ValueError(f"Conflicting cluster assignments for {member}")
            assignments[member] = representative
    missing = sequences.keys() - assignments.keys()
    if missing:
        raise ValueError(f"MMseqs cluster output is incomplete: {len(missing)} missing sequences")
    return assignments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-fasta", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seq-id", type=float, choices=(IDENTITY_THRESHOLD,), default=IDENTITY_THRESHOLD)
    parser.add_argument("--coverage", type=float, default=0.0,
                        help="Alignment coverage for benchmark filtering AND family clustering; default 0 (identity only). If no AFDB negatives survive, rerun with 0.8.")
    parser.add_argument("--negative-pair-table", type=Path, action="append", default=[],
                        help="AFDB negative TSV with chain1/chain2. If identity-only search removes every negative pair, automatically retry with coverage 0.8 at identity 0.3.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.coverage) or not 0 <= args.coverage <= 1:
        raise SystemExit("--coverage must be in [0, 1].")
    if args.threads < 1:
        raise SystemExit("--threads must be positive.")
    executable = shutil.which("mmseqs")
    if executable is None:
        raise SystemExit("MMseqs2 is required but was not found on PATH.")
    # Separate namespaces allow a benchmark ID to also appear in training.
    sequences = merge_fastas(args.fasta)
    benchmarks = merge_fastas(args.benchmark_fasta)
    version = subprocess.check_output([executable, "version"], text=True).strip()
    signature = {
        "schema": 2, "mmseqs_version": version,
        "identity": IDENTITY_THRESHOLD, "coverage": args.coverage,
        "alignment_mode": 3, "seq_id_mode": 0, "cov_mode": 0,
        "automatic_coverage_fallback": bool(args.negative_pair_table),
        "inputs": [
            {"role": role, "path": str(path.resolve()),
             "sha256": file_sha256(path)}
            for role, paths in (("training", args.fasta), ("benchmark", args.benchmark_fasta), ("negative_pairs", args.negative_pair_table))
            for path in paths
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = args.output_dir / "run_metadata.json"
    previous = json.loads(metadata.read_text()) if metadata.is_file() else {}
    reusable = previous.get("request") == signature
    coverage = previous["effective_coverage"] if reusable and not args.overwrite else args.coverage
    if any(args.output_dir.iterdir()) and not reusable and not args.overwrite:
        raise SystemExit("MMseqs inputs/settings changed or old outputs lack provenance; use a new --output-dir or --overwrite.")
    if args.overwrite:
        # Remove only this script's raw results and temporary databases.
        for name in ("families_cluster.tsv", "families_rep_seq.fasta", "families_all_seqs.fasta", "benchmark_hits.tsv", "run_metadata.json"):
            (args.output_dir / name).unlink(missing_ok=True)
        for name in ("cluster_tmp", "search_tmp"):
            shutil.rmtree(args.output_dir / name, ignore_errors=True)
    combined = args.output_dir / "training_sequences.fasta"
    benchmark_fasta = args.output_dir / "benchmark_sequences.fasta"
    write_fasta(combined, sequences)
    write_fasta(benchmark_fasta, benchmarks)
    alignment_args = [
        "--min-seq-id", str(IDENTITY_THRESHOLD), "--alignment-mode", "3",
        "--seq-id-mode", "0", "-c", str(coverage), "--cov-mode", "0",
        "--threads", str(args.threads),
    ]
    hits = args.output_dir / "benchmark_hits.tsv"
    def search_benchmarks() -> None:
        if benchmarks:
            subprocess.run([
                executable, "easy-search", str(combined), str(benchmark_fasta),
                str(hits), str(args.output_dir / "search_tmp"),
                *alignment_args, "-s", "7.5",
                "--format-output", "query,target,fident,qcov,tcov",
            ], check=True)
        else:
            hits.write_text("")

    if not hits.is_file():
        search_benchmarks()
    excluded = read_benchmark_hits(hits, set(sequences), set(benchmarks), coverage)
    negative_counts = count_negative_pairs(args.negative_pair_table, set(sequences), excluded)
    if coverage == 0 and negative_counts and not any(
        counts["kept_after_sequence_filter"] for counts in negative_counts.values()
    ):
        print("Identity-only filtering removed all supplied negatives; retrying at identity 0.3, coverage 0.8.", file=sys.stderr)
        coverage = 0.8
        alignment_args[alignment_args.index("-c") + 1] = str(coverage)
        hits.unlink()
        shutil.rmtree(args.output_dir / "search_tmp", ignore_errors=True)
        search_benchmarks()
        excluded = read_benchmark_hits(hits, set(sequences), set(benchmarks), coverage)
        negative_counts = count_negative_pairs(args.negative_pair_table, set(sequences), excluded)
    cluster_tsv = args.output_dir / "families_cluster.tsv"
    if not cluster_tsv.is_file():
        subprocess.run([
            executable, "easy-cluster", str(combined), str(args.output_dir / "families"),
            str(args.output_dir / "cluster_tmp"), *alignment_args,
        ], check=True)
    assignments = read_clusters(cluster_tsv, sequences)
    mapping = args.output_dir / "family_mapping.tsv"
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("entity_id", "family_id", "length"))
        writer.writerows((identifier, assignments[identifier], len(sequence)) for identifier, sequence in sorted(sequences.items()))
    excluded_path = args.output_dir / "benchmark_entities.tsv"
    with excluded_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("entity_id",))
        writer.writerows((identifier,) for identifier in sorted(excluded))
    metadata.write_text(json.dumps({"request": signature, "effective_coverage": coverage}, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "filter_stats.json").write_text(json.dumps({
        "identity_threshold": IDENTITY_THRESHOLD, "requested_coverage": args.coverage,
        "effective_coverage": coverage, "direct_benchmark_hit_count": len(excluded),
        "negative_pair_counts": negative_counts,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {mapping} ({len(sequences)} training sequences)")
    print(f"Wrote {excluded_path} ({len(excluded)} direct benchmark hits; identity >= 0.3, coverage >= {coverage})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
