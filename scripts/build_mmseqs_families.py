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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

IDENTITY_THRESHOLD = 0.3
COVERAGE_THRESHOLD = 0.8


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


def distinct_search_sequences(sequences: dict[str, str]) -> dict[str, str]:
    """Search exact duplicates once; retain every original ID for clustering."""
    seen = set()
    distinct = {}
    for identifier in sorted(sequences):
        sequence = sequences[identifier]
        if sequence not in seen:
            seen.add(sequence)
            distinct[identifier] = sequence
    return distinct


def expand_search_hits(hits: set[str], sequences: dict[str, str]) -> set[str]:
    hit_sequences = {sequences[identifier] for identifier in hits}
    return {identifier for identifier, sequence in sequences.items() if sequence in hit_sequences}


def resolve_threads(requested: int | None) -> int:
    """Use CPUs visible to this process, capped by the job's CPU allocation."""
    if requested is not None and requested < 0:
        raise ValueError("--threads must be non-negative (0 selects automatically).")
    if requested:
        return requested
    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    limits = [process_cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            limits.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    for variable in ("SLURM_CPUS_PER_TASK", "NSLOTS"):
        value = os.environ.get(variable, "")
        if value.isdecimal() and int(value) > 0:
            limits.append(int(value))
    return max(1, min(limits))


def read_benchmark_hits(
    path: Path, training_ids: set[str], benchmark_ids: set[str], coverage: float,
    *, audit: dict | None = None,
) -> set[str]:
    """Read explicit fractional fident/qcov/tcov, including exact 0.3 hits."""
    excluded = set()
    matched_references = set()
    saved_rows = training_below80 = benchmark_below80 = either_below80 = 0
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
                matched_references.add(target)
                saved_rows += 1
                training_below80 += qcov < 0.8
                benchmark_below80 += tcov < 0.8
                either_below80 += min(qcov, tcov) < 0.8
    if audit is not None:
        audit.update({
            "saved_qualifying_alignment_rows": saved_rows,
            "matched_benchmark_reference_entity_count": len(matched_references),
            "saved_hits_training_coverage_below_0_8": training_below80,
            "saved_hits_benchmark_coverage_below_0_8": benchmark_below80,
            "saved_hits_either_coverage_below_0_8": either_below80,
            "saved_hits_both_coverages_at_least_0_8": saved_rows - either_below80,
            "interpretation": "Saved hits only: --max-accept 1 does not retain alternative alignments. Changing coverage requires a new search.",
        })
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
    parser.add_argument("--coverage", type=float, choices=(COVERAGE_THRESHOLD,), default=COVERAGE_THRESHOLD,
                        help="Required 0.8 coverage of both sequences for benchmark filtering AND family clustering (--cov-mode 0).")
    parser.add_argument("--negative-pair-table", type=Path, action="append", default=[],
                        help="AFDB negative TSV with chain1/chain2; records the exact number surviving the required identity/coverage filter.")
    parser.add_argument("--threads", type=int, default=None,
                        help="Worker threads; default/0 uses available CPUs, respecting CPU affinity and SLURM_CPUS_PER_TASK/NSLOTS.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        threads = resolve_threads(args.threads)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    executable = shutil.which("mmseqs")
    if executable is None:
        raise SystemExit("MMseqs2 is required but was not found on PATH.")
    # Separate namespaces allow a benchmark ID to also appear in training.
    sequences = merge_fastas(args.fasta)
    benchmarks = merge_fastas(args.benchmark_fasta)
    search_sequences = distinct_search_sequences(sequences)
    version = subprocess.check_output([executable, "version"], text=True).strip()
    signature = {
        "schema": 4, "mmseqs_version": version,
        "identity": IDENTITY_THRESHOLD, "coverage": args.coverage,
        "alignment_mode": 3, "seq_id_mode": 0, "cov_mode": 0,
        "coverage_required": True,
        "search_exact_duplicates_once": True,
        "search_sensitivity": 7.5, "search_max_accept": 1,
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
    reusable = reusable and previous.get("effective_coverage") == COVERAGE_THRESHOLD
    coverage = COVERAGE_THRESHOLD
    if any(args.output_dir.iterdir()) and not reusable and not args.overwrite:
        raise SystemExit("MMseqs inputs/settings changed or old outputs lack provenance; use a new --output-dir or --overwrite.")
    if args.overwrite:
        # Remove only this script's raw results and temporary databases.
        for name in ("families_cluster.tsv", "families_rep_seq.fasta", "families_all_seqs.fasta", "benchmark_hits.tsv", "run_metadata.json"):
            (args.output_dir / name).unlink(missing_ok=True)
        for name in ("cluster_tmp", "search_tmp"):
            shutil.rmtree(args.output_dir / name, ignore_errors=True)
    combined = args.output_dir / "training_sequences.fasta"
    search_fasta = args.output_dir / "training_search_sequences.fasta"
    benchmark_fasta = args.output_dir / "benchmark_sequences.fasta"
    write_fasta(combined, sequences)
    write_fasta(search_fasta, search_sequences)
    write_fasta(benchmark_fasta, benchmarks)
    alignment_args = [
        "--min-seq-id", str(IDENTITY_THRESHOLD), "--alignment-mode", "3",
        "--seq-id-mode", "0", "-c", str(coverage), "--cov-mode", "0",
        "--threads", str(threads),
    ]
    print(
        f"MMseqs: {threads} threads; {len(sequences):,} training IDs, "
        f"{len(search_sequences):,} distinct query sequences, {len(benchmarks):,} benchmark targets. "
        "Identity 0.3; coverage 0.8 on both sequences; search sensitivity 7.5.", file=sys.stderr, flush=True,
    )
    hits = args.output_dir / "benchmark_hits.tsv"
    hit_audit: dict = {}

    def search_benchmarks() -> None:
        if benchmarks:
            started = time.monotonic()
            print(f"Starting benchmark search (coverage {coverage}); prefilter progress follows.", file=sys.stderr, flush=True)
            subprocess.run([
                executable, "easy-search", str(search_fasta), str(benchmark_fasta),
                str(hits), str(args.output_dir / "search_tmp"),
                *alignment_args, "-s", "7.5", "--max-accept", "1",
                "--format-output", "query,target,fident,qcov,tcov",
            ], check=True)
            print(f"Benchmark search finished in {time.monotonic() - started:.1f}s.", file=sys.stderr, flush=True)
        else:
            hits.write_text("")

    def excluded_training_entities() -> set[str]:
        representative_hits = read_benchmark_hits(
            hits, set(search_sequences), set(benchmarks), coverage, audit=hit_audit,
        )
        return expand_search_hits(representative_hits, sequences)

    if not hits.is_file():
        search_benchmarks()
    excluded = excluded_training_entities()
    negative_counts = count_negative_pairs(args.negative_pair_table, set(sequences), excluded)
    cluster_tsv = args.output_dir / "families_cluster.tsv"
    if not cluster_tsv.is_file():
        started = time.monotonic()
        print(f"Starting family clustering of all {len(sequences):,} training IDs.", file=sys.stderr, flush=True)
        subprocess.run([
            executable, "easy-cluster", str(combined), str(args.output_dir / "families"),
            str(args.output_dir / "cluster_tmp"), *alignment_args,
        ], check=True)
        print(f"Family clustering finished in {time.monotonic() - started:.1f}s.", file=sys.stderr, flush=True)
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
        "coverage_mode": 0,
        "benchmark_reference_entity_count": len(benchmarks),
        "benchmark_matching_training_entity_count": len(excluded),
        "saved_hit_audit": hit_audit,
        "benchmark_entities_sha256": file_sha256(excluded_path),
        "threads": threads, "training_entity_count": len(sequences),
        "distinct_search_sequence_count": len(search_sequences),
        "negative_pair_counts": negative_counts,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {mapping} ({len(sequences)} training sequences)")
    print(f"Wrote {excluded_path} ({len(excluded)} direct benchmark hits; identity >= 0.3, coverage >= {coverage})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
