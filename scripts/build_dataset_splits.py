#!/usr/bin/env python3
"""Build deterministic, homology-component-safe SurPass data manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import (  # noqa: E402
    build_dataset_splits,
    read_records,
    write_split_manifests,
    write_tsv,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join pair tables to a precomputed family mapping, split whole "
            "positive interaction components, exclude benchmark-matching sequences, "
            "and optionally generate matched synthetic hard negatives."
        )
    )
    parser.add_argument(
        "--family-mapping",
        type=Path,
        action="append",
        default=[],
        help=(
            "TSV/CSV/JSONL with entity_id, family_id, and preferably length. "
            "Repeat to combine mappings. Required: this script deliberately "
            "does not invoke MMseqs; run MMseqs separately and supply its "
            "normalized family mapping here."
        ),
    )
    parser.add_argument(
        "--pair-table",
        type=Path,
        action="append",
        default=[],
        help=(
            "Pair TSV/CSV/JSONL. Repeat for multiple positive, curated, or "
            "static-negative sources. Rows without the positive column are "
            "treated as positive except for afdb_neg/pdb_neg sources."
        ),
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=[],
        help=(
            "Stable source name corresponding to each --pair-table. Repeat in "
            "the same order; names are used for source-specific manifests."
        ),
    )
    parser.add_argument(
        "--source-label",
        choices=("positive", "negative", "from-table"),
        action="append",
        default=[],
        help=(
            "Label policy corresponding to each --pair-table. Repeat in the "
            "same order. Use from-table to read --positive-column."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--benchmark-families",
        type=Path,
        help="Optional one-column or tabular benchmark family/entity list.",
    )
    parser.add_argument("--benchmark-column", default="family_id")
    parser.add_argument(
        "--benchmark-entities", type=Path,
        help="Direct training-sequence hits from build_mmseqs_families.py; excludes only pairs containing these entities.",
    )
    parser.add_argument(
        "--benchmark-filter-stats", type=Path,
        help="Filtering policy/provenance JSON; defaults to filter_stats.json beside --benchmark-entities when available.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-count", type=int, default=0)
    parser.add_argument(
        "--pdb-negative-ratio", type=float, default=1.0,
        help="Distinct negative family-pair clusters per positive PPI cluster in each pdb_ppi split (default: 1); written separately as pdb_neg.",
    )
    parser.add_argument(
        "--pdb-negative-pairs-per-cluster", type=int, default=0,
        help="Candidate pairs per PDB negative family-pair cluster; 0 (default) matches the positive split's mean pairs per cluster, subject to available distinct member pairs.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Candidate-ranking worker processes (default: 1). Selection stays deterministic.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bars/status messages; the final size report remains on stderr and stdout contains the JSON summary.",
    )

    parser.add_argument("--entity-column", default="entity_id")
    parser.add_argument("--family-column", default="family_id")
    parser.add_argument("--length-column", default="length")
    parser.add_argument("--chain1-column", default="chain1")
    parser.add_argument("--chain2-column", default="chain2")
    parser.add_argument("--positive-column", default="source_is_positive")
    parser.add_argument("--length1-column", default="length1")
    parser.add_argument("--length2-column", default="length2")
    return parser


def _read_benchmark_values(path: Path | None, column: str, *, progress: bool = False) -> set[str]:
    if path is None:
        return set()
    records = read_records(path, progress=progress)
    if not records:
        return set()
    if column not in records[0]:
        if len(records[0]) != 1:
            raise ValueError(
                f"Benchmark table {path} has no {column!r} column; available "
                f"columns are {sorted(records[0])}."
            )
        column = next(iter(records[0]))
    return {
        str(record[column]).strip()
        for record in records
        if str(record.get(column, "")).strip()
    }


def _default_source_name(path: Path) -> str:
    """Match the standard preparation outputs to the training source names."""
    standard = {
        "verified_pdb_pairs.tsv": "pdb_ppi",
        "verified_heterodimers.tsv": "pdb_ppi",
        "verified_heterodimers.clustered.tsv": "pdb_ppi",
        "verified_positive_pairs.dissimilar.tsv": "afdb_pos",
        "verified_negative_pairs.dissimilar.tsv": "afdb_neg",
    }
    return standard.get(path.name, f"{path.parent.name}.{path.stem}")


def _read_filter_stats(path: Path | None, entities_path: Path | None, entities: set[str]) -> dict | None:
    if path is None and entities_path is not None:
        candidate = entities_path.parent / "filter_stats.json"
        if candidate.is_file():
            path = candidate
    if path is None:
        return None
    stats = json.loads(path.read_text())
    if not isinstance(stats, dict):
        raise ValueError(f"Benchmark filter stats must be a JSON object: {path}")
    if stats.get("effective_coverage") not in (None, 0.8) or stats.get("coverage_mode") not in (None, 0):
        raise ValueError(f"{path}: benchmark hits were not generated with required coverage 0.8 on both sequences; rerun build_mmseqs_families.py and rebuild splits.")
    if stats.get("identity_threshold") not in (None, 0.3):
        raise ValueError(f"{path}: benchmark identity threshold must be 0.3; rerun build_mmseqs_families.py and rebuild splits.")
    expected_count = stats.get("benchmark_matching_training_entity_count", stats.get("direct_benchmark_hit_count"))
    if expected_count is not None and expected_count != len(entities):
        raise ValueError(f"{path}: matched training-ID count does not match --benchmark-entities; supply files from the same run.")
    if stats.get("benchmark_entities_sha256"):
        if entities_path is None:
            raise ValueError("Benchmark filter provenance requires --benchmark-entities.")
        with entities_path.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        if checksum != stats["benchmark_entities_sha256"]:
            raise ValueError(f"{path}: benchmark exclusion file checksum mismatch; supply files from the same run.")
    return stats


def _report_benchmark_filter(stats: dict) -> None:
    references = stats["benchmark_reference_entity_count"]
    print("\nBenchmark reference sequences: " + (f"{references:,}" if references is not None else "not recorded in supplied filter metadata"), file=sys.stderr)
    print(
        f"Benchmark-matching training IDs: {stats['benchmark_matching_training_entity_count']:,} "
        f"({stats['benchmark_matching_pair_referenced_entity_count']:,} referenced in pair tables; "
        f"{stats['benchmark_matching_unreferenced_entity_count']:,} unreferenced).",
        file=sys.stderr,
    )
    policy = stats["benchmark_filter"]
    if policy is not None:
        print(
            f"Benchmark filter policy: identity >= {policy.get('identity_threshold', 'unknown')}; "
            f"effective coverage {policy.get('effective_coverage', 'unknown')}.", file=sys.stderr,
        )
    else:
        print("Benchmark filter policy: unavailable; exclusion IDs alone do not record identity/coverage.", file=sys.stderr)
    print("Validation fraction targets components; per-source sample/cluster fractions are reported in stats.json.", file=sys.stderr)


def _write_size_report(output_dir: Path, stats: dict) -> Path:
    """Report final pair rows and source-scoped PPI clusters, including zeros."""
    source_counts = stats["source_counts"]
    cluster_counts = stats["source_cluster_counts"]
    sources = sorted({source for counts in source_counts.values() for source in counts})
    standard_order = ("pdb_ppi", "pdb_neg", "afdb_pos", "afdb_neg")
    sources = [source for source in standard_order if source in sources] + [
        source for source in sources if source not in standard_order
    ]
    rows = []
    for split in ("train", "validation", "excluded"):
        split_rows = [
            {
                "split": split,
                "source": source,
                "cluster_count": cluster_counts[split].get(source, 0),
                "sample_count": source_counts[split].get(source, 0),
            }
            for source in sources
        ]
        rows.extend(split_rows)
        rows.append({
            "split": split,
            "source": "TOTAL",
            "cluster_count": sum(row["cluster_count"] for row in split_rows),
            "sample_count": sum(row["sample_count"] for row in split_rows),
        })

    path = write_tsv(
        output_dir / "subset_sizes.tsv", rows,
        preferred_columns=("split", "source", "cluster_count", "sample_count"),
    )
    table = [("Split", "Source", "Clusters", "Samples")] + [
        (row["split"], row["source"], f"{row['cluster_count']:,}", f"{row['sample_count']:,}")
        for row in rows
    ]
    widths = [max(len(row[index]) for row in table) for index in range(4)]
    print("\nFinal subset sizes (clusters = PPI clusters; samples = pair rows):", file=sys.stderr)
    for split, source, clusters, samples in table:
        print(
            f"{split:<{widths[0]}}  {source:<{widths[1]}}  "
            f"{clusters:>{widths[2]}}  {samples:>{widths[3]}}",
            file=sys.stderr,
        )
    print(f"Saved size report: {path}", file=sys.stderr)
    print("Excluded cluster counts include partially excluded clusters; see source_cluster_retention_counts in stats.json.", file=sys.stderr)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.family_mapping:
        parser.error(
            "at least one --family-mapping is required. This script consumes "
            "precomputed family mappings and does not invoke MMseqs."
        )
    if not args.pair_table:
        parser.error("at least one --pair-table is required")
    if args.source_name and len(args.source_name) != len(args.pair_table):
        parser.error("--source-name must be repeated once per --pair-table")
    if args.source_label and len(args.source_label) != len(args.pair_table):
        parser.error("--source-label must be repeated once per --pair-table")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.hard_negative_count < 0:
        parser.error("--hard-negative-count must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not math.isfinite(args.pdb_negative_ratio) or args.pdb_negative_ratio < 0:
        parser.error("--pdb-negative-ratio must be finite and non-negative")
    if args.pdb_negative_pairs_per_cluster < 0:
        parser.error("--pdb-negative-pairs-per-cluster must be non-negative")

    progress = not args.no_progress
    family_tables = [read_records(path, progress=progress) for path in args.family_mapping]
    family_rows = [row for table in family_tables for row in table]
    pair_tables = [read_records(path, progress=progress) for path in args.pair_table]
    if args.source_label:
        for table, policy in zip(pair_tables, args.source_label):
            if policy == "from-table":
                continue
            value = policy == "positive"
            for row in table:
                row[args.positive_column] = value
    empty_family_tables = [
        str(path)
        for path, rows in zip(args.family_mapping, family_tables)
        if not rows
    ]
    if empty_family_tables:
        parser.error(f"family mappings are empty: {', '.join(empty_family_tables)}")
    empty_pair_tables = [
        str(path)
        for path, rows in zip(args.pair_table, pair_tables)
        if not rows
    ]
    if empty_pair_tables:
        parser.error(f"pair tables are empty: {', '.join(empty_pair_tables)}")

    benchmark_families = _read_benchmark_values(
        args.benchmark_families, args.benchmark_column, progress=progress,
    )
    benchmark_entities = _read_benchmark_values(args.benchmark_entities, "entity_id", progress=progress)
    benchmark_filter_stats = _read_filter_stats(args.benchmark_filter_stats, args.benchmark_entities, benchmark_entities)
    manifests = build_dataset_splits(
        family_rows,
        pair_tables,
        benchmark_families=benchmark_families,
        benchmark_entities=benchmark_entities,
        benchmark_filter_stats=benchmark_filter_stats,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        hard_negative_count=args.hard_negative_count,
        entity_column=args.entity_column,
        family_column=args.family_column,
        length_column=args.length_column,
        chain1_column=args.chain1_column,
        chain2_column=args.chain2_column,
        positive_column=args.positive_column,
        length1_column=args.length1_column,
        length2_column=args.length2_column,
        source_names=(
            args.source_name
            if args.source_name
            else [
                _default_source_name(path)
                for path in args.pair_table
            ]
        ),
        pdb_negative_ratio=args.pdb_negative_ratio,
        pdb_negative_pairs_per_cluster=args.pdb_negative_pairs_per_cluster,
        workers=args.workers,
        progress=progress,
    )
    paths = write_split_manifests(args.output_dir, manifests, progress=progress)
    _report_benchmark_filter(manifests["stats"])
    paths["subset_sizes"] = _write_size_report(args.output_dir, manifests["stats"])
    print(
        json.dumps(
            {
                "manifests": {name: str(path) for name, path in paths.items()},
                "stats": manifests["stats"],
                "leakage": manifests["leakage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDataset build interrupted; worker processes stopped.", file=sys.stderr, flush=True)
        raise SystemExit(130)
