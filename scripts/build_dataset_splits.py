#!/usr/bin/env python3
"""Build deterministic, homology-component-safe SurPass data manifests."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-count", type=int, default=0)
    parser.add_argument(
        "--pdb-negative-ratio", type=float, default=1.0,
        help="Negative clusters per positive PPI cluster in each pdb_ppi split (default: 1). Each generated negative is one cluster; written separately as pdb_neg.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Candidate-ranking worker processes (default: 1). Selection stays deterministic.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bars/status messages on stderr; stdout always contains the JSON summary.",
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
    manifests = build_dataset_splits(
        family_rows,
        pair_tables,
        benchmark_families=benchmark_families,
        benchmark_entities=_read_benchmark_values(args.benchmark_entities, "entity_id", progress=progress),
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
        workers=args.workers,
        progress=progress,
    )
    paths = write_split_manifests(args.output_dir, manifests, progress=progress)
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
