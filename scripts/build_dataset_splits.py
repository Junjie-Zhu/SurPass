#!/usr/bin/env python3
"""Build deterministic, homology-component-safe SurPass data manifests."""

from __future__ import annotations

import argparse
import json
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
            "connected components, exclude benchmark-connected components, "
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
            "treated as known positives."
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-families",
        type=Path,
        help="Optional one-column or tabular benchmark family/entity list.",
    )
    parser.add_argument("--benchmark-column", default="family_id")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-count", type=int, default=0)
    parser.add_argument("--synthetic-confidence", type=float, default=0.5)

    parser.add_argument("--entity-column", default="entity_id")
    parser.add_argument("--family-column", default="family_id")
    parser.add_argument("--length-column", default="length")
    parser.add_argument("--chain1-column", default="chain1")
    parser.add_argument("--chain2-column", default="chain2")
    parser.add_argument("--positive-column", default="source_is_positive")
    parser.add_argument("--length1-column", default="length1")
    parser.add_argument("--length2-column", default="length2")
    return parser


def _read_benchmark_values(path: Path | None, column: str) -> set[str]:
    if path is None:
        return set()
    records = read_records(path)
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
    if not 0.0 < args.synthetic_confidence < 1.0:
        parser.error("--synthetic-confidence must be between 0 and 1")

    family_tables = [read_records(path) for path in args.family_mapping]
    family_rows = [row for table in family_tables for row in table]
    pair_tables = [read_records(path) for path in args.pair_table]
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
        args.benchmark_families, args.benchmark_column
    )
    manifests = build_dataset_splits(
        family_rows,
        pair_tables,
        benchmark_families=benchmark_families,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        hard_negative_count=args.hard_negative_count,
        synthetic_confidence=args.synthetic_confidence,
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
                f"{path.parent.name}.{path.stem}"
                for path in args.pair_table
            ]
        ),
    )
    paths = write_split_manifests(args.output_dir, manifests)
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
    raise SystemExit(main())
