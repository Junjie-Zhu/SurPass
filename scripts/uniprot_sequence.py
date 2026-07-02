#!/usr/bin/env python3
"""Fetch protein sequences for UniProt accessions.

This script is standalone and uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence
import wget


BASE_URL = "https://rest.uniprot.org"
USER_AGENT = "uniprot-sequence-fetcher/1.0"


def fetch_entry(
    accession: str,
    output_dir: str = ".",
    dataset: str = "uniprotkb",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch one UniProt entry as JSON from the public REST API."""
    accession = accession.strip()
    if not accession:
        raise ValueError("Accession must not be empty")

    url = f"{BASE_URL}/{dataset.strip()}/{accession}.json"

    # use wget to download the json file
    wget.download(url, out=os.path.join(output_dir, f"{accession}.json"))
    with open(os.path.join(output_dir, f"{accession}.json"), "r") as f:
        entry = json.load(f)
    return entry


def _response_charset(response: Any) -> str:
    content_type = response.headers.get("content-type", "")
    for part in content_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value
    return "utf-8"


def extract_sequence(entry: dict[str, Any]) -> str:
    """Extract the amino-acid sequence value from a UniProt JSON entry."""
    sequence = entry.get("sequence")
    if not isinstance(sequence, dict):
        raise ValueError("No sequence value found in entry")

    value = sequence.get("value")
    if not isinstance(value, str) or not value:
        raise ValueError("No sequence value found in entry")
    return value


def entry_label(entry: dict[str, Any], fallback: str) -> str:
    """Return a stable FASTA label for the entry."""
    for key in ("primaryAccession", "uniProtkbId", "uniParcId", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def format_fasta(label: str, sequence: str, width: int = 60) -> str:
    """Format one sequence as FASTA."""
    if width <= 0:
        raise ValueError("FASTA width must be greater than zero")
    wrapped = "\n".join(
        sequence[index : index + width] for index in range(0, len(sequence), width)
    )
    return f">{label}\n{wrapped}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch sequences for one or more UniProt accessions."
    )
    parser.add_argument("accessions", nargs="+", help="UniProt accession IDs")
    parser.add_argument(
        "--dataset",
        default="uniprotkb",
        help="UniProt dataset to query, such as uniprotkb, uniparc, or uniref",
    )
    parser.add_argument(
        "--output",
        choices=("fasta", "raw"),
        default="fasta",
        help="Output format. raw prints sequence only for one ID, or TSV for many IDs.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=60,
        help="FASTA sequence line width",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results: list[str] = []

    for accession in args.accessions:
        entry = fetch_entry(accession, dataset=args.dataset, timeout=args.timeout)
        sequence = extract_sequence(entry)
        label = entry_label(entry, accession)

        if args.output == "fasta":
            results.append(format_fasta(label, sequence, width=args.width))
        elif len(args.accessions) == 1:
            results.append(sequence)
        else:
            results.append(f"{label}\t{sequence}")

    print("\n".join(results))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
