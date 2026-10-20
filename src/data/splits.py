"""Leakage-safe, deterministic family-component dataset splitting.

The utilities operate on ordinary mappings so they can be used from tests,
notebooks, or the thin command-line wrapper without a dataframe dependency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

Record = dict[str, Any]
FamilyMapping = Mapping[str, str] | Iterable[Mapping[str, Any]]


PAIR_MANIFEST_COLUMNS = [
    "pair_id",
    "chain1",
    "chain2",
    "family1",
    "family2",
    "length1",
    "length2",
    "source_is_positive",
    "negative_kind",
    "sample_confidence",
    "source_table",
    "split",
    "component_id",
    "exclusion_reason",
]


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "positive", "pos"}:
        return True
    if lowered in {"0", "false", "no", "n", "negative", "neg"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean label.")


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(float(value))
    if parsed <= 0:
        raise ValueError(f"Sequence length must be positive, got {value!r}.")
    return parsed


def canonical_family_pair(family1: str, family2: str) -> tuple[str, str]:
    family1, family2 = str(family1), str(family2)
    return tuple(sorted((family1, family2)))


def read_records(path: str | Path) -> list[Record]:
    """Read a TSV, CSV, or JSON-lines table into plain dictionaries."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number} must contain a JSON object."
                    )
                records.append(value)
        return records

    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Input table has no header: {path}")
        return [dict(row) for row in reader]


def normalize_family_mapping(
    family_mapping: FamilyMapping,
    *,
    entity_column: str = "entity_id",
    family_column: str = "family_id",
    length_column: str = "length",
) -> tuple[dict[str, str], dict[str, int]]:
    """Validate family assignments and collect optional entity lengths."""
    entity_to_family: dict[str, str] = {}
    entity_lengths: dict[str, int] = {}

    if isinstance(family_mapping, Mapping):
        rows: Iterable[tuple[str, Any]] = family_mapping.items()
        for raw_entity, raw_value in rows:
            entity = str(raw_entity).strip()
            if isinstance(raw_value, Mapping):
                family = str(raw_value.get(family_column, "")).strip()
                length = _as_optional_int(raw_value.get(length_column))
            else:
                family = str(raw_value).strip()
                length = None
            if not entity or not family:
                raise ValueError("Family mapping contains an empty entity or family.")
            entity_to_family[entity] = family
            if length is not None:
                entity_lengths[entity] = length
    else:
        for row_number, row in enumerate(family_mapping, start=2):
            if entity_column not in row or family_column not in row:
                raise ValueError(
                    f"Family mapping row {row_number} must contain "
                    f"{entity_column!r} and {family_column!r}."
                )
            entity = str(row[entity_column]).strip()
            family = str(row[family_column]).strip()
            if not entity or not family:
                raise ValueError(
                    f"Family mapping row {row_number} has an empty identifier."
                )
            previous = entity_to_family.get(entity)
            if previous is not None and previous != family:
                raise ValueError(
                    f"Entity {entity!r} belongs to both {previous!r} and {family!r}."
                )
            entity_to_family[entity] = family
            length = _as_optional_int(row.get(length_column))
            if length is not None:
                entity_lengths[entity] = length

    if not entity_to_family:
        raise ValueError("Family mapping is empty.")
    return entity_to_family, entity_lengths


def annotate_pair_tables(
    pair_tables: Sequence[Sequence[Mapping[str, Any]]],
    entity_to_family: Mapping[str, str],
    *,
    entity_lengths: Mapping[str, int] | None = None,
    chain1_column: str = "chain1",
    chain2_column: str = "chain2",
    positive_column: str = "source_is_positive",
    length1_column: str = "length1",
    length2_column: str = "length2",
    source_names: Sequence[str] | None = None,
) -> list[Record]:
    """Join pair rows to families while preserving negative provenance."""
    if not pair_tables:
        raise ValueError("At least one pair table is required.")
    if source_names is not None and len(source_names) != len(pair_tables):
        raise ValueError("source_names must have one name per pair table.")

    lengths = dict(entity_lengths or {})
    annotated: list[Record] = []
    for table_index, table in enumerate(pair_tables):
        source_name = (
            str(source_names[table_index])
            if source_names is not None
            else f"table_{table_index}"
        )
        for row_index, raw_row in enumerate(table):
            row = dict(raw_row)
            if chain1_column not in row or chain2_column not in row:
                raise ValueError(
                    f"{source_name} row {row_index + 2} must contain "
                    f"{chain1_column!r} and {chain2_column!r}."
                )
            chain1 = str(row[chain1_column]).strip()
            chain2 = str(row[chain2_column]).strip()
            missing = [
                chain
                for chain in (chain1, chain2)
                if chain not in entity_to_family
            ]
            if missing:
                raise ValueError(
                    f"{source_name} row {row_index + 2} contains entities absent "
                    f"from the family mapping: {sorted(set(missing))}"
                )

            source_is_positive = _as_bool(
                row.get(positive_column), default=True
            )
            negative_kind = ""
            confidence = float(
                row.get("sample_confidence")
                or row.get("confidence")
                or 1.0
            )
            if not source_is_positive:
                negative_kind = str(
                    row.get("negative_kind")
                    or row.get("negative_source")
                    or "curated"
                )

            length1 = _as_optional_int(row.get(length1_column))
            length2 = _as_optional_int(row.get(length2_column))
            length1 = length1 if length1 is not None else lengths.get(chain1)
            length2 = length2 if length2 is not None else lengths.get(chain2)
            if length1 is not None:
                lengths.setdefault(chain1, length1)
            if length2 is not None:
                lengths.setdefault(chain2, length2)

            row.update(
                {
                    "pair_id": str(
                        row.get("pair_id") or f"{source_name}:{row_index}"
                    ),
                    "chain1": chain1,
                    "chain2": chain2,
                    "family1": str(entity_to_family[chain1]),
                    "family2": str(entity_to_family[chain2]),
                    "length1": length1,
                    "length2": length2,
                    "source_is_positive": source_is_positive,
                    "negative_kind": negative_kind,
                    "sample_confidence": confidence,
                    "source_table": str(row.get("source_table") or source_name),
                }
            )
            annotated.append(row)

    if not annotated:
        raise ValueError("Pair tables contain no rows.")
    return annotated


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def connected_family_components(
    families: Iterable[str],
    pair_records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build deterministic components from all observed family-pair edges."""
    family_values = sorted({str(family) for family in families})
    if not family_values:
        raise ValueError("Cannot build components without families.")
    union_find = _UnionFind(family_values)
    for row in pair_records:
        family1, family2 = str(row["family1"]), str(row["family2"])
        if family1 not in union_find.parent or family2 not in union_find.parent:
            raise ValueError("Pair record references an unknown family.")
        union_find.union(family1, family2)

    groups: dict[str, list[str]] = defaultdict(list)
    for family in family_values:
        groups[union_find.find(family)].append(family)
    ordered_groups = sorted((sorted(group) for group in groups.values()), key=lambda x: x[0])
    components = {
        f"component_{index:06d}": group
        for index, group in enumerate(ordered_groups)
    }
    family_to_component = {
        family: component_id
        for component_id, group in components.items()
        for family in group
    }
    return family_to_component, components


def assign_component_splits(
    components: Mapping[str, Sequence[str]],
    *,
    benchmark_families: Iterable[str] = (),
    validation_fraction: float = 0.1,
    seed: int = 42,
    component_weights: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Assign whole components to train, validation, or excluded."""
    if not 0.0 <= float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    benchmark = {str(family) for family in benchmark_families}
    assignments: dict[str, str] = {}
    eligible: list[str] = []
    for component_id, families in sorted(components.items()):
        if benchmark.intersection(map(str, families)):
            assignments[component_id] = "excluded"
        else:
            eligible.append(component_id)

    if not eligible:
        return assignments
    rng = random.Random(int(seed))
    rng.shuffle(eligible)
    weights = {
        component_id: max(1, int((component_weights or {}).get(component_id, 1)))
        for component_id in eligible
    }
    validation_ids: set[str] = set()
    if validation_fraction > 0.0 and len(eligible) > 1:
        target = sum(weights.values()) * float(validation_fraction)
        current = 0
        for component_id in eligible[:-1]:
            before = abs(target - current)
            after = abs(target - (current + weights[component_id]))
            if after <= before or not validation_ids:
                validation_ids.add(component_id)
                current += weights[component_id]
        if len(validation_ids) == len(eligible):
            validation_ids.remove(eligible[-1])

    for component_id in eligible:
        assignments[component_id] = (
            "validation" if component_id in validation_ids else "train"
        )
    return assignments


def known_positive_family_pairs(
    pair_records: Iterable[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    return {
        canonical_family_pair(str(row["family1"]), str(row["family2"]))
        for row in pair_records
        if _as_bool(row.get("source_is_positive"), default=True)
    }


def _stable_tie_breaker(seed: int, *values: Any) -> str:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_entity_lengths(
    pair_records: Sequence[Mapping[str, Any]],
    entity_lengths: Mapping[str, int] | None,
) -> dict[str, int]:
    lengths = {str(key): int(value) for key, value in (entity_lengths or {}).items()}
    for row in pair_records:
        for chain_key, length_key in (("chain1", "length1"), ("chain2", "length2")):
            length = _as_optional_int(row.get(length_key))
            if length is not None:
                lengths.setdefault(str(row[chain_key]), length)
    return lengths


def generate_hard_negative_candidates(
    pair_records: Sequence[Mapping[str, Any]],
    entity_to_family: Mapping[str, str],
    *,
    count: int,
    entity_lengths: Mapping[str, int] | None = None,
    family_splits: Mapping[str, str] | None = None,
    seed: int = 42,
    synthetic_confidence: float = 0.5,
) -> list[Record]:
    """Generate deterministic endpoint-rewired, length/degree matched negatives.

    Candidate family pairs known positive anywhere in the supplied tables are
    forbidden. A candidate also stays within the anchor's train/validation
    split, so synthetic negatives cannot create cross-split family leakage.
    """
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return []
    if not 0.0 < float(synthetic_confidence) < 1.0:
        raise ValueError("synthetic_confidence must be between 0 and 1.")

    positive_rows = [
        dict(row)
        for row in pair_records
        if _as_bool(row.get("source_is_positive"), default=True)
        and str(row.get("split", "train")) != "excluded"
    ]
    if not positive_rows:
        raise ValueError("Hard-negative generation requires known-positive pairs.")

    lengths = _collect_entity_lengths(pair_records, entity_lengths)
    missing_lengths = sorted(
        entity for entity in entity_to_family if entity not in lengths
    )
    if missing_lengths:
        preview = ", ".join(missing_lengths[:5])
        raise ValueError(
            "Length-matched hard negatives require lengths for every mapped "
            f"entity; missing {len(missing_lengths)} (for example: {preview})."
        )

    known_positive = known_positive_family_pairs(pair_records)
    family_neighbors: dict[str, set[str]] = defaultdict(set)
    for family1, family2 in known_positive:
        family_neighbors[family1].add(family2)
        family_neighbors[family2].add(family1)
    family_degree = {
        family: len(family_neighbors.get(family, set()))
        for family in set(entity_to_family.values())
    }

    inferred_splits: dict[str, str] = dict(family_splits or {})
    for row in pair_records:
        split = str(row.get("split", "train"))
        if split in {"train", "validation", "excluded"}:
            inferred_splits.setdefault(str(row["family1"]), split)
            inferred_splits.setdefault(str(row["family2"]), split)

    existing_entity_pairs = {
        tuple(sorted((str(row["chain1"]), str(row["chain2"]))))
        for row in pair_records
    }
    entity_sources: dict[str, set[str]] = defaultdict(set)
    for row in pair_records:
        source = str(row.get("source_table", "unknown"))
        entity_sources[str(row["chain1"])].add(source)
        entity_sources[str(row["chain2"])].add(source)
    entities = sorted(entity_to_family)
    templates = sorted(positive_rows, key=lambda row: str(row.get("pair_id", "")))
    random.Random(int(seed)).shuffle(templates)

    generated: list[Record] = []
    exhausted_templates: set[tuple[str, bool]] = set()
    while len(generated) < count:
        made_progress = False
        for template in templates:
            if len(generated) >= count:
                break
            for reverse in (False, True):
                template_key = (str(template.get("pair_id", "")), reverse)
                if template_key in exhausted_templates:
                    continue
                anchor_key, target_key = (
                    ("chain2", "chain1") if reverse else ("chain1", "chain2")
                )
                anchor = str(template[anchor_key])
                target = str(template[target_key])
                template_source = str(template.get("source_table", "unknown"))
                anchor_family = str(entity_to_family[anchor])
                target_family = str(entity_to_family[target])
                split = inferred_splits.get(
                    anchor_family, str(template.get("split", "train"))
                )
                if split == "excluded":
                    exhausted_templates.add(template_key)
                    continue

                candidates: list[tuple[float, str, str]] = []
                for candidate in entities:
                    candidate_family = str(entity_to_family[candidate])
                    if template_source not in entity_sources.get(candidate, set()):
                        continue
                    if candidate == anchor or candidate_family == anchor_family:
                        continue
                    if inferred_splits.get(candidate_family, split) != split:
                        continue
                    if canonical_family_pair(anchor_family, candidate_family) in known_positive:
                        continue
                    entity_pair = tuple(sorted((anchor, candidate)))
                    if entity_pair in existing_entity_pairs:
                        continue

                    length_score = abs(lengths[candidate] - lengths[target]) / max(
                        lengths[target], 1
                    )
                    degree_score = abs(
                        family_degree.get(candidate_family, 0)
                        - family_degree.get(target_family, 0)
                    ) / max(family_degree.get(target_family, 0), 1)
                    score = float(length_score + degree_score)
                    tie = _stable_tie_breaker(
                        seed, template.get("pair_id"), reverse, candidate
                    )
                    candidates.append((score, tie, candidate))

                if not candidates:
                    exhausted_templates.add(template_key)
                    continue
                score, _, candidate = min(candidates)
                candidate_family = str(entity_to_family[candidate])
                existing_entity_pairs.add(tuple(sorted((anchor, candidate))))
                pair_index = len(generated)
                generated.append(
                    {
                        "pair_id": f"synthetic_{pair_index:06d}",
                        "chain1": anchor,
                        "chain2": candidate,
                        "family1": anchor_family,
                        "family2": candidate_family,
                        "length1": lengths[anchor],
                        "length2": lengths[candidate],
                        "source_is_positive": False,
                        "negative_kind": "synthetic",
                        "sample_confidence": float(synthetic_confidence),
                        "source_table": f"{template_source}.synthetic",
                        "split": split,
                        "component_id": "",
                        "exclusion_reason": "",
                        "matched_positive_pair_id": str(
                            template.get("pair_id", "")
                        ),
                        "match_score": score,
                    }
                )
                made_progress = True
                break
        if not made_progress:
            break
    return generated


def validate_family_leakage(
    pair_partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    benchmark_families: Iterable[str] = (),
) -> Record:
    family_sets = {
        split: {
            str(row[key])
            for row in rows
            for key in ("family1", "family2")
        }
        for split, rows in pair_partitions.items()
    }
    train_validation = sorted(
        family_sets.get("train", set()).intersection(
            family_sets.get("validation", set())
        )
    )
    benchmark = {str(family) for family in benchmark_families}
    benchmark_in_kept = sorted(
        benchmark.intersection(
            family_sets.get("train", set())
            | family_sets.get("validation", set())
        )
    )
    return {
        "passed": not train_validation and not benchmark_in_kept,
        "train_validation_family_overlap": train_validation,
        "benchmark_families_in_kept_splits": benchmark_in_kept,
        "family_counts": {
            split: len(families) for split, families in family_sets.items()
        },
    }


def build_dataset_splits(
    family_mapping: FamilyMapping,
    pair_tables: Sequence[Sequence[Mapping[str, Any]]],
    *,
    benchmark_families: Iterable[str] = (),
    validation_fraction: float = 0.1,
    seed: int = 42,
    hard_negative_count: int = 0,
    synthetic_confidence: float = 0.5,
    entity_column: str = "entity_id",
    family_column: str = "family_id",
    length_column: str = "length",
    chain1_column: str = "chain1",
    chain2_column: str = "chain2",
    positive_column: str = "source_is_positive",
    length1_column: str = "length1",
    length2_column: str = "length2",
    source_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build all pair, family, component, leakage, and statistics manifests."""
    entity_to_family, entity_lengths = normalize_family_mapping(
        family_mapping,
        entity_column=entity_column,
        family_column=family_column,
        length_column=length_column,
    )
    rows = annotate_pair_tables(
        pair_tables,
        entity_to_family,
        entity_lengths=entity_lengths,
        chain1_column=chain1_column,
        chain2_column=chain2_column,
        positive_column=positive_column,
        length1_column=length1_column,
        length2_column=length2_column,
        source_names=source_names,
    )
    all_families = sorted(set(entity_to_family.values()))
    family_to_component, components = connected_family_components(
        all_families, rows
    )

    component_weights = defaultdict(int)
    for row in rows:
        component_weights[family_to_component[str(row["family1"])]] += 1
    benchmark = {
        entity_to_family.get(str(value), str(value))
        for value in benchmark_families
    }
    component_splits = assign_component_splits(
        components,
        benchmark_families=benchmark,
        validation_fraction=validation_fraction,
        seed=seed,
        component_weights=component_weights,
    )
    family_splits = {
        family: component_splits[component_id]
        for family, component_id in family_to_component.items()
    }

    partitions: dict[str, list[Record]] = {
        "train": [],
        "validation": [],
        "excluded": [],
    }
    positive_family_pairs = known_positive_family_pairs(rows)
    for raw_row in rows:
        row = dict(raw_row)
        component_id = family_to_component[str(row["family1"])]
        split = component_splits[component_id]
        conflicts_with_positive = (
            not _as_bool(row.get("source_is_positive"), default=True)
            and canonical_family_pair(row["family1"], row["family2"])
            in positive_family_pairs
        )
        if conflicts_with_positive:
            split = "excluded"
        row["component_id"] = component_id
        row["split"] = split
        if conflicts_with_positive:
            row["exclusion_reason"] = "negative_conflicts_known_positive_family_pair"
        elif split == "excluded":
            row["exclusion_reason"] = "component_touches_benchmark_family"
        else:
            row["exclusion_reason"] = ""
        partitions[split].append(row)

    synthetic = generate_hard_negative_candidates(
        [*partitions["train"], *partitions["validation"], *partitions["excluded"]],
        entity_to_family,
        count=hard_negative_count,
        entity_lengths=entity_lengths,
        family_splits=family_splits,
        seed=seed,
        synthetic_confidence=synthetic_confidence,
    )
    for row in synthetic:
        partitions[str(row["split"])].append(row)

    family_entities: dict[str, list[str]] = defaultdict(list)
    for entity, family in entity_to_family.items():
        family_entities[family].append(entity)
    family_manifest = [
        {
            "entity_id": entity,
            "family_id": family,
            "component_id": family_to_component[family],
            "split": family_splits[family],
            "length": entity_lengths.get(entity),
            "benchmark_family": family in benchmark,
        }
        for entity, family in sorted(entity_to_family.items())
    ]
    component_manifest = []
    for component_id, families in sorted(components.items()):
        component_rows = [
            row
            for row in rows
            if family_to_component[str(row["family1"])] == component_id
        ]
        component_manifest.append(
            {
                "component_id": component_id,
                "split": component_splits[component_id],
                "family_count": len(families),
                "entity_count": sum(
                    len(family_entities[family]) for family in families
                ),
                "pair_count": len(component_rows),
                "positive_pair_count": sum(
                    _as_bool(row.get("source_is_positive"), default=True)
                    for row in component_rows
                ),
                "families": list(families),
            }
        )

    leakage = validate_family_leakage(
        partitions, benchmark_families=benchmark
    )
    if not leakage["passed"]:
        raise RuntimeError(f"Family leakage detected: {leakage}")

    stats = {
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "entity_count": len(entity_to_family),
        "family_count": len(all_families),
        "component_count": len(components),
        "benchmark_family_count": len(benchmark),
        "pair_counts": {
            split: len(split_rows) for split, split_rows in partitions.items()
        },
        "positive_counts": {
            split: sum(
                _as_bool(row.get("source_is_positive"), default=True)
                for row in split_rows
            )
            for split, split_rows in partitions.items()
        },
        "negative_kind_counts": {
            split: dict(
                sorted(
                    (
                        kind,
                        sum(
                            1
                            for row in split_rows
                            if not _as_bool(
                                row.get("source_is_positive"), default=True
                            )
                            and str(row.get("negative_kind")) == kind
                        ),
                    )
                    for kind in {
                        str(row.get("negative_kind"))
                        for row in split_rows
                        if not _as_bool(
                            row.get("source_is_positive"), default=True
                        )
                    }
                )
            )
            for split, split_rows in partitions.items()
        },
        "requested_hard_negatives": int(hard_negative_count),
        "generated_hard_negatives": len(synthetic),
        "conflicting_negative_count": sum(
            row.get("exclusion_reason")
            == "negative_conflicts_known_positive_family_pair"
            for row in partitions["excluded"]
        ),
    }
    return {
        **partitions,
        "families": family_manifest,
        "components": component_manifest,
        "leakage": leakage,
        "stats": stats,
    }


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def write_tsv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred_columns: Sequence[str] = (),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    observed = {str(key) for row in rows for key in row}
    fieldnames = [
        column for column in preferred_columns if column in observed or not rows
    ]
    fieldnames.extend(sorted(observed.difference(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return path
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _serialize_cell(row.get(key)) for key in fieldnames}
            )
    return path


def write_split_manifests(
    output_dir: str | Path,
    manifests: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the seven required manifest groups with stable filenames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "validation", "excluded"):
        paths[split] = write_tsv(
            output_dir / f"{split}.tsv",
            manifests[split],
            preferred_columns=PAIR_MANIFEST_COLUMNS,
        )
        by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in manifests[split]:
            source = str(row.get("source_table", "unknown"))
            source = source.removesuffix(".synthetic")
            by_source[source].append(row)
        for source, rows in by_source.items():
            safe_source = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in source
            )
            write_tsv(
                output_dir / "sources" / f"{safe_source}.{split}.tsv",
                rows,
                preferred_columns=PAIR_MANIFEST_COLUMNS,
            )
    paths["families"] = write_tsv(
        output_dir / "families.tsv",
        manifests["families"],
        preferred_columns=(
            "entity_id",
            "family_id",
            "component_id",
            "split",
            "length",
            "benchmark_family",
        ),
    )
    paths["components"] = write_tsv(
        output_dir / "components.tsv",
        manifests["components"],
        preferred_columns=(
            "component_id",
            "split",
            "family_count",
            "entity_count",
            "pair_count",
            "positive_pair_count",
            "families",
        ),
    )
    for name in ("leakage", "stats"):
        path = output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifests[name], handle, indent=2, sort_keys=True)
            handle.write("\n")
        paths[name] = path
    leakage = manifests["leakage"]
    write_tsv(
        output_dir / "leakage_report.tsv",
        [
            {
                "check": name,
                "passed": not bool(value) if isinstance(value, list) else bool(value),
                "details": value,
            }
            for name, value in leakage.items()
            if name != "passed"
        ],
        preferred_columns=("check", "passed", "details"),
    )
    return paths
