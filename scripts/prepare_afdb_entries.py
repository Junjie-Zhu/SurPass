"""
    Prepare AFDB domain entries for training.

    Structures and sequences are read from local files. Nothing is downloaded.

    1. Load positive and negative DDI tables and collect unique domain ids.
    2. Verify each domain PDB, save processed chain features, and write sequences.
    3. Save pairwise distance labels for positive pairs that contain an
       inter-domain CB contact.
    4. Search every verified domain against the test set, then drop positive
       and negative pairs that contain a similar domain.
    5. Cluster the remaining positive-pair and negative-pair domains.
"""

import argparse
import gzip
import os
import pickle
import shutil
import sys
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

from prepare_pdb_entries import (  
    PDBDataSelector,
    cluster_heterodimer_ppi_pairs,
    extract_pdb_features,
    load_uniprot_fasta_by_accession,
    parse_mmseqs_cluster_tsv,
    run_mmseqs_cluster,
    split_homodimer_heterodimer_pairs,
)


DOMAIN_EXTENSIONS = (".pdb", ".pdb.gz", ".cif", ".cif.gz")
CHAIN_KEYS = (
    "atom14_positions",
    "atom14_mask",
    "residue_type",
    "residue_index",
    "chain_index",
    "cb_positions",
    "cb_mask",
    "resseq",
)
VERIFIED_PAIR_COLUMNS = [
    "chain1",
    "chain2",
    "label_path",
    "min_inter_domain_distance",
    "PAIRID",
    "HASH",
    "CLUSTER",
    "LEN",
]
REJECTED_PAIR_COLUMNS = VERIFIED_PAIR_COLUMNS + [
    "reason",
    "error_type",
    "error_message",
]
CLUSTER_OUTPUTS = (
    "verified_homodimers.tsv",
    "verified_heterodimers.tsv",
    "verified_homodimers.clustered.tsv",
    "verified_homodimers.duplicate.tsv",
    "verified_heterodimers.clustered.tsv",
    "verified_heterodimers.duplicate.tsv",
)
NEGATIVE_CLUSTER_OUTPUTS = (
    "verified_negative_homodimers.tsv",
    "verified_negative_heterodimers.tsv",
    "verified_negative_homodimers.clustered.tsv",
    "verified_negative_homodimers.duplicate.tsv",
    "verified_negative_heterodimers.clustered.tsv",
    "verified_negative_heterodimers.duplicate.tsv",
)


def _split_two_field(value: str) -> Tuple[str, str]:
    text = "" if pd.isna(value) else str(value).strip()
    left, separator, right = text.partition(":")
    if separator != ":":
        return "", ""
    return left.strip(), right.strip()


def load_pair_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find pair metadata: {path}")

    df = pd.read_csv(path)
    required = {"PAIRID", "HASH", "CLUSTER", "LEN"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    split_ids = df["PAIRID"].map(_split_two_field)
    split_lengths = df["LEN"].map(_split_two_field)
    df = df.copy()
    df["domain1"] = [item[0] for item in split_ids]
    df["domain2"] = [item[1] for item in split_ids]
    df["len1"] = pd.to_numeric([item[0] for item in split_lengths], errors="coerce")
    df["len2"] = pd.to_numeric([item[1] for item in split_lengths], errors="coerce")

    malformed = df["domain1"].eq("") | df["domain2"].eq("")
    n_malformed = int(malformed.sum())
    if n_malformed:
        print(f"Skipping {n_malformed} malformed PAIRID rows in {path.name}")
        df = df.loc[~malformed].copy()
    return df.reset_index(drop=True)


def collect_domain_metadata(posi: pd.DataFrame, nega: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for table, source in ((posi, "positive"), (nega, "negative")):
        left = table[["domain1", "len1"]].rename(
            columns={"domain1": "domain_id", "len1": "metadata_length"}
        )
        right = table[["domain2", "len2"]].rename(
            columns={"domain2": "domain_id", "len2": "metadata_length"}
        )
        part = pd.concat([left, right], ignore_index=True)
        part["source"] = source
        frames.append(part)

    long = pd.concat(frames, ignore_index=True)
    long = long.loc[long["domain_id"].astype(str).str.len() > 0].copy()
    long["_order"] = np.arange(len(long))
    long["_missing_length"] = long["metadata_length"].isna()
    lengths = (
        long.sort_values(["_missing_length", "_order"])
        .drop_duplicates("domain_id", keep="first")
        .loc[:, ["domain_id", "metadata_length"]]
    )
    positive_ids = set(long.loc[long["source"].eq("positive"), "domain_id"])
    negative_ids = set(long.loc[long["source"].eq("negative"), "domain_id"])
    lengths["in_positive"] = lengths["domain_id"].isin(positive_ids)
    lengths["in_negative"] = lengths["domain_id"].isin(negative_ids)
    return lengths.reset_index(drop=True)


def domain_shard(domain_id: str) -> str:
    accession = str(domain_id).rsplit("_", 1)[0]
    if len(accession) < 2:
        raise ValueError(f"Domain id {domain_id!r} has no accession shard.")
    return accession[-2:]


def resolve_domain_pdb(dompdb_dir: Path, domain_id: str) -> Optional[Path]:
    folder = dompdb_dir / domain_shard(domain_id)
    for extension in DOMAIN_EXTENSIONS:
        path = folder / f"{domain_id}{extension}"
        if path.is_file():
            return path
    return None


def extract_domain_features(
    structure_path: Path,
    min_chain_length: int,
    max_chain_length: int,
):
    if structure_path.name.endswith(".gz"):
        plain_suffix = ".cif" if structure_path.name.endswith(".cif.gz") else ".pdb"
        fd, tmp_name = tempfile.mkstemp(suffix=plain_suffix)
        os.close(fd)
        try:
            with gzip.open(structure_path, "rb") as src, open(tmp_name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return extract_pdb_features(
                tmp_name,
                min_chain_length=min_chain_length,
                max_chain_length=max_chain_length,
                process_all_model=False,
            )
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    return extract_pdb_features(
        str(structure_path),
        min_chain_length=min_chain_length,
        max_chain_length=max_chain_length,
        process_all_model=False,
    )


def _error_record(domain_id: str, stage: str, error_type: str, error_message: str) -> Dict[str, str]:
    return {
        "domain_id": domain_id,
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
    }


def _chain_from_feature_dict(pdb_dict: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    if len(pdb_dict) == 0:
        return None, "no protein chain inside the length window"
    if len(pdb_dict) > 1:
        chain_ids = ", ".join(str(chain_id) for chain_id in pdb_dict)
        return None, f"expected one protein chain, found {len(pdb_dict)}: {chain_ids}"
    return next(iter(pdb_dict.values())), None


def _validated_chain(chain: Dict, domain_id: str) -> Tuple[Optional[Dict], Optional[Dict[str, str]]]:
    missing = [key for key in CHAIN_KEYS if key not in chain]
    if missing:
        return None, _error_record(
            domain_id,
            "verify_single_domain",
            "ValueError",
            f"chain feature is missing keys: {missing}",
        )
    sequence = str(chain["resseq"])
    n_positions = int(np.asarray(chain["cb_positions"]).shape[0])
    if not sequence:
        return None, _error_record(
            domain_id,
            "verify_single_domain",
            "ValueError",
            "chain sequence is empty",
        )
    if len(sequence) != n_positions:
        return None, _error_record(
            domain_id,
            "verify_single_domain",
            "ValueError",
            f"sequence length {len(sequence)} does not match {n_positions} CB positions",
        )
    return chain, None


def build_concatenated_cb_label(
    chain1: Dict,
    chain2: Dict,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Full CB distance matrix for chain1 followed by chain2.

    The returned minimum is taken from the inter-domain block only.
    """
    cb1 = np.asarray(chain1["cb_positions"], dtype=np.float32)
    cb2 = np.asarray(chain2["cb_positions"], dtype=np.float32)
    mask1 = np.asarray(chain1["cb_mask"], dtype=bool)
    mask2 = np.asarray(chain2["cb_mask"], dtype=bool)
    n1 = int(cb1.shape[0])

    cb_pos = np.concatenate([cb1, cb2], axis=0)
    cb_mask = np.concatenate([mask1, mask2], axis=0)
    pairwise_dist = np.linalg.norm(
        cb_pos[:, None, :] - cb_pos[None, :, :],
        axis=2,
    ).astype(np.float32)
    pairwise_mask = cb_mask[:, None] & cb_mask[None, :]

    cross = pairwise_dist[:n1, n1:]
    cross_mask = pairwise_mask[:n1, n1:]
    valid = cross[cross_mask]
    min_dist = float(np.min(valid)) if valid.size else float("nan")
    return pairwise_dist, pairwise_mask, min_dist


def _write_fasta(records: Sequence[Tuple[str, str]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def _map_tasks(func, tasks: Sequence, workers: int, chunksize: int, desc: str) -> List:
    tasks = list(tasks)
    if workers > 1 and len(tasks) > 1:
        with mp.Pool(workers) as pool:
            iterator = pool.imap_unordered(func, tasks, chunksize=max(int(chunksize), 1))
            return list(tqdm(iterator, total=len(tasks), desc=desc))
    return [func(task) for task in tqdm(tasks, desc=desc)]


class AFDBVerifier:
    def __init__(
        self,
        root_dir: str,
        dompdb_dir: str,
        min_chain_length: int = 30,
        max_chain_length: int = 1024,
        num_workers: int = 1,
        chunksize: int = 1,
        overwrite: bool = False,
        contact_threshold: float = 8.0,
        max_pair_elements: int = 4000000,
    ):
        self.root_dir = Path(root_dir)
        self.dompdb_dir = Path(dompdb_dir)
        self.min_chain_length = int(min_chain_length)
        self.max_chain_length = int(max_chain_length)
        self.num_workers = int(num_workers)
        self.chunksize = int(chunksize)
        self.overwrite = bool(overwrite)
        self.contact_threshold = float(contact_threshold)
        self.max_pair_elements = int(max_pair_elements)

    def verify_domains(self, domain_ids: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        os.makedirs(self.root_dir / "processed", exist_ok=True)
        domain_ids = [str(domain_id) for domain_id in domain_ids]
        results = _map_tasks(
            self.verify_single_domain,
            domain_ids,
            workers=self.num_workers,
            chunksize=self.chunksize,
            desc="Step 2/5 Verify domain PDBs",
        )

        verified_rows = []
        error_rows = []
        fasta_records = {}
        for result in results:
            if result.get("error") is not None:
                error_rows.append(result["error"])
                continue
            verified_rows.append(
                {
                    "domain_id": result["domain_id"],
                    "parsed_length": result["parsed_length"],
                    "pdb_path": result["pdb_path"],
                }
            )
            fasta_records[result["domain_id"]] = result["sequence"]

        verified = pd.DataFrame.from_records(verified_rows)
        errors = pd.DataFrame.from_records(
            error_rows,
            columns=["domain_id", "stage", "error_type", "error_message"],
        )
        fasta_path = self.root_dir / "processed_domains.fasta"
        _write_fasta(
            [(name, fasta_records[name]) for name in sorted(fasta_records)],
            fasta_path,
        )
        print(f"Processed domain FASTA written: {fasta_path} ({len(fasta_records)} domains)")
        print(
            f"Verification errors: {len(errors)}/{len(domain_ids)} domains"
        )
        if not errors.empty:
            print("Example verification errors:")
            print(errors.head(10).to_string(index=False))
        return verified, errors

    def verify_single_domain(self, domain_id: str) -> Dict:
        domain_id = str(domain_id)
        pickle_path = self.root_dir / "processed" / f"{domain_id}.pkl"
        try:
            if pickle_path.exists() and not self.overwrite:
                with open(pickle_path, "rb") as handle:
                    chain = pickle.load(handle)
                chain, error = self._accept_chain(chain, domain_id)
                if error is not None:
                    return {"error": error}
                structure_path = resolve_domain_pdb(self.dompdb_dir, domain_id)
                return {
                    "domain_id": domain_id,
                    "sequence": str(chain["resseq"]),
                    "parsed_length": len(chain["resseq"]),
                    "pdb_path": "" if structure_path is None else str(structure_path),
                    "error": None,
                }

            structure_path = resolve_domain_pdb(self.dompdb_dir, domain_id)
            if structure_path is None:
                folder = self.dompdb_dir / domain_shard(domain_id)
                return {
                    "error": _error_record(
                        domain_id,
                        "verify_single_domain",
                        "FileNotFoundError",
                        f"missing structure under {folder} ({', '.join(DOMAIN_EXTENSIONS)})",
                    )
                }

            pdb_dict = extract_domain_features(
                structure_path,
                min_chain_length=self.min_chain_length,
                max_chain_length=self.max_chain_length,
            )
            chain, chain_error = _chain_from_feature_dict(pdb_dict)
            if chain_error is not None:
                return {
                    "error": _error_record(
                        domain_id,
                        "verify_single_domain",
                        "ValueError",
                        chain_error,
                    )
                }
            chain, error = self._accept_chain(chain, domain_id)
            if error is not None:
                return {"error": error}

            with open(pickle_path, "wb") as handle:
                pickle.dump(chain, handle)
            return {
                "domain_id": domain_id,
                "sequence": str(chain["resseq"]),
                "parsed_length": len(chain["resseq"]),
                "pdb_path": str(structure_path),
                "error": None,
            }
        except Exception as exc:
            return {
                "error": _error_record(
                    domain_id,
                    "verify_single_domain",
                    type(exc).__name__,
                    str(exc),
                )
            }

    def _accept_chain(self, chain: Dict, domain_id: str) -> Tuple[Optional[Dict], Optional[Dict[str, str]]]:
        chain, error = _validated_chain(chain, domain_id)
        if error is not None:
            return None, error
        length = len(chain["resseq"])
        if length < self.min_chain_length or length > self.max_chain_length:
            return None, _error_record(
                domain_id,
                "verify_single_domain",
                "ValueError",
                f"chain length {length} is outside [{self.min_chain_length}, {self.max_chain_length}]",
            )
        return chain, None

    def label_positive_pairs(
        self,
        pairs: pd.DataFrame,
        verified_ids: Iterable[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        os.makedirs(self.root_dir / "labels", exist_ok=True)
        verified_ids = set(verified_ids)
        tasks = pairs.loc[:, ["domain1", "domain2", "PAIRID", "HASH", "CLUSTER", "LEN"]]
        tasks = tasks.drop_duplicates(["domain1", "domain2"])
        present = tasks["domain1"].isin(verified_ids) & tasks["domain2"].isin(verified_ids)
        ready = tasks.loc[present].to_dict("records")
        missing_rows = []
        for pair in tasks.loc[~present].to_dict("records"):
            domain1 = str(pair["domain1"])
            domain2 = str(pair["domain2"])
            missing = [domain for domain in (domain1, domain2) if domain not in verified_ids]
            missing_rows.append(
                {
                    "chain1": domain1,
                    "chain2": domain2,
                    "label_path": str(self.root_dir / "labels" / f"{domain1}__{domain2}.pkl"),
                    "min_inter_domain_distance": "",
                    "PAIRID": pair.get("PAIRID", ""),
                    "HASH": pair.get("HASH", ""),
                    "CLUSTER": pair.get("CLUSTER", ""),
                    "LEN": pair.get("LEN", ""),
                    "reason": "missing_domain",
                    "error_type": "FileNotFoundError",
                    "error_message": f"domain did not verify: {', '.join(missing)}",
                }
            )
        results = _map_tasks(
            self.label_single_pair,
            ready,
            workers=self.num_workers,
            chunksize=self.chunksize,
            desc="Step 3/5 Label positive pairs",
        )

        verified_rows = [result["record"] for result in results if result["status"] == "verified"]
        rejected_rows = missing_rows + [
            result["record"] for result in results if result["status"] != "verified"
        ]
        verified = pd.DataFrame.from_records(verified_rows, columns=VERIFIED_PAIR_COLUMNS)
        rejected = pd.DataFrame.from_records(rejected_rows, columns=REJECTED_PAIR_COLUMNS)

        verified_path = self.root_dir / "verified_positive_pairs.tsv"
        rejected_path = self.root_dir / "rejected_positive_pairs.tsv"
        verified.to_csv(verified_path, sep="\t", index=False)
        rejected.to_csv(rejected_path, sep="\t", index=False)
        print(f"Verified positive pairs written: {verified_path} ({len(verified)} pairs)")
        print(f"Rejected positive pairs written: {rejected_path} ({len(rejected)} pairs)")
        if not rejected.empty:
            print(rejected["reason"].value_counts().to_string())
        return verified, rejected

    def label_single_pair(self, pair: Dict) -> Dict:
        domain1 = str(pair["domain1"])
        domain2 = str(pair["domain2"])
        label_path = self.root_dir / "labels" / f"{domain1}__{domain2}.pkl"
        record = {
            "chain1": domain1,
            "chain2": domain2,
            "label_path": str(label_path),
            "min_inter_domain_distance": "",
            "PAIRID": pair.get("PAIRID", ""),
            "HASH": pair.get("HASH", ""),
            "CLUSTER": pair.get("CLUSTER", ""),
            "LEN": pair.get("LEN", ""),
            "reason": "",
            "error_type": "",
            "error_message": "",
        }
        feature1 = self.root_dir / "processed" / f"{domain1}.pkl"
        feature2 = self.root_dir / "processed" / f"{domain2}.pkl"
        if not feature1.is_file() or not feature2.is_file():
            missing = [name for name, path in ((domain1, feature1), (domain2, feature2)) if not path.is_file()]
            record["reason"] = "missing_domain"
            record["error_type"] = "FileNotFoundError"
            record["error_message"] = f"missing processed features: {', '.join(missing)}"
            return {"status": "rejected", "record": record}

        if label_path.is_file() and not self.overwrite:
            return {"status": "verified", "record": record}

        try:
            with open(feature1, "rb") as handle:
                chain1 = pickle.load(handle)
            with open(feature2, "rb") as handle:
                chain2 = pickle.load(handle)
            n1 = int(np.asarray(chain1["cb_positions"]).shape[0])
            n2 = int(np.asarray(chain2["cb_positions"]).shape[0])
            if n1 * n2 > self.max_pair_elements:
                record["reason"] = "oversized"
                record["error_type"] = "ValueError"
                record["error_message"] = (
                    f"pair size {n1}x{n2} exceeds max_pair_elements={self.max_pair_elements}"
                )
                return {"status": "rejected", "record": record}

            pairwise_dist, pairwise_mask, min_dist = build_concatenated_cb_label(chain1, chain2)
            record["min_inter_domain_distance"] = "" if not np.isfinite(min_dist) else float(min_dist)
            if not np.isfinite(min_dist):
                record["reason"] = "no_atoms"
                record["error_type"] = "ValueError"
                record["error_message"] = "no masked inter-domain CB pairs"
                return {"status": "rejected", "record": record}
            if min_dist >= self.contact_threshold:
                record["reason"] = "no_contact"
                record["error_message"] = (
                    f"minimum inter-domain CB distance {min_dist:.3f} "
                    f">= {self.contact_threshold}"
                )
                if label_path.exists() and self.overwrite:
                    label_path.unlink()
                return {"status": "rejected", "record": record}

            with open(label_path, "wb") as handle:
                pickle.dump(
                    {
                        "pairwise_dist": pairwise_dist.astype(np.float32),
                        "pairwise_mask": pairwise_mask.astype(bool),
                    },
                    handle,
                )
            return {"status": "verified", "record": record}
        except Exception as exc:
            record["reason"] = "error"
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)
            return {"status": "rejected", "record": record}


def build_test_set_fasta(root_dir: Path, benchmark_tsv: str, uniprot_fasta: str) -> Path:
    benchmark_path = Path(benchmark_tsv)
    uniprot_path = Path(uniprot_fasta)
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Cannot find benchmark TSV: {benchmark_path}")
    if not uniprot_path.is_file():
        raise FileNotFoundError(f"Cannot find UniProt FASTA: {uniprot_path}")

    test_set_data = pd.read_csv(benchmark_path, sep="\t")
    test_set_pairs = [str(pair).split("_") for pair in test_set_data["Protein pairs"]]
    test_set_labels = test_set_data["Category"].tolist()
    unique_uniprot_ids = np.unique(np.array(test_set_pairs, dtype=object).flatten())
    print(f"Number of unique uniprot ids in test set: {len(unique_uniprot_ids)}")

    raw_uniprot_fasta = load_uniprot_fasta_by_accession(str(uniprot_path))
    test_set_fasta: Dict[str, str] = {}
    for uniprot_id in unique_uniprot_ids:
        query_id = str(uniprot_id).strip()
        sequence = raw_uniprot_fasta.get(query_id, "")
        if not sequence and "-" in query_id:
            sequence = raw_uniprot_fasta.get(query_id.split("-", maxsplit=1)[0], "")
        test_set_fasta[query_id] = sequence

    test_set_fasta_path = root_dir / "test_set.fasta"
    missing_items = []
    with open(test_set_fasta_path, "w", encoding="utf-8") as handle:
        for uniprot_id, sequence in tqdm(
            test_set_fasta.items(),
            desc="Writing test set FASTA",
            unit="sequence",
        ):
            if sequence:
                handle.write(f">{uniprot_id}\n{sequence}\n")
            else:
                missing_items.append(uniprot_id)
    print(f"Number of missing items in test set: {len(missing_items)}")
    print(f"Test set FASTA written: {test_set_fasta_path}")

    missing_ids = set(missing_items)
    kept_pairs = []
    kept_labels = []
    for pair, label in zip(test_set_pairs, test_set_labels):
        if pair[0] in missing_ids or pair[1] in missing_ids:
            continue
        kept_pairs.append("_".join(pair))
        kept_labels.append(str(label).strip() == "positive")
    test_set_table = pd.DataFrame({"Protein pairs": kept_pairs, "Category": kept_labels})
    test_set_table_path = root_dir / "test_set.tsv"
    test_set_table.to_csv(test_set_table_path, sep="\t", index=False)
    print(f"Test set data written: {test_set_table_path} ({len(test_set_table)} pairs)")
    return test_set_fasta_path


def write_domain_table(
    domain_ids: Iterable[str],
    sequences: Dict[str, str],
    table_path: Path,
    fasta_path: Path,
    description: str,
    require_sequences: bool = True,
) -> pd.DataFrame:
    rows = []
    missing = []
    for domain_id in sorted(set(str(domain_id) for domain_id in domain_ids)):
        sequence = sequences.get(domain_id, "")
        if not sequence:
            missing.append(domain_id)
            continue
        rows.append({"id": domain_id, "sequence": sequence})
    if missing and require_sequences:
        raise RuntimeError(
            f"{len(missing)} {description} have no sequence. Example: {missing[:5]}"
        )
    if missing:
        print(
            f"{len(missing)} {description} have no sequence and are left out of {fasta_path.name}. "
            f"Example: {missing[:5]}"
        )
    table = pd.DataFrame.from_records(rows, columns=["id", "sequence"])
    table.to_csv(table_path, sep="\t", index=False)
    _write_fasta([(row["id"], row["sequence"]) for _, row in table.iterrows()], fasta_path)
    print(f"{description} table written: {table_path} ({len(table)} domains)")
    print(f"{description} FASTA written: {fasta_path}")
    return table


def _without_similar_domains(
    pairs: pd.DataFrame,
    left_column: str,
    right_column: str,
    similar_ids: set,
) -> pd.DataFrame:
    if pairs.empty or not similar_ids:
        return pairs.reset_index(drop=True)
    contains_similar = pairs[left_column].astype(str).isin(similar_ids) | pairs[right_column].astype(str).isin(
        similar_ids
    )
    return pairs.loc[~contains_similar].reset_index(drop=True)


def filter_pairs_against_test_set(
    root_dir: Path,
    positive_pairs: pd.DataFrame,
    negative_pairs: pd.DataFrame,
    verified_domain_ids: Sequence[str],
    sequences: Dict[str, str],
    test_set_fasta: Path,
    identity_threshold: float,
    coverage: float,
    mmseqs_prefix: str,
    overwrite: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    domain_table_path = root_dir / "verified_domains.sequences.tsv"
    domain_fasta_path = root_dir / "verified_domains.fasta"
    domain_table = write_domain_table(
        verified_domain_ids,
        sequences,
        domain_table_path,
        domain_fasta_path,
        "Verified-domain",
    )

    if domain_table.empty:
        print("No verified domains to compare with the test set; no pairs removed for similarity.")
        similar_ids: set = set()
    else:
        selector = PDBDataSelector(root_dir=str(root_dir))
        mmseqs_tsv = selector.run_mmseqs_cross_set_search(
            query_fasta=str(test_set_fasta),
            target_fasta=str(domain_fasta_path),
            output_prefix=mmseqs_prefix,
            min_seq_id=identity_threshold,
            coverage=coverage,
            overwrite=overwrite,
        )
        if Path(mmseqs_tsv).stat().st_size == 0:
            print("MMSeqs2 result is empty; no domains are similar to the test set.")
            similar_ids = set()
        else:
            kept, _, _ = selector.filter_dissimilar_pdb_chains(
                pdb_subset_table=str(domain_table_path),
                mmseqs_tsv=mmseqs_tsv,
                identity_threshold=identity_threshold,
                output_table_filename="verified_domains.dissimilar.txt",
                output_fasta_filename="verified_domains.dissimilar.fasta",
            )
            similar_ids = set(domain_table["id"].astype(str)) - set(kept["id"].astype(str))
        print(f"Domains similar to the test set: {len(similar_ids)}")

    filtered_positive = _without_similar_domains(positive_pairs, "chain1", "chain2", similar_ids)
    filtered_negative = _without_similar_domains(negative_pairs, "domain1", "domain2", similar_ids)
    filtered_negative = filtered_negative.copy()
    filtered_negative["chain1"] = filtered_negative["domain1"].astype(str)
    filtered_negative["chain2"] = filtered_negative["domain2"].astype(str)

    positive_path = root_dir / "verified_positive_pairs.dissimilar.tsv"
    negative_path = root_dir / "verified_negative_pairs.dissimilar.tsv"
    filtered_positive.to_csv(positive_path, sep="\t", index=False)
    filtered_negative.to_csv(negative_path, sep="\t", index=False)
    print(
        f"Positive pairs kept after test-set filter: {positive_path} "
        f"({len(filtered_positive)}/{len(positive_pairs)} pairs)"
    )
    print(
        f"Negative pairs kept after test-set filter: {negative_path} "
        f"({len(filtered_negative)}/{len(negative_pairs)} pairs)"
    )

    positive_domain_ids: List[str] = []
    if not filtered_positive.empty:
        positive_domain_ids = sorted(
            set(filtered_positive["chain1"].astype(str)) | set(filtered_positive["chain2"].astype(str))
        )
    write_domain_table(
        positive_domain_ids,
        sequences,
        root_dir / "positive_domains.dissimilar.txt",
        root_dir / "positive_domains.dissimilar.fasta",
        "Remaining positive-pair domain",
    )
    negative_domain_ids: List[str] = []
    if not filtered_negative.empty:
        negative_domain_ids = sorted(
            set(filtered_negative["chain1"].astype(str)) | set(filtered_negative["chain2"].astype(str))
        )
    write_domain_table(
        negative_domain_ids,
        sequences,
        root_dir / "negative_domains.dissimilar.txt",
        root_dir / "negative_domains.dissimilar.fasta",
        "Remaining negative-pair domain",
        require_sequences=False,
    )
    return filtered_positive, filtered_negative


def cluster_homodimer_ppi_pairs(
    homodimer_tsv: str,
    member_to_rep: Dict[str, str],
    output_clustered_tsv: str,
) -> pd.DataFrame:
    """Assign each homodimer to the cluster of its shared representative."""
    df_homo = pd.read_csv(homodimer_tsv, sep="\t")
    if df_homo.empty:
        df_homo["cluster_rep_chain1"] = pd.Series(dtype=str)
        df_homo["cluster_rep_chain2"] = pd.Series(dtype=str)
        df_homo["ppi_cluster_id"] = pd.Series(dtype=str)
        df_homo.to_csv(output_clustered_tsv, sep="\t", index=False)
        print(f"Homodimer PPI clusters written: {output_clustered_tsv} (0 pairs, 0 clusters)")
        return df_homo

    rep1 = df_homo["chain1"].map(lambda chain: member_to_rep.get(chain, chain))
    rep2 = df_homo["chain2"].map(lambda chain: member_to_rep.get(chain, chain))
    pair_keys = []
    for left, right in zip(rep1, rep2):
        if left == right:
            pair_keys.append((left,))
        else:
            pair_keys.append(tuple(sorted((left, right))))
    unique_keys = sorted(set(pair_keys))
    key_to_id = {key: f"ppi_cluster_{i:06d}" for i, key in enumerate(unique_keys, start=1)}
    df_homo = df_homo.copy()
    df_homo["cluster_rep_chain1"] = list(rep1)
    df_homo["cluster_rep_chain2"] = list(rep2)
    df_homo["ppi_cluster_id"] = [key_to_id[key] for key in pair_keys]
    df_homo.to_csv(output_clustered_tsv, sep="\t", index=False)
    print(
        f"Homodimer PPI clusters written: {output_clustered_tsv} "
        f"({len(df_homo)} pairs, {len(unique_keys)} clusters)"
    )
    return df_homo


def _write_empty_cluster_tables(root_dir: Path, output_names: Sequence[str]) -> None:
    empty = pd.DataFrame(columns=["chain1", "chain2", "label_path", "ppi_cluster_id"])
    for name in output_names:
        empty.to_csv(root_dir / name, sep="\t", index=False)


def cluster_remaining_pairs(
    root_dir: Path,
    pairs_tsv: str,
    fasta_path: str,
    threads: int,
    overwrite: bool,
    cluster90_prefix: str,
    cluster30_prefix: str,
    output_names: Sequence[str],
    label: str,
) -> None:
    fasta = Path(fasta_path)
    if not fasta.is_file() or fasta.stat().st_size == 0:
        print(f"No {label} domains left after the test-set filter; skipping self clustering.")
        _write_empty_cluster_tables(root_dir, output_names)
        return

    cluster90 = run_mmseqs_cluster(
        fasta_path=str(fasta),
        output_prefix=cluster90_prefix,
        min_seq_id=0.9,
        coverage=0.9,
        cov_mode=0,
        threads=threads,
        overwrite=overwrite,
    )
    cluster30 = run_mmseqs_cluster(
        fasta_path=str(fasta),
        output_prefix=cluster30_prefix,
        min_seq_id=0.3,
        coverage=0.0,
        cov_mode=0,
        threads=threads,
        overwrite=overwrite,
    )
    member_to_rep90, _ = parse_mmseqs_cluster_tsv(cluster90["cluster_tsv"])
    member_to_rep30, _ = parse_mmseqs_cluster_tsv(cluster30["cluster_tsv"])

    homo_name, hetero_name, homo_clustered, homo_duplicate, hetero_clustered, hetero_duplicate = output_names
    homodimer_path = root_dir / homo_name
    heterodimer_path = root_dir / hetero_name
    df_homo, df_hetero = split_homodimer_heterodimer_pairs(
        verified_pairs_tsv=pairs_tsv,
        member_to_rep90=member_to_rep90,
        output_homo_tsv=str(homodimer_path),
        output_hetero_tsv=str(heterodimer_path),
    )

    df_hetero_clustered = cluster_heterodimer_ppi_pairs(
        heterodimer_tsv=str(heterodimer_path),
        member_to_rep30=member_to_rep30,
        output_clustered_tsv=str(root_dir / hetero_clustered),
    )
    df_hetero_duplicate = cluster_heterodimer_ppi_pairs(
        heterodimer_tsv=str(heterodimer_path),
        member_to_rep30=member_to_rep90,
        output_clustered_tsv=str(root_dir / hetero_duplicate),
    )
    df_homo_clustered = cluster_homodimer_ppi_pairs(
        homodimer_tsv=str(homodimer_path),
        member_to_rep=member_to_rep30,
        output_clustered_tsv=str(root_dir / homo_clustered),
    )
    df_homo_duplicate = cluster_homodimer_ppi_pairs(
        homodimer_tsv=str(homodimer_path),
        member_to_rep=member_to_rep90,
        output_clustered_tsv=str(root_dir / homo_duplicate),
    )
    print(
        f"Step 5 complete ({label}): "
        f"{len(df_homo)} homodimers, {len(df_hetero)} heterodimers, "
        f"{df_hetero_clustered['ppi_cluster_id'].nunique() if not df_hetero_clustered.empty else 0} heterodimer clusters, "
        f"{df_hetero_duplicate['ppi_cluster_id'].nunique() if not df_hetero_duplicate.empty else 0} heterodimer duplicate clusters, "
        f"{df_homo_clustered['ppi_cluster_id'].nunique() if not df_homo_clustered.empty else 0} homodimer clusters, "
        f"{df_homo_duplicate['ppi_cluster_id'].nunique() if not df_homo_duplicate.empty else 0} homodimer duplicate clusters."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AFDB domain entries for SurPass training.")
    parser.add_argument("-r", "--root-dir", default="./AFDB_DDI", help="Output root directory.")
    parser.add_argument(
        "-b",
        "--benchmark-tsv",
        default="./benchmarks/positives_and_negatives.tsv",
        help="Benchmark TSV containing 'Protein pairs' and 'Category' columns.",
    )
    parser.add_argument(
        "-u",
        "--uniprot-fasta",
        default="./uniprot_sprot.fasta.gz",
        help="Local UniProt FASTA (.gz) used to build the test-set FASTA.",
    )
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    parser.add_argument("-k", "--chunksize", type=int, default=32, help="Chunk size for multiprocessing.")
    parser.add_argument("-m", "--min-chain-len", type=int, default=20, help="Minimum chain length to keep.")
    parser.add_argument("-M", "--max-chain-len", type=int, default=1024, help="Maximum chain length to keep.")
    parser.add_argument(
        "-P",
        "--max-pair-elements",
        type=int,
        default=4000000,
        help="Skip positive pairs where len(domain1)*len(domain2) exceeds this value.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=8.0,
        help="Maximum inter-domain CB distance, in angstroms, accepted as a positive contact.",
    )
    parser.add_argument("-c", "--coverage", type=float, default=0.8, help="MMSeqs2 coverage for the test-set search.")
    parser.add_argument(
        "-t",
        "--identity-threshold",
        type=float,
        default=0.3,
        help="Drop pairs that contain a domain at or above this identity to the test set.",
    )
    parser.add_argument(
        "-p",
        "--mmseqs-prefix",
        default="test_vs_verified_domains",
        help="Output prefix for the test-set versus verified-domain MMSeqs2 search.",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="Recompute domain features, positive labels, and MMSeqs2 results when outputs already exist.",
    )
    return parser


def main(args: argparse.Namespace):
    if args.min_chain_len < 1 or args.max_chain_len < args.min_chain_len:
        raise ValueError("Chain length window must satisfy 1 <= min_chain_len <= max_chain_len.")
    if args.contact_threshold <= 0:
        raise ValueError("contact_threshold must be positive.")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")

    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    dompdb_dir = root_dir / "dompdbs"
    if not dompdb_dir.is_dir():
        raise FileNotFoundError(f"Cannot find domain PDB directory: {dompdb_dir}")

    print("Step 1/5 Load AFDB metadata")
    posi = load_pair_table(root_dir / "posi_all.csv")
    nega = load_pair_table(root_dir / "nega_all.csv")
    domain_meta = collect_domain_metadata(posi, nega)
    print(
        f"Metadata pairs: {len(posi)} positive, {len(nega)} negative; "
        f"{len(domain_meta)} unique domains "
        f"({int(domain_meta['in_positive'].sum())} in the positive table, "
        f"{int(domain_meta['in_negative'].sum())} in the negative table)"
    )

    verifier = AFDBVerifier(
        root_dir=str(root_dir),
        dompdb_dir=str(dompdb_dir),
        min_chain_length=args.min_chain_len,
        max_chain_length=args.max_chain_len,
        num_workers=args.workers,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
        contact_threshold=args.contact_threshold,
        max_pair_elements=args.max_pair_elements,
    )
    verified_domains, verify_errors = verifier.verify_domains(domain_meta["domain_id"].tolist())
    if verified_domains.empty:
        verified_domains = pd.DataFrame(columns=["domain_id", "parsed_length", "pdb_path"])
    verified_domains = verified_domains.merge(domain_meta, on="domain_id", how="left")
    verified_domains = verified_domains.sort_values("domain_id").reset_index(drop=True)
    verified_domain_path = root_dir / "verified_domains.tsv"
    error_path = root_dir / "verify_errors.csv"
    verified_domains.to_csv(verified_domain_path, sep="\t", index=False)
    verify_errors.to_csv(error_path, index=False)
    print(f"Verified domains written: {verified_domain_path} ({len(verified_domains)} domains)")
    print(f"Verification errors written: {error_path}")

    sequences = {}
    fasta_path = root_dir / "processed_domains.fasta"
    if fasta_path.is_file():
        current_name = None
        chunks: List[str] = []
        with open(fasta_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_name is not None:
                        sequences[current_name] = "".join(chunks)
                    current_name = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
        if current_name is not None:
            sequences[current_name] = "".join(chunks)

    verified_pairs, _ = verifier.label_positive_pairs(
        posi,
        verified_domains["domain_id"].tolist(),
    )
    print(f"Step 3 complete: {len(verified_pairs)} positive pairs contain an inter-domain contact.")

    print("Step 4/5 Remove domains similar to the test set")
    test_set_fasta = build_test_set_fasta(root_dir, args.benchmark_tsv, args.uniprot_fasta)
    filtered_pairs, _ = filter_pairs_against_test_set(
        root_dir=root_dir,
        positive_pairs=verified_pairs,
        negative_pairs=nega,
        verified_domain_ids=verified_domains["domain_id"].astype(str).tolist(),
        sequences=sequences,
        test_set_fasta=test_set_fasta,
        identity_threshold=args.identity_threshold,
        coverage=args.coverage,
        mmseqs_prefix=args.mmseqs_prefix,
        overwrite=args.overwrite,
    )

    print("Step 5/5 Cluster remaining positive and negative pairs")
    cluster_remaining_pairs(
        root_dir=root_dir,
        pairs_tsv=str(root_dir / "verified_positive_pairs.dissimilar.tsv"),
        fasta_path=str(root_dir / "positive_domains.dissimilar.fasta"),
        threads=args.workers,
        overwrite=args.overwrite,
        cluster90_prefix="positive_domains_c90_cov90",
        cluster30_prefix="positive_domains_c30",
        output_names=CLUSTER_OUTPUTS,
        label="positive",
    )
    cluster_remaining_pairs(
        root_dir=root_dir,
        pairs_tsv=str(root_dir / "verified_negative_pairs.dissimilar.tsv"),
        fasta_path=str(root_dir / "negative_domains.dissimilar.fasta"),
        threads=args.workers,
        overwrite=args.overwrite,
        cluster90_prefix="negative_domains_c90_cov90",
        cluster30_prefix="negative_domains_c30",
        output_names=NEGATIVE_CLUSTER_OUTPUTS,
        label="negative",
    )
    if filtered_pairs.empty:
        print("No positive pairs remained after the test-set filter.")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
