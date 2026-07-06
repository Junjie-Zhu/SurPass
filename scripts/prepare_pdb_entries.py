"""
    Prepare PDB entries for training model
    1. download pdb chains (fasta) from rcsb
    2. gather test set ppi from external source
    3. collect all sequences, calculate similarity with MMSeqs2
    4. filter out pdb chains of high similarity against test set (30% identity)
    5. find interacting pdb chains in the filtered set (by structure, atom distance < 4.5A)

    To do:
    Get available pdb chains for test set proteins
"""

import os
import argparse
import multiprocessing as mp
from itertools import combinations
from collections import defaultdict
import wget
import gzip
import shutil
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import biotite.database.uniprot as uniprot
import biotite.structure.io as strucio
import biotite.structure as struc
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from src.common.residue_constants import (
        restype_name_to_atom14_names, 
        restype_3to1,
        restype_1to3,
        restype_order_with_x,
    )
except:
    from residue_constants import (
        restype_name_to_atom14_names, 
        restype_3to1,
        restype_1to3,
        restype_order_with_x,
    )


class DataRetriever:
    def __init__(
        self,
        root_dir: str,
        num_workers: int = 32,
    ):
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.pdb_sequences_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
        )
        self.source_map_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/index/source.idx"
        )
        self.resolution_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/index/resolu.idx"
        )
        self.pdb_entry_type_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt"
        )
        self.pdb_deposition_date_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
        )
        self.pdb_availability_url = (
            "https://files.wwpdb.org/pub/pdb/compatible/pdb_bundle/pdb_bundle_index.txt"
        )
        self.pdb_chain_cath_uniprot_url = (
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_cath_uniprot.tsv.gz"
        )
        self.metadata = {
            i.split("/")[-1]: i for i in [
                self.pdb_sequences_url,
                self.source_map_url,
                self.resolution_url,
                self.pdb_entry_type_url,
                self.pdb_deposition_date_url,
                self.pdb_availability_url,
                self.pdb_chain_cath_uniprot_url,
            ]
        }

    def download_metadata(self):
        os.makedirs(self.root_dir, exist_ok=True)
        metadata_items = list(self.metadata.items())
        for filename, url in tqdm(metadata_items, desc="Step 1/7 Download metadata"):
            if not os.path.exists(os.path.join(self.root_dir, filename)):
                print(f"Downloading {filename}...")
                wget.download(url, out=os.path.join(self.root_dir, filename))
            else:
                print(f"File {filename} already exists")

    def unzip_metadata(self):
        for filename in tqdm(list(self.metadata.keys()), desc="Step 1/7 Unzip metadata"):
            if not filename.endswith(".gz") or os.path.exists(os.path.join(self.root_dir, filename.replace(".gz", ""))):
                continue
            with gzip.open(os.path.join(self.root_dir, filename), 'rb') as f_in:
                with open(os.path.join(self.root_dir, filename.replace(".gz", "")), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


class PDBDataSelector:
    def __init__(
        self,
        root_dir: str,
        min_length: int = None,
        max_length: int = None,
        molecule_type: str = None,
        experiment_types: List[str] = None,
        oligomeric_min: int = None,
        worst_resolution: float = None,
        remove_non_standard_residues: bool = True,
        remove_pdb_unavailable: bool = True,
        num_workers: int = 32,
    ):
        self.root_dir = Path(root_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.molecule_type = molecule_type
        self.experiment_types = experiment_types
        self.oligomeric_min = oligomeric_min
        self.worst_resolution = worst_resolution
        self.remove_non_standard_residues = remove_non_standard_residues
        self.remove_pdb_unavailable = remove_pdb_unavailable
        self.num_workers = num_workers

    @staticmethod
    def _read_fasta_records(fasta_path: Path) -> List[Tuple[str, str]]:
        records: List[Tuple[str, str]] = []
        header: Optional[str] = None
        sequence_lines: List[str] = []

        with open(fasta_path, "r", encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"Step 4/7 Parse {fasta_path.name}"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        records.append((header, "".join(sequence_lines)))
                    header = line[1:].strip()
                    sequence_lines = []
                else:
                    sequence_lines.append(line.upper())

        if header is not None:
            records.append((header, "".join(sequence_lines)))
        return records

    @staticmethod
    def _parse_seqres_record(header: str, sequence: str) -> Optional[Dict]:
        params = header.split()
        if len(params) < 3:
            return None

        chain_id = params[0]
        if "_" not in chain_id:
            return None

        try:
            length = int(params[2].split(":")[1])
        except (IndexError, ValueError):
            length = len(sequence)

        molecule_type = params[1].split(":")[1] if ":" in params[1] else params[1]
        return {
            "id": chain_id,
            "pdb": chain_id.split("_")[0].lower(),
            "chain": chain_id.split("_")[1],
            "length": length,
            "molecule_type": molecule_type.lower(),
            "name": " ".join(params[3:]) if len(params) > 3 else "",
            "sequence": sequence,
        }

    def _parse_experiment_type_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        file_path = self.root_dir / "pdb_entry_type.txt"
        if not file_path.exists():
            return mapping

        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    parts = line.strip().split()
                if len(parts) >= 3:
                    mapping[parts[0].lower()] = parts[2].lower()
        return mapping

    def _parse_resolution_map(self) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        file_path = self.root_dir / "resolu.idx"
        if not file_path.exists():
            return mapping

        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                if ";" in line:
                    left, right = [x.strip() for x in line.split(";", maxsplit=1)]
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    left, right = parts[0], parts[-1]
                try:
                    mapping[left.lower()] = float(right)
                except ValueError:
                    continue
        return mapping

    def _parse_unavailable_pdbs(self) -> set:
        file_path = self.root_dir / "pdb_bundle_index.txt"
        if not file_path.exists():
            return set()

        unavailable = set()
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                pdb_id = line.strip().lower()
                if pdb_id:
                    unavailable.add(pdb_id)
        return unavailable

    def _parse_source_map(self) -> Dict[str, str]:
        source_map: Dict[str, str] = {}
        file_path = self.root_dir / "source.idx"
        if not file_path.exists():
            return source_map

        skip_tokens = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"Step 7/7 Parse {file_path.name}"):
                params = line.strip().split()
                if not params:
                    continue
                if params[0] in skip_tokens:
                    continue
                source_map[params[0].lower()] = " ".join(params[1:])

        for key in ["protein", "idcode", "------"]:
            source_map.pop(key, None)
        return source_map

    @staticmethod
    def _is_standard_sequence(sequence: str) -> bool:
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        return all(residue in standard_aa for residue in sequence)

    @staticmethod
    def _write_fasta(df: pd.DataFrame, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            for _, row in df.iterrows():
                handle.write(f">{row['id']}\n{row['sequence']}\n")

    def process_pdb_chain_sequences(
        self,
        seqres_filename: str = "pdb_seqres.txt",
        output_table_filename: str = "pdb_seqres.subset.txt",
        output_fasta_filename: str = "pdb_seqres.subset.fasta",
    ) -> Tuple[pd.DataFrame, str, str]:
        seqres_path = self.root_dir / seqres_filename
        if not seqres_path.exists():
            raise FileNotFoundError(f"Cannot find sequence file: {seqres_path}")

        records = []
        for header, sequence in tqdm(self._read_fasta_records(seqres_path), desc=f"Step 4/7 Parse {seqres_path.name}"):
            record = self._parse_seqres_record(header, sequence)
            if record is not None:
                records.append(record)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            raise ValueError(f"No chain records parsed from {seqres_path}")

        if self.min_length is not None:
            df = df.loc[df["length"] >= self.min_length]
        if self.max_length is not None:
            df = df.loc[df["length"] <= self.max_length]
        if self.molecule_type is not None:
            df = df.loc[df["molecule_type"] == self.molecule_type.lower()]

        if self.remove_non_standard_residues:
            df = df.loc[df["sequence"].map(self._is_standard_sequence)]

        if self.oligomeric_min is not None:
            chain_counts = df.groupby("pdb")["id"].transform("count")
            df = df.loc[chain_counts >= self.oligomeric_min]

        if self.experiment_types:
            exp_map = self._parse_experiment_type_map()
            df["experiment_type"] = df["pdb"].map(exp_map)
            normalized_exp_types = {exp_type.lower() for exp_type in self.experiment_types}
            df = df.loc[df["experiment_type"].isin(normalized_exp_types)]

        if self.worst_resolution is not None:
            resolution_map = self._parse_resolution_map()
            df["resolution"] = df["pdb"].map(resolution_map)
            df = df.loc[df["resolution"].notna() & (df["resolution"] <= self.worst_resolution)]

        if self.remove_pdb_unavailable:
            unavailable = self._parse_unavailable_pdbs()
            if unavailable:
                df = df.loc[~df["pdb"].isin(unavailable)]

        df = df.reset_index(drop=True)
        output_table_path = self.root_dir / output_table_filename
        output_fasta_path = self.root_dir / output_fasta_filename

        df.to_csv(output_table_path, sep="\t", index=False)
        self._write_fasta(df, output_fasta_path)
        print(f"PDB chain subset written: {output_table_path} ({len(df)} chains)")
        print(f"PDB chain FASTA written: {output_fasta_path}")
        return df, str(output_table_path), str(output_fasta_path)

    def run_mmseqs_cross_set_search(
        self,
        query_fasta: str,
        target_fasta: str,
        output_prefix: str = "test_vs_pdb",
        min_seq_id: float = 0.3,
        coverage: float = 0.8,
        overwrite: bool = False,
    ) -> str:
        if shutil.which("mmseqs") is None:
            raise RuntimeError(
                "MMseqs2 not found. Install with: conda install -c conda-forge -c bioconda mmseqs2"
            )

        query_fasta_path = Path(query_fasta)
        target_fasta_path = Path(target_fasta)
        output_dir = self.root_dir / "mmseqs2"
        output_dir.mkdir(parents=True, exist_ok=True)

        target_db = output_dir / f"{output_prefix}_target_db"
        query_db = output_dir / f"{output_prefix}_query_db"
        result_db = output_dir / f"{output_prefix}_result_db"
        tmp_dir = output_dir / "tmp"
        result_tsv = output_dir / f"{output_prefix}.tsv"

        if result_tsv.exists() and not overwrite:
            print(f"MMSeqs2 results already exist: {result_tsv}")
            return str(result_tsv)

        if overwrite and result_tsv.exists():
            result_tsv.unlink()

        commands = [
            ["mmseqs", "createdb", str(target_fasta_path), str(target_db)],
            ["mmseqs", "createdb", str(query_fasta_path), str(query_db)],
            [
                "mmseqs",
                "search",
                str(query_db),
                str(target_db),
                str(result_db),
                str(tmp_dir),
                "--min-seq-id",
                str(min_seq_id),
                "-c",
                str(coverage),
                "--cov-mode",
                "1",
            ],
            [
                "mmseqs",
                "convertalis",
                str(query_db),
                str(target_db),
                str(result_db),
                str(result_tsv),
                "--format-output",
                "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits",
            ],
        ]
        for cmd in tqdm(commands, desc="Step 5/7 MMSeqs2 workflow"):
            subprocess.run(cmd, check=True)

        print(f"MMSeqs2 cross-set search written: {result_tsv}")
        return str(result_tsv)

    def filter_dissimilar_pdb_chains(
        self,
        pdb_subset_table: str,
        mmseqs_tsv: str,
        identity_threshold: float = 0.3,
        output_table_filename: str = "pdb_seqres.dissimilar.txt",
        output_fasta_filename: str = "pdb_seqres.dissimilar.fasta",
    ) -> Tuple[pd.DataFrame, str, str]:
        df_subset = pd.read_csv(pdb_subset_table, sep="\t")
        if df_subset.empty:
            raise ValueError(f"PDB subset table is empty: {pdb_subset_table}")

        mmseqs_columns = [
            "query",
            "target",
            "fident",
            "alnlen",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bits",
        ]
        df_mmseqs = pd.read_csv(mmseqs_tsv, sep="\t", names=mmseqs_columns)

        if df_mmseqs.empty:
            print("MMSeqs2 result is empty; keeping all chains as dissimilar.")
            df_dissimilar = df_subset.copy()
        else:
            threshold = identity_threshold
            if df_mmseqs["fident"].max() > 1.0 and threshold <= 1.0:
                threshold *= 100.0
            similar_targets = set(
                df_mmseqs.loc[df_mmseqs["fident"] >= threshold, "target"].astype(str)
            )
            df_dissimilar = df_subset.loc[~df_subset["id"].astype(str).isin(similar_targets)].copy()

        df_dissimilar = df_dissimilar.reset_index(drop=True)
        output_table_path = self.root_dir / output_table_filename
        output_fasta_path = self.root_dir / output_fasta_filename

        df_dissimilar.to_csv(output_table_path, sep="\t", index=False)
        self._write_fasta(df_dissimilar, output_fasta_path)

        print(
            f"Dissimilar subset written: {output_table_path} "
            f"({len(df_dissimilar)} / {len(df_subset)} chains kept)"
        )
        print(f"Dissimilar FASTA written: {output_fasta_path}")
        return df_dissimilar, str(output_table_path), str(output_fasta_path)

    def gather_potential_ppi(
        self,
        pdb_chain_table: str,
        output_filename: str = "pdb_seqres.potential_ppi.tsv",
    ) -> Tuple[pd.DataFrame, str]:
        df = pd.read_csv(pdb_chain_table, sep="\t")
        if df.empty:
            raise ValueError(f"PDB chain table is empty: {pdb_chain_table}")

        source_map = self._parse_source_map()
        df["source"] = df["pdb"].astype(str).str.lower().map(source_map)
        df = df.loc[df["source"].notna() & (df["source"].astype(str).str.len() > 0)].copy()

        pair_records: List[Dict[str, str]] = []
        grouped = df.groupby(["pdb", "source"])
        for (pdb_id, source), group in tqdm(
            grouped,
            desc="Step 7/7 Build potential PPI pairs",
            total=grouped.ngroups,
        ):
            chain_ids = sorted(group["id"].astype(str).unique().tolist())
            if len(chain_ids) < 2:
                continue
            for chain1, chain2 in combinations(chain_ids, 2):
                pair_records.append(
                    {
                        "pdb": pdb_id,
                        "source": source,
                        "chain1": chain1,
                        "chain2": chain2,
                    }
                )

        df_pairs = pd.DataFrame.from_records(pair_records)
        output_path = self.root_dir / output_filename
        df_pairs.to_csv(output_path, sep="\t", index=False)
        print(f"Potential PPI pairs written: {output_path} ({len(df_pairs)} pairs)")
        return df_pairs, str(output_path)


def fetch_one_uniprot_sequence(uniprot_id: str) -> str:
    try:
        entry = fetch_entry(uniprot_id, dataset="uniprotkb", output_dir="./pdb_metadata/test_set")
        sequence = extract_sequence(entry)
        return f">{uniprot_id}\n{sequence}"
    except Exception as e:
        # print(f"Error fetching UniProt sequence for {uniprot_id}: {e}")
        return None


def parse_uniprot_accession(header: str) -> str:
    # UniProt FASTA headers usually start with sp|ACC|ENTRY or tr|ACC|ENTRY.
    first_token = header.strip().split(maxsplit=1)[0]
    header_parts = first_token.split("|")
    if len(header_parts) >= 2 and header_parts[1]:
        return header_parts[1]
    return first_token


def load_uniprot_fasta_by_accession(path: str) -> Dict[str, str]:
    fasta_by_accession: Dict[str, str] = {}
    current_header: Optional[str] = None
    sequence_chunks: List[str] = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for raw_line in tqdm(f, desc="Step 3/7 Load UniProt FASTA"):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    accession = parse_uniprot_accession(current_header)
                    if accession not in fasta_by_accession:
                        fasta_by_accession[accession] = "".join(sequence_chunks)
                current_header = line[1:]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)

    if current_header is not None:
        accession = parse_uniprot_accession(current_header)
        if accession not in fasta_by_accession:
            fasta_by_accession[accession] = "".join(sequence_chunks)

    return fasta_by_accession


def extract_pdb_features(pdb_path: str, min_chain_length: int = 30, max_chain_length: int = 1024):
    # only retain the first model
    structure = strucio.load_structure(pdb_path)
    try:
        struc_depth = structure.stack_depth()
        structure = structure[0]
    except AttributeError:
        struc_depth = 1

    atom14_positions = []
    atom14_mask = []
    residue_index = []
    chain_index = []
    residue_type = []
    residue_seq = []

    for residue in struc.residue_iter(structure):
        if residue[0].hetero:
            continue
        restype = restype_3to1.get(residue[0].res_name, "X")
        restype_3 = restype_1to3.get(restype, "UNK")
        restype_idx = restype_order_with_x[restype]

        atom14_names = restype_name_to_atom14_names.get(
            restype_3, restype_name_to_atom14_names["UNK"]
        )
        atom_name_to_idx = {
            atom_name: i for i, atom_name in enumerate(atom14_names) if atom_name
        }

        pos = np.zeros((14, 3), dtype=np.float32)
        mask = np.zeros((14,), dtype=np.float32)
        for atom in residue:
            atom_name = atom.atom_name.strip()
            atom_idx = atom_name_to_idx.get(atom_name)
            if atom_idx is None:
                continue
            pos[atom_idx] = atom.coord.astype(np.float32)
            mask[atom_idx] = 1.0

        atom14_positions.append(pos)
        atom14_mask.append(mask)
        residue_type.append(restype_idx)
        residue_seq.append(restype)
        residue_index.append(int(residue[0].res_id))
        chain_index.append(residue[0].chain_id)

    # get CB positions (when not available, use CA)
    atom14_positions_arr = np.array(atom14_positions, dtype=np.float32)
    atom14_mask_arr = np.array(atom14_mask, dtype=np.float32)
    residue_type_arr = np.array(residue_type, dtype=np.int64)
    residue_index_arr = np.array(residue_index, dtype=np.int64)
    chain_index_arr = np.array(chain_index)
    residue_seq_arr = np.array(residue_seq)

    cb_pos = atom14_positions_arr[:, 4, :]
    cb_mask = atom14_mask_arr[:, 4] > 0
    ca_pos = atom14_positions_arr[:, 1, :]
    ca_mask = atom14_mask_arr[:, 1] > 0
    cb_pos = np.where(cb_mask[:, None], cb_pos, ca_pos)
    cb_mask = cb_mask | ca_mask

    # split by chain_index
    chain_ids = np.unique(chain_index_arr)
    pdb_dict = {}
    for chain_id in chain_ids:
        chain_mask = chain_index_arr == chain_id
        chain_dict = {
            "atom14_positions": atom14_positions_arr[chain_mask],
            "atom14_mask": atom14_mask_arr[chain_mask],
            "residue_type": residue_type_arr[chain_mask],
            "residue_index": residue_index_arr[chain_mask],
            "chain_index": chain_index_arr[chain_mask],
            "cb_positions": cb_pos[chain_mask],
            "cb_mask": cb_mask[chain_mask],
            "resseq": "".join(residue_seq_arr[chain_mask].tolist()),
        }
        if len(chain_dict["atom14_positions"]) < min_chain_length or len(chain_dict["atom14_positions"]) > max_chain_length:
            continue
        pdb_dict[chain_id] = chain_dict
    return pdb_dict


def identify_interacting_chains(
    pdb_dict: Dict[str, Dict[str, np.ndarray]],
):
    chain_combinations = combinations(pdb_dict.keys(), 2)
    valid_combinations = {
        "chain1": [],
        "chain2": [],
        "pairwise_dist": [],
        "pairwise_mask": [],
    }
    for chain1, chain2 in chain_combinations:
        chain1_cb_pos = pdb_dict[chain1]["cb_positions"]
        chain2_cb_pos = pdb_dict[chain2]["cb_positions"]
        chain1_cb_mask = pdb_dict[chain1]["cb_mask"]
        chain2_cb_mask = pdb_dict[chain2]["cb_mask"]
        pair_elements = int(chain1_cb_pos.shape[0] * chain2_cb_pos.shape[0])

        # get the pairwise distance map
        pairwise_dist = np.linalg.norm(
            chain1_cb_pos[:, None, :] - chain2_cb_pos[None, :, :], 
            axis=2
        )
        pairwise_mask = chain1_cb_mask[:, None] & chain2_cb_mask[None, :]

        # check if the combination contains inter-chain contacts
        pdist = pairwise_dist[pairwise_mask]
        if pdist.size == 0:
            continue
        pdist_min = np.min(pdist)

        if pdist_min > 8.0:
            continue

        # get the full pairwise distance map
        cb_pos = np.concatenate([chain1_cb_pos, chain2_cb_pos], axis=0)
        cb_mask = np.concatenate([chain1_cb_mask, chain2_cb_mask], axis=0)
        pairwise_dist = np.linalg.norm(
            cb_pos[:, None, :] - cb_pos[None, :, :], 
            axis=2
        )
        pairwise_mask = cb_mask[:, None] & cb_mask[None, :]

        valid_combinations["chain1"].append(chain1)
        valid_combinations["chain2"].append(chain2)
        valid_combinations["pairwise_dist"].append(pairwise_dist)  # save the full pairwise distance map
        valid_combinations["pairwise_mask"].append(pairwise_mask)
    return valid_combinations


class PDBVerifier:
    # download and verify pdb files
    def __init__(
        self,
        root_dir: str,
        format: str = "pdb",
        min_chain_length: int = 30,
        max_chain_length: int = 1024,
        num_workers: int = 32,
        chunksize: int = 32,
        overwrite: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.num_workers = num_workers
        self.min_chain_length = min_chain_length
        self.max_chain_length = max_chain_length
        self.chunksize = chunksize
        self.overwrite = overwrite
        if format == "pdb":
            self.base_url = "https://files.rcsb.org/download/"
            self.extension = ".pdb"
        elif format == "mmtf":
            self.base_url = "https://mmtf.rcsb.org/v1.0/full/"
            self.extension = ".mmtf.gz"
        elif format == "cif" or format == "mmcif":
            self.base_url = "https://files.rcsb.org/download/"
            self.extension = ".cif.gz"
        elif format == "bcif":
            self.base_url = "https://models.rcsb.org/"
            self.extension = ".bcif.gz"
        else:
            raise ValueError(
                f"Invalid format: {format}. Must be 'pdb', 'mmtf', '(mm)cif' or 'bcif'."
            )

    def download_multiple_pdbs(self, pdb_ids: List[str]):
        os.makedirs(self.root_dir / "pdb", exist_ok=True)
        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(self.download_single_pdb, pdb_ids, chunksize=self.chunksize,),
                        total=len(pdb_ids),
                        desc="Downloading PDB files",
                        unit="file",
                    )
                )
        else:
            results = []
            for pdb_id in tqdm(pdb_ids, desc="Downloading PDB files", unit="file"):
                results.append(self.download_single_pdb(pdb_id))

        # summarize results
        downloaded_pdbs = [result for result in results if result is not None]
        print(f"Downloaded PDB files: {len(downloaded_pdbs)}/{len(pdb_ids)}")
        return downloaded_pdbs

    def download_single_pdb(self, pdb_id: str):
        pdb_id = pdb_id.strip().lower()
        pdb_path = self.root_dir / "pdb" / f"{pdb_id}{self.extension}"
        if pdb_path.exists() and not self.overwrite:
            return str(pdb_path)

        try:
            wget.download(
                f"{self.base_url}{pdb_id}{self.extension}",
                out=str(pdb_path),
                bar=None,
            )
            return str(pdb_path)
        except Exception as e:
            # print(f"Error downloading {pdb_id}: {e}")
            return None

    def verify_multiple_pdbs(self, pdb_ids: List[str]):
        os.makedirs(self.root_dir / "processed", exist_ok=True)  # save processed chain features
        os.makedirs(self.root_dir / "labels_full", exist_ok=True)  # save pairwise labels as separate files
        pair_records: List[Dict[str, str]] = []
        error_records: List[Dict[str, str]] = []
        chain_fasta_records: Dict[str, str] = {}

        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                result_iter = pool.imap_unordered(
                    self.verify_single_pdb,
                    pdb_ids,
                    chunksize=self.chunksize,
                )
                for result in tqdm(result_iter, total=len(pdb_ids), desc="Verifying PDB files", unit="file"):
                    if result is None:
                        continue
                    if result.get("error") is not None:
                        error_records.append(result["error"])
                        continue
                    pair_records.extend(result["pairs"])
                    for chain_name, chain_seq in result.get("chain_fasta_records", []):
                        chain_fasta_records[chain_name] = chain_seq
        else:
            for pdb_id in tqdm(pdb_ids, desc="Verifying PDB files", unit="file"):
                result = self.verify_single_pdb(pdb_id)
                if result is None:
                    continue
                if result.get("error") is not None:
                    error_records.append(result["error"])
                    continue
                pair_records.extend(result["pairs"])
                for chain_name, chain_seq in result.get("chain_fasta_records", []):
                    chain_fasta_records[chain_name] = chain_seq

        verified_pairs = pd.DataFrame.from_records(pair_records)
        output_path = self.root_dir / "verified_pdb_pairs.tsv"
        verified_pairs.to_csv(output_path, sep="\t", index=False)
        print(f"Verified PDB pairs written: {output_path} ({len(verified_pairs)} pairs)")

        chain_fasta_path = self.root_dir / "processed_chains.fasta"
        with open(chain_fasta_path, "w", encoding="utf-8") as f:
            for chain_name in sorted(chain_fasta_records):
                f.write(f">{chain_name}\n{chain_fasta_records[chain_name]}\n")
        print(f"Processed chain FASTA written: {chain_fasta_path} ({len(chain_fasta_records)} chains)")

        error_output_path = self.root_dir / "verify_errors.csv"
        error_df = pd.DataFrame.from_records(
            error_records,
            columns=["pdb", "stage", "error_type", "error_message"],
        )
        error_df.to_csv(error_output_path, index=False)
        print(
            f"Verification errors: {len(error_records)}/{len(pdb_ids)} PDB systems "
            f"(saved to {error_output_path})"
        )
        if not error_df.empty:
            print("Example verification errors:")
            print(error_df.head(10).to_string(index=False))
        return verified_pairs, str(output_path)

    def verify_single_pdb(self, pdb_id: str):
        # get interacting chains for given pdb id
        pdb_path = self.root_dir / "pdb" / f"{pdb_id}{self.extension}"
        if not pdb_path.exists():
            return None

        try:
            pdb_dict = extract_pdb_features(pdb_path, min_chain_length=self.min_chain_length, max_chain_length=self.max_chain_length)
            valid_combinations = identify_interacting_chains(pdb_dict)
            required_chains = np.unique(np.array(valid_combinations["chain1"] + valid_combinations["chain2"]))
            chain_fasta_records: List[Tuple[str, str]] = []
            for chain_id in required_chains:
                # save chain dict to pickle
                chain_dict = pdb_dict[chain_id]
                chain_name = f"{pdb_id}_{chain_id}"
                with open(self.root_dir / "processed" / f"{chain_name}.pkl", "wb") as f:
                    pickle.dump(chain_dict, f)
                chain_fasta_records.append((chain_name, chain_dict["resseq"]))

            # save pairwise labels and gather verified combinations
            verified_pairs: List[Dict[str, str]] = []
            for chain1, chain2, pairwise_dist, pairwise_mask in zip(
                valid_combinations["chain1"],
                valid_combinations["chain2"],
                valid_combinations["pairwise_dist"],
                valid_combinations["pairwise_mask"],
            ):
                chain1_name = f"{pdb_id}_{chain1}"
                chain2_name = f"{pdb_id}_{chain2}"
                pair_name = f"{chain1_name}__{chain2_name}"
                label_path = self.root_dir / "labels" / f"{pair_name}.pkl"
                with open(label_path, "wb") as f:
                    pickle.dump(
                        {
                            "pairwise_dist": pairwise_dist.astype(np.float32),
                            "pairwise_mask": pairwise_mask.astype(bool),
                        },
                        f,
                    )

                verified_pairs.append(
                    {
                        "pdb": pdb_id,
                        "chain1": chain1_name,
                        "chain2": chain2_name,
                        "label_path": str(label_path),
                    }
                )
            return {
                "pairs": verified_pairs,
                "chain_fasta_records": chain_fasta_records,
                "error": None,
            }
        except Exception as e:
            return {
                "pairs": [],
                "chain_fasta_records": [],
                "error": {
                    "pdb": pdb_id,
                    "stage": "verify_single_pdb",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            }


def run_mmseqs_cluster(
    fasta_path: str,
    output_prefix: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    overwrite: bool = False,
) -> Dict[str, str]:
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "MMseqs2 not found. Install with: conda install -c conda-forge -c bioconda mmseqs2"
        )

    fasta_path = Path(fasta_path)
    mmseqs_dir = fasta_path.parent / "mmseqs2_cluster"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)

    prefix_path = mmseqs_dir / output_prefix
    tmp_dir = mmseqs_dir / f"{output_prefix}_tmp"
    cluster_tsv = mmseqs_dir / f"{output_prefix}_cluster.tsv"
    rep_fasta = mmseqs_dir / f"{output_prefix}_rep_seq.fasta"
    all_fasta = mmseqs_dir / f"{output_prefix}_all_seqs.fasta"

    if cluster_tsv.exists() and not overwrite:
        print(f"MMSeqs2 cluster results already exist: {cluster_tsv}")
        return {
            "cluster_tsv": str(cluster_tsv),
            "rep_fasta": str(rep_fasta),
            "all_fasta": str(all_fasta),
        }

    cmd = [
        "mmseqs",
        "easy-cluster",
        str(fasta_path),
        str(prefix_path),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]
    subprocess.run(cmd, check=True)
    if not cluster_tsv.exists():
        raise RuntimeError(f"Expected MMSeqs2 cluster TSV not found: {cluster_tsv}")

    print(f"MMSeqs2 clustering written: {cluster_tsv}")
    return {
        "cluster_tsv": str(cluster_tsv),
        "rep_fasta": str(rep_fasta),
        "all_fasta": str(all_fasta),
    }


def parse_mmseqs_cluster_tsv(cluster_tsv: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    member_to_rep: Dict[str, str] = {}
    rep_to_members: Dict[str, List[str]] = defaultdict(list)
    with open(cluster_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep_id, member_id = parts[0], parts[1]
            member_to_rep[member_id] = rep_id
            rep_to_members[rep_id].append(member_id)
            if rep_id not in member_to_rep:
                member_to_rep[rep_id] = rep_id
    return member_to_rep, dict(rep_to_members)


def split_homodimer_heterodimer_pairs(
    verified_pairs_tsv: str,
    member_to_rep90: Dict[str, str],
    output_homo_tsv: str,
    output_hetero_tsv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_pairs = pd.read_csv(verified_pairs_tsv, sep="\t")

    df_pairs["cluster90_chain1"] = df_pairs["chain1"].map(lambda x: member_to_rep90.get(x, x))
    df_pairs["cluster90_chain2"] = df_pairs["chain2"].map(lambda x: member_to_rep90.get(x, x))
    is_homodimer = df_pairs["cluster90_chain1"] == df_pairs["cluster90_chain2"]
    df_homo = df_pairs.loc[is_homodimer].reset_index(drop=True)
    df_hetero = df_pairs.loc[~is_homodimer].reset_index(drop=True)

    df_homo.to_csv(output_homo_tsv, sep="\t", index=False)
    df_hetero.to_csv(output_hetero_tsv, sep="\t", index=False)
    print(f"Homodimers written: {output_homo_tsv} ({len(df_homo)} pairs)")
    print(f"Heterodimers written: {output_hetero_tsv} ({len(df_hetero)} pairs)")
    return df_homo, df_hetero


def cluster_heterodimer_ppi_pairs(
    heterodimer_tsv: str,
    member_to_rep30: Dict[str, str],
    output_clustered_tsv: str,
) -> pd.DataFrame:
    df_hetero = pd.read_csv(heterodimer_tsv, sep="\t")

    rep1 = df_hetero["chain1"].map(lambda x: member_to_rep30.get(x, x))
    rep2 = df_hetero["chain2"].map(lambda x: member_to_rep30.get(x, x))
    pair_keys = [tuple(sorted((a, b))) for a, b in zip(rep1, rep2)]
    df_hetero["cluster30_chain1"] = [k[0] for k in pair_keys]
    df_hetero["cluster30_chain2"] = [k[1] for k in pair_keys]

    unique_keys = sorted(set(pair_keys))
    key_to_id = {key: f"ppi_cluster_{i:06d}" for i, key in enumerate(unique_keys, start=1)}
    df_hetero["ppi_cluster_id"] = [key_to_id[key] for key in pair_keys]
    df_hetero.to_csv(output_clustered_tsv, sep="\t", index=False)
    print(
        f"Heterodimer PPI clusters written: {output_clustered_tsv} "
        f"({len(df_hetero)} pairs, {len(unique_keys)} clusters)"
    )
    return df_hetero


def _parse_experiment_types(value: str) -> List[str]:
    return [exp_type.strip() for exp_type in value.split(",") if exp_type.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PDB entries for SurPass training.")
    parser.add_argument("-r", "--root-dir", default="./pdb_metadata", help="Metadata/output root directory.")
    parser.add_argument(
        "-b",
        "--benchmark-tsv",
        default="./benchmarks/positives_and_negatives.tsv",
        help="Benchmark TSV containing 'Protein pairs' column.",
    )
    parser.add_argument(
        "-u",
        "--uniprot-fasta",
        default="./uniprot_sprot.fasta.gz",
        help="UniProt FASTA (.gz) used to build test-set FASTA.",
    )
    parser.add_argument(
        "-f",
        "--pdb-format",
        default="pdb",
        choices=["pdb", "mmtf", "cif", "mmcif", "bcif"],
        help="Downloaded PDB structure format.",
    )
    parser.add_argument("-w", "--workers", type=int, default=32, help="Number of worker processes.")
    parser.add_argument("-k", "--chunksize", type=int, default=32, help="Chunk size for multiprocessing.")
    parser.add_argument("-m", "--min-chain-len", type=int, default=30, help="Minimum chain length to keep.")
    parser.add_argument("-M", "--max-chain-len", type=int, default=1024, help="Maximum chain length to keep.")
    parser.add_argument("-Z", "--max-pdb-residues", type=int, default=10000, help="Skip PDB entries above this total residue count.")
    parser.add_argument("-P", "--max-pair-elements", type=int, default=4000000, help="Skip chain pairs where len(chain1)*len(chain2) exceeds this value.")
    parser.add_argument(
        "-e",
        "--experiment-types",
        default="diffraction,EM",
        help="Comma-separated experiment types, e.g. diffraction,EM.",
    )
    parser.add_argument("-R", "--worst-resolution", type=float, default=5.0, help="Maximum allowed resolution.")
    parser.add_argument("-c", "--coverage", type=float, default=0.8, help="MMSeqs2 coverage.")
    parser.add_argument(
        "-t",
        "--identity-threshold",
        type=float,
        default=0.3,
        help="Filter threshold for test-set similarity against PDB chains.",
    )
    parser.add_argument(
        "-p",
        "--mmseqs-prefix",
        default="test_vs_pdb_subset",
        help="Output prefix for MMSeqs2 intermediate/result files.",
    )
    parser.add_argument("-O", "--overwrite-pdb", action="store_true", help="Re-download PDB files if present.")
    return parser


def main(args: argparse.Namespace):
    root_dir = Path(args.root_dir)
    experiment_types = _parse_experiment_types(args.experiment_types)

    # # first retrieve metadata
    # retriever = DataRetriever(root_dir=str(root_dir), num_workers=args.workers)
    # retriever.download_metadata()
    # retriever.unzip_metadata()

    # # next process test set ppi
    # test_set_data = pd.read_csv(args.benchmark_tsv, sep="\t")
    # test_set_pairs = [pair.split("_") for pair in test_set_data["Protein pairs"]]
    # test_set_labels = test_set_data["Category"].tolist()
    # unique_uniprot_ids = np.unique(np.array(test_set_pairs).flatten())
    # print(f"Number of unique uniprot ids in test set: {len(unique_uniprot_ids)}")

    # # step 3: get fasta for test set uniprot ids
    # test_set_fasta_path = root_dir / "test_set.fasta"
    # raw_uniprot_fasta = load_uniprot_fasta_by_accession(args.uniprot_fasta)
    # test_set_fasta: Dict[str, str] = {}
    # for uniprot_id in unique_uniprot_ids:
    #     query_id = str(uniprot_id).strip()
    #     sequence = raw_uniprot_fasta.get(query_id, "")
    #     if not sequence and "-" in query_id:
    #         sequence = raw_uniprot_fasta.get(query_id.split("-", maxsplit=1)[0], "")
    #     test_set_fasta[query_id] = sequence

    # missing_items = []
    # with open(test_set_fasta_path, "w") as f:
    #     for uniprot_id, sequence in tqdm(test_set_fasta.items(), desc="Writing test set FASTA", unit="sequence"):
    #         if sequence:
    #             f.write(f">{uniprot_id}\n{sequence}\n")
    #         else:
    #             missing_items.append(uniprot_id)
    # print(f"Number of missing items in test set: {len(missing_items)}")
    # print(f"Test set FASTA written: {test_set_fasta_path}")

    # # remove unavailable pairs from test_set_pairs and save
    # missing_pair_indices = []
    # for i, pair in enumerate(test_set_pairs):
    #     if pair[0] in missing_items or pair[1] in missing_items:
    #         missing_pair_indices.append(i)
    # test_set_pairs = ["_".join(pair) for i, pair in enumerate(test_set_pairs) if i not in missing_pair_indices]
    # test_set_labels = [label.strip() == "positive" for i, label in enumerate(test_set_labels) if i not in missing_pair_indices]
    # test_set_data = pd.DataFrame({"Protein pairs": test_set_pairs, "Category": test_set_labels})
    # test_set_data.to_csv(root_dir / "test_set.tsv", sep="\t", index=False)
    # print(f"Test set data written: {root_dir / 'test_set.tsv'} ({len(test_set_data)} pairs)")

    # # step 4: process pdb chain sequences
    # selector = PDBDataSelector(
    #     root_dir=str(root_dir),
    #     min_length=args.min_chain_len,
    #     max_length=args.max_chain_len,
    #     molecule_type="protein",
    #     experiment_types=experiment_types,
    #     oligomeric_min=2,  # at least 2 chains
    #     worst_resolution=args.worst_resolution,
    #     num_workers=args.workers,
    # )
    # _, pdb_subset_table_path, pdb_subset_fasta_path = selector.process_pdb_chain_sequences(
    #     seqres_filename="pdb_seqres.txt",
    #     output_table_filename="pdb_seqres.subset.txt",
    #     output_fasta_filename="pdb_seqres.subset.fasta",
    # )

    # # step 5: run MMSeqs2 search (test set -> pdb subset)
    # mmseqs_result_path = selector.run_mmseqs_cross_set_search(
    #     query_fasta=str(test_set_fasta_path),
    #     target_fasta=pdb_subset_fasta_path,
    #     output_prefix=args.mmseqs_prefix,
    #     min_seq_id=args.identity_threshold,
    #     coverage=args.coverage,
    # )

    # # step 6: filter out pdb chains of high similarity against test set
    # _, dissimilar_table_path, _ = selector.filter_dissimilar_pdb_chains(
    #     pdb_subset_table=pdb_subset_table_path,
    #     mmseqs_tsv=mmseqs_result_path,
    #     identity_threshold=args.identity_threshold,
    #     output_table_filename="pdb_seqres.dissimilar.txt",
    #     output_fasta_filename="pdb_seqres.dissimilar.fasta",
    # )

    # # step 7: gather potential ppi (same pdb id, same taxon source)
    # ppi_pairs, _ = selector.gather_potential_ppi(
    #     pdb_chain_table=dissimilar_table_path,
    #     output_filename="pdb_seqres.potential_ppi.tsv",
    # )

    ppi_pairs = pd.read_csv(root_dir / "pdb_seqres.potential_ppi.tsv", sep="\t")

    # step 8: download unique pdb files
    pdb_downloader = PDBVerifier(
        root_dir=str(root_dir),
        format=args.pdb_format,
        min_chain_length=args.min_chain_len,
        max_chain_length=args.max_chain_len,
        num_workers=args.workers,
        chunksize=args.chunksize,
        overwrite=args.overwrite_pdb,
    )
    pdb_id_unique = ppi_pairs["pdb"].unique().tolist()
    # pdb_downloader.download_multiple_pdbs(pdb_id_unique)

    # step 9: verify true positive pairs from pdb
    verified_pairs, verified_pairs_path = pdb_downloader.verify_multiple_pdbs(pdb_id_unique)
    print(f"Step 9 complete: {len(verified_pairs)} verified PPI pairs saved to {verified_pairs_path}")

    # # step 10: cluster processed training protein chains
    # cluster_fasta_path = root_dir / "processed_chains.fasta"
    # verified_pairs_path = root_dir / "verified_pdb_pairs.tsv"
    # if not cluster_fasta_path.exists():
    #     raise FileNotFoundError(f"Cannot find training FASTA for step 10: {cluster_fasta_path}")
    # if not verified_pairs_path.exists():
    #     raise FileNotFoundError(f"Cannot find verified pair table for step 10: {verified_pairs_path}")

    # cluster90_outputs = run_mmseqs_cluster(
    #     fasta_path=str(cluster_fasta_path),
    #     output_prefix="processed_chains_c90_cov90",
    #     min_seq_id=0.9,
    #     coverage=0.9,
    #     cov_mode=0,
    #     threads=32,
    #     overwrite=False,
    # )
    # cluster30_outputs = run_mmseqs_cluster(
    #     fasta_path=str(cluster_fasta_path),
    #     output_prefix="processed_chains_c30",
    #     min_seq_id=0.3,
    #     coverage=0.0,
    #     cov_mode=0,
    #     threads=32,
    #     overwrite=False,
    # )

    # member_to_rep90, _ = parse_mmseqs_cluster_tsv(cluster90_outputs["cluster_tsv"])
    # member_to_rep30, _ = parse_mmseqs_cluster_tsv(cluster30_outputs["cluster_tsv"])

    # homodimer_path = root_dir / "verified_homodimers.tsv"
    # heterodimer_path = root_dir / "verified_heterodimers.tsv"
    # df_homo, df_hetero = split_homodimer_heterodimer_pairs(
    #     verified_pairs_tsv=str(verified_pairs_path),
    #     member_to_rep90=member_to_rep90,
    #     output_homo_tsv=str(homodimer_path),
    #     output_hetero_tsv=str(heterodimer_path),
    # )

    # heterodimer_clustered_path = root_dir / "verified_heterodimers.clustered.tsv"
    # heterodimer_duplicate_path = root_dir / "verified_heterodimers.duplicate.tsv"
    # df_hetero_clustered = cluster_heterodimer_ppi_pairs(
    #     heterodimer_tsv=str(heterodimer_path),
    #     member_to_rep30=member_to_rep30,
    #     output_clustered_tsv=str(heterodimer_clustered_path),
    # )
    # df_hetero_duplicate = cluster_heterodimer_ppi_pairs(
    #     heterodimer_tsv=str(heterodimer_path),
    #     member_to_rep30=member_to_rep90,
    #     output_clustered_tsv=str(heterodimer_duplicate_path),
    # )
    # print(
    #     "Step 10 complete: "
    #     f"{len(df_homo)} homodimers, {len(df_hetero)} heterodimers, "
    #     f"{df_hetero_clustered['ppi_cluster_id'].nunique() if not df_hetero_clustered.empty else 0} heterodimer clusters."
    #     f"{df_hetero_duplicate['ppi_cluster_id'].nunique() if not df_hetero_duplicate.empty else 0} heterodimer duplicate clusters."
    # )

if __name__ == "__main__":
    main(build_arg_parser().parse_args())
