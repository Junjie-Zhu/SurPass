"""
    Prepare PDB entries for training model
    1. download pdb chains (fasta) from rcsb
    2. link pdb ids with uniprot ids
    3. filter ppi pairs recorded by given tsv, where structure is available
"""

import os
import wget
import gzip
import shutil
import glob
from collections import defaultdict

import numpy as np
import pandas as pd


class DataRetriever:
    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = root_dir
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
        for filename, url in self.metadata.items():
            if not os.path.exists(os.path.join(self.root_dir, filename)):
                print(f"Downloading {filename}...")
                wget.download(url, out=os.path.join(self.root_dir, filename))
            else:
                print(f"File {filename} already exists")

    def unzip_metadata(self):
        for filename in self.metadata.keys():
            if not filename.endswith(".gz") or os.path.exists(os.path.join(self.root_dir, filename.replace(".gz", ""))):
                continue
            with gzip.open(os.path.join(self.root_dir, filename), 'rb') as f_in:
                with open(os.path.join(self.root_dir, filename.replace(".gz", "")), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def _parse_uniprot_id(self):
        uniprot_mapping = {}
        with gzip.open(
            os.path.join(self.root_dir, "pdb_chain_cath_uniprot.tsv.gz"), "rt"
        ) as f:
            for line in f:
                try:
                    pdb, chain, uniprot_id, cath_id = line.strip().split("\t")
                    key = f"{pdb}_{chain}"
                    uniprot_mapping[key] = uniprot_id
                except ValueError:
                    continue
        return uniprot_mapping


def get_seq_ppi_data(seq_ppi_root):
    pos_data = glob.glob(os.path.join(seq_ppi_root, "*_pos_rr.txt"))
    # neg_data = glob.glob(os.path.join(seq_ppi_root, "*_neg_rr.txt"))

    # read pos data
    pos_pairs = []
    for pos_file in pos_data:
        with open(pos_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(" ")
                pos_pairs.append(data)
    pos_pairs = np.array(pos_pairs)
    unique_pos_chains = np.unique(pos_pairs.flatten())
    return pos_pairs, unique_pos_chains

if __name__ == "__main__":
    retriever = DataRetriever(root_dir="./pdb_metadata")
    retriever.download_metadata()
    retriever.unzip_metadata()
    uniprot_id_dict = retriever._parse_uniprot_id()

    pos_pairs, unique_pos_chains = get_seq_ppi_data(seq_ppi_root="./seq_ppi_data")

    # one uniprot id can correspond to multiple pdb chains, reverse the dictionary
    uniprot_pdb_dict = defaultdict(list)
    for pdb, uniprot in uniprot_id_dict.items():
        if uniprot not in uniprot_pdb_dict:
            uniprot_pdb_dict[uniprot] = [pdb]
        else:
            uniprot_pdb_dict[uniprot].append(pdb)
    # uniprot_pdb_dict = {k: set(v) for k, v in uniprot_pdb_dict.items()}

    # check if we can find structure for pos pairs
    valid_pos_pairs = []
    for pos_pair in pos_pairs:
        pdb_pos0, pdb_pos1 = uniprot_pdb_dict.get(pos_pair[0], []), uniprot_pdb_dict.get(pos_pair[1], [])
        if len(pdb_pos0) == 0 or len(pdb_pos1) == 0:
            continue
        
        # # check if the same pdb id can be found for both pos chains
        # pdb_pos0 = set([pdb.split("_")[0].lower() for pdb in pdb_pos0])
        # pdb_pos1 = set([pdb.split("_")[0].lower() for pdb in pdb_pos1])
        # if len(pdb_pos0 & pdb_pos1) == 0:
        #     continue

        valid_pos_pairs.append(pos_pair)
    print(f"Number of valid pos pairs: {len(valid_pos_pairs)}")
    print(f"Number of pos pairs: {len(pos_pairs)}")


