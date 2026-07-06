
import os
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from E1.batch_preparer import E1BatchPreparer
from E1.modeling import E1ForMaskedLM

FEATURE_DTYPES = {
    "atom14_positions": torch.float32,
    "atom14_mask": torch.bool,
    "cb_positions": torch.float32,
    "cb_mask": torch.bool,
    "residue_type": torch.long,
    "residue_index": torch.long,
}


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    device = "cuda:0"
    model = E1ForMaskedLM.from_pretrained("Profluent-Bio/E1-600m").to(device)
    model.eval()

    verified_ppi_pairs = pd.read_csv("/lustre/home/acct-clschf/clschf/jjzhu/datasets/ppi_dataset/pdb_metadata/verified_heterodimers.tsv", sep="\t")
    all_chains = set(verified_ppi_pairs["chain1"].tolist() + verified_ppi_pairs["chain2"].tolist())

    print(f"all chains: {len(all_chains)}")

    # feed verified seqs to model and get embeddings
    feature_path = Path("/lustre/home/acct-clschf/clschf/jjzhu/datasets/ppi_dataset/pdb_metadata/processed")
    embedding_path = Path("/lustre/home/acct-clschf/clschf/jjzhu/datasets/ppi_dataset/pdb_metadata/embedded")
    embedding_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "verified": len(all_chains),
        "saved": 0,
        "missing_features": 0,
        "failed": 0,
        "residues": 0,
        "embedding_dim": None,
    }
    batch_preparer = E1BatchPreparer()
    errors = []

    for chain_id in tqdm(sorted(all_chains), desc="Extracting PLM embeddings"):
        feature_file = feature_path / f"{chain_id}.pkl"
        output_file = embedding_path / f"{chain_id}.pt"

        if not feature_file.exists():
            stats["missing_features"] += 1
            errors.append(f"{chain_id}: missing feature file {feature_file}")
            continue

        try:
            with open(feature_file, "rb") as f:
                feature_dict = pickle.load(f)
            seq = feature_dict["resseq"]

            plm_emb = generate_embeddings_from_request(
                request_payload={"chain_id": chain_id, "sequence": seq},
                model=model,
                device=device,
                msa_directory="",
                embed_mode="single",
                batch_preparer=batch_preparer,
            )
            features = tensorize_feature_dict(feature_dict)
            features["plm_emb"] = plm_emb

            feature_len = _infer_feature_length(features)
            if feature_len is not None and feature_len != plm_emb.shape[0]:
                raise ValueError(
                    f"feature length {feature_len} does not match embedding length {plm_emb.shape[0]}"
                )

            torch.save(features, output_file)
            stats["saved"] += 1
            stats["residues"] += int(plm_emb.shape[0])
            stats["embedding_dim"] = int(plm_emb.shape[-1])
        except Exception as exc:
            stats["failed"] += 1
            errors.append(f"{chain_id}: {type(exc).__name__}: {exc}")

    print("\nEmbedding extraction summary")
    print(f"verified chains with sequences: {stats['verified']}")
    print(f"saved chains: {stats['saved']}")
    print(f"missing feature files: {stats['missing_features']}")
    print(f"failed chains: {stats['failed']}")
    print(f"total embedded residues: {stats['residues']}")
    print(f"embedding dim: {stats['embedding_dim']}")
    print(f"output directory: {embedding_path}")
    if errors:
        print("example errors:")
        for error in errors[:10]:
            print(f"  - {error}")


def tensorize_feature_dict(feature_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    features = {}
    for key, value in feature_dict.items():
        tensor = _to_feature_tensor(key, value)
        if tensor is not None:
            features[key] = tensor
    return features


def _to_feature_tensor(key: str, value: Any) -> Optional[torch.Tensor]:
    if key == "chain_index":
        return _encode_chain_index(value)

    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        return None

    dtype = FEATURE_DTYPES.get(key)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _encode_chain_index(chain_index: Any) -> torch.Tensor:
    if isinstance(chain_index, torch.Tensor):
        return chain_index.long()

    values = chain_index.tolist() if isinstance(chain_index, np.ndarray) else list(chain_index)
    mapping = {value: i for i, value in enumerate(sorted(set(values)))}
    return torch.tensor([mapping[value] for value in values], dtype=torch.long)


def _batched(items, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _infer_feature_length(features: Dict[str, torch.Tensor]) -> Optional[int]:
    for key in ("residue_type", "residue_index", "atom14_positions", "cb_positions", "cb_mask", "atom14_mask"):
        value = features.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _is_cuda_device(device: Any) -> bool:
    """Check if the device is a CUDA device."""
    if hasattr(device, "type"):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def _build_sequence_input(
    chain_request: Dict[str, Any],
    msa_directory: str,
    embed_mode: str
) -> str:
    """
    Build the input sequence based on embed_mode:
    - 'single': return the one-letter amino acid sequence directly.
    - 'msa': attempt to load .a3m from msa_directory, sample context,
             and return 'context,sequence' if successful; otherwise fallback to single.
    """
    chain_seq_1 = chain_request["sequence"]
    return chain_seq_1


def generate_embeddings_from_request(
    request_payload: Dict[str, Any],
    model,
    device,
    msa_directory: str,
    embed_mode: str,
    batch_preparer: Optional[E1BatchPreparer] = None,
) -> torch.Tensor:
    """
    Generate embeddings for all protein chains in the request.

    Args:
        request_payload: contains 'chain_requests' list; each item must have
                         'chain_id' and 'sequence'. If embed_mode='msa', also
                         require 'msa_basename'.
        model: E1ForMaskedLM instance (already loaded and in eval mode).
        device: torch.device.
        msa_directory: directory containing .a3m files (used only for MSA mode).
        embed_mode: 'single' or 'msa'.
        batch_preparer: E1BatchPreparer instance; creates a new one if None.

    Returns:
        dict {chain_id: numpy.ndarray} of shape (L, D), where L is number of
        residues and D is embedding dimension.
    """
    if batch_preparer is None:
        batch_preparer = E1BatchPreparer()

    autocast_enabled = _is_cuda_device(device)

    sequence_input = _build_sequence_input(request_payload, msa_directory, embed_mode)
    batch = batch_preparer.get_batch_kwargs([sequence_input], device=device)

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled)
        if hasattr(torch, "autocast")
        else nullcontext()
    )

    with torch.inference_mode():
        with autocast_ctx:
            outputs = model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        embeddings = outputs.embeddings  # shape: (batch, seq_len, hidden_dim)

        # Extract residue-level embeddings for the last sequence (target protein)
        last_sequence_selector = batch["sequence_ids"] == batch["sequence_ids"].max(dim=1)[0][:, None]
        residue_selector = ~(batch_preparer.get_boundary_token_mask(batch["input_ids"]))
        last_sequence_residue_selector = last_sequence_selector & residue_selector
        last_sequence_embeddings = embeddings[0, last_sequence_residue_selector[0]]

    return last_sequence_embeddings.to(dtype=torch.float32).cpu()


if __name__ == "__main__":
    main()