# SurPass

SurPass predicts a rectangular inter-protein distance/contact map from two
single-sequence PLM embeddings. It does not use an MSA. The primary PPI score
is a length-normalized pool of inter-chain contact probabilities.

## Model

`ResOnly` uses:

1. a shared per-chain PLM adapter and short self-attention stack;
2. shared bidirectional cross-attention;
3. an `L1 x L2` pair map containing projected residue-pair physics and
   length-normalized positional features;
4. quadratic depthwise/dilated pair-map updates; and
5. a coarse inter-chain distance head.

There is no square `L x L` pair track, cubic axial/triangle attention,
windowing, row chunking, or activation checkpoint flag. Dense inference has an
explicit combined-length limit of 2,048 residues.

The v2 architecture requires retraining. Legacy square-pair checkpoints are
not shape-compatible; new checkpoints carry `architecture: inter_pair_v2` and
fail clearly when loaded into a different architecture.

## Environment

```bash
conda env create -f env.yaml
conda activate surpass
```

The physics initializer needs the CALVADOS and Mpipi parameter files expected
by `src/common/calvados` and `src/common/mpipi`. Tests and CPU smoke benchmarks
can request zero-initialized physics features.

## Data preparation

The PDB and AFDB scripts produce structures, labels, sequences, and source
tables:

```bash
python scripts/prepare_pdb_entries.py --help
python scripts/prepare_afdb_entries.py --help
python scripts/prepare_plm_embeddings.py --help
```

Create one MMseqs family mapping jointly across every PDB/AFDB/benchmark
sequence. Then build immutable leakage-safe manifests:

```bash
python scripts/build_mmseqs_families.py \
  --fasta PDB_PPI/pdb_seqres.dissimilar.fasta \
  --fasta AFDB_DDI/processed_domains.fasta \
  --benchmark-fasta PDB_PPI/test_set.fasta \
  --output-dir splits/mmseqs

python scripts/build_dataset_splits.py \
  --family-mapping splits/mmseqs/family_mapping.tsv \
  --pair-table pdb_pairs.tsv --source-name pdb_ppi --source-label positive \
  --pair-table afdb_positive_pairs.tsv --source-name afdb_pos --source-label positive \
  --pair-table afdb_negative_pairs.tsv --source-name afdb_neg --source-label negative \
  --benchmark-families splits/mmseqs/benchmark_entities.tsv \
  --benchmark-column entity_id \
  --hard-negative-count 10000 \
  --output-dir splits
```

Every pair table must contain `chain1`, `chain2`, and
`source_is_positive`; positive rows additionally retain `label_path`. The
family mapping contains `entity_id`, `family_id`, and `length`.

All pair edges in a connected homology component stay in one split.
Components touching a benchmark family are excluded. Synthetic negatives are
rejected when their family pair is known positive and are emitted with lower
confidence than curated negatives. Source-specific files are written under
`splits/sources/` and are the manifests referenced by `configs/train.yaml`.

## Train and evaluate

```bash
python -m src.train data.root_dir=/path/to/data wandb.enabled=false

python -m src.inference \
  ckpt_dir=/path/to/best_pair_auprc.pth \
  data.root_dir=/path/to/PDB_PPI \
  data.embedding_dir=/path/to/PDB_PPI/test_embeddings
```

Inference writes every requested pair. Inputs above the dense length contract
are marked `unsupported_length` rather than silently disappearing. Official
metrics use the primary `p_pair` score and report coverage.

## Benchmarks

```bash
python scripts/benchmark_runtime.py --device cuda --precision bf16
python benchmark/run_rf2-ppi.py --help
```

Report PLM/MSA preparation separately from model-only latency. RF2-PPI is read
from the sibling checkout and is never modified by the adapter.

## Tests

```bash
python -m unittest discover -s tests
```
