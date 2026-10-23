# SurPass

SurPass predicts a rectangular inter-protein distance/contact map from two
single-sequence PLM embeddings. It does not use an MSA. The primary PPI score
is the maximum valid inter-chain contact probability: one contact at or above the
configured threshold is enough for a positive PPI call.

## Model

`ResOnly` interleaves the following updates for each of `n_layers`:

1. shared per-chain self-attention after a shared PLM adapter;
2. shared bidirectional cross-attention with simultaneous chain updates;
3. quadratic depthwise/dilated updates of the `L1 x L2` pair map; and
4. symmetric token-to-pair updates.

The initial pair map retains projected residue-pair physics and normalized
positional features. A coarse distance head converts the final pair map into
contact probabilities; a symmetric pooled-token head supplies an auxiliary
protein loss. There is no pair-to-token feedback. Self-attention is quadratic
in each chain length; the pair track costs O(L1*L2).

All attention and transition branches use pre-layer normalization and residual
addition. Their output projections start at zero, so the branches initially
preserve their inputs, including the initial physics/position features. The
token-to-pair sum/product also receives normalized tokens. SwiGLU halves its
input width; every following projection uses that resulting width.

There is no square `L x L` pair track, cubic axial/triangle attention,
windowing, row chunking, or activation checkpoint flag. Dense inference has an
explicit combined-length limit of 2,048 residues.

These corrections require retraining. New checkpoints carry
`architecture: inter_pair_v3` and reject v2 or legacy square-pair checkpoints.
The old `cross_n_layers`, `pair_n_layers`, and top-k configuration entries have
been removed; `n_layers` controls the interleaved stack.

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

Search each processed training chain/domain directly against the benchmark
sequences at **identity 0.3**, then cluster the PDB/AFDB training sequences
jointly (benchmarks are not included in training clusters). Use the processed
FASTAs because chain/domain extraction can change alignments relative to the
original full-length sequences. Then build immutable leakage-safe manifests:

```bash
python /path/to/SurPass/scripts/build_mmseqs_families.py \
  --fasta PDB_PPI/processed_chains.fasta \
  --fasta AFDB_DDI/processed_domains.fasta \
  --benchmark-fasta PDB_PPI/test_set.fasta \
  --negative-pair-table AFDB_DDI/verified_negative_pairs.dissimilar.tsv \
  --output-dir splits/mmseqs

python /path/to/SurPass/scripts/build_dataset_splits.py \
  --family-mapping splits/mmseqs/family_mapping.tsv \
  --pair-table PDB_PPI/verified_heterodimers.tsv --source-name pdb_ppi --source-label positive \
  --pair-table AFDB_DDI/verified_positive_pairs.dissimilar.tsv --source-name afdb_pos --source-label positive \
  --pair-table AFDB_DDI/verified_negative_pairs.dissimilar.tsv --source-name afdb_neg --source-label negative \
  --benchmark-entities splits/mmseqs/benchmark_entities.tsv \
  --pdb-negative-ratio 1.0 \
  --output-dir splits
```

Every pair table must contain `chain1` and `chain2`; positive rows additionally
retain `label_path`. `--source-label` supplies labels for raw tables without
`source_is_positive`. Standard `afdb_neg` sources default to negative and
`pdb_ppi`/`afdb_pos` to positive. The family mapping contains `entity_id`,
`family_id`, and `length`.

The MMseqs identity threshold is fixed at **0.3**, using `--alignment-mode 3`
and `--seq-id-mode 0` to calculate identity from identical residues over aligned
columns, rather than a score estimate (see the [MMseqs2 identity documentation](https://github.com/soedinglab/MMseqs2/wiki#how-does-mmseqs2-compute-the-sequence-identity)). Coverage defaults to **0.0** (identity
only). Supplying `--negative-pair-table` enables the requested fallback: if
every supplied negative pair is removed by direct benchmark hits, retry with
**coverage 0.8 on both sequences** (`--cov-mode 0`) while keeping identity at
0.3. The same effective threshold/coverage is used for family clustering.
`filter_stats.json` records the effective coverage and surviving negative-pair
counts. `--coverage 0.8` also selects that policy explicitly. MMseqs is a
heuristic alignment search; these thresholds apply to the alignments it finds.

`benchmark_entities.tsv` now lists **training IDs with direct benchmark hits**.
Pass it through `--benchmark-entities`, which removes pairs containing those
IDs before building positive interaction components. This prevents benchmark
exclusion from spreading through interaction partners. The legacy
`--benchmark-families` option deliberately excludes all members of supplied
families; it also filters pairs before component construction.

Whole positive interaction components stay in one split. Negative observations
do not connect positive components. A curated negative is excluded if its two
families land in different splits, if its exact entity pair is known positive,
or if its pair cluster would otherwise occur in both splits. In the last case,
the cluster keeps the split with more eligible rows (ties choose training).
Homologous family pairs alone do not invalidate curated negative labels.

Existing `ppi_cluster_id` values are preserved. Otherwise, AFDB `CLUSTER` values
become source-qualified PPI cluster IDs; raw PDB heterodimers are grouped by
the unordered pair of their 30% sequence families. No deduplication discards
member pairs. Training still samples one pair from each PPI cluster, so
`stats.json` reports **both pair and cluster counts per source and split**,
alongside exclusion reasons. Input estimates are diagnostics, not target counts
that justify relaxing the sequence filter.

PDB synthetic negatives are generated **after** splitting, separately within
train and validation, at one negative per `pdb_ppi` positive cluster by default
(`--pdb-negative-ratio`; zero disables them). Each generated negative has a
unique `ppi_cluster_id`. Candidates reuse processed PDB chains and embeddings;
known positive family pairs, benchmark-matching entities, cross-split pairs,
and duplicate entity pairs are excluded. These are presumed noninteractions.
Sample confidence is removed from manifests, datasets, collated batches, and
losses: protein/focal losses use unweighted sample means and distance loss
uses a mean over valid residue pairs. Loss-term weights and source sampling
probabilities remain configurable.

MMseqs outputs carry input checksums and parameter/version metadata. Reusing
an output directory with changed inputs/settings or old results without
metadata fails instead of silently using stale clusters; choose a new directory
or pass `--overwrite` to rebuild that script's outputs.

Source-specific files are written under `splits/sources/`:
`pdb_ppi.{train,validation}.tsv` contains only PDB positives,
`pdb_neg.{train,validation}.tsv` contains generated PDB negatives, and
`afdb_pos`/`afdb_neg` retain their supplied labels. All split files are emitted,
even when empty. Generation shortages produce warnings and requested/generated
counts per split in `stats.json`. Homology constraints can make the requested
count impossible; the generator never crosses split boundaries to fill it.
An optional `--hard-negative-count` adds synthetic candidates from all positive
sources in separate `<source>.synthetic` manifests; these extra sources must be
added explicitly to training if desired.

Run the example from the dataset parent directory, so its default `splits/`
output matches `${data.root_dir}/splits`. Training and preview resolve standard
filenames using the source name and `data.split_dir`; `train_tsv` and
`validation_tsv` are optional overrides, resolved relative to the source data
directory. The default config samples the four sources equally, giving a
balanced positive/negative mixture.

The split builder shows progress by default while reading tables, annotating
pairs, connecting families, generating negatives, and writing manifests. Bars
show elapsed time and processing rate; negative-generation bars also show the
requested/generated pair count and template attempts. Status/progress goes to
stderr, leaving stdout as a machine-readable JSON summary. `--no-progress`
disables these messages (generation-shortage warnings are still reported).

Hard-negative matching indexes candidates by source, split, degree, and length.
It searches length buckets in exact score order instead of scanning every
entity for each negative. The score, seeded tie-breaking, duplicate rejection,
and family-leakage rules are unchanged. Component summaries also use single-pass
counts rather than rescanning the pair table for every component.

`--workers 1` is the default and often fastest after indexing. To parallelize
expensive candidate ranking, append `--workers 4` (or another CPU count within
your job allocation) to the same command. The parent commits proposals in
order and resolves collisions, producing the same manifests as a serial run.
Many equal-length/equal-degree candidates can benefit from workers; cheap
indexed queries can be slower because of process startup and communication.
Workers can increase memory use, depending on the platform's process start
method. Ctrl-C stops the worker pool and exits with status 130.

Positive crops always center both contiguous windows on an observed contact
under one shared residue budget. A positive source with no valid contact raises
an error rather than silently becoming a negative crop. Negatives use random
contiguous windows. There is no interface/random probability setting.

## Train and evaluate

```bash
python -m src.train data.root_dir=/path/to/data wandb.enabled=false

python -m src.inference \
  ckpt_dir=/path/to/best_pair_auprc.pth \
  data.root_dir=/path/to/PDB_PPI \
  data.embedding_dir=/path/to/PDB_PPI/test_embeddings
```

The protein loss uses noisy-OR, `P(PPI) = 1 - product(1 - p_ij)`, over valid
residue pairs. It is computed directly from distance logits in float32 log space
with stable log-sum-exp/softplus/expm1 operations; it never converts a saturated
probability back to a logit. Negative bags therefore retain gradients even when
their predicted probability rounds to one. `SurpassOutput.protein_logit` always
contains this training logit, including during validation loss evaluation.
The focal contact loss also uses logits/softplus directly, preserving corrective
gradients for confidently wrong contact predictions.

Noisy-OR is a useful differentiable OR, but neighboring contacts are correlated
and many weak probabilities can saturate it on large maps. Direct contact and
distance supervision remain essential. The distance head starts with a sparse
contact bias (`model.contact_prior: 0.0001`), and gradient clipping defaults to
1.0. These improve initialization/numerics; they do not establish calibration
or guarantee convergence. Compare held-out metrics after retraining.

Inference, preview, validation AUROC/AUPRC, and checkpoint selection all use
`max(p_ij)` over valid contacts. Thus `pred_pair` is true exactly when at least
one valid contact reaches the score threshold; many individually weak contacts
cannot accumulate into a positive inference call. The default 0.5 threshold
should be calibrated on validation data before production use.

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
