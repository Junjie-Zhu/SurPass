# SurPass

SurPass predicts a rectangular inter-protein distance/contact map from two
single-sequence PLM embeddings. It does not use an MSA. The primary PPI score
is the maximum valid inter-chain contact probability: one contact at or above the
configured threshold is enough for a positive PPI call.

## Reviewed training revision

Training and inference use the latest settings by default: `python -m src.train`
and `python -m src.inference ckpt_dir=/path/to/checkpoint.pth`.
Keep routine experiments in `configs/train.yaml` / `configs/inference.yaml`
or command-line overrides; a new config file is not needed for each experiment.
See [the revision run guide](docs/training_revision.md) for exact bins, fixed
validation, complete EMA checkpoints, periodic full inference, resume, and the
CUDA bf16/float32 diagnostic. The `*_revision` names are compatibility aliases;
`*_respair` preserves legacy settings for existing runs.

## Model

### ResPair (RFD4 / PPLM adaptation)

`ResPair` is a separate sequence-only model in `src/model/res_pair.py`. The
original `ResOnly` class, weights, and `inter_pair_v3` checkpoint format remain
available. Train the new architecture with:

```bash
python -m src.train
python -m src.preview ckpt_dir=logs/<run>/checkpoints/epoch_N.pth
python -m src.inference ckpt_dir=logs/<run>/checkpoints/epoch_N.pth
```

The default `train` and `inference` configs select the latest `ResPair`. Python
callers can still use `from src.model import ResPair, ResOnly`. ResPair checkpoints
carry `architecture: res_pair_v1`; cross-architecture loading is rejected.
ResPair requires training; existing ResOnly checkpoints cannot initialize it
through the ordinary checkpoint loader.

The ResPair presets use **12 layers**, with pair updates on layers
**2, 4, 6, 8, 10, and 12** (`model.pair_update_every=2`). All 12 layers
update the single representations. The current defaults use one trunk pass. The legacy `*_respair` profiles
use two passes, executing 24 single blocks and 12 pair blocks with shared weights.

Each ResPair block performs shared per-chain self-attention, then simultaneous
bidirectional cross-attention with a learned **pair bias inside the attention
softmax**. Its pair update order is **single-to-pair Hadamard update, inter/intra
triangle multiplication, pair transition**, each added residually. There are
no Conv2d modules in ResPair. The normalized single-to-pair projections are
symmetrized as `(a_i * b_j + b_i * a_j) / 2` to preserve chain-swap equivariance.

The triangle update follows the rectangular inter/intra contraction in
[PPLM's InterTriangleMultiplication_S](https://github.com/junliu621/PPLM/blob/main/pplm_contact/Module.py),
adapted to sequence-only input. A triangle joins an interchain pair to a learned
intrachain pair on either partner. Intrachain context comes from the current
single representations, with weights shared between chains. It is generated
in row chunks and discarded; there is no persistent square pair trunk and no
PDB/coordinate input. This is an adaptation of PPLM, rather than its original
structure-derived intrachain feature encoder.

Recycling can be enabled with `model.recycle_passes=2` (the current default is 1). Each pass
receives the original input embeddings plus normalized/projected states from
the preceding pass, using shared trunk weights. Earlier trunk passes run
without gradients; input embeddings still receive gradients through the final
pass. States reset on every forward call. This follows RFD4 SI Algorithm S1,
pp.45-46: **hidden-state recycling**, distinct from conditioning on a previous
structure prediction. Set `model.recycle_passes=1` to disable feedback. The
pass count is fixed here; RFD4's Poisson training schedule is not enabled.

Input embeddings, physics projection, pair positions, and chain gather plans
are computed once per forward and reused across passes. Each attention block
computes Q/K/V and gates once per chain and projects the pair bias once for
both directions. Single-to-pair projections also supply the triangle context.
Trainable projections are never cached across batches or optimizer steps.
The contiguous layout supplies residue positions in linear time rather than
constructing a square same-chain comparison matrix. Autocast weight caching
is disabled inside recycled trunks to preserve final-pass gradients.

`model.triangle_hidden_dim` controls contraction width (64 in the presets).
`model.triangle_chunk_size=64` limits transient intrachain rows; `0` uses all
rows. `model.triangle_checkpoint=true` recomputes chunk activations during
backward to reduce saved memory; disable it to trade memory for less
recomputation. Chunking is algebraically exact, with normal floating-point
roundoff. `model.pair_update_every=2` is the default and uses RFD4's
every-other-block schedule (blocks 2, 4, ...); `1` updates pairs every block.
Layer count and pair-update frequency change the trained module layout and
must match when restoring a checkpoint. Earlier two-layer/every-layer ResPair
checkpoints cannot be loaded directly into this 12-layer preset.

Unlike ResOnly's convolutional pair track, triangle arithmetic costs
O(L1*L2*(L1+L2)*hidden), plus intrachain feature projections. Chunking reduces
temporary memory, not that arithmetic cost. The predicted pair map remains
rectangular. Benchmark GPU memory/time for long-chain workloads before a full
training run; CPU smoke checks do not establish GPU throughput or prediction
quality.

Validation: the test suite checks dense-reference triangle outputs and
gradients, attention bias placement, active-update chain swapping, mixed-length
padding, empty context, recycling/caching, checkpoint separation, configuration
selection, the 12-layer/even-layer pair schedule, and float32/bfloat16
optimizer steps. The current preset has 43,338,993 parameters. A short CPU
forward with the full model dimensions passed; reduced-width tests also cover
optimizer steps through all 12 layers. These checks do not establish whether
2,048 residues fit on a GPU. The feature comparison is in
`docs/rfd4_features.html`.

### ResOnly (original model)

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
  --min-seq-id 0.3 --coverage 0.8 \
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
columns (see the [MMseqs2 identity documentation](https://github.com/soedinglab/MMseqs2/wiki#how-does-mmseqs2-compute-the-sequence-identity)).
**Coverage 0.8 on both sequences is required** (`-c 0.8 --cov-mode 0`) for
benchmark filtering and family clustering. These are the defaults, and the
CLI rejects other identity/coverage thresholds. `--negative-pair-table`
records the negative-pair retention count; it never relaxes thresholds.
`filter_stats.json` records the effective policy and surviving pair counts.
MMseqs is a heuristic alignment search; thresholds apply to the alignments
it finds.

Earlier identity-only outputs (`effective_coverage: 0.0`) must be rebuilt:
rerun `build_mmseqs_families.py` with a new output directory or `--overwrite`,
then rerun `build_dataset_splits.py` using the new mapping and benchmark
exclusions. The split builder rejects a supplied filter sidecar that records
coverage 0.0. Filtering the old saved hits alone is insufficient because the
search retained only one accepted hit per query and omitted alternatives.

The preparation scripts use 80% coverage of the training sequence (test
queries, training targets, `--cov-mode 1`). The build stage additionally
requires 80% coverage of the benchmark sequence (`--cov-mode 0`). Thus its
coverage requirement applies to both aligned partners. Benchmark filtering
always removes individual matching training entities before splitting;
it does not propagate benchmark exclusions through interaction components.

`benchmark_entities.tsv` now lists **training IDs with direct benchmark hits**.
Pass it through `--benchmark-entities`, which removes pairs containing those
IDs before building positive interaction components. This prevents benchmark
exclusion from spreading through interaction partners. The legacy
`--benchmark-families` option deliberately excludes all members of supplied
families; it also filters pairs before component construction.

`filter_stats.json` distinguishes benchmark reference sequences, matched
training IDs, and saved-hit coverage diagnostics. The split builder reads
this sidecar automatically beside `--benchmark-entities`, or accepts
`--benchmark-filter-stats PATH`; it validates the matching-ID count and,
for new reports, the exclusion-file checksum. It records the effective
identity/coverage policy in `stats.json`. Legacy `benchmark_entity_count`
means excluded **training IDs**, and `benchmark_family_count` counts the
explicit family blocklist. New `benchmark_matching_*` fields report
matched training IDs/families, with pair-referenced and unused IDs split
apart. `source_benchmark_filter_counts` records per-source losses.

The saved search uses one accepted hit per query; coverage statistics
describe those saved hits only. Selecting another coverage policy requires
rerunning the search, since alternative qualifying hits were not saved.

Whole positive interaction components stay in one split. Negative observations
do not connect positive components. A curated negative is excluded if its two
families land in different splits, if its exact entity pair is known positive,
or if its pair cluster would otherwise occur in both splits. In the last case,
the cluster keeps the split with more eligible rows (ties choose training).
Homologous family pairs alone do not invalidate curated negative labels.

`--validation-fraction` is a target for weighted components, not a
per-source stratified pair split. When negative endpoints belong to
independently assigned components, both must land in validation: with
validation probability f, approximately f² negatives enter validation,
(1-f)² enter training, and 2f(1-f) cross the boundary and are excluded.
Shared components change these proportions. `source_validation_fractions`
reports actual retained sample and cluster proportions per source.

Existing `ppi_cluster_id` values are preserved. Otherwise, AFDB `CLUSTER` values
become source-qualified PPI cluster IDs; raw PDB heterodimers are grouped by
the unordered pair of their 30% sequence families. No deduplication discards
member pairs. Training still samples one pair from each PPI cluster, so
`stats.json` reports **both pair and cluster counts per source and split**,
alongside exclusion reasons. Input estimates are diagnostics, not target counts
that justify relaxing the sequence filter.
An excluded row can belong to a cluster that still has retained members;
therefore excluded cluster counts cannot be added to train/validation
counts to recover the input cluster total. `source_cluster_retention_counts`
separates fully excluded clusters from partially excluded clusters.

PDB synthetic negatives are generated **after** splitting, separately within
train and validation. `--pdb-negative-ratio 1.0` requests one distinct negative
family-pair cluster per positive PPI cluster; zero disables generation. Each
negative cluster is then enriched with distinct pairs of existing PDB chains
from those two families in the same split. By default,
`--pdb-negative-pairs-per-cluster 0` targets the positive split's mean number of
pairs per cluster (rounded up). Set an explicit value, such as 16, to choose
the candidate density. The target is capped by available eligible pairs.

The enrichment samples family-member combinations without constructing the
full Cartesian product. It preserves the family-pair cluster IDs so adding
candidates increases within-cluster variety, rather than inflating dataset
length. Known positive family pairs, benchmark-matching entities, cross-split
pairs, and existing entity pairs remain excluded. Generated candidates reuse
processed chains and embeddings. They are presumed noninteractions, and do
not need structural labels or sample-confidence weights.

`pdb_negative_counts` reports requested/generated clusters, target pairs per
cluster, and actual negative pairs separately. `generated_pdb_negatives` counts
pair rows; `generated_pdb_negative_clusters` counts family-pair clusters.

The MMseqs builder selects available CPU threads automatically, respecting
CPU affinity and `SLURM_CPUS_PER_TASK`/`NSLOTS`. Set `--threads N` explicitly
to use the CPU count allocated to your job (`--threads 0` also selects
automatically). The script prints the selected thread count, query/target
counts, and elapsed time for search and clustering. The prefilter can take
many minutes at sensitivity 7.5; its ETA applies to that stage, not the
whole build. A static captured progress line alone does not prove a hang.

Benchmark search runs once per **identical sequence**, then expands each
hit back to every original training ID. Only one accepted benchmark hit
per query is needed for exclusion, so search uses `--max-accept 1`; this
reduces alignment work, not the preceding prefilter calculation (see the
[MMseqs2 search tuning documentation](https://github.com/soedinglab/MMseqs2/wiki#how-to-control-the-speed-of-the-search)).
The identity cutoff stays 0.3, search sensitivity stays 7.5, and family
clustering still receives all original sequences.

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

Validation is sampled each epoch using two settings inherited by both training
configs:

```yaml
data:
  validation_samples: 3000
  validation_negative_ratio: 1.0
```

`validation_samples` is the global number of cluster draws across all GPUs,
not a per-rank batch count. Each selected cluster supplies one sampled member
pair, as in training. A ratio of 1.0 means one negative per positive (50%
negative); 2.0 means two negatives per positive. Sampling uses the dataset
source weights within each class, without repeating clusters within an epoch.
If a class has insufficient clusters, the total budget is capped to preserve
the requested ratio, up to integer rounding; logs record the actual counts.
Set `validation_samples: null` to evaluate the full validation split, keeping
its natural class ratio.

The class/source pools are indexed once from manifest metadata without loading
features. Each epoch draws only the configured indices, with a deterministic
seed based on the epoch. This selection takes O(sample budget) work; embedding
loading and model evaluation happen only for selected samples. Distributed
ranks receive disjoint shards without padding duplicates. Validation runs the
underlying model without DDP forward broadcasts, allowing uneven shard sizes.
The global evaluated positive/negative counts are logged with validation
metrics. A new draw each epoch adds sampling variation to AUROC/AUPRC; keep
budget and ratio fixed when comparing runs.

The split builder shows progress by default while reading tables, annotating
pairs, connecting families, generating negatives, and writing manifests. Bars
show elapsed time and processing rate; negative-generation bars also show the
requested/generated pair count and template attempts. Status/progress goes to
stderr, leaving stdout as a machine-readable JSON summary. After writing all
manifests, the builder prints exact cluster and sample counts for every source
in train, validation, and excluded, with totals for each split. Clusters are
unique PPI clusters within each source; samples are pair rows, including
generated negatives. The same counts are saved to `splits/subset_sizes.tsv`
and its path is included in the JSON summary. `--no-progress` disables
progress messages but keeps this final size report and shortage warnings.

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

Inference keeps a least-recently-used CPU feature cache bounded by
`data.embedding_cache_mb` (MiB of tensor data, default 256). Set it to 0 to
disable caching. The budget applies separately to every inference rank and
every active DataLoader worker; it does not bound model, batch, prefetch, or
allocator memory. Keep `data.num_workers=0` initially when diagnosing host-RAM
limits. Loaded embeddings that are views of larger serialized storage are
compacted before caching, preserving their values.

## Benchmarks

The runtime script defaults to the current `inference.yaml` config: 12
layers, pair updates every other layer, one trunk pass, batch size one,
and **2,048 total residues (1,024 + 1,024)**. It composes inherited Hydra
configuration and selects the configured model rather than always constructing
ResOnly. Synthetic inputs exclude PLM generation, disk I/O, and data loading.
No checkpoint is needed to test the architecture's time and memory.

For a quick capacity check and one cold forward timing:

```bash
python scripts/benchmark_runtime.py --device cuda --precision bf16 --warmup 0 --repeats 1
```

For warm timing, use the defaults (one warmup and three timed forwards):

```bash
python scripts/benchmark_runtime.py --device cuda --precision bf16
```

Each iteration prints progress and elapsed seconds. The final summary prints
mean/median seconds per batch, seconds per sample, and peak CUDA allocated and
reserved memory in GiB. CUDA is synchronized around each timed iteration.
Memory peaks include model weights, synthetic inputs, and warmup allocations.
CSV/JSON reports are saved as `benchmark_runtime.csv` and
`benchmark_runtime.json`, including model settings, per-iteration timings,
output shapes, GPU identity, and any failure stage. Out-of-memory errors during
input allocation or execution are saved and cause a nonzero exit status.

The first run includes cold-start overhead; use warm timing for steady-state
latency. Passing an inference fit check does not establish training fit. To
measure a training step, including loss, backward, gradient clipping, and the
optimizer update/state allocations:

```bash
python scripts/benchmark_runtime.py --config configs/train.yaml \
  --device cuda --precision bf16 --mode training --batch-size 1 \
  --lengths 2048 --output-json training_2048.json --output-csv training_2048.csv
```

Training timing excludes EMA, DDP, scheduler updates, and data preparation;
reserve additional memory for the full training application. `--batch-size 2`
tests the configured training batch size. To test a different chain balance,
use `--lengths 2048 --p1-length 512` (512 + 1,536). To sweep lengths or adjust
memory-related settings, use:

```bash
python scripts/benchmark_runtime.py --device cuda --lengths 256 512 1024 2048
python scripts/benchmark_runtime.py --device cuda --override model.triangle_chunk_size=32
```

`--checkpoint /path/to/model.pth` optionally loads matching trained weights.
`--config configs/inference_respair.yaml` selects the legacy ResPair profile. For an
installation check without a GPU, `--cpu-smoke` uses a tiny two-layer model,
a real pair update, and length 16; it does not test the full 12-layer capacity.

RF2-PPI adapter usage:

```bash
python benchmark/run_rf2-ppi.py --help
```

Report PLM/MSA preparation separately from model-only latency. RF2-PPI is read
from the sibling checkout and is never modified by the adapter.

## Tests

```bash
python -m unittest discover -s tests
```
