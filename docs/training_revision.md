# Reviewed SurPass training revision

Run A is a new 30-epoch ResPair training run, with exact 1 Å interior distance bins, max pooling selected, primary protein loss weight zero, and the auxiliary protein objective retained at 0.1. The latest settings live in `configs/train.yaml` and `configs/inference.yaml` and are loaded by default. Edit these files or use command-line overrides for routine experiments; no new experiment config is needed. The old `*_revision` names are compatibility aliases, while `*_respair` keeps the previous legacy settings.

```bash
python -m src.train
```

On the cluster, keep your existing distributed launcher without a `--config-name` argument. Edit `data.root_dir` and `epoch_inference.config.data` paths to match the actual split, FASTA, test table, and embeddings. Configuration defaults retain the downloaded run's 12 layers, one recycle, batch 8, source weights, contact/distance weights, and 256 total crop residues.

The revised head has 22 classes: `d<2`, twenty intervals `[2,3)` through `[21,22)`, and `d>=22`. Contact probability sums the first seven classes, exactly `d<8`. Exactly 8 Å is noncontact. Old 20-class checkpoints are incompatible with this experiment; start from scratch. Keep the legacy configuration when evaluating an old checkpoint.

Validation is drawn once and saved in `validation_manifest.json`, including member rows, crop seeds, actual crop coordinates, and a split fingerprint. It is replayed lazily each epoch and included in checkpoints. To reuse that panel for a later matched run, set `data.validation_manifest=/absolute/path/validation_manifest.json`.

Both unweighted and 0.01-positive-weighted AP are reported. `validation_pair_auprc` remains the weighted selection metric; `validation_pair_auprc_unweighted` is additional information. Validation component losses are logged. Clipping is disabled (`optimizer.max_grad_norm=0`). The new scheduler warms up for 1,000 optimizer updates and decays from 1e-4 to 1e-5 over the planned 30-epoch update horizon. LR and update counters are logged to experiment tracking and the resolved schedule is saved in checkpoints. Set shorter warmup explicitly for very small smoke-test datasets.

EMA updates after optimizer steps and is used for validation. Best and regular checkpoints contain live weights, EMA state, optimizer/scheduler state, per-rank RNG, resolved configuration, update counters, best weighted AP, and the validation panel. A best checkpoint's `model_state_dict` is now **live** state; select its `ema_state_dict` for evaluation using the default inference configuration.

```bash
python -m src.inference \
  ckpt_dir=/absolute/path/checkpoints/best_pair_auprc.pth
```

Full EMA inference runs after periodic checkpoint saves at epochs 10, 20, 30, and at a final non-multiple epoch. Each `epoch_inference/epoch_NNNN/` contains the reusable resolved inference configuration, predictions, both AP weights, AUROC, coverage, runtime, and checkpoint hash/completion status. Overlength inputs remain in the output as unsupported. Best-checkpoint updates do not trigger extra full inference. Scoring reuses the model and bounded cache, restores training mode/live weights/RNG, and does not advance training state. `epoch_inference.every_n_epochs` must match `checkpoint_interval`.

To reproduce an epoch's inference settings directly:

```bash
python -m src.inference \
  --config-dir /absolute/path/run/epoch_inference/epoch_0010 \
  --config-name config \
  output_dir=/absolute/path/reproduced_inference
```

To resume training at an epoch boundary:

```bash
python -m src.train \
  ckpt_dir=/absolute/path/checkpoints/epoch_10.pth
```

Keep the original optimizer settings, epoch horizon, world size, data paths, and model/bin configuration. A changed cosine horizon is rejected rather than silently restarting or stretching the schedule. If the job stopped after an epoch save but before full inference finished, resume completes that inference event in the source run directory; an already completed matching event is skipped. Full mid-epoch replay is outside this revision. `ema_dir` is superseded by complete `ckpt_dir` checkpoints. An old weights-only checkpoint can only be a model-only warm start under its matching legacy model/bin configuration; it cannot supply missing EMA resume history.

The precision check must run on a CUDA GPU supporting bf16:

```bash
python -m scripts.compare_precision
```

It uses a fixed class/length-spread inference panel and compares bf16 against a float32 reference with TF32 disabled. With no checkpoint it explicitly reports `initialized_model`; pass `ckpt_dir` for a matching complete revision checkpoint to compare its EMA inference. It also runs 100 matched training updates from identical fresh initialization on four prepared batches replayed in each precision. `precision_check.*` controls panel size, updates, batch-buffer size, and output directory. Reports include per-example scores, both AP weights, ties, score differences, loss trajectories, numerical failures, and GPU/PyTorch/backend settings. These are numerical smoke checks, not proof of full convergence equivalence. The tool is separate from the pilot and does not modify pilot checkpoints.

Run B is not launched automatically: after reviewing Run A, change only `loss.protein.weight=1.0` and reuse its validation manifest and matched training settings.
