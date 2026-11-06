# Full versus cropped inference

This diagnostic loads the existing checkpoint and existing ESMC embeddings. It does not train, re-embed sequences, change model defaults, or alter the checkpoint contract. It can score the full benchmark or a small pair panel.

From the SurPass repository, start with the included **15-pair diagnostic panel**:

```bash
python -m scripts.diagnose_inference \
  --config logs/SURPASS_REVISION_2026-11-05_14-25-31/epoch_inference/epoch_0050/config.yaml \
  --panel configs/diagnostics/crop_panel.tsv \
  --output-dir logs/crop_epoch50_panel \
  --crop-size 256 --batch-size 8
```

The panel includes low/high-scoring positives and high-scoring labeled negatives in five length bins. It was chosen from existing predictions for diagnosis and is not an unbiased performance sample. Pair labels never determine crop placement or scores.

For the **whole benchmark**, omit `--panel` and use a different output directory. Each pair is scored in full, followed by every combination of contiguous chain windows. The 256-residue budget is shared between chains exactly as in training; unused budget from a short chain goes to its partner. Windows overlap by half their width by default, and the last window always reaches the sequence end. The cropped protein-pair score is the maximum across all windows. Short pairs fitting within the budget reuse the full prediction exactly. Crop windows are batched to reduce overhead.

Keep the default **8 Å** score first to isolate the context experiment. Optionally add `--cutoffs 8 12` to compute both scores from the same logits, with no checkpoint-contract change. `--stride-fraction 1` gives a faster nonoverlapping grid but can make boundary effects more pronounced; use the same setting for positives and negatives and record it as a separate experiment.

## Add attention and representation checks

The supplied trace panel contains four long pairs and is a subset of the scoring panel. Use a new directory:

```bash
python -m scripts.diagnose_inference \
  --config logs/SURPASS_REVISION_2026-11-05_14-25-31/epoch_inference/epoch_0050/config.yaml \
  --panel configs/diagnostics/crop_panel.tsv \
  --diagnostic-pairs configs/diagnostics/trace_pairs.tsv \
  --output-dir logs/crop_epoch50_traced \
  --crop-size 256 --batch-size 8 --save-maps
```

Only the listed diagnostic pairs get extra forwards and layer tracing. The diagnostic views include the full protein, each winning crop, and a crop centered on the full prediction's peak. Native-contact-centered views can be added when structural contact coordinates are available. Those additional views never enter the grid prediction or benchmark metric.

**Cached embeddings:** if the full-protein embedding for a residue is vector `x`, every window containing that residue gets precisely the same vector `x`. This is direct slicing of the saved tensor, with no ESMC call. SurPass's own position features and hidden representations are still allowed to change with context; that is what the experiment measures.

**Same coordinates:** suppose a pair of residues has full-sequence indices `(400, 90)`. In a crop beginning at `(350, 50)`, the same pair is at crop indices `(50, 40)`. We compare those two probabilities, not just the two map maxima, which can correspond to entirely different residues. All indices are zero-based sequence indices, not PDB residue numbers; crop ends are exclusive.

Without native structures, tracked sites are explicitly labeled `full_predicted_peak_d8` or `crop_predicted_peak_d8`. They are model-selected hypotheses, not known interfaces. An optional `--native-contacts contacts.tsv` accepts:

```text
Protein pairs	i	j
P13501_P51681	20	35
```

The line above illustrates the format only; it is not an asserted native contact. Supply verified sequence-coordinate mappings. Native coordinates are used for tracking and diagnostic views, never for grid scoring. At most 16 native contacts per pair are accepted by default; `--max-native-contacts` changes that bound.

## Outputs

- `predictions.tsv`: full/crop protein-pair scores, winning crop boundaries and peak coordinates, auxiliary scores, window counts, and elapsed times. `p_pair` is cropped; `p_pair_full` is the full baseline. Unsupported pairs remain present with their status.
- `metrics.csv`: full and cropped AUROC/AP at positive weights 1 and 0.01, on identical supported pairs. A small selected panel's metric is descriptive only.
- `run.json`: resolved settings, checkpoint and input-file hashes, code hashes, scoring scope, and coordinate convention. Keep embedding files unchanged during a resumed run; their whole directory is not hashed.
- `pairs/<input row index>/windows.tsv`: scores and boundaries of every grid window.
- `tracked_contacts.tsv` within each pair directory: full versus crop probabilities at the same coordinates, with the origin of each tracked site.
- Optional `contact_maps.npz`: full maps and winning crop maps; boundaries are in the corresponding result/window files.

Traced pairs additionally contain:

- `attention.tsv`: selected **cross-attention** rows, per layer, recycle, head, and direction. Includes probability to the tracked partner residue, attention mass to tracked partner residues, entropy, and enrichment above uniform attention. Native-key mass is only populated when native contacts are supplied. A small tracked mass is not necessarily small mass to the entire biological interface, because only selected sites are tracked.
- `representation_comparison.tsv`: cosine similarity, relative L2 change, RMS values/ratio, and exact equality at the same global residues or contact pairs. `esm_input` should be exactly equal. Stages include projected embeddings, self-/cross-attention blocks, pair blocks, and triangle sums before/after normalization.
- `representation_statistics.tsv`: bounded, deterministic samples of activation magnitudes. These global samples can represent different cells in the full and crop; the same-coordinate vector comparisons above provide the direct comparison.
- `representation_vectors.npz` plus `representation_vector_index.tsv`: the selected vectors, without dumping every full-protein representation.
- `tracked_contacts_diagnostics.tsv` and `diagnostic_notes.json`: probabilities in the additional diagnostic views and observer details.

The observer reconstructs selected attention rows in float32 from the current Q/K and pair bias. It does not replace the model's fused attention. These probabilities can differ slightly from native reduced-precision arithmetic. It checks that observed full-map scores match the ordinary forward within a small numerical tolerance and removes its hooks after each view.

## Execution details

`--precision float32` or `--precision bfloat16` overrides the saved inference precision. CPU uses float32. `--device cuda:1` selects another GPU. This command runs as one process; do not launch it with a multi-rank launcher. Full forwards use batch one; `--batch-size` applies to crops. If a crop batch runs out of memory, the command stops without treating a partial maximum as a completed prediction; lower the batch size and choose a new output directory.

Completed pair results are saved atomically. `--resume` reuses them only if the run identity matches. It rejects changed checkpoint/input hashes, crop settings, selected pairs, precision, or diagnostic settings. Embedding files must remain immutable.

For this investigation, compare positive and negative behavior together. Maximizing over many windows can also increase false positives. A score increase alone does not demonstrate improved ranking or recovery of a native interface. No training adaptation is needed for these checks.
