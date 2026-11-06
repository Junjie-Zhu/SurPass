"""Compare full-sequence and tiled inference; optionally trace selected pairs.

Run from the repository: python -m scripts.diagnose_inference --help
No training code, checkpoint weights, or model defaults are modified.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm.auto import tqdm

from src.data.labels import contact_bin_count
from src.inference import InferencePairDataset
from src.runtime.checkpoint import load_checkpoint
from src.runtime.config import create_model
from src.runtime.crop_inference import centered_window, crop_features, forward_maps, score_pair
from src.runtime.device import normalize_precision
from src.runtime.inference_diagnostics import InferenceDiagnostics, compare_representations
from src.runtime.metrics import binary_classification_metrics
from src.utils.ddp_utils import seed_everything


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def read_panel(path):
    if path is None:
        return None
    with Path(path).open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "Protein pairs" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing Protein pairs column")
        return {row["Protein pairs"].strip() for row in reader}


def read_contacts(path, maximum):
    result = {}
    if path is not None:
        with Path(path).open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"Protein pairs", "i", "j"} <= set(reader.fieldnames or []):
                raise ValueError("Contact TSV needs Protein pairs, i, j (zero-based sequence indices)")
            for row in reader:
                pair = row["Protein pairs"].strip()
                result.setdefault(pair, set()).add((int(row["i"]), int(row["j"])))
        if any(len(contacts) > maximum for contacts in result.values()):
            raise ValueError(f"More than {maximum} native contacts in one pair; use a selected panel or raise --max-native-contacts")
    return {pair: sorted(contacts) for pair, contacts in result.items()}


def write_table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def trace_pair(model, first, second, metadata, result, *, device, precision, cutoffs, crop_size, directory):
    anchors = result["anchors"]
    full_window = result["full_window"]
    with InferenceDiagnostics(model, first, second, full_window, anchors) as full_trace:
        replay = forward_maps(model, [(first, second, metadata)], device, precision, cutoffs)[0]
    # Verify observer hooks do not materially change the actual scored maps.
    for name in cutoffs:
        np.testing.assert_allclose(replay["maps"][name], result["full"]["maps"][name], atol=2e-6, rtol=2e-5)
    views = list(dict.fromkeys(winner["window"] for winner in result["winners"].values()))
    for anchor in anchors:
        if anchor["origin"].startswith("full_predicted_peak") or anchor["origin"] == "native_supplied":
            view = centered_window(len(first["mask"]), len(second["mask"]), anchor["i"], anchor["j"], crop_size)
            if view not in views:
                views.append(view)
    attention = [dict(view="full", **row) for row in full_trace.attention]
    statistics = [dict(view="full", **row) for row in full_trace.statistics]
    comparisons, contacts, vector_arrays, vector_index = [], [], {}, []

    def save_vectors(trace, view):
        for key, value in trace.vectors.items():
            identifier = f"v{len(vector_arrays)}"
            vector_arrays[identifier] = value
            vector_index.append(dict(array=identifier, view=view, stage=key[0], layer=key[1],
                recycle=key[2], side=key[3], i=key[4], j=key[5]))
    save_vectors(full_trace, "full")
    for number, window in enumerate(views):
        if window == full_window:
            trace, prediction = full_trace, replay
        else:
            a, b = crop_features(first, second, window)
            with InferenceDiagnostics(model, a, b, window, anchors) as trace:
                prediction = forward_maps(model, [(a, b, metadata)], device, precision, cutoffs)[0]
        view = f"crop_{number}"
        attention.extend(dict(view=view, **window.as_dict(), **row) for row in trace.attention)
        statistics.extend(dict(view=view, **window.as_dict(), **row) for row in trace.statistics)
        comparisons.extend(dict(view=view, **window.as_dict(), **row) for row in compare_representations(full_trace, trace))
        save_vectors(trace, view)
        for anchor in anchors:
            i, j = anchor["i"], anchor["j"]
            if window.contains(i, j):
                for name in cutoffs:
                    contacts.append(dict(view=view, **window.as_dict(), **anchor, cutoff=name,
                        full_probability=float(result["full"]["maps"][name][i, j]),
                        crop_probability=float(prediction["maps"][name][i-window.start1, j-window.start2])))
    write_table(directory/"attention.tsv", attention)
    write_table(directory/"representation_statistics.tsv", statistics)
    write_table(directory/"representation_comparison.tsv", comparisons)
    write_table(directory/"tracked_contacts_diagnostics.tsv", contacts)
    write_table(directory/"representation_vector_index.tsv", vector_index)
    np.savez_compressed(directory/"representation_vectors.npz", **vector_arrays)
    (directory/"diagnostic_notes.json").write_text(json.dumps(dict(
        attention="Float32 reconstruction of selected cross-attention rows from current Q/K and pair bias; native fused forward is unchanged.",
        representation_statistics="Deterministically sampled cells; vector comparisons use the same global residue/contact coordinates.",
        index_convention="Zero-based sequence coordinates, crop ends exclusive; layer/recycle one-based, embedding/input stage zero.",
        native_contacts_supplied=any(a["origin"] == "native_supplied" for a in anchors),
        observer_full_map_check_passed=True), indent=2)+"\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Saved resolved epoch inference config.yaml")
    p.add_argument("--checkpoint", type=Path, help="Optional override of cfg.ckpt_dir")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--panel", type=Path, help="Optional pair TSV; omit for the full benchmark")
    p.add_argument("--diagnostic-pairs", type=Path, help="Only these pairs get attention/representation tracing")
    p.add_argument("--native-contacts", type=Path, help="Optional TSV: Protein pairs, i, j; zero-based sequence coordinates")
    p.add_argument("--max-native-contacts", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=256, help="Total residues across both chains")
    p.add_argument("--stride-fraction", type=float, default=.5, help="0.5 = half-window stride; 1 = no overlap")
    p.add_argument("--batch-size", type=int, default=8, help="Crop windows per forward; full forwards use batch one")
    p.add_argument("--cutoffs", nargs="+", type=float, default=[8.], help="Distance cutoffs in Angstrom; first is primary")
    p.add_argument("--precision", choices=["float32", "bfloat16"])
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--test-set-tsv", type=Path)
    p.add_argument("--fasta", type=Path)
    p.add_argument("--embedding-dir", type=Path)
    p.add_argument("--save-maps", action="store_true", help="Full maps and winning-crop maps only")
    p.add_argument("--resume", action="store_true", help="Reuse completed pairs only when run identity matches")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.crop_size < 2 or not 0 < args.stride_fraction <= 1 or args.max_native_contacts < 1:
        raise ValueError("Require batch_size >= 1, crop_size >= 2, 0 < stride_fraction <= 1, max_native_contacts >= 1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run on one GPU/process; crop windows are batched internally")
    cfg = OmegaConf.load(args.config)
    for flag, field in [(args.test_set_tsv, "test_set_tsv"), (args.fasta, "fasta_path"), (args.embedding_dir, "embedding_dir")]:
        if flag is not None:
            cfg.data[field] = str(flag)
    checkpoint = args.checkpoint or Path(cfg.ckpt_dir)
    panel, diagnostic_pairs = read_panel(args.panel), read_panel(args.diagnostic_pairs) or set()
    native = read_contacts(args.native_contacts, args.max_native_contacts)
    device = torch.device(args.device)
    precision = normalize_precision(args.precision or str(cfg.runtime.precision), device)
    seed_everything(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", False)))
    model = create_model(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, weights=str(cfg.runtime.checkpoint_weights))
    if diagnostic_pairs and getattr(model, "architecture", None) != "res_pair_v1":
        raise ValueError("Layer diagnostics require a ResPair checkpoint")
    cutoffs = {f"d{radius:g}": contact_bin_count(radius, cfg.data.distance_bin_start, cfg.data.distance_bin_end,
        cfg.data.distance_bin_count, convention=str(cfg.data.distance_bin_convention)) for radius in args.cutoffs}
    data = InferencePairDataset(cfg.data.test_set_tsv, cfg.data.fasta_path, cfg.data.embedding_dir,
        maximum_length=int(cfg.data.maximum_length), embedding_cache_mb=256)
    all_pairs = {item["pair"] for item in data.items} | {row["Protein pairs"] for row in data.unsupported_rows}
    if panel is not None and panel-all_pairs:
        raise ValueError(f"Unknown panel pairs: {sorted(panel-all_pairs)}")
    selected = all_pairs if panel is None else panel
    if diagnostic_pairs-selected or set(native)-selected:
        raise ValueError("Diagnostic/contact pairs must belong to the selected scoring set")
    indices = [i for i, item in enumerate(data.items) if item["pair"] in selected]
    if not indices:
        raise ValueError("No supported pairs selected")
    identity = dict(config=OmegaConf.to_container(cfg, resolve=True), checkpoint_sha256=file_hash(checkpoint),
        test_set_sha256=file_hash(cfg.data.test_set_tsv), fasta_sha256=file_hash(cfg.data.fasta_path),
        crop_size=args.crop_size, stride_fraction=args.stride_fraction, crop_batch_size=args.batch_size,
        cutoffs=cutoffs, precision=precision, device=str(device),
        panel=sorted(selected), diagnostic_pairs=sorted(diagnostic_pairs), native_contacts=native,
        save_maps=args.save_maps, torch=torch.__version__,
        implementation_sha256={name: file_hash(path) for name, path in [
            ("runner", Path(__file__)),
            ("crop", Path(__file__).resolve().parents[1]/"src/runtime/crop_inference.py"),
            ("diagnostics", Path(__file__).resolve().parents[1]/"src/runtime/inference_diagnostics.py")]})
    # Normalize tuples to JSON lists before resume comparison.
    identity = json.loads(json.dumps(identity))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/"run.json"
    if manifest.exists():
        if not args.resume:
            raise ValueError("Output already contains a run; use a new directory or --resume")
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError("Resume identity differs; use a new output directory")
    elif args.resume and any(out.iterdir()):
        raise ValueError("Cannot resume a nonempty output without run.json")
    manifest.write_text(json.dumps(dict(identity=identity, weights=str(cfg.runtime.checkpoint_weights),
        input_embeddings="Slices of cached full-sequence embeddings; ESMC is not re-run.",
        crop_placement="All grid windows, independent of labels/native contacts.",
        scope="selected diagnostic panel" if panel is not None else "full supported benchmark",
        embedding_identity_limit="Embedding directory/files must remain immutable for resume; full tensor-directory hashes are not computed.",
        coordinate_convention="Zero-based residue indices; stop positions exclusive"), indent=2)+"\n")
    summaries = []
    for index in tqdm(indices, desc="Full versus crops"):
        item = data.items[index]
        directory = out/"pairs"/f"{item['index']:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        completed = directory/"result.json"
        if args.resume and completed.exists():
            summaries.append(json.loads(completed.read_text()))
            continue
        first, second, metadata = data[index]
        try:
            result = score_pair(model, first, second, metadata, device=device, precision=precision,
                cutoffs=cutoffs, crop_size=args.crop_size, stride_fraction=args.stride_fraction,
                batch_size=args.batch_size, native_contacts=native.get(metadata["pair"], ()))
            write_table(directory/"windows.tsv", result["windows"])
            write_table(directory/"tracked_contacts.tsv", result["tracked"])
            if args.save_maps:
                arrays = {f"full_{name}": values for name, values in result["full"]["maps"].items()}
                arrays.update({f"winner_{name}": winner["maps"][name] for name, winner in result["winners"].items()})
                np.savez_compressed(directory/"contact_maps.npz", **arrays)
            if metadata["pair"] in diagnostic_pairs:
                trace_pair(model, first, second, metadata, result, device=device, precision=precision,
                    cutoffs=cutoffs, crop_size=args.crop_size, directory=directory)
            summary = result["summary"]
            del result
        except torch.cuda.OutOfMemoryError as exc:
            # A partial window maximum is not a complete prediction.
            summary = {"Protein pairs": metadata["pair"], "Category": metadata["label"],
                "combined_length": metadata["combined_length"], "status": "out_of_memory", "error": str(exc)}
            gc.collect()
            torch.cuda.empty_cache()
            write_table(out/"predictions.tsv", summaries+[summary])
            raise RuntimeError(f"OOM on {metadata['pair']}; lower --batch-size and use a new output directory") from exc
        temporary = completed.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2)+"\n")
        temporary.replace(completed)
        summaries.append(summary)
    unsupported = [{key: value for key, value in row.items() if key != "_index"} for row in data.unsupported_rows
                   if row["Protein pairs"] in selected]
    write_table(out/"predictions.tsv", summaries+unsupported)
    metrics = []
    for name in cutoffs:
        for mode in ("full", "crop"):
            for weight in (1., .01):
                values = binary_classification_metrics([row[f"{mode}_{name}"] for row in summaries],
                    [row["Category"] for row in summaries], positive_weight=weight)
                metrics.append(dict(head=f"{mode}_{name}", positive_weight=weight, AUROC=values["auroc"],
                    AUPRC=values["auprc"], N=len(summaries), N_pos=sum(row["Category"] for row in summaries),
                    N_neg=sum(not row["Category"] for row in summaries), coverage=len(summaries)/len(selected)))
    # Match the existing comma-delimited metrics convention.
    with (out/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Scored {len(summaries)} pairs; unsupported {len(unsupported)}; wrote {out/'predictions.tsv'}")


if __name__ == "__main__":
    main()
