"""Label-blind tiled inference using slices of existing residue embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, product
import time

import numpy as np
import torch

from src.data.cropping import slice_complex_features, split_total_crop_budget
from src.inference import inference_collate_fn
from src.runtime.device import autocast_context, to_device


@dataclass(frozen=True)
class Window:
    start1: int
    stop1: int
    start2: int
    stop2: int

    def contains(self, i, j):
        return self.start1 <= i < self.stop1 and self.start2 <= j < self.stop2

    def as_dict(self):
        return {key: getattr(self, key) for key in ("start1", "stop1", "start2", "stop2")}


def window_starts(length, width, stride):
    if min(length, width, stride) < 1 or width > length or stride > width:
        raise ValueError("Require 1 <= stride <= width <= length")
    last = length - width
    return sorted(set([*range(0, last + 1, stride), last]))


def tile_windows(length1, length2, crop_size=256, stride_fraction=.5):
    if not 0 < stride_fraction <= 1:
        raise ValueError("stride_fraction must be in (0, 1]")
    width1, width2 = split_total_crop_budget(length1, length2, crop_size)
    starts1 = window_starts(length1, width1, max(1, int(width1*stride_fraction)))
    starts2 = window_starts(length2, width2, max(1, int(width2*stride_fraction)))
    return [Window(a, a+width1, b, b+width2) for a, b in product(starts1, starts2)]


def centered_window(length1, length2, i, j, crop_size=256):
    width1, width2 = split_total_crop_budget(length1, length2, crop_size)
    start1 = min(max(0, i-width1//2), length1-width1)
    start2 = min(max(0, j-width2//2), length2-width2)
    return Window(start1, start1+width1, start2, start2+width2)


def crop_features(first, second, window):
    """Index selection copies values exactly; the PLM is never re-executed."""
    return (slice_complex_features(first, torch.arange(window.start1, window.stop1)),
            slice_complex_features(second, torch.arange(window.start2, window.stop2)))


@torch.inference_mode()
def forward_maps(model, samples, device, precision, cutoffs):
    batch, _ = inference_collate_fn(samples)
    with autocast_context(device, precision):
        output = model(to_device(batch, device))
    probability = output.pair_logits.float().softmax(-1)
    maps = {name: probability[..., :count].sum(-1).masked_fill(~output.pair_mask, 0).cpu().numpy()
            for name, count in cutoffs.items()}
    auxiliary = output.aux_protein_logit.detach().float().cpu()
    result = []
    for index, (first, second, _) in enumerate(samples):
        n1, n2 = len(first["mask"]), len(second["mask"])
        result.append(dict(maps={name: values[index, :n1, :n2].copy() for name, values in maps.items()},
            aux_logit=float(auxiliary[index]), aux_probability=float(auxiliary[index].sigmoid())))
    return result


def map_summary(values, window):
    i, j = np.unravel_index(int(np.argmax(values)), values.shape)
    return dict(score=float(values[i, j]), i=int(i)+window.start1, j=int(j)+window.start2,
                n_ge_05=int((values >= .5).sum()))


def contact_record(anchor, window, full, crop, cutoff):
    i, j = anchor["i"], anchor["j"]
    return dict(anchor_origin=anchor["origin"], i=i, j=j, cutoff=cutoff,
        **window.as_dict(), local_i=i-window.start1, local_j=j-window.start2,
        full_probability=float(full[i, j]),
        crop_probability=float(crop[i-window.start1, j-window.start2]))


def score_pair(model, first, second, metadata, *, device, precision, cutoffs,
               crop_size=256, stride_fraction=.5, batch_size=8, native_contacts=()):
    """All windows enter the maximum, for either class; no labels place crops.

    Native contacts only enter diagnostics, never crop placement or scoring.
    The returned maps retain full maps and one winning crop per cutoff only.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n1, n2 = len(first["mask"]), len(second["mask"])
    full_window = Window(0, n1, 0, n2)
    for i, j in native_contacts:
        if not (0 <= i < n1 and 0 <= j < n2):
            raise ValueError(f"Contact {(i, j)} is outside sequence lengths {(n1, n2)}")
    windows = tile_windows(n1, n2, crop_size, stride_fraction)
    started = time.perf_counter()
    full = forward_maps(model, [(first, second, metadata)], device, precision, cutoffs)[0]
    full_seconds = time.perf_counter()-started
    full_summaries = {name: map_summary(values, full_window) for name, values in full["maps"].items()}
    anchors = [{"i": item["i"], "j": item["j"], "origin": f"full_predicted_peak_{name}"}
               for name, item in full_summaries.items()]
    anchors += [{"i": int(i), "j": int(j), "origin": "native_supplied"} for i, j in native_contacts]
    tracked, window_rows, winners = [], [], {}
    crop_started = time.perf_counter()
    iterator = iter(windows)
    while group := list(islice(iterator, batch_size)):
        if group == [full_window]:
            predictions = [full]  # Exactly identical for already-short pairs.
        else:
            predictions = forward_maps(model,
                [(*crop_features(first, second, window), metadata) for window in group],
                device, precision, cutoffs)
        for window, prediction in zip(group, predictions):
            row = dict(**window.as_dict(), aux_logit=prediction["aux_logit"], aux_probability=prediction["aux_probability"])
            for name, values in prediction["maps"].items():
                summary = map_summary(values, window)
                row.update({f"{name}_{key}": value for key, value in summary.items()})
                if name not in winners or summary["score"] > winners[name]["summary"]["score"]:
                    winners[name] = dict(window=window, summary=summary, maps=prediction["maps"],
                        aux_logit=prediction["aux_logit"], aux_probability=prediction["aux_probability"])
                for anchor in anchors:
                    if window.contains(anchor["i"], anchor["j"]):
                        tracked.append(contact_record(anchor, window, full["maps"][name], values, name))
            window_rows.append(row)
    crop_seconds = time.perf_counter()-crop_started
    summary = {"Protein pairs": metadata["pair"], "Category": bool(metadata["label"]),
        "combined_length": n1+n2, "status": "scored", "n_windows": len(windows),
        "full_seconds": full_seconds, "crop_seconds": crop_seconds,
        "p_aux_protein_full": full["aux_probability"], "aux_logit_full": full["aux_logit"]}
    for name, winner in winners.items():
        summary[f"full_{name}"] = full_summaries[name]["score"]
        summary[f"crop_{name}"] = winner["summary"]["score"]
        summary.update({f"winner_{name}_{key}": value for key, value in winner["window"].as_dict().items()})
        for key in ("i", "j"):
            summary[f"full_{name}_{key}"] = full_summaries[name][key]
            summary[f"crop_{name}_{key}"] = winner["summary"][key]
        anchor = {"i": winner["summary"]["i"], "j": winner["summary"]["j"],
                  "origin": f"crop_predicted_peak_{name}"}
        tracked.append(contact_record(anchor, winner["window"], full["maps"][name], winner["maps"][name], name))
        anchors.append(anchor)
    primary = next(iter(cutoffs))
    summary.update(p_pair_full=summary[f"full_{primary}"], p_pair=summary[f"crop_{primary}"],
                   p_aux_protein_crop_winner=winners[primary]["aux_probability"])
    return dict(summary=summary, windows=window_rows, tracked=tracked, anchors=anchors,
                full=full, full_window=full_window, winners=winners)
