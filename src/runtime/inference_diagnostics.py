"""Temporary read-only hooks for ResPair inference; no attention replacement."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch


class InferenceDiagnostics:
    """Capture selected rows of attention and bounded representation samples.

    Attention is reconstructed in float32 from the current Q/K and pair bias;
    the model still executes its original fused attention unchanged. Vectors
    are retained only at tracked global residue/contact coordinates.
    """

    def __init__(self, model, first, second, window, anchors, sample_cells=1024):
        if sample_cells < 1:
            raise ValueError("sample_cells must be positive")
        self.model, self.first, self.second = model, first, second
        self.window, self.anchors, self.sample_cells = window, anchors, sample_cells
        self.attention, self.statistics, self.vectors, self.handles = [], [], {}, []
        self.calls = defaultdict(int)

    def _next(self, key):
        value = self.calls[key]
        self.calls[key] += 1
        return value

    def _summary(self, value, stage, layer, recycle, side):
        flat = value.detach().reshape(-1, value.shape[-1])
        indices = torch.linspace(0, len(flat)-1, min(len(flat), self.sample_cells), device=value.device).long()
        sample = flat.index_select(0, indices).float()
        self.statistics.append(dict(stage=stage, layer=layer, recycle=recycle, side=side,
            shape=list(value.shape), sampled_cells=len(indices),
            sampled_mean=float(sample.mean()), sampled_std=float(sample.std(correction=0)),
            sampled_rms=float(sample.square().mean().sqrt()), finite=bool(torch.isfinite(sample).all())))

    def _single(self, value, stage, layer, recycle, side):
        self._summary(value, stage, layer, recycle, side)
        start, stop = ((self.window.start1, self.window.stop1) if side == "A"
                       else (self.window.start2, self.window.stop2))
        indices = sorted({anchor["i" if side == "A" else "j"] for anchor in self.anchors})
        for index in indices:
            if start <= index < stop:
                key = (stage, layer, recycle, side, index, -1)
                self.vectors[key] = value[0, index-start].detach().float().cpu().numpy().copy()

    def _pair(self, value, stage, layer, recycle):
        self._summary(value, stage, layer, recycle, "pair")
        for anchor in self.anchors:
            i, j = anchor["i"], anchor["j"]
            if self.window.contains(i, j):
                key = (stage, layer, recycle, "pair", i, j)
                self.vectors[key] = value[0, i-self.window.start1, j-self.window.start2].detach().float().cpu().numpy().copy()

    def _cross_attention(self, module, args, layer):
        first, second, mask1, mask2, pair = args
        recycle = self._next(("attention", layer))+1
        prepared = (module._prepare(first), module._prepare(second))
        for direction, q, k, oriented_pair, key_mask, query_start, key_start, first_side in [
            ("A_to_B", prepared[0][0], prepared[1][1], pair, mask2,
             self.window.start1, self.window.start2, True),
            ("B_to_A", prepared[1][0], prepared[0][1], pair.transpose(1, 2), mask1,
             self.window.start2, self.window.start1, False),
        ]:
            present = [a for a in self.anchors if self.window.contains(a["i"], a["j"])]
            queries = sorted({a["i" if first_side else "j"]-query_start for a in present})
            if not queries or not bool(key_mask.any()):
                continue
            qi = torch.tensor(queries, device=q.device)
            bias = module.pair_bias(module.pair_norm(oriented_pair.index_select(1, qi))).permute(0, 3, 1, 2)
            logits = q.index_select(2, qi).float() @ k.float().transpose(-1, -2) / math.sqrt(module.head_dim)
            logits = (logits + bias.float()).masked_fill(~key_mask[:, None, None, :], -torch.inf)
            weights = logits.softmax(-1)[0].detach().cpu().numpy()
            valid = key_mask[0].detach().cpu().numpy().astype(bool)
            count = int(valid.sum())
            tracked_keys = sorted({a["j" if first_side else "i"]-key_start for a in present})
            native_keys = sorted({a["j" if first_side else "i"]-key_start for a in present
                                  if a["origin"] == "native_supplied"})
            lookup = {query: row for row, query in enumerate(queries)}
            for anchor in present:
                query = anchor["i" if first_side else "j"]-query_start
                key = anchor["j" if first_side else "i"]-key_start
                for head, head_weights in enumerate(weights):
                    row = head_weights[lookup[query]]
                    positive = row[row > 0]
                    entropy = float(-(positive*np.log(positive)).sum())
                    self.attention.append(dict(layer=layer, recycle=recycle, direction=direction, head=head,
                        i=anchor["i"], j=anchor["j"], anchor_origin=anchor["origin"], valid_keys=count,
                        probability_to_partner=float(row[key]), uniform_probability=1/count,
                        partner_enrichment=float(row[key])*count,
                        tracked_key_mass=float(row[tracked_keys].sum()),
                        native_key_mass=float(row[native_keys].sum()) if native_keys else None,
                        entropy_nats=entropy, entropy_fraction=entropy/math.log(count) if count > 1 else 0.,
                        max_probability=float(row.max()), max_key_global=int(row.argmax())+key_start))

    def __enter__(self):
        if not hasattr(self.model, "cross_blocks") or not hasattr(self.model, "pair_blocks"):
            raise ValueError("Attention/representation diagnostics currently require ResPair")
        self._single(self.first["plm_emb"][None], "esm_input", 0, 0, "A")
        self._single(self.second["plm_emb"][None], "esm_input", 0, 0, "B")

        def embedding_hook(module, args, value):
            n1 = self.window.stop1-self.window.start1
            self._single(value[:, :n1], "embedding", 0, 0, "A")
            self._single(value[:, n1:], "embedding", 0, 0, "B")
        self.handles.append(self.model.residue_embedder.register_forward_hook(embedding_hook))
        for index, block in enumerate(self.model.residue_blocks):
            layer = index+1
            def residue_hook(module, args, value, layer=layer):
                call = self._next(("residue", layer))
                self._single(value, "self_attention_block", layer, call//2+1, "A" if call % 2 == 0 else "B")
            self.handles.append(block.register_forward_hook(residue_hook))
        for index, block in enumerate(self.model.cross_blocks):
            layer = index+1
            def attention_hook(module, args, layer=layer):
                self._cross_attention(module, args, layer)
            def cross_hook(module, args, values, layer=layer):
                recycle = self._next(("cross", layer))+1
                self._single(values[0], "cross_attention_block", layer, recycle, "A")
                self._single(values[1], "cross_attention_block", layer, recycle, "B")
            self.handles.append(block.register_forward_pre_hook(attention_hook))
            self.handles.append(block.register_forward_hook(cross_hook))
        for name, block in self.model.pair_blocks.items():
            layer = int(name)+1
            def pair_hook(module, args, value, layer=layer):
                self._pair(value, "pair_block", layer, self._next(("pair", layer))+1)
            def pre_norm_hook(module, args, layer=layer):
                self._pair(args[0], "triangle_pre_norm", layer, self._next(("triangle_pre", layer))+1)
            def post_norm_hook(module, args, value, layer=layer):
                self._pair(value, "triangle_post_norm", layer, self._next(("triangle_post", layer))+1)
            self.handles.append(block.register_forward_hook(pair_hook))
            self.handles.append(block.triangle.output_norm.register_forward_pre_hook(pre_norm_hook))
            self.handles.append(block.triangle.output_norm.register_forward_hook(post_norm_hook))
        def head_hook(module, args, value):
            self._pair(value, "distance_logits", 0, 0)
        self.handles.append(self.model.pair_head.register_forward_hook(head_hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compare_representations(full, crop):
    rows = []
    for key, value in crop.vectors.items():
        if key not in full.vectors:
            continue
        baseline = full.vectors[key]
        norm_full, norm_crop = float(np.linalg.norm(baseline)), float(np.linalg.norm(value))
        stage, layer, recycle, side, i, j = key
        rows.append(dict(stage=stage, layer=layer, recycle=recycle, side=side, i=i, j=j,
            cosine=float(np.dot(baseline, value)/(norm_full*norm_crop)) if norm_full*norm_crop else None,
            relative_l2=float(np.linalg.norm(value-baseline)/norm_full) if norm_full else None,
            rms_full=float(np.sqrt(np.mean(baseline**2))), rms_crop=float(np.sqrt(np.mean(value**2))),
            rms_ratio=norm_crop/norm_full if norm_full else None,
            exactly_equal=bool(np.array_equal(value, baseline))))
    return rows
