from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import ResOnly, ResPair
from src.model.components.res_pair import (
    InterIntraTriangle, PairBiasedCrossAttention, SymmetricSingleToPair,
)
from src.model.loss import SurpassLoss
from src.runtime.checkpoint import load_checkpoint, save_checkpoint
from src.runtime.config import create_model


def model(**kwargs):
    settings = dict(n_layers=2, dim_token=24, dim_pair=12, n_heads=3,
                    plm_in_dim=8, plm_out_dim=12, index_emb_dim=8,
                    distance_bin_count=8, triangle_hidden_dim=8,
                    triangle_chunk_size=2, recycle_passes=2,
                    pair_update_every=1,
                    allow_missing_physics=True)
    settings.update(kwargs)
    return ResPair(**settings)


def batch(first, second, padding=0):
    length_1, length_2 = len(first), len(second)
    width = length_1 + length_2 + padding
    return {
        "plm_emb": torch.cat((first, second, torch.randn(padding, 8)))[None],
        "residue_type": (torch.arange(width) % 21)[None],
        "chain_index": torch.cat((torch.zeros(length_1), torch.ones(length_2),
                                   torch.zeros(padding))).long()[None],
        "p1_length": torch.tensor([length_1]), "p2_length": torch.tensor([length_2]),
    }


def activate(module):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "output_projection" in name:
                parameter.normal_(std=0.05)


class TriangleTest(unittest.TestCase):
    def test_chunked_contraction_and_gradients_match_dense_pplm_equation(self):
        torch.manual_seed(41)
        chunked = InterIntraTriangle(6, 4, chunk_size=2).double().train()
        activate(chunked)
        dense = copy.deepcopy(chunked)
        dense.chunk_size = 0
        dense.checkpoint_chunks = False
        single = SymmetricSingleToPair(8, 6).double()
        masks = (torch.tensor([[True, True, True, True, False]]),
                 torch.tensor([[True, True, False]]))
        pair_mask = masks[0][:, :, None] & masks[1][:, None, :]
        first, second = single(torch.randn(1, 5, 8, dtype=torch.double)), single(torch.randn(1, 3, 8, dtype=torch.double))
        pair = torch.randn(1, 5, 3, 6, dtype=torch.double, requires_grad=True)
        factors = tuple(tuple(x.detach().requires_grad_() for x in f) for f in (first, second))
        actual = chunked(pair, *factors, *masks, pair_mask)

        # Independent dense implementation, including both chain orientations.
        normalized = dense.norm(pair)
        def gated(value):
            value, gate = value.chunk(2, -1)
            return value * gate.sigmoid()
        inter = gated(dense.inter_projection(normalized)) * pair_mask[..., None]
        intra = []
        for f, mask in zip(factors, masks):
            a, b = f
            raw = (a[:, :, None] * b[:, None, :] + b[:, :, None] * a[:, None, :]) / 2
            intra.append(gated(dense.intra_projection(dense.norm(raw))) * (mask[:, :, None] & mask[:, None, :])[..., None])
        hidden = torch.einsum("bikc,bkjc->bijc", intra[0], inter)
        hidden = hidden + torch.einsum("bilc,bjlc->bijc", inter, intra[1])
        expected = dense.output_projection(dense.output_norm(hidden)) * dense.output_gate(normalized).sigmoid() * pair_mask[..., None]
        torch.testing.assert_close(actual, expected)
        inputs = (pair, *factors[0], *factors[1])
        actual_grads = torch.autograd.grad(actual.square().sum(), (*inputs, *chunked.parameters()), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (*inputs, *dense.parameters()))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)

    def test_triangle_uses_both_intrachain_contexts_and_ignores_padding(self):
        torch.manual_seed(42)
        module = InterIntraTriangle(6, 4, chunk_size=2).eval()
        activate(module)
        pair = torch.randn(1, 4, 3, 6)
        first = (torch.randn(1, 4, 6), torch.randn(1, 4, 6))
        second = (torch.randn(1, 3, 6), torch.randn(1, 3, 6))
        mask_1, mask_2 = torch.tensor([[True, True, True, False]]), torch.tensor([[True, True, False]])
        mask = mask_1[:, :, None] & mask_2[:, None, :]
        baseline = module(pair, first, second, mask_1, mask_2, mask)
        for side in (0, 1):
            factors = [tuple(x.clone() for x in first), tuple(x.clone() for x in second)]
            factors[side][0][:, 0] += torch.randn_like(factors[side][0][:, 0]) * 5
            changed = module(pair, *factors, mask_1, mask_2, mask)
            self.assertGreater((baseline - changed).abs().sum().item(), 1e-5)
        pair[~mask] = 1000
        for factors, valid in ((first, mask_1), (second, mask_2)):
            for value in factors:
                value[~valid] = 1000
        torch.testing.assert_close(module(pair, first, second, mask_1, mask_2, mask), baseline)


class PairAttentionTest(unittest.TestCase):
    def test_pair_bias_changes_attention_and_receives_gradients(self):
        torch.manual_seed(43)
        module = PairBiasedCrossAttention(12, 6, 3).eval()
        activate(module)
        a, b, z = torch.randn(1, 4, 12), torch.randn(1, 3, 12), torch.randn(1, 4, 3, 6, requires_grad=True)
        ma, mb = torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)
        out = module(a, b, ma, mb, z)
        changed = module(a, b, ma, mb, torch.zeros_like(z))
        self.assertGreater(sum((x-y).abs().sum().item() for x,y in zip(out, changed)), 1e-5)
        sum(x.square().sum() for x in out).backward()
        self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertGreater(module.pair_bias.weight.grad.abs().sum().item(), 0)

    def test_bias_is_inside_softmax_and_masked_before_normalization(self):
        torch.manual_seed(44)
        module = PairBiasedCrossAttention(12, 6, 3).eval()
        activate(module)
        a, b, z = torch.randn(1, 4, 12), torch.randn(1, 3, 12), torch.randn(1, 4, 3, 6)
        ma, mb = torch.tensor([[True, True, True, False]]), torch.tensor([[True, True, False]])
        prepared_a, prepared_b = module._prepare(a), module._prepare(b)
        q, _, _, gate = prepared_a
        _, k, v, _ = prepared_b
        bias = module.pair_bias(module.pair_norm(z)).permute(0, 3, 1, 2)
        scores = (q @ k.transpose(-1, -2)) / (module.head_dim ** 0.5) + bias
        weights = scores.masked_fill(~mb[:, None, None], -torch.inf).softmax(-1)
        expected = module.output_projection((weights @ v).transpose(1, 2).flatten(2) * gate) * ma[..., None]
        actual = module._attend(prepared_a, prepared_b, ma, mb, bias)
        torch.testing.assert_close(actual, expected)


class ResPairTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)

    def test_swap_and_padding_after_active_updates_and_recycling(self):
        net = model().eval()
        activate(net)
        a, b = torch.randn(5, 8), torch.randn(3, 8)
        forward, reverse, padded = batch(a, b), batch(b, a), batch(a, b, padding=7)
        # Swap the true residue identities as well as the PLM features.
        reverse["residue_type"] = torch.cat((forward["residue_type"][:, 5:], forward["residue_type"][:, :5]), dim=1)
        with torch.inference_mode():
            x, y, p = net(forward), net(reverse), net(padded)
        self.assertEqual(x.pair_logits.shape, (1, 5, 3, 8))
        torch.testing.assert_close(x.pair_logits, y.pair_logits.transpose(1, 2), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(x.protein_logit, y.protein_logit)
        torch.testing.assert_close(x.aux_protein_logit, y.aux_protein_logit)
        torch.testing.assert_close(x.pair_logits, p.pair_logits)
        self.assertFalse(any(isinstance(m, torch.nn.Conv2d) for m in net.modules()))

    def test_mixed_length_batch_and_empty_context_are_finite(self):
        net = model().eval()
        activate(net)
        first = batch(torch.randn(5, 8), torch.randn(3, 8))
        second = batch(torch.randn(2, 8), torch.randn(4, 8), padding=2)
        empty = batch(torch.randn(3, 8), torch.empty(0, 8), padding=5)
        combined = {key: torch.cat([x[key] for x in (first, second, empty)]) for key in first}
        with torch.inference_mode():
            result, alone = net(combined), net(second)
        torch.testing.assert_close(result.pair_logits[1, :2, :4], alone.pair_logits[0], atol=1e-5, rtol=1e-4)
        self.assertEqual(result.pair_mask.sum((1, 2)).tolist(), [15, 8, 0])
        self.assertTrue(torch.isfinite(result.pair_logits).all())
        self.assertTrue(torch.isfinite(result.protein_logit).all())
        self.assertTrue(torch.isfinite(result.aux_protein_logit).all())

    def test_update_order(self):
        net = model(n_layers=1, recycle_passes=1).eval()
        events, handles = [], []
        block = net.pair_blocks["0"]
        for name in ("single_to_pair", "triangle", "transition"):
            handles.append(getattr(block, name).register_forward_hook(lambda m, a, o, n=name: events.append(n)))
        net(batch(torch.randn(4, 8), torch.randn(3, 8)))
        for handle in handles:
            handle.remove()
        self.assertEqual(events, ["single_to_pair", "single_to_pair", "triangle", "transition"])

    def test_recycling_uses_original_embeddings_and_detached_states(self):
        net = model(recycle_passes=3).train()
        data = batch(torch.randn(4, 8), torch.randn(3, 8))
        trunk_inputs, trunk_outputs, grad_modes = [], [], []
        original_trunk = net._trunk
        def record(*args):
            trunk_inputs.append(tuple(x.detach().clone() for x in args[:3]))
            grad_modes.append(torch.is_grad_enabled())
            result = original_trunk(*args)
            trunk_outputs.append(tuple(x.detach().clone() for x in result))
            return result
        with patch.object(net, "_trunk", side_effect=record), \
             patch.object(net.residue_embedder, "forward", wraps=net.residue_embedder.forward) as embedding, \
             patch.object(net.residue_embedder, "cross_pair", wraps=net.residue_embedder.cross_pair) as pairs, \
             patch.object(net.residue_embedder.physics_projection, "forward", wraps=net.residue_embedder.physics_projection.forward) as physics:
            output = net(data)
        self.assertEqual(grad_modes, [False, False, True])
        self.assertEqual((embedding.call_count, pairs.call_count, physics.call_count), (1, 1, 1))
        for r in (1, 2):
            for i, projection in enumerate((net.recycle_single, net.recycle_single, net.recycle_pair)):
                # Ps(0) and Pz(0) initially equal zero.
                expected = trunk_inputs[0][i] + projection(trunk_outputs[r-1][i])
                torch.testing.assert_close(trunk_inputs[r][i], expected)
        (output.pair_logits.square().mean() + output.aux_protein_logit.square().mean()).backward()
        self.assertGreater(net.residue_embedder.plm_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(net.recycle_single[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(net.recycle_pair[1].weight.grad.abs().sum().item(), 0)

    def test_backward_optimizer_and_no_stale_cache_in_float_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            net = model().train()
            data = batch(torch.randn(4, 8), torch.randn(3, 8))
            labels = {"crop_is_positive": torch.tensor([True]), "source_is_positive": torch.tensor([True]),
                      "label_inter_bins": torch.full((1, 4, 3), 7),
                      "label_inter_mask": torch.ones(1, 4, 3, dtype=torch.bool)}
            labels["label_inter_bins"][0, 0, 0] = 0
            loss_fn = SurpassLoss(net.contact_bins, {"protein": 1, "contact": 1, "distance": .1, "aux_protein": .1})
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
                    output = net(data)
                    loss, _ = loss_fn(output, labels)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in net.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                opt.step()
            self.assertGreater(net.cross_blocks[0].pair_bias.weight.grad.abs().sum().item(), 0)
            self.assertGreater(net.pair_blocks["0"].triangle.intra_projection.weight.grad.abs().sum().item(), 0)

    def test_single_pass_has_no_unused_trainable_parameters(self):
        net = model(recycle_passes=1).train()
        output = net(batch(torch.randn(4, 8), torch.randn(3, 8)))
        (output.pair_logits.square().mean() + output.aux_protein_logit.square().mean()).backward()
        self.assertEqual([n for n,p in net.named_parameters() if p.grad is None], [])

    def test_separate_checkpoint_identity_and_roundtrip(self):
        net = model().eval()
        data = batch(torch.randn(4, 8), torch.randn(3, 8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "respair.pth"
            save_checkpoint(path, net)
            restored = model().eval()
            payload = load_checkpoint(path, restored)
            self.assertEqual(payload["architecture"], "res_pair_v1")
            torch.testing.assert_close(net(data).pair_logits, restored(data).pair_logits)
            old = ResOnly(n_layers=1, dim_token=24, dim_pair=12, n_heads=3,
                          plm_in_dim=8, plm_out_dim=12, index_emb_dim=8,
                          distance_bin_count=8, allow_missing_physics=True)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(path, old)
            save_checkpoint(path, old)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(path, net)

    def test_configuration_and_pair_update_frequency(self):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            for preset, name in (("train", "ResPair"), ("inference", "ResPair"),
                                 ("train_respair", "ResPair"), ("inference_respair", "ResPair")):
                cfg = compose(config_name=preset, overrides=["model.dim_token=24", "model.dim_pair=12",
                    "model.n_heads=3", "model.plm_in_dim=8", "model.plm_out_dim=12", "model.index_emb_dim=8",
                    "model.allow_missing_physics=true"])
                self.assertEqual(type(create_model(cfg)).__name__, name)
                if name == "ResPair":
                    self.assertEqual(cfg.model.n_layers, 12)
                    self.assertEqual(cfg.model.pair_update_every, 2)
        net = model(n_layers=4, pair_update_every=2)
        self.assertEqual(list(net.pair_blocks), ["1", "3"])
        cfg = OmegaConf.create({"model": {"name": "typo"}})
        with self.assertRaisesRegex(ValueError, "model.name"):
            create_model(cfg)
        for kwargs in ({"recycle_passes": 0}, {"triangle_chunk_size": -1}, {"pair_update_every": 0}):
            with self.assertRaises(ValueError):
                model(**kwargs)
        invalid = batch(torch.randn(4, 8), torch.randn(3, 8))
        invalid["padding_mask"] = torch.zeros(1, 7, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "padding_mask"):
            model()(invalid)

    def test_twelve_layer_schedule_executes_pairs_only_on_even_layers(self):
        net = model(n_layers=12, pair_update_every=2, recycle_passes=1).eval()
        self.assertEqual(list(net.pair_blocks), ["1", "3", "5", "7", "9", "11"])
        events = []
        handles = [block.register_forward_hook(
            lambda module, args, output, key=key: events.append(int(key) + 1)
        ) for key, block in net.pair_blocks.items()]
        with torch.inference_mode():
            output = net(batch(torch.randn(4, 8), torch.randn(3, 8)))
        for handle in handles:
            handle.remove()
        self.assertEqual(events, [2, 4, 6, 8, 10, 12])
        self.assertTrue(torch.isfinite(output.pair_logits).all())


if __name__ == "__main__":
    unittest.main()
