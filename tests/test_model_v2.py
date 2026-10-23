from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import pair_contact_summary
from src.model.surpass import ResOnly
from src.model.loss import SurpassLoss
from src.model.components.transformer import ResidueTransformer
from src.model.components.inter_pair import InterPairBlock
from src.runtime.checkpoint import load_checkpoint, save_checkpoint


def _model() -> ResOnly:
    torch.manual_seed(0)
    return ResOnly(
        n_layers=1,
        dim_token=32,
        dim_pair=16,
        n_heads=4,
        plm_in_dim=8,
        plm_out_dim=16,
        index_emb_dim=8,
        distance_bin_count=8,
        allow_missing_physics=True,
    ).eval()


def _batch(first: torch.Tensor, second: torch.Tensor, padding: int = 0):
    sequence = torch.cat(
        (first, second, torch.randn(padding, first.shape[-1])),
        dim=0,
    )
    length_1 = first.shape[0]
    length_2 = second.shape[0]
    residue_type = torch.zeros(sequence.shape[0], dtype=torch.long)
    chain_index = torch.cat(
        (
            torch.zeros(length_1, dtype=torch.long),
            torch.ones(length_2, dtype=torch.long),
            torch.zeros(padding, dtype=torch.long),
        )
    )
    return {
        "plm_emb": sequence.unsqueeze(0),
        "residue_type": residue_type.unsqueeze(0),
        "chain_index": chain_index.unsqueeze(0),
        "p1_length": torch.tensor([length_1]),
        "p2_length": torch.tensor([length_2]),
    }


class ModelV2Test(unittest.TestCase):
    def test_rectangular_output_and_padding_invariance(self):
        model = _model()
        torch.manual_seed(1)
        first = torch.randn(4, 8)
        second = torch.randn(3, 8)
        with torch.inference_mode():
            plain = model(_batch(first, second))
            padded = model(_batch(first, second, padding=5))
        self.assertEqual(tuple(plain.pair_logits.shape), (1, 4, 3, 8))
        torch.testing.assert_close(plain.pair_logits, padded.pair_logits)
        torch.testing.assert_close(plain.protein_logit, padded.protein_logit)

    def test_chain_swap_transposes_map_and_preserves_score(self):
        model = _model()
        torch.manual_seed(2)
        first = torch.randn(5, 8)
        second = torch.randn(3, 8)
        with torch.inference_mode():
            forward = model(_batch(first, second))
            reverse = model(_batch(second, first))
        torch.testing.assert_close(
            forward.pair_logits,
            reverse.pair_logits.transpose(1, 2),
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            forward.protein_logit,
            reverse.protein_logit,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_checkpoint_roundtrip(self):
        model = _model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_checkpoint(path, model, epoch=3)
            restored = _model()
            payload = load_checkpoint(path, restored)
        self.assertEqual(payload["architecture"], "inter_pair_v3")
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_residual_blocks_start_as_identity_on_valid_features(self):
        single = torch.randn(2, 5, 32)
        mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
        block = ResidueTransformer(32, 4)
        torch.testing.assert_close(block(single, mask), single * mask[..., None])
        pair = torch.randn(2, 5, 3, 16)
        pair_mask = mask[..., None].expand(-1, -1, 3)
        torch.testing.assert_close(
            InterPairBlock(16)(pair, pair_mask), pair * pair_mask[..., None],
        )

    def test_pair_initializer_preserves_input_feature_gradients(self):
        model = _model()
        output = model(_batch(torch.randn(4, 8), torch.randn(3, 8)))
        output.pair_logits.square().mean().backward()
        # A zero-initialized Transition must not replace its input.
        grad = model.residue_embedder.pair_left.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_training_loss_backward_and_optimizer_step_in_float32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = _model().train()
            batch = _batch(torch.randn(4, 8), torch.randn(3, 8))
            batch = {key: value.expand(2, *value.shape[1:]).clone() for key, value in batch.items()}
            labels = {
                "crop_is_positive": torch.tensor([True, False]),
                "source_is_positive": torch.tensor([True, False]),
                "label_inter_bins": torch.full((2, 4, 3), 7),
                "label_inter_mask": torch.ones((2, 4, 3), dtype=torch.bool),
            }
            labels["label_inter_bins"][0, 0, 0] = 0
            criterion = SurpassLoss(model.contact_bins, {
                "protein": 1, "contact": 1, "distance": 0.1, "aux_protein": 0.1,
            })
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
                    output = model(batch)
                    loss, _ = criterion(output, labels)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters() if parameter.grad is not None
                ))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    def test_chain_swap_after_residual_updates_are_active(self):
        model = _model()
        with torch.no_grad():
            # Initial zero residual branches would hide ordering/masking bugs.
            for name, parameter in model.named_parameters():
                if "output_projection" in name or "axis_projection" in name or "transition_out" in name:
                    parameter.normal_(std=0.02)
        first, second = torch.randn(5, 8), torch.randn(3, 8)
        forward = model(_batch(first, second))
        reverse = model(_batch(second, first))
        torch.testing.assert_close(forward.pair_logits, reverse.pair_logits.transpose(1, 2))
        torch.testing.assert_close(forward.protein_logit, reverse.protein_logit)

    def test_old_architecture_checkpoint_is_rejected(self):
        model = _model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pth"
            save_checkpoint(path, model, architecture="inter_pair_v2")
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(path, model)

    def test_rectangular_contact_count_is_not_double_counted(self):
        logits = torch.full((1, 2, 3, 4), -8.0)
        logits[..., -1] = 8.0
        logits[0, 1, 2, 0] = 12.0
        logits[0, 1, 2, -1] = -12.0
        probability, count = pair_contact_summary(
            logits,
            torch.ones((1, 2, 3), dtype=torch.bool),
            contact_bins=1,
            score_threshold=0.5,
        )
        self.assertEqual(count.tolist(), [1])
        self.assertGreater(float(probability.item()), 0.99)


if __name__ == "__main__":
    unittest.main()
