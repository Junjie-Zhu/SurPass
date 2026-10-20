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
from src.runtime.checkpoint import load_checkpoint, save_checkpoint


def _model() -> ResOnly:
    torch.manual_seed(0)
    return ResOnly(
        n_layers=1,
        cross_n_layers=1,
        pair_n_layers=1,
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
        self.assertEqual(payload["architecture"], "inter_pair_v2")
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, actual)

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
            topk_fraction=0.1,
            maximum_topk=4,
        )
        self.assertEqual(count.tolist(), [1])
        self.assertGreater(float(probability.item()), 0.99)


if __name__ == "__main__":
    unittest.main()
