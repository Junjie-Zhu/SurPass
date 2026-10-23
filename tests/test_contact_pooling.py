"""OR semantics and gradient stability for protein-level supervision."""

import unittest
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import pair_contact_summary
from src.model.loss import InterDistanceLoss, ProteinBindLoss, InterContactFocalLoss, max_contact_probability, noisy_or_contact_logit


class ContactPoolingTest(unittest.TestCase):
    def test_noisy_or_matches_explicit_probability_and_masks_padding(self):
        p = torch.tensor([[[0.1, 0.2, 0.99]]])
        logits = torch.stack((p.log(), torch.log1p(-p)), dim=-1).requires_grad_()
        mask = torch.tensor([[[True, True, False]]])
        logit = noisy_or_contact_logit(logits, mask, 1)
        torch.testing.assert_close(logit.sigmoid(), torch.tensor([0.28]))
        F.binary_cross_entropy_with_logits(logit, torch.zeros(1)).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[0, 0, :2, 0] > 0).all())
        self.assertEqual(logits.grad[0, 0, 2].abs().sum().item(), 0)

    def test_negative_gradient_survives_saturated_large_bag(self):
        logits = torch.zeros((1, 128, 128, 2), requires_grad=True)
        logit = noisy_or_contact_logit(logits, torch.ones((1, 128, 128)).bool(), 1)
        loss = F.binary_cross_entropy_with_logits(logit, torch.zeros(1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 1000)
        torch.testing.assert_close(logits.grad[..., 0], torch.full((1, 128, 128), 0.5))

    def test_very_unlikely_positive_keeps_gradient(self):
        logits = torch.tensor([[[[-80.0, 0.0]]]], requires_grad=True)
        logit = noisy_or_contact_logit(logits, torch.ones((1, 1, 1)).bool(), 1)
        torch.testing.assert_close(logit, torch.tensor([-80.0]))
        F.binary_cross_entropy_with_logits(logit, torch.ones(1)).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 0, 0, 0].item(), -0.99)

    def test_all_masked_bag_is_finite_and_has_zero_score_and_gradient(self):
        logits = torch.randn((1, 2, 2, 4), requires_grad=True)
        mask = torch.zeros((1, 2, 2), dtype=torch.bool)
        logit = noisy_or_contact_logit(logits, mask, 2)
        self.assertTrue(torch.isfinite(logit).all())
        logit.sum().backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)
        self.assertEqual(max_contact_probability(torch.ones_like(mask).float(), mask).item(), 0)

    def test_one_contact_suffices_without_accumulating_weak_contacts(self):
        p = torch.full((2, 100, 100), 0.01)
        p[0, 35, 24] = 0.6
        # Even high scores in padding cannot create contacts.
        p[:, -1, :] = 0.99
        mask = torch.ones_like(p, dtype=torch.bool)
        mask[:, -1, :] = False
        logits = torch.stack((p.log(), torch.log1p(-p)), dim=-1)
        score, count = pair_contact_summary(
            logits, mask, contact_bins=1, score_threshold=0.5,
        )
        torch.testing.assert_close(score, torch.tensor([0.6, 0.01]))
        self.assertEqual(count.tolist(), [1, 0])
        self.assertEqual((score >= 0.5).tolist(), [True, False])

    def test_bfloat16_logits_reduce_in_float32(self):
        logits = torch.tensor([[[[-20.0, 0.0], [20.0, 0.0]]]], dtype=torch.bfloat16)
        logits.requires_grad_()
        logit = noisy_or_contact_logit(logits, torch.ones((1, 1, 2)).bool(), 1)
        self.assertEqual(logit.dtype, torch.float32)
        F.binary_cross_entropy_with_logits(logit, torch.zeros(1)).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_losses_use_unweighted_means_and_masked_distances(self):
        logits = torch.tensor([-2.0, 1.0], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        torch.testing.assert_close(ProteinBindLoss()(logits, targets),
                                   F.binary_cross_entropy_with_logits(logits, targets))
        pairs = torch.tensor([[[[0.0, 2.0], [4.0, 0.0]]],
                              [[[1.0, 0.0], [0.0, 5.0]]]], requires_grad=True)
        bins = torch.tensor([[[0, 1]], [[1, 0]]])
        mask = torch.tensor([[[True, False]], [[True, True]]])
        expected = F.cross_entropy(pairs[mask], bins[mask])
        actual = InterDistanceLoss()(pairs, bins, mask)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(pairs.grad[~mask].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(pairs.grad).all())
        empty = InterDistanceLoss()(pairs, bins, torch.zeros_like(mask))
        self.assertEqual(empty.item(), 0)

    def test_focal_contact_loss_recovers_from_confident_wrong_predictions(self):
        logits = torch.tensor([[[[-80.0, 0.0], [80.0, 0.0]]]], requires_grad=True)
        loss = InterContactFocalLoss(1)(
            logits, torch.tensor([[[0, 1]]]), torch.ones((1, 1, 2)).bool(),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 0, 0, 0].item(), -0.99)
        self.assertGreater(logits.grad[0, 0, 1, 0].item(), 0.99)


if __name__ == "__main__":
    unittest.main()
