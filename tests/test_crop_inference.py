"""Inference-only regression tests: coverage, coordinate mapping, and hooks."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from scripts.diagnose_inference import file_hash, main, trace_pair
from src.model.ema import EMAWrapper
from src.model.res_pair import ResPair
from src.runtime.checkpoint import save_checkpoint
from src.runtime.crop_inference import Window, crop_features, forward_maps, score_pair, tile_windows
from src.runtime.inference_diagnostics import InferenceDiagnostics, compare_representations


def features(length, offset=0):
    return dict(plm_emb=torch.arange(length*8).reshape(length, 8).float()/100+offset,
        residue_type=torch.arange(length) % 20, residue_index=torch.arange(length),
        chain_index=torch.zeros(length).long(), mask=torch.ones(length).bool())


class ContextSensitiveModel(torch.nn.Module):
    """Known score law: identical residue vectors score lower in longer context."""
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        length1, length2 = batch["p1_length"], batch["p2_length"]
        n1, n2 = int(length1.max()), int(length2.max())
        rows = torch.arange(n1)[None].expand(len(length1), -1)
        columns = length1[:, None]+torch.arange(n2)[None]
        first = batch["plm_emb"][:, :, 0].gather(1, rows)
        second = batch["plm_emb"][:, :, 0].gather(1, columns)
        logits = torch.zeros(len(length1), n1, n2, 22)
        logits[..., 0] = first[:, :, None]+second[:, None, :]-.1*(length1+length2)[:, None, None]
        mask = (torch.arange(n1)[None, :, None] < length1[:, None, None]) & (torch.arange(n2)[None, None, :] < length2[:, None, None])
        return SimpleNamespace(pair_logits=logits, pair_mask=mask, aux_protein_logit=first.mean(-1))


def score(net, a, b, **kwargs):
    return score_pair(net, a, b, dict(pair="a_b", label=True, combined_length=len(a["mask"])+len(b["mask"])),
        device=torch.device("cpu"), precision="float32", cutoffs={"d8": 7, "d12": 11}, crop_size=8, **kwargs)


@pytest.mark.parametrize("lengths", [(1, 1), (1, 19), (19, 1), (3, 19), (7, 12), (17, 18)])
@pytest.mark.parametrize("stride", [.5, 1.])
def test_tiles_cover_every_residue_pair_and_obey_budget(lengths, stride):
    windows = tile_windows(*lengths, crop_size=8, stride_fraction=stride)
    covered = {(i, j) for w in windows for i in range(w.start1, w.stop1) for j in range(w.start2, w.stop2)}
    assert covered == {(i, j) for i in range(lengths[0]) for j in range(lengths[1])}
    assert len(windows) == len(set(windows))
    assert all((w.stop1-w.start1)+(w.stop2-w.start2) <= 8 for w in windows)


def test_crop_preserves_embedding_values_and_global_residue_identity():
    first, second = features(7), features(9, 1)
    a, b = crop_features(first, second, Window(2, 6, 3, 7))
    assert torch.equal(a["plm_emb"], first["plm_emb"][2:6])
    assert torch.equal(b["plm_emb"], second["plm_emb"][3:7])
    assert a["residue_index"].tolist() == [2, 3, 4, 5]
    a["plm_emb"].zero_()
    assert first["plm_emb"][2:6].count_nonzero() > 0


def test_batching_native_contacts_and_labels_do_not_change_crop_scoring():
    first, second = features(7), features(9, 1)
    baseline = score(ContextSensitiveModel(), first, second, batch_size=1)
    native = score(ContextSensitiveModel(), first, second, batch_size=4, native_contacts=[(2, 4)])
    for cutoff in ("d8", "d12"):
        assert baseline["summary"][f"crop_{cutoff}"] == native["summary"][f"crop_{cutoff}"]
        assert baseline["winners"][cutoff]["window"] == native["winners"][cutoff]["window"]
    assert baseline["windows"] == native["windows"]
    for row in native["tracked"]:
        i, j = row["i"], row["j"]
        assert row["full_probability"] == native["full"]["maps"][row["cutoff"]][i, j]
        assert row["crop_probability"] > row["full_probability"]
        assert row["local_i"] == i-row["start1"]
        assert row["local_j"] == j-row["start2"]
    flipped = score_pair(ContextSensitiveModel(), first, second, dict(pair="a_b", label=False),
        device=torch.device("cpu"), precision="float32", cutoffs={"d8": 7}, crop_size=8)
    assert flipped["summary"]["p_pair"] == baseline["summary"]["p_pair"]


def test_short_inputs_are_exactly_unchanged_and_reuse_full_forward():
    model = ContextSensitiveModel()
    result = score(model, features(3), features(4))
    assert result["summary"]["p_pair"] == result["summary"]["p_pair_full"]
    assert result["summary"]["n_windows"] == 1
    assert model.calls == 1


@pytest.fixture
def tiny_model():
    settings = dict(n_layers=2, dim_token=16, dim_pair=8, n_heads=4, plm_in_dim=8,
        plm_out_dim=8, index_emb_dim=4, distance_bin_count=22, distance_bin_convention="left_closed",
        contact_threshold=8., triangle_hidden_dim=4, triangle_chunk_size=2, triangle_checkpoint=False,
        recycle_passes=2, pair_update_every=1, protein_pooling="max", allow_missing_physics=True)
    with patch("src.model.components.embedder.build_pair_bin_energy_table", return_value=torch.zeros(21, 21, 4, 22)):
        torch.manual_seed(123)
        net = ResPair(**settings).eval()
        with torch.no_grad():
            for name, parameter in net.named_parameters():
                if "output_projection.weight" in name:
                    parameter.normal_(std=.03)
        yield net, settings


def test_real_model_hooks_preserve_outputs_and_track_same_residues(tiny_model):
    net, _ = tiny_model
    first, second = features(7), features(8, 1)
    metadata = dict(pair="a_b", label=True)
    anchors = [dict(i=3, j=4, origin="native_supplied")]
    full_window = Window(0, 7, 0, 8)
    baseline = forward_maps(net, [(first, second, metadata)], torch.device("cpu"), "float32", {"d8": 7})[0]
    with torch.inference_mode(), InferenceDiagnostics(net, first, second, full_window, anchors) as full:
        observed = forward_maps(net, [(first, second, metadata)], torch.device("cpu"), "float32", {"d8": 7})[0]
    np.testing.assert_array_equal(observed["maps"]["d8"], baseline["maps"]["d8"])
    assert len(full.attention) == 2*2*2*4  # layers, recycles, directions, heads
    assert {row["recycle"] for row in full.attention} == {1, 2}
    assert all(0 <= row["entropy_fraction"] <= 1.000001 for row in full.attention)
    assert all(0 <= row["probability_to_partner"] <= 1 for row in full.attention)
    assert all(row["valid_keys"] == (8 if row["direction"] == "A_to_B" else 7) for row in full.attention)
    assert all(row["native_key_mass"] == row["probability_to_partner"] for row in full.attention)
    window = Window(1, 5, 2, 6)
    a, b = crop_features(first, second, window)
    with torch.inference_mode(), InferenceDiagnostics(net, a, b, window, anchors) as crop:
        forward_maps(net, [(a, b, metadata)], torch.device("cpu"), "float32", {"d8": 7})
    comparison = compare_representations(full, crop)
    inputs = [row for row in comparison if row["stage"] == "esm_input"]
    assert len(inputs) == 2 and all(row["exactly_equal"] and row["relative_l2"] == 0 for row in inputs)
    assert {row["stage"] for row in comparison} >= {"triangle_pre_norm", "triangle_post_norm", "pair_block", "embedding", "distance_logits"}
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in net.modules())


def test_hook_cleanup_on_exception(tiny_model):
    net, _ = tiny_model
    with pytest.raises(RuntimeError, match="injected"):
        with InferenceDiagnostics(net, features(3), features(4), Window(0, 3, 0, 4), []):
            raise RuntimeError("injected")
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in net.modules())


def test_observer_preserves_bfloat16_forward_and_finite_attention(tiny_model):
    net, _ = tiny_model
    first, second = features(5), features(6, 1)
    metadata = dict(pair="a_b", label=True)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        baseline = forward_maps(net, [(first, second, metadata)], torch.device("cpu"), "float32", {"d8": 7})[0]
        with InferenceDiagnostics(net, first, second, Window(0, 5, 0, 6), [dict(i=2, j=3, origin="native_supplied")]) as trace:
            observed = forward_maps(net, [(first, second, metadata)], torch.device("cpu"), "float32", {"d8": 7})[0]
    np.testing.assert_array_equal(baseline["maps"]["d8"], observed["maps"]["d8"])
    assert all(np.isfinite(row["entropy_nats"]) for row in trace.attention)


def test_complete_command_ema_metrics_diagnostics_and_resume(tmp_path, tiny_model):
    net, settings = tiny_model
    for identifier, length in [("a", 3), ("b", 3), ("c", 7), ("d", 8), ("e", 30)]:
        torch.save({"plm_emb": features(length)["plm_emb"]}, tmp_path/f"{identifier}.pt")
    fasta = tmp_path/"proteins.fasta"
    fasta.write_text("".join(f">{identifier}\n{'A'*length}\n" for identifier, length in [("a", 3), ("b", 3), ("c", 7), ("d", 8), ("e", 30)]))
    pairs = tmp_path/"test.tsv"
    pairs.write_text("Protein pairs\tCategory\na_b\tTrue\nc_d\tFalse\na_e\tFalse\n")
    panel = tmp_path/"diagnostic.tsv"
    panel.write_text("Protein pairs\nc_d\n")
    contacts = tmp_path/"contacts.tsv"
    contacts.write_text("Protein pairs\ti\tj\nc_d\t2\t3\n")
    checkpoint = tmp_path/"model.pth"
    ema = EMAWrapper(net, .999, [""])
    ema.register()
    save_checkpoint(checkpoint, net, ema=ema)
    digest = file_hash(checkpoint)
    cfg = OmegaConf.create(dict(model=dict(name="ResPair", **settings), seed=42,
        ckpt_dir=str(checkpoint), runtime=dict(precision="float32", checkpoint_weights="ema"),
        data=dict(test_set_tsv=str(pairs), fasta_path=str(fasta), embedding_dir=str(tmp_path), maximum_length=20,
                  distance_bin_start=2., distance_bin_end=22., distance_bin_count=22,
                  distance_bin_convention="left_closed", contact_threshold=8.)))
    config = tmp_path/"config.yaml"
    OmegaConf.save(cfg, config)
    out = tmp_path/"output"
    args = ["--config", str(config), "--output-dir", str(out), "--crop-size", "8",
            "--batch-size", "3", "--cutoffs", "8", "12", "--diagnostic-pairs", str(panel),
            "--native-contacts", str(contacts), "--save-maps"]
    main(args)
    assert file_hash(checkpoint) == digest
    with (out/"predictions.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 3
    assert rows[0]["p_pair"] == rows[0]["p_pair_full"]
    assert rows[-1]["status"].startswith("unsupported_length")
    with (out/"metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == 8 and all(int(row["N"]) == 2 for row in metrics)
    case = out/"pairs/000001"
    assert json.loads((case/"diagnostic_notes.json").read_text())["observer_full_map_check_passed"]
    assert json.loads((case/"diagnostic_notes.json").read_text())["native_contacts_supplied"]
    assert (case/"attention.tsv").exists() and (case/"representation_vectors.npz").exists()
    with (case/"representation_comparison.tsv").open() as f:
        comparisons = list(csv.DictReader(f, delimiter="\t"))
    assert all(row["exactly_equal"] == "True" for row in comparisons if row["stage"] == "esm_input")
    with patch("scripts.diagnose_inference.score_pair", side_effect=AssertionError("should reuse completed pairs")):
        main(args+["--resume"])
    with pytest.raises(ValueError, match="Resume identity"):
        main(args+["--resume", "--stride-fraction", "1"])
