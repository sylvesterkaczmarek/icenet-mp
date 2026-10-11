"""Independently recompute saved forecast scores and verify experiment invariants.

Works on the compact public evidence files, without original NetCDF or checkpoints.
The original source archives and optional full processor weights can be kept outside
Git; hashes in the experiment manifest still identify exactly what was used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def score(prediction, truth, ocean):
    assert prediction.shape == truth.shape
    delta = (prediction[:, ocean] - truth[:, ocean]).astype(np.float64)
    return {
        "mae_percentage_points": float(np.abs(delta).mean() * 100.0),
        "rmse_percentage_points": float(np.sqrt(np.square(delta).mean()) * 100.0),
        "bias_percentage_points": float(delta.mean() * 100.0),
        "binary_ice_area_abs_error_cells": float(
            np.abs(
                (prediction[:, ocean] >= 0.15).sum(axis=1)
                - (truth[:, ocean] >= 0.15).sum(axis=1)
            ).mean()
        ),
    }


def audit(base):
    design = load_json(base / "dataset_design.json")
    manifest = load_json(base / "download_manifest.json")
    encoder = load_json(base / "encoder_results.json")
    summary = load_json(base / "summary.json")

    assert len(manifest["files"]) == 180
    assert len({item["date"] for item in manifest["files"]}) == 180
    assert len([item for item in manifest["files"] if item["split"] == "training"]) == 120
    assert len([item for item in manifest["files"] if item["split"] == "validation"]) == 31
    assert len([item for item in manifest["files"] if item["split"] == "testing"]) == 29
    assert sha256(base / "download_manifest.json") == design["raw_manifest_sha256"]
    assert sha256(base / "observed_sic_paired.npz") == design["prepared_sha256"]
    assert sha256(base / "latent_sequences.npz") == encoder["latents_sha256"]
    assert sha256(base / "dataset_design.json") == encoder["data_design_sha256"]
    assert design["excluded_status_flag_bit_128"] is True
    assert design["years_disjoint"] is True

    with np.load(base / "observed_sic_paired.npz", allow_pickle=False) as source:
        train = source["train"]
        val = source["val"]
        test = source["test"]
        mask = source["mask"].astype(bool)
        assert len(train) == 120 and len(val) == 31 and len(test) == 29
        assert mask.shape == (64, 64) and int(mask.sum()) == 3584
        assert np.isfinite(train).all() and np.isfinite(test).all()
        assert np.all((train >= 0) & (train <= 1))
        assert np.all((test >= 0) & (test <= 1))
        assert source["train_dates"][-1] < source["val_dates"][0] < source["test_dates"][0]

    with np.load(base / "latent_sequences.npz", allow_pickle=False) as latents:
        assert latents["train"].shape == (120, 2, 32, 32)
        assert latents["val"].shape == (31, 2, 32, 32)
        assert latents["test"].shape == (29, 2, 32, 32)
        # Torch nearest-neighbour 64->32 takes source grid coordinates [0, 2, ...].
        active_latent = mask[::2, ::2]
        assert int(active_latent.sum()) == encoder["target_latent_mask_pixel_count"]
        targets = latents["train"][1:, :, active_latent]
        std = float(targets.std(dtype=np.float64))
        assert abs(std - encoder["training_target_latent_std"]) < 1e-10
        assert abs(std - summary["training_only_target_latent_std"]) < 1e-10

    expected_seeds = (477, 478, 479)
    expected_t = (16, 32, 64)
    assert summary["runs"] == 18
    assert summary["train_pairs"] == 119
    assert summary["validation_pairs"] == 30
    assert summary["test_pairs"] == 28
    assert summary["seeds"] == list(expected_seeds)
    assert summary["training_timesteps"] == list(expected_t)
    assert summary["diffusion_plateau_arms"] == 18
    assert summary["diffusion_budget_limited_arms"] == 0

    persistence = score(test[:-1], test[1:], mask)
    assert abs(persistence["mae_percentage_points"] - summary["persistence_mae_pp"]) < 1e-8
    assert abs(persistence["rmse_percentage_points"] - summary["persistence_rmse_pp"]) < 1e-8

    pair_groups = {}
    audited = 0
    original_driver = Path(__file__).with_name("run_diffusion.py")
    driver_sha = sha256(original_driver)
    processor = Path(__file__).parents[2] / "cryocast/models/processors/diffusion.py"
    processor_sha = sha256(processor) if processor.exists() else None
    for file in sorted((base / "training_results").glob("*/metrics.json")):
        run = load_json(file)
        key = (run["seed"], run["training_timesteps"])
        assert run["seed"] in expected_seeds and run["training_timesteps"] in expected_t
        assert run["x0_scale"] in (1.0, std)
        assert run["converged_by_plateau"]
        assert run["trained_updates"] <= 5000
        assert run["driver_sha256"] == driver_sha
        if processor_sha is not None:
            assert run["processor_sha256"] == processor_sha
        assert run["training_examples"] == 119
        assert run["validation_examples"] == 30
        assert run["testing_examples"] == 28
        pair_groups.setdefault(key, []).append(run)
        with np.load(file.parent / "test_predictions.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(run["samplers"])
            for n, metrics in run["samplers"].items():
                predicted = saved[n].astype(np.float32)
                assert predicted.shape == (28, 64, 64)
                assert np.isfinite(predicted).all()
                calc = score(predicted, test[1:], mask)
                for metric, value in calc.items():
                    assert np.isclose(value, metrics[metric], atol=5e-6), (
                        file,
                        n,
                        metric,
                        value,
                        metrics[metric],
                    )
                per_date = [
                    float(np.abs(p[mask] - truth[mask]).mean() * 100.0)
                    for p, truth in zip(predicted, test[1:], strict=True)
                ]
                np.testing.assert_allclose(
                    per_date, metrics["per_date_mae_percentage_points"], atol=5e-6
                )
        audited += 1

    assert audited == 18
    for seed in expected_seeds:
        for t in expected_t:
            pairs = pair_groups[(seed, t)]
            assert len(pairs) == 2
            a, b = pairs
            assert {a["x0_scale"], b["x0_scale"]} == {1.0, std}
            for field in (
                "initial_state_sha256",
                "encoder_sha256",
                "driver_sha256",
                "design_sha256",
                "processor_sha256",
            ):
                assert a[field] == b[field], (seed, t, field)
            assert a["encoder_sha256"] == encoder["encoder_final_sha256"]
            assert a["design_sha256"] == encoder["data_design_sha256"]

    results = {(r["seed"], r["training_timesteps"], r["x0_scale"]): r
               for group in pair_groups.values() for r in group}
    for pair in summary["paired_comparisons"]:
        t, ddim = pair["training_timesteps"], str(pair["sampling_steps"])
        raw = np.mean([
            results[(seed, t, 1.0)]["samplers"][ddim]["mae_percentage_points"]
            for seed in expected_seeds
        ])
        scaled = np.mean([
            results[(seed, t, std)]["samplers"][ddim]["mae_percentage_points"]
            for seed in expected_seeds
        ])
        assert np.isclose(raw, pair["raw_mae_pp"], atol=1e-8)
        assert np.isclose(scaled, pair["scaled_mae_pp"], atol=1e-8)
        assert np.isclose(scaled - raw, pair["scaled_minus_raw_mae_pp"], atol=1e-7)

    print("AUDIT PASS: 180 verified manifest entries, 119/30/28 date pairs")
    print("AUDIT PASS: 3584 training-only valid cells, latent std",std)
    print("AUDIT PASS: 18 paired runs, all original 28-day/sampler predictions")
    print("AUDIT PASS: all physical metrics, per-date scores, parameter hashes")
    print("AUDIT PASS: all 12 held-out summary comparisons and persistence")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(os.environ.get("CRYOCAST_X0_EVIDENCE_DIR", Path(__file__).parent)),
    )
    args = parser.parse_args()
    audit(args.dir)
