"""Matched one-day sea-ice forecast pilot for IceNet PR 478.

Run from the PR checkout with its development environment:
    python run_comparison.py --input input_slice.npz --design design.json --output results
This is a small empirical pilot, not an operational forecast qualification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from icenet_mp.models.processors import DiffusionProcessor
from icenet_mp.types import DataSpace


class MaskedMSE(nn.Module):
    """Use a fixed training-derived ocean mask for velocity loss."""

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask[None, None, None].float())

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.mask.expand_as(prediction)
        return ((prediction - target).square() * weights).sum() / weights.sum()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes and values, independently of torch serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    difference = prediction[..., mask] - truth[..., mask]
    return {
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "bias": float(difference.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="6b2764b241f630d6146ab90c9877c7706aa9fa99")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design: dict[str, Any] = json.loads(args.design.read_text())
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == design["input_sha256"]
    with np.load(args.input, allow_pickle=False) as data:
        sic = data["sic"].copy()
        mask = data["mask"].copy()
        dates = data["dates"].copy()
    assert sic.shape == (10, 64, 64) and mask.dtype == np.bool_
    mean = float(sic[:6, mask].mean())
    std = float(sic[:6, mask].std())
    assert np.isclose(mean, design["training_observation_mean"])
    assert np.isclose(std, design["training_observation_std"])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    norm = np.where(mask[None], (sic - mean) / std, 0).astype(np.float32)
    # First channel is the target representation; second is the fixed ocean mask.
    combined = np.stack([norm, np.broadcast_to(mask, norm.shape)], axis=1).astype(np.float32)
    x = torch.from_numpy(combined[:, None])
    y = torch.from_numpy(norm[:, None, None])
    mask_tensor = torch.from_numpy(mask)
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    baseline_source = subprocess.check_output([
        "git", "show", args.baseline_ref + ":icenet_mp/models/processors/diffusion.py"
    ], cwd=repo)
    baseline_file = args.output / "baseline_diffusion.py"
    baseline_file.write_bytes(baseline_source)
    spec = importlib.util.spec_from_file_location(
        "icenet_mp.models.processors._x0scale_reference", baseline_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = {
        "data_space": DataSpace(name="history_and_mask", channels=2, shape=(64, 64)),
        "data_space_target": DataSpace(name="sea_ice", channels=1, shape=(64, 64)),
        "n_history_steps": 1, "n_forecast_steps": 1, "target_channel_offset": 0,
        "timesteps": 32, "ddim_steps": 32, "eta": 0.0,
        "start_out_channels": 8, "time_embed_dim": 256,
        "dropout_rate": 0.0, "use_autoregressive": True,
        "loss": MaskedMSE(mask_tensor),
    }
    result: dict[str, Any] = {"design": design, "source_head": head,
        "baseline_ref": args.baseline_ref, "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu", "runs": []}
    predictions: dict[str, np.ndarray] = {}
    total_start = time.monotonic()
    for seed in design["seeds"]:
        torch.manual_seed(seed)
        template = DiffusionProcessor(x0_scale=1.0, **copy.deepcopy(kwargs))
        initial = copy.deepcopy(template.state_dict())
        initial_hash = state_hash(initial)
        torch.save(initial, args.output / f"initial-{seed}.pt")
        # Assert the unscaled treatment is the unchanged pre-PR implementation.
        reference = module.DiffusionProcessor(**copy.deepcopy(kwargs))
        reference.load_state_dict(initial)
        torch.manual_seed(seed * 10000)
        before = reference._rollout_training(x[:5], y[1:6])
        torch.manual_seed(seed * 10000)
        after = template._rollout_training(x[:5], y[1:6])
        torch.testing.assert_close(before.loss, after.loss, rtol=0, atol=0)
        torch.testing.assert_close(before.prediction, after.prediction, rtol=0, atol=0)
        for scale in design["scales"]:
            model = DiffusionProcessor(x0_scale=scale, **copy.deepcopy(kwargs))
            model.load_state_dict(initial)
            assert state_hash(model.state_dict()) == initial_hash
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            losses = []
            start = time.monotonic()
            model.train()
            for step in range(design["training_steps"]):
                # Same timestep draws and Gaussian noise for both scale arms.
                torch.manual_seed(seed * 100000 + step)
                optimizer.zero_grad(set_to_none=True)
                output = model.rollout(x[:5], y[1:6])
                loss = output.loss
                assert loss is not None and torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 30 == 0:
                    print(json.dumps({"seed": seed, "scale": scale, "step": step+1,
                                      "loss": losses[-1]}), flush=True)
            model.eval()
            checkpoint = f"seed-{seed}-scale-{scale:g}.pt"
            torch.save(model.state_dict(), args.output / checkpoint)
            forecasts = []
            with torch.no_grad():
                for target_index in [6, 7, 8, 9]:
                    members = []
                    for member in range(design["sampling"]["ensemble_members"]):
                        torch.manual_seed(seed * 100000 + target_index * 100 + member)
                        pred = model.rollout(x[target_index-1:target_index]).prediction
                        members.append((pred[0, 0, 0].numpy() * std + mean).copy())
                    forecasts.append(np.stack(members).mean(axis=0))
                if scale == 1.0:
                    reference.load_state_dict(model.state_dict())
                    reference.eval()
                    torch.manual_seed(seed * 10000 + 77)
                    a = reference.rollout(x[6:7]).prediction
                    torch.manual_seed(seed * 10000 + 77)
                    b = model.rollout(x[6:7]).prediction
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            forecasts_np = np.stack(forecasts)
            assert np.isfinite(forecasts_np).all()
            predictions[f"seed_{seed}_scale_{scale:g}"] = forecasts_np
            raw = metrics(forecasts_np[1:], sic[7:10], mask)
            bounded = metrics(np.clip(forecasts_np[1:], 0, 1), sic[7:10], mask)
            per_date = [{"date": str(dates[i]), **metrics(forecasts_np[k], sic[i], mask)}
                        for k, i in enumerate([6, 7, 8, 9])]
            run = {"seed": seed, "scale": scale, "initial_state_sha256": initial_hash,
                   "final_state_sha256": state_hash(model.state_dict()),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "training_steps": len(losses), "training_losses": losses,
                   "seconds": time.monotonic()-start, "test_raw": raw,
                   "test_clipped": bounded, "per_date": per_date,
                   "default_equals_base_training_and_sampling": scale == 1.0}
            result["runs"].append(run)
            (args.output / "results.json").write_text(json.dumps(result, indent=2))
            np.savez_compressed(args.output / "predictions.npz", **predictions)
            print("RESULT", json.dumps({k:v for k,v in run.items() if k!='training_losses'}), flush=True)
    result["persistence"] = metrics(sic[6:9], sic[7:10], mask)
    result["training_mean_field"] = metrics(np.broadcast_to(sic[:6].mean(axis=0),sic[7:10].shape),sic[7:10],mask)
    result["total_seconds"] = time.monotonic()-total_start
    result["complete"] = True
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print("COMPLETE", result["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
