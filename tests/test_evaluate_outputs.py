"""Exercise real tiny-model checkpoints and preserve the legacy sampling RNG."""
import json
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from baseline1 import evaluate, score_npz
from baseline1.data import HEEWConditionDataset
from baseline1.models import build_condition_model, build_diffusion, checkpoint_payload
from baseline1.runtime import seed_everything
from model9_NS_transformer.samplers.dpm_sampler import DPMSolverSampler
from test_baseline1_data import make_csvs
from test_baseline1_models import tiny_config


@pytest.mark.parametrize("sampler", ["dpm_solver", "ddpm", "ddim"])
def test_evaluate_archive_and_exact_legacy_samples(tmp_path, monkeypatch, sampler):
    energy, weather = make_csvs(tmp_path)
    config = tiny_config()
    config.update(run={"seed": 42}, eval={"batch_size": 8, "num_scenarios": 3})
    config["data"].update(energy_path=str(energy), weather_path=str(weather), weather_feature_set="pv10")
    config["diffusion"].update(sampler=sampler, dpm_solver_steps=2,
                               parameterization="x_start" if sampler == "dpm_solver" else "eps")
    dataset = HEEWConditionDataset(energy, weather, "test")
    seed_everything(42)
    condition = build_condition_model(config, dataset.condition_channels).eval()
    diffusion = build_diffusion(config).eval()
    checkpoint = tmp_path / "tiny.pt"
    torch.save(checkpoint_payload(config, condition, diffusion, 3, {}), checkpoint)
    solver = DPMSolverSampler(diffusion, torch.device("cpu"), "x_start") if sampler == "dpm_solver" else None

    # Reference: original evaluate.py order, including the DataLoader RNG draw.
    seed_everything(42)
    expected = []
    with torch.no_grad():
        for conditions, year, target, dates in DataLoader(dataset, batch_size=8, shuffle=False):
            point = condition(conditions, year)
            tiled = point[:, None].expand(len(point), 3, 24, 4).reshape(-1, 24, 4)
            if sampler == "dpm_solver":
                residual = solver.sample(S=2, conditioning=tiled, shape=tiled.shape, verbose=False)
            elif sampler == "ddpm":
                residual = diffusion.p_sample_loop(tiled, tiled, tiled.shape)
            else:
                residual = diffusion.fast_sample(tiled, tiled, tiled.shape, 0.)
            generated = (residual + tiled).reshape(len(point), 3, 24, 4).permute(0, 1, 3, 2)
            physical = dataset.denormalize(generated)
            physical[:, :, 3].clamp_(min=0)
            expected.append(physical.numpy())

    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint),
                                     "--outdir", str(out), "--device", "cpu", "--no-plots"])
    evaluate.main()
    with np.load(out / "baseline1_scenarios.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["scenarios"], np.concatenate(expected))
        np.testing.assert_array_equal(saved["target_mean"], dataset.target_mean)
        np.testing.assert_array_equal(saved["target_std"], dataset.target_std)
        assert saved["conditions"].shape == (1, 18, 24)
        assert saved["deterministic"].shape == (1, 4, 24)
    result = json.loads((out / "global_metrics.json").read_text())
    assert result["training_timesteps"] == 4
    assert result["sampling_steps"] == (4 if sampler == "ddpm" else 2)
    assert result["sample_seed"] == 42 and result["VS_pair_count"] == 3456
    assert result["checkpoint_epoch"] == 3
    rescored = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(out / "baseline1_scenarios.npz"),
                                     "--outdir", str(rescored), "--no-plots"])
    score_npz.main()
    offline = json.loads((rescored / "global_metrics.json").read_text())
    for key in ("ES", "VS", "macro_nCRPS", "PV_R2", "PV_IS", "PV_RMSE_Z"):
        assert offline[key] == result[key]
