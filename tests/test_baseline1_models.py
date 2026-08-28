import torch

from baseline1.models import build_condition_model, build_diffusion
from model9_NS_transformer.samplers.dpm_sampler import DPMSolverSampler


def tiny_config():
    return {
        "data": {"seq_len": 24},
        "condition_model": {
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "use_pv_year_head": True,
            "pv_year_film_scale": 0.2,
        },
        "diffusion_model": {
            "hidden_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "patch_size": 4,
            "stride": 2,
            "padding_patch": "end",
        },
        "diffusion": {
            "timesteps": 4,
            "sampling_timesteps": 2,
            "parameterization": "x_start",
        },
    }


def test_condition_and_d3u_diffusion_shapes():
    config = tiny_config()
    condition_model = build_condition_model(config, condition_channels=18).eval()
    conditions = torch.randn(2, 18, 24)
    pv_year = torch.zeros(2, 1, 24)
    deterministic = condition_model(conditions, pv_year)
    assert deterministic.shape == (2, 24, 4)

    diffusion = build_diffusion(config).eval()
    residual = torch.randn(2, 24, 4)
    timestep = torch.tensor([0, 1])
    noise = torch.randn_like(residual)
    noisy = diffusion.q_sample(residual, timestep, noise)
    predicted = diffusion(noisy, timestep, deterministic)
    assert predicted.shape == residual.shape
    with torch.no_grad():
        sampled = diffusion.p_sample_loop(
            deterministic, deterministic, residual.shape
        )
    assert sampled.shape == residual.shape
    assert torch.isfinite(sampled).all()

    solver = DPMSolverSampler(diffusion, torch.device("cpu"), "x_start")
    with torch.no_grad():
        solved = solver.sample(
            S=2,
            conditioning=deterministic,
            shape=residual.shape,
            verbose=False,
        )
    assert solved.shape == residual.shape
    assert torch.isfinite(solved).all()
