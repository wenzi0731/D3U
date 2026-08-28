from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baseline1.data import HEEWConditionDataset
from baseline1.metrics import summarize
from baseline1.models import build_condition_model, build_diffusion
from baseline1.runtime import (
    choose_device,
    load_checkpoint,
    resolve_path,
    seed_everything,
)
from model9_NS_transformer.samplers.dpm_sampler import DPMSolverSampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D3U Baseline 1.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--energy-path", default=None)
    parser.add_argument("--weather-path", default=None)
    parser.add_argument("--output-dir", default="evaluation_results/baseline1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-scenarios", type=int, default=None)
    parser.add_argument(
        "--sampler", choices=["dpm_solver", "ddpm", "ddim"], default=None
    )
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    data_cfg = config["data"]
    dataset = HEEWConditionDataset(
        energy_path=resolve_path(args.energy_path or data_cfg["energy_path"]),
        weather_path=resolve_path(args.weather_path or data_cfg["weather_path"]),
        split="test",
        weather_feature_set=data_cfg["weather_feature_set"],
    )
    evaluation_set = dataset
    if args.max_days is not None:
        if args.max_days <= 0:
            parser.error("--max-days must be positive")
        evaluation_set = Subset(dataset, range(min(args.max_days, len(dataset))))
    loader = DataLoader(
        evaluation_set,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    condition_model = build_condition_model(
        config, dataset.condition_channels
    ).to(device)
    condition_model.load_state_dict(checkpoint["condition_model"], strict=True)
    diffusion = build_diffusion(config).to(device)
    diffusion.load_state_dict(checkpoint["diffusion_model"], strict=True)
    condition_model.eval()
    diffusion.eval()

    scenario_count = int(
        args.num_scenarios or config["eval"]["num_scenarios"]
    )
    if scenario_count <= 0:
        parser.error("--num-scenarios must be positive")
    sampler = args.sampler or config["diffusion"].get("sampler", "ddpm")
    parameterization = config["diffusion"].get("parameterization", "x_start")
    if parameterization == "x_start" and sampler != "dpm_solver":
        parser.error(
            "x_start checkpoints require --sampler dpm_solver; DDPM/DDIM in the "
            "legacy implementation expect noise parameterization"
        )
    dpm_sampler = None
    if sampler == "dpm_solver":
        dpm_sampler = DPMSolverSampler(
            diffusion,
            device,
            parameterization,
        )
    all_scenarios = []
    all_targets = []
    all_points = []
    all_dates: list[str] = []
    seed_everything(seed)  # Reset immediately before probabilistic sampling.
    with torch.no_grad():
        for conditions, pv_year, target, dates in loader:
            conditions = conditions.to(device)
            pv_year = pv_year.to(device)
            target = target.to(device)
            point = condition_model(conditions, pv_year)  # [B,T,C]
            batch_size = point.shape[0]
            tiled_point = point[:, None].expand(
                batch_size, scenario_count, 24, 4
            ).reshape(batch_size * scenario_count, 24, 4)
            shape = tiled_point.shape
            if sampler == "dpm_solver":
                residual = dpm_sampler.sample(
                    S=int(config["diffusion"].get("dpm_solver_steps", 20)),
                    conditioning=tiled_point,
                    shape=shape,
                    verbose=False,
                )
            elif sampler == "ddpm":
                residual = diffusion.p_sample_loop(
                    tiled_point, tiled_point, shape
                )
            else:
                residual = diffusion.fast_sample(
                    tiled_point,
                    tiled_point,
                    shape,
                    float(config["diffusion"].get("eta", 0.0)),
                )
            generated = (residual + tiled_point).reshape(
                batch_size, scenario_count, 24, 4
            ).permute(0, 1, 3, 2)
            point_channels_first = point.permute(0, 2, 1)
            scenarios_phys = dataset.denormalize(generated).cpu()
            targets_phys = dataset.denormalize(target).cpu()
            points_phys = dataset.denormalize(point_channels_first).cpu()
            scenarios_phys[:, :, 3, :].clamp_(min=0.0)
            points_phys[:, 3, :].clamp_(min=0.0)
            all_scenarios.append(scenarios_phys)
            all_targets.append(targets_phys.cpu())
            all_points.append(points_phys)
            all_dates.extend(str(value) for value in dates)

    scenarios = torch.cat(all_scenarios).numpy()
    targets = torch.cat(all_targets).numpy()
    points = torch.cat(all_points).numpy()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "baseline1_scenarios.npz"
    np.savez_compressed(
        archive,
        scenarios=scenarios,
        targets=targets,
        deterministic=points,
        dates=np.asarray(all_dates),
        channel_names=np.asarray(dataset.target_cols),
        seed=seed,
        sampler=sampler,
        num_scenarios=scenario_count,
    )
    metrics = summarize(scenarios, targets, dataset.target_cols)
    summary = {
        "method": "D3U-condition-only-baseline1",
        "checkpoint": str(resolve_path(args.checkpoint)),
        "split": "test-2022",
        "num_days": len(all_dates),
        "num_scenarios": scenario_count,
        "sampler": sampler,
        "seed": seed,
        "archive": str(archive),
        **metrics,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
