from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from baseline1.data import HEEWConditionDataset
from baseline1.models import build_condition_model, build_diffusion, checkpoint_payload
from baseline1.runtime import (
    choose_device,
    load_config,
    resolve_path,
    seed_everything,
    seed_worker,
)


def make_loaders(config: dict, device: torch.device):
    data = config["data"]
    common = {
        "energy_path": resolve_path(data["energy_path"]),
        "weather_path": resolve_path(data["weather_path"]),
        "weather_feature_set": data["weather_feature_set"],
    }
    train_set = HEEWConditionDataset(split="train", **common)
    val_set = HEEWConditionDataset(split="val", **common)
    generator = torch.Generator().manual_seed(int(config["run"]["seed"]))
    kwargs = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": int(config["run"].get("num_workers", 0)),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "pin_memory": device.type == "cuda",
    }
    return (
        train_set,
        val_set,
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
    )


def condition_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for conditions, pv_year, target, _ in loader:
        conditions = conditions.to(device)
        pv_year = pv_year.to(device)
        target = target.transpose(1, 2).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(conditions, pv_year)
        loss = nn.functional.mse_loss(prediction, target)
        if training:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * target.shape[0]
        count += target.shape[0]
    return total / max(count, 1)


def diffusion_epoch(
    diffusion,
    condition_model,
    loader,
    device,
    optimizer=None,
    gradient_clip: float = 1.0,
) -> float:
    training = optimizer is not None
    diffusion.train(training)
    condition_model.eval()
    total = 0.0
    count = 0
    for conditions, pv_year, target, _ in loader:
        conditions = conditions.to(device)
        pv_year = pv_year.to(device)
        target = target.transpose(1, 2).to(device)
        with torch.no_grad():
            deterministic = condition_model(conditions, pv_year)
        residual = target - deterministic
        batch_size = residual.shape[0]
        timesteps = torch.randint(
            0, diffusion.num_timesteps, (batch_size,), device=device
        )
        noise = torch.randn_like(residual)
        noisy = diffusion.q_sample(residual, timesteps, noise=noise)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = diffusion(noisy, timesteps, deterministic)
        parameterization = config_parameterization(diffusion)
        objective = noise if parameterization == "noise" else residual
        loss = nn.functional.mse_loss(prediction, objective)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), gradient_clip)
            optimizer.step()
        total += float(loss.detach()) * batch_size
        count += batch_size
    return total / max(count, 1)


def config_parameterization(diffusion) -> str:
    parameterization = getattr(diffusion.args, "parameterization", "x_start")
    if parameterization not in {"noise", "x_start"}:
        raise ValueError("diffusion.parameterization must be noise or x_start")
    return parameterization


def train_condition(config, model, train_loader, val_loader, device, run_dir):
    settings = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["condition_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(1, int(settings["condition_epochs"]) + 1):
        train_loss = condition_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = condition_epoch(model, val_loader, device)
        print(
            f"condition epoch={epoch:04d} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(
                {"condition_model": best_state, "config": config, "epoch": epoch},
                run_dir / "best_condition.pt",
            )
        else:
            bad_epochs += 1
        if bad_epochs >= int(settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Condition training produced no checkpoint.")
    model.load_state_dict(best_state, strict=True)
    return best_loss


def train_diffusion(
    config,
    condition_model,
    diffusion,
    train_loader,
    val_loader,
    device,
    run_dir,
):
    for parameter in condition_model.parameters():
        parameter.requires_grad_(False)
    settings = config["train"]
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=float(settings["diffusion_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(1, int(settings["diffusion_epochs"]) + 1):
        train_loss = diffusion_epoch(
            diffusion,
            condition_model,
            train_loader,
            device,
            optimizer,
            float(settings["gradient_clip"]),
        )
        # Use a deterministic validation-noise stream for every epoch.
        torch.manual_seed(int(config["run"]["seed"]) + epoch)
        with torch.no_grad():
            val_loss = diffusion_epoch(
                diffusion, condition_model, val_loader, device
            )
        metrics = {"train/diffusion_mse": train_loss, "val/diffusion_mse": val_loss}
        print(
            f"diffusion epoch={epoch:04d} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            torch.save(
                checkpoint_payload(
                    config, condition_model, diffusion, epoch, metrics
                ),
                run_dir / "best_baseline1.pt",
            )
            (run_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
        else:
            bad_epochs += 1
        if bad_epochs >= int(settings["patience"]):
            break
    return best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train D3U condition-only Baseline 1.")
    parser.add_argument("--config", default="baseline1/configs/heew.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--condition-checkpoint", default=None)
    parser.add_argument(
        "--stage", choices=["all", "condition", "diffusion"], default="all"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = choose_device(config["run"].get("device", "auto"))
    train_set, _, train_loader, val_loader = make_loaders(config, device)
    run_dir = resolve_path(config["run"]["output_root"]) / (
        f"{config['run']['name']}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    condition_model = build_condition_model(
        config, train_set.condition_channels
    ).to(device)
    if args.condition_checkpoint:
        condition_checkpoint = torch.load(
            resolve_path(args.condition_checkpoint), map_location="cpu"
        )
        condition_model.load_state_dict(
            condition_checkpoint["condition_model"], strict=True
        )
    elif args.stage in {"all", "condition"}:
        train_condition(
            config, condition_model, train_loader, val_loader, device, run_dir
        )
    elif (run_dir / "best_condition.pt").exists():
        condition_checkpoint = torch.load(
            run_dir / "best_condition.pt", map_location="cpu"
        )
        condition_model.load_state_dict(
            condition_checkpoint["condition_model"], strict=True
        )
    else:
        raise FileNotFoundError(
            "Diffusion-only training requires --condition-checkpoint or "
            f"{run_dir / 'best_condition.pt'}."
        )

    if args.stage in {"all", "diffusion"}:
        diffusion = build_diffusion(config).to(device)
        train_diffusion(
            config,
            condition_model,
            diffusion,
            train_loader,
            val_loader,
            device,
            run_dir,
        )
    print(json.dumps({"run_dir": str(run_dir), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
