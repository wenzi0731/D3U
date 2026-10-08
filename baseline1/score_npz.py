"""Rescore existing physical D3U scenarios; no training or sampling."""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np

from baseline1.data import HEEWConditionDataset, TARGET_COLUMNS
from baseline1.reporting import require_empty_output, write_report
from baseline1.runtime import load_checkpoint, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--outdir", "--output-dir", required=True)
    parser.add_argument("--checkpoint", help="Required only for legacy archives without training normalization")
    parser.add_argument("--energy-path")
    parser.add_argument("--weather-path")
    parser.add_argument("--seed", type=int, help="Metric subsampling/plot seed, not a new sampling seed")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    source = resolve_path(args.npz)
    destination = require_empty_output(resolve_path(args.outdir))
    with np.load(source, allow_pickle=False) as payload:
        if not {"scenarios", "targets", "dates"}.issubset(payload.files):
            raise ValueError("Archive requires scenarios, targets and dates")
        samples, targets = payload["scenarios"], payload["targets"]
        dates = payload["dates"].astype(str).tolist()
        labels = payload["channel_names"].astype(str).tolist() if "channel_names" in payload.files else list(TARGET_COLUMNS)
        if labels != list(TARGET_COLUMNS):
            raise ValueError("Expected Electricity, Heat, Cooling, PV channel order")
        archive_seed = int(payload["seed"].item()) if "seed" in payload.files else None
        metadata = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload.files else {}
        if "sampler" in payload.files:
            metadata["sampler"] = str(payload["sampler"].item())
        if {"target_mean", "target_std"}.issubset(payload.files):
            mean, std = payload["target_mean"], payload["target_std"]
            normalization_source = "training statistics stored in source NPZ"
        elif "target_mean" in payload.files or "target_std" in payload.files:
            raise ValueError("Archive must contain both target_mean and target_std")
        else:
            mean = std = None

    if mean is None:
        if not args.checkpoint:
            parser.error("Legacy NPZ lacks training statistics: supply --checkpoint and the original CSV files. Never normalize with test data.")
        checkpoint = load_checkpoint(args.checkpoint)
        data = checkpoint["config"]["data"]
        dataset = HEEWConditionDataset(
            resolve_path(args.energy_path or data["energy_path"]),
            resolve_path(args.weather_path or data["weather_path"]),
            split="test", weather_feature_set=data["weather_feature_set"],
        )
        positions = {date: i for i, date in enumerate(dataset.dates)}
        if any(date not in positions for date in dates):
            raise ValueError("Saved dates do not match the supplied test dataset")
        matching = [positions[date] for date in dates]
        expected = dataset.denormalize(dataset.targets[matching]).numpy()
        if targets.shape != expected.shape or not np.allclose(targets, expected, rtol=1e-5, atol=1e-5):
            raise ValueError("Saved targets differ from supplied data; refusing to infer normalization")
        mean, std = dataset.target_mean, dataset.target_std
        normalization_source = "recomputed on 2014-2020 from supplied original CSVs; legacy checkpoint has no data hash"
        metadata.update(checkpoint=str(resolve_path(args.checkpoint)),
                        checkpoint_epoch=checkpoint.get("epoch"),
                        partial_test=len(dates) != len(dataset))

    seed = args.seed if args.seed is not None else archive_seed if archive_seed is not None else 42
    metadata.update(
        method="D3U-condition-only-baseline1", split="test-2022",
        source_npz=str(source), archive=str(source), rescored_without_sampling=True,
        sample_seed=archive_seed, normalization_source=normalization_source,
    )
    result = write_report(
        samples, targets, mean, std, dates, labels, destination,
        seed=seed, no_plots=args.no_plots, metadata=metadata,
        precision_recall_k=int(metadata.get("precision_recall_k", 5)),
        max_precision_samples=int(metadata.get("max_precision_samples", 10_000)),
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
