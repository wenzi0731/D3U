from __future__ import annotations

import numpy as np


def crps_map(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pointwise ensemble CRPS for samples [N,S,C,T] and target [N,C,T]."""
    values = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if values.ndim != 4 or truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("Expected samples [N,S,C,T] and target [N,C,T].")
    term1 = np.mean(np.abs(values - truth[:, None]), axis=1)
    ordered = np.sort(values, axis=1)
    ensemble_size = values.shape[1]
    coefficients = (2 * np.arange(ensemble_size) - ensemble_size + 1).reshape(
        1, ensemble_size, 1, 1
    )
    mean_pairwise = 2.0 * np.sum(coefficients * ordered, axis=1) / ensemble_size**2
    return term1 - 0.5 * mean_pairwise


def summarize(samples: np.ndarray, target: np.ndarray, labels: list[str]) -> dict[str, float]:
    ensemble_mean = samples.mean(axis=1)
    error = ensemble_mean - target
    scores = crps_map(samples, target)
    result: dict[str, float] = {}
    for channel, label in enumerate(labels):
        channel_target = target[:, channel]
        channel_crps = scores[:, channel]
        lower = np.quantile(samples[:, :, channel], 0.05, axis=1)
        upper = np.quantile(samples[:, :, channel], 0.95, axis=1)
        result[f"{label}_RMSE"] = float(np.sqrt(np.mean(error[:, channel] ** 2)))
        result[f"{label}_MAE"] = float(np.mean(np.abs(error[:, channel])))
        result[f"{label}_CRPS"] = float(np.mean(channel_crps))
        result[f"{label}_nCRPS"] = float(
            np.sum(channel_crps) / (np.sum(np.abs(channel_target)) + 1e-8)
        )
        result[f"{label}_Coverage90"] = float(
            np.mean((channel_target >= lower) & (channel_target <= upper))
        )
        result[f"{label}_IntervalWidth90"] = float(np.mean(upper - lower))
    result["mean_nCRPS"] = float(
        np.sum(scores) / (np.sum(np.abs(target)) + 1e-8)
    )
    return result
