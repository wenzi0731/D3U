from __future__ import annotations

from pathlib import Path

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


def compute_global_pearson_matrix(total_data: np.ndarray) -> np.ndarray:
    """Compute channel Pearson correlations over all days and hours."""
    values = np.asarray(total_data, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got {values.shape}.")
    flattened = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    return np.corrcoef(flattened, rowvar=False)


def _save_pearson_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: str | Path,
    labels: list[str],
    cmap: str,
    vmin: float,
    vmax: float,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def save_global_pearson_plot(
    total_real: np.ndarray,
    save_path: str | Path,
    labels: list[str],
) -> Path:
    """Save the real-data global Pearson heatmap for compatibility."""
    correlation = compute_global_pearson_matrix(total_real)
    return _save_pearson_heatmap(
        correlation,
        "Global Pearson Correlation",
        save_path,
        labels,
        "coolwarm",
        -1.0,
        1.0,
    )


def save_global_pearson_comparison(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    output_dir: str | Path,
    labels: list[str],
) -> dict[str, float]:
    """Save real/generated Pearson matrices and their absolute error."""
    real = np.asarray(total_real, dtype=np.float64)
    generated = np.asarray(total_generated, dtype=np.float64)
    if real.ndim != 3 or generated.ndim != 4:
        raise ValueError(
            f"Expected real [N,C,T] and generated [N,S,C,T], got "
            f"{real.shape} and {generated.shape}."
        )
    if generated.shape[0] != real.shape[0] or generated.shape[2:] != real.shape[1:]:
        raise ValueError("Real and generated Pearson inputs are not aligned.")
    if len(labels) != real.shape[1]:
        raise ValueError("labels length does not match the channel count.")

    real_correlation = compute_global_pearson_matrix(real)
    generated_correlation = compute_global_pearson_matrix(
        generated.reshape(-1, generated.shape[2], generated.shape[3])
    )
    absolute_error = np.abs(generated_correlation - real_correlation)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        (
            real_correlation,
            "Real Global Pearson",
            "real_global_pearson.png",
            "coolwarm",
            -1.0,
            1.0,
        ),
        (
            generated_correlation,
            "Generated Global Pearson",
            "generated_global_pearson.png",
            "coolwarm",
            -1.0,
            1.0,
        ),
        (
            absolute_error,
            "Pearson Absolute Error",
            "pearson_difference.png",
            "Reds",
            0.0,
            1.0,
        ),
    )
    for matrix, title, filename, cmap, vmin, vmax in plot_specs:
        _save_pearson_heatmap(
            matrix,
            title,
            destination / filename,
            labels,
            cmap,
            vmin,
            vmax,
        )
    return {
        "Pearson_MAE": float(absolute_error.mean()),
        "Pearson_RMSE": float(np.sqrt(np.mean(absolute_error**2))),
    }


def save_random_timeseries_plots(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    dates: list[str],
    output_dir: str | Path,
    labels: list[str],
    seed: int = 42,
    n_plots: int = 50,
) -> list[Path]:
    """Save seeded random daily plots matching the 2-stages evaluator."""
    import matplotlib.pyplot as plt

    real = np.asarray(total_real)
    generated = np.asarray(total_generated)
    if real.ndim != 3 or generated.ndim != 4:
        raise ValueError(
            f"Expected real [N,C,T] and generated [N,S,C,T], got "
            f"{real.shape} and {generated.shape}."
        )
    if generated.shape[0] != real.shape[0] or generated.shape[2:] != real.shape[1:]:
        raise ValueError("Real and generated time-series inputs are not aligned.")
    if len(dates) != real.shape[0]:
        raise ValueError("dates length does not match the number of days.")
    if len(labels) != real.shape[1]:
        raise ValueError("labels length does not match the channel count.")
    if n_plots < 0:
        raise ValueError("n_plots must be non-negative.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(seed)
    selected = random_generator.choice(
        real.shape[0],
        size=min(n_plots, real.shape[0]),
        replace=False,
    )
    saved_paths: list[Path] = []
    for index in selected:
        figure, axes = plt.subplots(
            len(labels),
            1,
            figsize=(10, 3 * len(labels)),
            sharex=True,
        )
        if len(labels) == 1:
            axes = [axes]
        for channel, label in enumerate(labels):
            for scenario in range(min(100, generated.shape[1])):
                axes[channel].plot(
                    generated[index, scenario, channel],
                    color="red",
                    alpha=0.1,
                    linewidth=1,
                )
            axes[channel].plot(
                real[index, channel],
                color="black",
                linewidth=2,
                linestyle="--",
                label="Ground Truth",
            )
            axes[channel].set_title(label)
            axes[channel].grid(True, alpha=0.3)
            axes[channel].legend()
        date_text = dates[index]
        figure.suptitle(f"Generated scenarios - {date_text}")
        figure.tight_layout()
        save_path = destination / f"{date_text}.png"
        figure.savefig(save_path, dpi=200)
        plt.close(figure)
        saved_paths.append(save_path)
    return saved_paths
