import matplotlib
import numpy as np

from baseline1.metrics import (
    compute_global_pearson_matrix,
    crps_map,
    save_global_pearson_comparison,
    save_global_pearson_plot,
    save_random_timeseries_plots,
    summarize,
)


matplotlib.use("Agg")


def test_crps_and_interval_metrics_are_exact_for_constant_ensemble():
    target = np.ones((2, 4, 24), dtype=np.float32)
    samples = np.ones((2, 5, 4, 24), dtype=np.float32)

    np.testing.assert_allclose(crps_map(samples, target), 0.0)
    metrics = summarize(samples, target, ["E", "H", "C", "PV"])
    assert metrics["mean_nCRPS"] == 0.0
    assert metrics["E_Coverage90"] == 1.0
    assert metrics["E_IntervalWidth90"] == 0.0


def test_pearson_and_random_timeseries_outputs(tmp_path):
    rng = np.random.default_rng(42)
    real = rng.normal(size=(3, 4, 24))
    generated = np.repeat(real[:, None], repeats=5, axis=1)
    labels = ["Electricity", "Heat", "Cooling", "PV"]

    correlation = compute_global_pearson_matrix(real)
    assert correlation.shape == (4, 4)
    np.testing.assert_allclose(np.diag(correlation), 1.0)

    pearson_dir = tmp_path / "pearson"
    pearson_metrics = save_global_pearson_comparison(
        real,
        generated,
        pearson_dir,
        labels,
    )
    assert pearson_metrics["Pearson_MAE"] < 1e-12
    assert pearson_metrics["Pearson_RMSE"] < 1e-12
    assert {path.name for path in pearson_dir.glob("*.png")} == {
        "real_global_pearson.png",
        "generated_global_pearson.png",
        "pearson_difference.png",
    }

    global_path = save_global_pearson_plot(
        real,
        tmp_path / "global_pearson.png",
        labels,
    )
    assert global_path.exists()

    random_paths = save_random_timeseries_plots(
        real,
        generated,
        ["2022-01-01", "2022-01-02", "2022-01-03"],
        tmp_path / "random_timeseries_50",
        labels,
        seed=42,
        n_plots=2,
    )
    assert len(random_paths) == 2
    assert all(path.exists() for path in random_paths)
