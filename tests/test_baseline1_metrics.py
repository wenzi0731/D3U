import numpy as np

from baseline1.metrics import crps_map, summarize


def test_crps_and_interval_metrics_are_exact_for_constant_ensemble():
    target = np.ones((2, 4, 24), dtype=np.float32)
    samples = np.ones((2, 5, 4, 24), dtype=np.float32)

    np.testing.assert_allclose(crps_map(samples, target), 0.0)
    metrics = summarize(samples, target, ["E", "H", "C", "PV"])
    assert metrics["mean_nCRPS"] == 0.0
    assert metrics["E_Coverage90"] == 1.0
    assert metrics["E_IntervalWidth90"] == 0.0
