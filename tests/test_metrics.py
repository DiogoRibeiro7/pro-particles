from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pro_particles.metrics.metrics import (
    elpd_from_lppd,
    lppd_from_logpdf,
    mean_and_se,
    mmd2_empirical,
    mmd2_to_dirac,
    negative_log_likelihood,
)


def test_lppd_elpd() -> None:
    logpdf = np.zeros((5, 3), dtype=np.float64)
    lppd = lppd_from_logpdf(logpdf)
    assert np.allclose(lppd, 0.0)
    assert elpd_from_lppd(lppd) == 0.0


def test_negative_log_likelihood() -> None:
    logpdf = np.array([0.0, -1.0], dtype=np.float64)
    assert negative_log_likelihood(logpdf) == 1.0


def test_mmd2_constant_kernel() -> None:
    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.ones((a.shape[0], b.shape[0]), dtype=np.float64)

    x = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([[2.0], [3.0]], dtype=np.float64)
    assert mmd2_empirical(x=x, y=y, kernel=k) == 0.0
    assert mmd2_to_dirac(samples=x, x=np.array([0.0]), kernel=k) == 0.0


def test_mean_and_se() -> None:
    deltas = np.array([1.0, 3.0], dtype=np.float64)
    mean, se = mean_and_se(deltas)
    assert mean == 2.0
    assert np.isclose(se, 1.0)
