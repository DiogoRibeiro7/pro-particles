from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def lppd_from_logpdf(logpdf: ArrayF) -> ArrayF:
    """Log point-wise predictive density (D.4, p. 67).

    logpdf: shape (S, n) where S is number of posterior samples.
    Returns: shape (n,)
    """
    if logpdf.ndim != 2:
        raise ValueError("logpdf must be 2D (S, n).")
    s = logpdf.shape[0]
    if s <= 0:
        raise ValueError("logpdf must have S > 0.")
    m = np.max(logpdf, axis=0)
    lse = m + np.log(np.mean(np.exp(logpdf - m), axis=0))
    return lse


def elpd_from_lppd(lppd: ArrayF) -> float:
    """Expected log predictive density (D.4, p. 67)."""
    if lppd.ndim != 1:
        raise ValueError("lppd must be 1D.")
    return float(np.sum(lppd))


def negative_log_likelihood(logpdf: ArrayF) -> float:
    """Negative log likelihood (NLL) from logpdf values."""
    if logpdf.ndim != 1:
        raise ValueError("logpdf must be 1D.")
    return float(-np.sum(logpdf))


def mmd2_to_dirac(
    *,
    samples: ArrayF,  # (m, d)
    x: ArrayF,        # (d,)
    kernel: Callable[[ArrayF, ArrayF], ArrayF],
) -> float:
    """MMD^2(P, δ_x) from definition in Section 1.1 (p. 5)."""
    if samples.ndim != 2:
        raise ValueError("samples must be 2D.")
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if samples.shape[1] != x.shape[0]:
        raise ValueError("dimension mismatch.")
    if samples.shape[0] < 2:
        raise ValueError("need at least 2 samples.")

    k_xx = kernel(x[None, :], x[None, :])[0, 0]
    k_XX = kernel(samples, samples)
    term1 = k_XX.mean()
    term2 = kernel(samples, x[None, :]).mean()
    return float(term1 + k_xx - 2.0 * term2)


def mmd2_empirical(
    *,
    x: ArrayF,  # (m, d)
    y: ArrayF,  # (n, d)
    kernel: Callable[[ArrayF, ArrayF], ArrayF],
) -> float:
    """Empirical MMD^2 between two samples."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("dimension mismatch.")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("need at least 2 samples in each set.")

    k_xx = kernel(x, x).mean()
    k_yy = kernel(y, y).mean()
    k_xy = kernel(x, y).mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def mean_and_se(deltas: ArrayF) -> Tuple[float, float]:
    """Mean and standard error for point-wise differences (D.4, p. 67)."""
    if deltas.ndim != 1:
        raise ValueError("deltas must be 1D.")
    n = deltas.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 deltas for standard error.")
    mean = float(np.mean(deltas))
    se = float(np.sqrt(np.sum((deltas - mean) ** 2) / (n * (n - 1))))
    return mean, se


def crps_ensemble(
    *,
    y_true: ArrayF,   # (n,)
    samples: ArrayF,  # (S, n)
) -> ArrayF:
    """CRPS via properscoring (D.4, p. 68).

    Uses properscoring.crps_ensemble if available.

    Parameters
    ----------
    y_true:
        Ground-truth values, shape (n,).
    samples:
        Ensemble samples, shape (S, n).

    Returns
    -------
    ArrayF
        CRPS values per observation, shape (n,).

    References
    ----------
    docs/scoring_rules.md (p. 68).
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D.")
    if samples.ndim != 2:
        raise ValueError("samples must be 2D (S, n).")
    if samples.shape[1] != y_true.shape[0]:
        raise ValueError("dimension mismatch.")
    try:
        import properscoring as ps
    except ImportError:
        # Ensemble CRPS: E|X - y| - 0.5 E|X - X'|
        s = samples.shape[0]
        if s < 2:
            raise ValueError("Need at least 2 samples for CRPS.")
        term1 = np.mean(np.abs(samples - y_true[None, :]), axis=0)
        diffs = np.abs(samples[:, None, :] - samples[None, :, :])
        term2 = 0.5 * np.mean(diffs, axis=(0, 1))
        return (term1 - term2).astype(np.float64)
    return ps.crps_ensemble(y_true, samples).astype(np.float64)
