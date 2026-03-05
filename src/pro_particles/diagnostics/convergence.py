from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def trace_plot_data(saved_particles: ArrayF) -> Dict[str, ArrayF]:
    """Compute per-step mean and std for trace plotting.

    Parameters
    ----------
    saved_particles:
        Array of shape (K, p, d) of saved particles.

    Returns
    -------
    Dict[str, ArrayF]
        Dictionary with keys:
        - "step": shape (K,)
        - "particle_means": shape (K, d)
        - "particle_stds": shape (K, d)
    """
    if saved_particles.ndim != 3:
        raise ValueError("saved_particles must be 3D (K, p, d).")

    k = saved_particles.shape[0]
    means = saved_particles.mean(axis=1)
    stds = saved_particles.std(axis=1, ddof=0)
    steps = np.arange(k, dtype=np.float64)

    return {
        "step": steps,
        "particle_means": means,
        "particle_stds": stds,
    }


def running_mean(saved_particles: ArrayF) -> ArrayF:
    """Cumulative mean of time-averaged samples.

    Parameters
    ----------
    saved_particles:
        Array of shape (K, p, d) of saved particles.

    Returns
    -------
    ArrayF
        Running mean of shape (K, d).
    """
    if saved_particles.ndim != 3:
        raise ValueError("saved_particles must be 3D (K, p, d).")

    k = saved_particles.shape[0]
    per_step_mean = saved_particles.mean(axis=1)
    cumsum = np.cumsum(per_step_mean, axis=0)
    denom = np.arange(1, k + 1, dtype=np.float64)[:, None]
    return cumsum / denom


def effective_sample_size(samples: ArrayF) -> ArrayF:
    """Estimate per-dimension effective sample size using autocorrelations.

    Parameters
    ----------
    samples:
        Array of shape (n, d).

    Returns
    -------
    ArrayF
        ESS per dimension, shape (d,).
    """
    if samples.ndim != 2:
        raise ValueError("samples must be 2D (n, d).")
    n, d = samples.shape
    if n < 2:
        raise ValueError("Need at least 2 samples for ESS.")

    ess = np.empty(d, dtype=np.float64)
    for j in range(d):
        x = samples[:, j].astype(np.float64, copy=False)
        x = x - np.mean(x)
        var = np.mean(x * x)
        if var == 0.0:
            ess[j] = 1.0
            continue

        max_lag = n - 1
        rho = []
        for lag in range(1, max_lag + 1):
            cov = np.mean(x[:-lag] * x[lag:])
            rho.append(cov / var)

        tau = 1.0
        for k in range(0, len(rho), 2):
            if k + 1 >= len(rho):
                break
            pair_sum = float(rho[k] + rho[k + 1])
            if pair_sum < 0.0:
                break
            tau += 2.0 * pair_sum

        ess[j] = float(n / tau)

    return ess


def split_rhat(samples: ArrayF) -> ArrayF:
    """Split R-hat-like diagnostic using within/between-variance.

    This treats each dimension independently and splits a single chain in half.
    Returns R-hat per dimension, shape (d,).
    """
    if samples.ndim != 2:
        raise ValueError("samples must be 2D (n, d).")
    n, d = samples.shape
    if n < 4:
        raise ValueError("Need at least 4 samples for split R-hat.")

    n_half = n // 2
    if n_half < 2:
        raise ValueError("Need at least 2 samples per half.")

    first = samples[:n_half]
    second = samples[-n_half:]
    means = np.stack([first.mean(axis=0), second.mean(axis=0)], axis=0)
    vars_ = np.stack([first.var(axis=0, ddof=1), second.var(axis=0, ddof=1)], axis=0)

    w = vars_.mean(axis=0)
    b = n_half * means.var(axis=0, ddof=1)
    var_hat = ((n_half - 1) / n_half) * w + (b / n_half)

    rhat = np.sqrt(var_hat / w)
    return rhat.astype(np.float64)


def trajectory_summary(
    saved_particles: ArrayF, last_n: Optional[int] = None
) -> Dict[str, ArrayF]:
    """Summarize the trajectory across saved steps.

    Returns per-dimension min/max/mean/std computed over step means.
    """
    if saved_particles.ndim != 3:
        raise ValueError("saved_particles must be 3D (K, p, d).")
    if last_n is not None and last_n <= 0:
        raise ValueError("last_n must be > 0.")

    if last_n is not None:
        sp = saved_particles[-last_n:]
    else:
        sp = saved_particles

    if sp.shape[0] == 0:
        d = saved_particles.shape[2]
        empty = np.empty((0, d), dtype=np.float64)
        return {
            "step_means": empty,
            "mean": np.empty((d,), dtype=np.float64),
            "std": np.empty((d,), dtype=np.float64),
            "min": np.empty((d,), dtype=np.float64),
            "max": np.empty((d,), dtype=np.float64),
        }

    step_means = sp.mean(axis=1)
    return {
        "step_means": step_means,
        "mean": step_means.mean(axis=0),
        "std": step_means.std(axis=0, ddof=0),
        "min": step_means.min(axis=0),
        "max": step_means.max(axis=0),
    }
