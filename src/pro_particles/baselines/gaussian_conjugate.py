from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianConjugateResult:
    """Analytic posterior for normal likelihood with Gaussian prior."""

    mean: ArrayF  # (d,)
    cov: ArrayF   # (d, d)


def gaussian_conjugate_posterior(
    *,
    x_obs: ArrayF,
    sigma: float,
    prior_mean: ArrayF,
    prior_var: float,
) -> GaussianConjugateResult:
    """Analytic posterior for N(theta, sigma^2 I) with theta ~ N(prior_mean, prior_var I).

    Used as a baseline when the model is Gaussian location. Matches the paper's
    Gaussian location settings (Appendix D.1, D.2).
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if prior_var <= 0.0:
        raise ValueError("prior_var must be > 0.")
    if x_obs.ndim != 2:
        raise ValueError("x_obs must be 2D.")
    if prior_mean.ndim != 1:
        raise ValueError("prior_mean must be 1D.")

    n, d = x_obs.shape
    if prior_mean.shape[0] != d:
        raise ValueError("prior_mean dimension mismatch.")

    x_bar = x_obs.mean(axis=0)
    post_var = 1.0 / (n / (sigma * sigma) + 1.0 / prior_var)
    post_cov = post_var * np.eye(d, dtype=np.float64)
    post_mean = post_var * (n * x_bar / (sigma * sigma) + prior_mean / prior_var)

    return GaussianConjugateResult(mean=post_mean, cov=post_cov)


def run_baseline_gaussian_conjugate(
    *,
    x_obs: ArrayF,
    sigma: float,
    prior_mean: ArrayF,
    prior_var: float,
    n_samples: int,
    seed: int,
) -> ArrayF:
    """Sample from the analytic conjugate posterior."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")

    res = gaussian_conjugate_posterior(
        x_obs=x_obs,
        sigma=sigma,
        prior_mean=prior_mean,
        prior_var=prior_var,
    )

    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(res.mean, res.cov, size=n_samples).astype(np.float64)
