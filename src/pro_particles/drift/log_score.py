from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior


ArrayF = NDArray[np.float64]


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def pro_drift_log_score(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    logpdf: Callable[[ArrayF, ArrayF], ArrayF],
    grad_logpdf_theta: Callable[[ArrayF, ArrayF], ArrayF],
) -> ArrayF:
    """Compute the PrO drift for the logarithmic scoring rule.

    Parameters
    ----------
    particles:
        Current particles, shape (p, d).
    x_obs:
        Observations, shape (n, xdim) or (n,) for 1D.
    lam_n:
        Temperature/scaling λ_n.
    prior:
        Prior object providing grad_log_pdf.
    logpdf:
        Function (theta, x_obs) -> log p_theta(x) for each x, shape (n,).
    grad_logpdf_theta:
        Function (theta, x_obs) -> ∇_theta log p_theta(x) for each x, shape (n, d).

    Returns
    -------
    drift:
        Drift per particle, shape (p, d).
    """
    particles = _ensure_2d(particles, "particles")
    x_obs = _ensure_2d(x_obs, "x_obs")

    if not np.all(np.isfinite(particles)):
        raise ValueError("particles contains non-finite values.")
    if not np.all(np.isfinite(x_obs)):
        raise ValueError("x_obs contains non-finite values.")
    if lam_n <= 0.0:
        raise ValueError("lam_n must be > 0.")

    p, d = particles.shape
    n = x_obs.shape[0]
    if p < 2:
        raise ValueError("Need at least p>=2 particles for leave-one-out interaction.")
    if n < 1:
        raise ValueError("Need at least one observation.")

    # log p_{theta_j}(x_i) and ∇ log p_{theta_j}(x_i)
    logp = np.empty((p, n), dtype=np.float64)
    gradlogp = np.empty((p, n, d), dtype=np.float64)
    for j in range(p):
        theta_j = particles[j]
        logp[j] = logpdf(theta_j, x_obs)
        gradlogp[j] = grad_logpdf_theta(theta_j, x_obs)

    drift = np.zeros((p, d), dtype=np.float64)

    # Leave-one-out log-average using log-sum-exp (stable)
    for j in range(p):
        mask = np.ones(p, dtype=bool)
        mask[j] = False

        # logsumexp across l != j
        lse = np.logaddexp.reduce(logp[mask], axis=0)  # (n,)
        log_avg = lse - np.log(float(p - 1))          # (n,)

        # ratio: exp(logp_j - log_avg) * ∇ logp_j
        w = np.exp(logp[j] - log_avg)                 # (n,)
        score_grad = (w[:, None] * gradlogp[j]).mean(axis=0)  # (d,)

        # SDE drift sign convention:
        # dθ = -{ λ score_grad - ∇ log prior } dt + sqrt(2) dB
        drift[j] = -(lam_n * score_grad - prior.grad_log_pdf(particles[j]))

    return drift
