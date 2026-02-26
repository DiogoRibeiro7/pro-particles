from __future__ import annotations

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


def wq_mmd2_gaussian_location_fast(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, d)
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
    leave_one_out: bool,
) -> ArrayF:
    """Vectorized W(Q) for Gaussian location MMD² (docs/scoring_rules.md, p. 71)."""
    particles = _ensure_2d(particles, "particles")
    x_obs = _ensure_2d(x_obs, "x_obs")

    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0.")
    if m < 2:
        raise ValueError("m must be >= 2.")

    p, d = particles.shape
    n = x_obs.shape[0]
    if leave_one_out and p < 2:
        raise ValueError("leave_one_out requires p >= 2.")

    eps = rng.normal(size=(p, n, m, d))
    eps2 = rng.normal(size=(p, n, m, d))
    y = particles[:, None, None, :] + sigma * eps
    y2 = particles[:, None, None, :] + sigma * eps2

    wq = np.zeros((p, d), dtype=np.float64)

    for i in range(n):
        yi = y[:, i, :, :]  # (p, m, d)
        y2i = y2[:, i, :, :]  # (p, m, d)

        diff = y2i[None, :, None, :, :] - yi[:, None, :, None, :]  # (p, p, m, m, d)
        dist2 = np.sum(diff * diff, axis=-1)
        k = np.exp(-0.5 * dist2 / (lengthscale**2))
        grad = (k[..., None] * diff) / (lengthscale**2)
        term1 = grad.mean(axis=(2, 3))  # (p, p, d)

        diff_x = x_obs[i][None, None, :] - yi  # (p, m, d)
        dist2_x = np.sum(diff_x * diff_x, axis=-1)
        k_x = np.exp(-0.5 * dist2_x / (lengthscale**2))
        term2 = (k_x[..., None] * diff_x / (lengthscale**2)).mean(axis=1)  # (p, d)

        if leave_one_out:
            diag = np.diagonal(term1, axis1=0, axis2=1).T
            term1_sum = term1.sum(axis=1) - diag
            term1_avg = term1_sum / float(p - 1)
        else:
            term1_avg = term1.mean(axis=1)

        term1_avg = term1_avg.reshape(p, d)
        term2 = term2.reshape(p, d)
        wq += (term1_avg - term2)

    return wq / float(n)


def drift_mmd2_gaussian_location_fast(
    *,
    particles: ArrayF,
    x_obs: ArrayF,
    lam_n: float,
    prior: GaussianPrior,
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
    leave_one_out: bool,
) -> ArrayF:
    wq = wq_mmd2_gaussian_location_fast(
        particles=particles,
        x_obs=x_obs,
        sigma=sigma,
        lengthscale=lengthscale,
        m=m,
        rng=rng,
        leave_one_out=leave_one_out,
    )
    prior_grad = np.array([prior.grad_log_pdf(theta) for theta in particles])
    return -(lam_n * wq - prior_grad)


def wq_mmd2_linear_regression_fast(
    *,
    particles: ArrayF,  # (p, d)
    x: ArrayF,          # (n, d)
    y: ArrayF,          # (n,)
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
    leave_one_out: bool,
) -> ArrayF:
    """Vectorized W(Q) for linear regression MMD² (Appendix D.3/D.4)."""
    particles = _ensure_2d(particles, "particles")
    x = _ensure_2d(x, "x")
    y = y.reshape(-1).astype(np.float64, copy=False)

    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0.")
    if m < 2:
        raise ValueError("m must be >= 2.")

    p, d = particles.shape
    n = x.shape[0]
    if leave_one_out and p < 2:
        raise ValueError("leave_one_out requires p >= 2.")

    wq = np.zeros((p, d), dtype=np.float64)

    for i in range(n):
        mu = x[i] @ particles.T  # (p,)
        eps1 = rng.normal(size=(p, m))
        eps2 = rng.normal(size=(p, m))
        y1 = mu[:, None] + sigma * eps1
        y2 = mu[:, None] + sigma * eps2

        diff = y2[None, :, None, :] - y1[:, None, :, None]  # (p,p,m,m)
        dist2 = diff * diff
        k = np.exp(-0.5 * dist2 / (lengthscale**2))
        grad = (k * diff) / (lengthscale**2)
        term1 = grad.mean(axis=(2, 3))  # (p,p)

        diff_x = y[i] - y1  # (p,m)
        dist2_x = diff_x * diff_x
        k_x = np.exp(-0.5 * dist2_x / (lengthscale**2))
        term2 = (k_x * diff_x / (lengthscale**2)).mean(axis=1)  # (p,)

        if leave_one_out:
            term1_sum = term1.sum(axis=1) - np.diagonal(term1)
            term1_avg = term1_sum / float(p - 1)
        else:
            term1_avg = term1.mean(axis=1)

        wq += (term1_avg - term2)[:, None] * x[i][None, :]

    return wq / float(n)


def drift_mmd2_linear_regression_fast(
    *,
    particles: ArrayF,
    x: ArrayF,
    y: ArrayF,
    lam_n: float,
    prior: GaussianPrior,
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
    leave_one_out: bool,
) -> ArrayF:
    wq = wq_mmd2_linear_regression_fast(
        particles=particles,
        x=x,
        y=y,
        sigma=sigma,
        lengthscale=lengthscale,
        m=m,
        rng=rng,
        leave_one_out=leave_one_out,
    )
    prior_grad = np.array([prior.grad_log_pdf(theta) for theta in particles])
    return -(lam_n * wq - prior_grad)
