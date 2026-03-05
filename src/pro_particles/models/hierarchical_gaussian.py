from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def hierarchical_gaussian_logpdf(theta: ArrayF, x_obs: ArrayF, sigma_y: float) -> ArrayF:
    """Log-likelihood per observation for hierarchical Gaussian means.

    theta = [alpha, mu_0, ..., mu_{G-1}]
    x_obs: (n, 2) with columns [group_id, y]
    """
    if sigma_y <= 0.0:
        raise ValueError("sigma_y must be > 0.")
    if theta.ndim != 1:
        raise ValueError("theta must be 1D (d,).")
    if x_obs.ndim != 2 or x_obs.shape[1] != 2:
        raise ValueError("x_obs must be 2D with shape (n, 2).")

    group_ids = x_obs[:, 0].astype(int)
    y = x_obs[:, 1]
    n_groups = theta.shape[0] - 1
    if np.any(group_ids < 0) or np.any(group_ids >= n_groups):
        raise ValueError("group_ids must be in [0, n_groups).")

    mus = theta[1:]
    mu_y = mus[group_ids]
    z = (y - mu_y) / sigma_y
    return -0.5 * (z * z) - np.log(sigma_y) - 0.5 * np.log(2.0 * np.pi)


def hierarchical_gaussian_grad_logpdf_theta(
    theta: ArrayF, x_obs: ArrayF, sigma_y: float
) -> ArrayF:
    """Gradient of log-likelihood per observation, shape (n, d)."""
    if sigma_y <= 0.0:
        raise ValueError("sigma_y must be > 0.")
    if theta.ndim != 1:
        raise ValueError("theta must be 1D (d,).")
    if x_obs.ndim != 2 or x_obs.shape[1] != 2:
        raise ValueError("x_obs must be 2D with shape (n, 2).")

    group_ids = x_obs[:, 0].astype(int)
    y = x_obs[:, 1]
    n_groups = theta.shape[0] - 1
    if np.any(group_ids < 0) or np.any(group_ids >= n_groups):
        raise ValueError("group_ids must be in [0, n_groups).")

    d = theta.shape[0]
    grads = np.zeros((x_obs.shape[0], d), dtype=np.float64)
    mus = theta[1:]
    mu_y = mus[group_ids]
    grad_mu = (y - mu_y) / (sigma_y * sigma_y)
    grads[np.arange(x_obs.shape[0]), group_ids + 1] = grad_mu
    return grads
