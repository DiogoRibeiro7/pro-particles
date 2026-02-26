from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pro_particles.kernels.rbf import grad_gaussian_kernel_wrt_first_arg
from pro_particles.priors.gaussian import GaussianPrior


ArrayF = NDArray[np.float64]


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def pro_drift_mmd2_location_gaussian(
    *,
    particles: ArrayF,      # (p, d), interpreted as Gaussian mean μ
    x_obs: ArrayF,          # (n, d)
    lam_n: float,
    prior: GaussianPrior,
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
) -> ArrayF:
    """PrO drift for squared MMD, specialised to a Gaussian location model.

    Notes
    -----
    - Model: Y | μ ~ N(μ, σ^2 I).
    - Kernel: RBF with given lengthscale.
    - Uses reparameterised Monte Carlo gradients.

    Returns
    -------
    drift: (p, d)
    """
    particles = _ensure_2d(particles, "particles")
    x_obs = _ensure_2d(x_obs, "x_obs")

    if not np.all(np.isfinite(particles)):
        raise ValueError("particles contains non-finite values.")
    if not np.all(np.isfinite(x_obs)):
        raise ValueError("x_obs contains non-finite values.")
    if lam_n <= 0.0:
        raise ValueError("lam_n must be > 0.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if m < 2:
        raise ValueError("m must be >= 2.")

    p, d = particles.shape
    n = x_obs.shape[0]
    if p < 2:
        raise ValueError("Need p>=2 particles for leave-one-out interaction.")
    if x_obs.shape[1] != d:
        raise ValueError("x_obs second dimension must match particle dimension d.")

    drift = np.zeros((p, d), dtype=np.float64)

    eps_j = rng.normal(size=(p, m, d))
    eps_l = rng.normal(size=(p, m, d))

    for j in range(p):
        mu_j = particles[j]
        interaction_sum = np.zeros(d, dtype=np.float64)

        for k in range(p):
            if k == j:
                continue
            mu_l = particles[k]

            Y = mu_j[None, :] + sigma * eps_j[j]  # (m, d)
            Yp = mu_l[None, :] + sigma * eps_l[k] # (m, d)

            grad_over_data = np.zeros(d, dtype=np.float64)

            for i in range(n):
                x_i = x_obs[i]  # (d,)

                # E[∇ k(Y, Y')] via all pairings (m x m)
                Y_exp = Y[:, None, :]   # (m, 1, d)
                Yp_exp = Yp[None, :, :] # (1, m, d)
                grad_k_YYp = grad_gaussian_kernel_wrt_first_arg(
                    Y_exp, Yp_exp, lengthscale
                )  # (m, m, d)
                term1 = grad_k_YYp.mean(axis=(0, 1))

                # E[∇ k(Y, x_i)]
                grad_k_Yx = grad_gaussian_kernel_wrt_first_arg(Y, x_i[None, :], lengthscale)
                term2 = grad_k_Yx.mean(axis=0)

                grad_over_data += (term1 - term2)

            grad_over_data /= float(n)
            interaction_sum += grad_over_data

        interaction_avg = interaction_sum / float(p - 1)
        drift[j] = -(lam_n * interaction_avg - prior.grad_log_pdf(mu_j))

    return drift
