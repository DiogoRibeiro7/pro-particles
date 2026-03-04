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


def drift_crps(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    grad_L_crps: Callable[[ArrayF, ArrayF, ArrayF], ArrayF],
    leave_one_out: bool,
) -> ArrayF:
    """CRPS drift using a user-supplied ∇_1 L_crps(θ, θ'; x).

    Note: The paper does not provide an explicit W(Q) formula for CRPS.
    This implementation mirrors the MMD^2 structure and requires grad_L_crps.
    """
    particles = _ensure_2d(particles, "particles")
    x_obs = _ensure_2d(x_obs, "x_obs")

    if lam_n <= 0.0:
        raise ValueError("lam_n must be > 0.")

    p, d = particles.shape
    n = x_obs.shape[0]
    if n < 1:
        raise ValueError("Need at least one observation.")
    if leave_one_out and p < 2:
        raise ValueError("leave_one_out requires p >= 2.")

    drift = np.zeros((p, d), dtype=np.float64)

    for j in range(p):
        interaction = np.zeros((n, d), dtype=np.float64)
        for k in range(p):
            if leave_one_out and k == j:
                continue
            for i in range(n):
                interaction[i] += grad_L_crps(particles[j], particles[k], x_obs[i])

        if leave_one_out:
            interaction /= float(p - 1)
        else:
            interaction /= float(p)

        wq = interaction.mean(axis=0)
        drift[j] = -(lam_n * wq - prior.grad_log_pdf(particles[j]))

    if not np.all(np.isfinite(drift)):
        raise ValueError("drift_crps produced non-finite values.")

    return drift
