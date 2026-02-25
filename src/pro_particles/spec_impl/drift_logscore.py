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


def drift_logscore(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    logpdf: Callable[[ArrayF, ArrayF], ArrayF],
    grad_logpdf_theta: Callable[[ArrayF, ArrayF], ArrayF],
    leave_one_out: bool,
) -> ArrayF:
    """Log-score drift from docs/scoring_rules.md (p. 71) and docs/spec.md §2.

    Implements W(Q)[theta] for the log score and plugs it into the drift term
    from docs/spec.md §2 (SDE with sqrt(2)).
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

    logp = np.empty((p, n), dtype=np.float64)
    gradlogp = np.empty((p, n, d), dtype=np.float64)
    for j in range(p):
        theta_j = particles[j]
        logp[j] = logpdf(theta_j, x_obs)
        gradlogp[j] = grad_logpdf_theta(theta_j, x_obs)

    dP = np.exp(logp)
    grad_dP = dP[:, :, None] * gradlogp

    drift = np.zeros((p, d), dtype=np.float64)

    for j in range(p):
        if leave_one_out:
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            denom = dP[mask].mean(axis=0)  # 1/(p-1) sum_{l!=j} dP_l(x_i)
        else:
            denom = dP.mean(axis=0)        # 1/p sum_{l} dP_l(x_i)

        if not np.all(np.isfinite(denom)) or np.any(denom == 0.0):
            raise ValueError("Non-finite or zero denominator in W(Q) for log-score.")

        wq = (grad_dP[j] / denom[:, None]).mean(axis=0)

        drift[j] = -(lam_n * wq - prior.grad_log_pdf(particles[j]))

    if not np.all(np.isfinite(drift)):
        raise ValueError("drift_logscore produced non-finite values.")

    return drift
