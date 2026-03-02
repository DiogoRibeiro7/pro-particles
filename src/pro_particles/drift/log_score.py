from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.drift_logscore import drift_logscore


ArrayF = NDArray[np.float64]


def pro_drift_log_score(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    logpdf: Callable[[ArrayF, ArrayF], ArrayF],
    grad_logpdf_theta: Callable[[ArrayF, ArrayF], ArrayF],
) -> ArrayF:
    """Backward-compatible wrapper around spec_impl drift (leave-one-out)."""
    return drift_logscore(
        particles=particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        logpdf=logpdf,
        grad_logpdf_theta=grad_logpdf_theta,
        leave_one_out=True,
    )
