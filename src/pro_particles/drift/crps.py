from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.drift_crps import drift_crps


ArrayF = NDArray[np.float64]


def pro_drift_crps(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    grad_L_crps: Callable[[ArrayF, ArrayF, ArrayF], ArrayF],
) -> ArrayF:
    """Backward-compatible wrapper around spec_impl drift (leave-one-out)."""
    return drift_crps(
        particles=particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        grad_L_crps=grad_L_crps,
        leave_one_out=True,
    )
