from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.drift_mmd2 import drift_mmd2


ArrayF = NDArray[np.float64]


def pro_drift_mmd2_generic(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    grad_L_mmd: Callable[[ArrayF, ArrayF, ArrayF], ArrayF],
    leave_one_out: bool,
) -> ArrayF:
    """Backward-compatible wrapper around spec_impl drift_mmd2."""
    return drift_mmd2(
        particles=particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        grad_L_mmd=grad_L_mmd,
        leave_one_out=leave_one_out,
    )
