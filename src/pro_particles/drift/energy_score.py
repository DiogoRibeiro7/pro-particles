from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.drift_energy_score import drift_energy_score


ArrayF = NDArray[np.float64]


def pro_drift_energy_score(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    grad_L_energy: Callable[[ArrayF, ArrayF, ArrayF], ArrayF],
) -> ArrayF:
    """Backward-compatible wrapper around spec_impl drift (leave-one-out)."""
    return drift_energy_score(
        particles=particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        grad_L_energy=grad_L_energy,
        leave_one_out=True,
    )
