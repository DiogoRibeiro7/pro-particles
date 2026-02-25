from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pro_particles.drift.mmd2_generic import pro_drift_mmd2_generic
from pro_particles.priors.gaussian import GaussianPrior


def test_mmd2_generic_smoke() -> None:
    rng = np.random.default_rng(0)
    particles = rng.normal(size=(4, 2)).astype(np.float64)
    x_obs = rng.normal(size=(5, 2)).astype(np.float64)

    def grad_L_mmd(theta, theta2, x):
        return (theta - theta2) + 0.1 * x

    drift = pro_drift_mmd2_generic(
        particles=particles,
        x_obs=x_obs,
        lam_n=1.0,
        prior=GaussianPrior(prior_var=10.0),
        grad_L_mmd=grad_L_mmd,
        leave_one_out=True,
    )

    assert drift.shape == particles.shape
    assert np.isfinite(drift).all()
