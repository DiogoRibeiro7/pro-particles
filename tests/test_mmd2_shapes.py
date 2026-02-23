from __future__ import annotations

import numpy as np

from pro_particles.drift.mmd2 import pro_drift_mmd2_location_gaussian
from pro_particles.priors.gaussian import GaussianPrior


def test_pro_drift_mmd2_shapes() -> None:
    rng = np.random.default_rng(0)
    p, d = 6, 2
    n = 10

    particles = rng.normal(size=(p, d)).astype(np.float64)
    x_obs = rng.normal(size=(n, d)).astype(np.float64)

    prior = GaussianPrior(prior_var=4.0)
    drift = pro_drift_mmd2_location_gaussian(
        particles=particles,
        x_obs=x_obs,
        lam_n=1.5,
        prior=prior,
        sigma=0.9,
        lengthscale=1.2,
        m=16,
        rng=rng,
    )
    assert drift.shape == (p, d)
    assert np.all(np.isfinite(drift))
