from __future__ import annotations

import numpy as np

from pro_particles.drift.log_score import pro_drift_log_score
from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior


def test_pro_drift_log_score_shapes() -> None:
    rng = np.random.default_rng(0)
    p, d = 8, 1
    n = 20
    particles = rng.normal(size=(p, d)).astype(np.float64)
    x_obs = rng.normal(size=(n, 1)).astype(np.float64)
    prior = GaussianPrior(prior_var=5.0)
    lam_n = 3.0
    sigma = 1.2

    drift = pro_drift_log_score(
        particles=particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        logpdf=lambda th, xx: normal_logpdf(th, xx, sigma),
        grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma),
    )
    assert drift.shape == (p, d)
    assert np.all(np.isfinite(drift))
