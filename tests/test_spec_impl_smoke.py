from __future__ import annotations

import numpy as np

from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl import SpecConfig, drift_logscore, run_em


def test_spec_impl_smoke() -> None:
    rng = np.random.default_rng(0)

    x_obs = rng.normal(loc=0.5, scale=1.0, size=(20, 1)).astype(np.float64)
    init_particles = rng.normal(size=(8, 1)).astype(np.float64)

    cfg = SpecConfig(
        p=8,
        dt_t=lambda step: 1e-3,
        B=10,
        K=200,
        thin=5,
        seed=0,
        lam_n=1.0,
    )

    def drift_fn(particles, x, lam_n, prior):
        return drift_logscore(
            particles=particles,
            x_obs=x,
            lam_n=lam_n,
            prior=prior,
            logpdf=lambda theta, xobs: normal_logpdf(theta, xobs, sigma=1.0),
            grad_logpdf_theta=lambda theta, xobs: normal_grad_logpdf_theta(theta, xobs, sigma=1.0),
            leave_one_out=True,
        )

    out = run_em(
        init_particles=init_particles,
        x_obs=x_obs,
        cfg=cfg,
        prior=GaussianPrior(prior_var=10.0),
        drift_fn=drift_fn,
    )

    final_particles = out["final_particles"]
    saved_particles = out["saved_particles"]
    time_avg_samples = out["time_avg_samples"]

    assert final_particles.shape == (cfg.p, 1)
    assert saved_particles.shape[1:] == (cfg.p, 1)
    assert time_avg_samples.shape[1:] == (1,)
    assert np.isfinite(final_particles).all()
    assert np.isfinite(saved_particles).all()
    assert np.isfinite(time_avg_samples).all()
