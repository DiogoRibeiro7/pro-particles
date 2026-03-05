from __future__ import annotations

import numpy as np

from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler import ProSampler, SamplerConfig


def test_pro_sampler_log_score_smoke() -> None:
    rng = np.random.default_rng(0)
    x_obs = rng.normal(size=(40, 1)).astype(np.float64)
    init_particles = rng.normal(size=(6, 1)).astype(np.float64)

    prior = GaussianPrior(prior_var=9.0)
    cfg = SamplerConfig(n_steps=300, burn_in=50, dt=1e-3, thin=10, seed=123)

    sampler = ProSampler(
        mode="log_score",
        lam_n=1.5,
        prior=prior,
        cfg=cfg,
        logpdf=lambda th, xx: normal_logpdf(th, xx, sigma=1.0),
        grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma=1.0),
    )

    result = sampler.sample(init_particles=init_particles, x_obs=x_obs)

    assert result.final_particles.shape == (6, 1)
    assert result.saved_particles.ndim == 3
    assert result.time_avg_samples.shape[1] == 1
    assert np.isfinite(result.final_particles).all()
    assert np.isfinite(result.time_avg_samples).all()
