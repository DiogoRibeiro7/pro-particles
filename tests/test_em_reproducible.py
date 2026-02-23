from __future__ import annotations

import numpy as np

from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler.em import SamplerConfig, sample_pro_posterior


def test_em_reproducible_given_seed() -> None:
    rng = np.random.default_rng(123)
    x_obs = rng.normal(size=(50, 1)).astype(np.float64)

    prior = GaussianPrior(prior_var=9.0)
    init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(12, 1)).astype(np.float64)

    cfg = SamplerConfig(n_steps=300, burn_in=50, dt=1e-3, thin=5, seed=7)
    lam_n = 2.0
    sigma = 1.0

    _, s1 = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="log_score",
        logpdf=lambda th, xx: normal_logpdf(th, xx, sigma),
        grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma),
    )
    _, s2 = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="log_score",
        logpdf=lambda th, xx: normal_logpdf(th, xx, sigma),
        grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma),
    )

    assert s1.shape == s2.shape
    assert np.allclose(s1, s2)
