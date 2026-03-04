from __future__ import annotations

import numpy as np

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler.em import SamplerConfig, sample_pro_posterior


def test_em_energy_score_smoke() -> None:
    rng = np.random.default_rng(10)
    x_obs = rng.normal(size=(30, 2)).astype(np.float64)

    prior = GaussianPrior(prior_var=9.0)
    init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(8, 2)).astype(np.float64)

    cfg = SamplerConfig(n_steps=200, burn_in=50, dt=1e-3, thin=10, seed=11)
    lam_n = 2.0

    final_particles, samples = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="energy_score",
        sigma=0.8,
        m=16,
    )

    assert final_particles.shape == (8, 2)
    assert samples.shape[1] == 2
    assert np.isfinite(final_particles).all()
    assert np.isfinite(samples).all()


def test_em_crps_smoke() -> None:
    rng = np.random.default_rng(12)
    x_obs = rng.normal(size=(40, 1)).astype(np.float64)

    prior = GaussianPrior(prior_var=9.0)
    init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(10, 1)).astype(np.float64)

    cfg = SamplerConfig(n_steps=200, burn_in=50, dt=1e-3, thin=10, seed=13)
    lam_n = 2.0

    final_particles, samples = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="crps",
        sigma=0.8,
        m=16,
    )

    assert final_particles.shape == (10, 1)
    assert samples.shape[1] == 1
    assert np.isfinite(final_particles).all()
    assert np.isfinite(samples).all()
