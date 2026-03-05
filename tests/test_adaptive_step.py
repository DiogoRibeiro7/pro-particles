from __future__ import annotations

import numpy as np

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl import SpecConfig
from pro_particles.spec_impl.config import AdaptiveStepConfig
from pro_particles.spec_impl.run import run_em


def test_adaptive_step_reduces_dt_for_large_drift() -> None:
    rng = np.random.default_rng(0)
    init_particles = rng.normal(size=(3, 1)).astype(np.float64)
    x_obs = rng.normal(size=(5, 1)).astype(np.float64)

    adaptive = AdaptiveStepConfig(
        enabled=True,
        max_drift_step=0.05,
        dt_min=1e-6,
        dt_max=1e-2,
        eps=1e-12,
    )

    cfg = SpecConfig(
        p=3,
        dt_t=lambda step: 1e-2,
        B=1,
        K=2,
        thin=1,
        seed=0,
        lam_n=1.0,
        adaptive_step=adaptive,
    )

    def drift_fn(particles, x, lam_n, prior):
        return np.full_like(particles, 1000.0)

    out = run_em(
        init_particles=init_particles,
        x_obs=x_obs,
        cfg=cfg,
        prior=GaussianPrior(prior_var=5.0),
        drift_fn=drift_fn,
        rng=rng,
    )

    assert out["final_particles"].shape == (3, 1)
