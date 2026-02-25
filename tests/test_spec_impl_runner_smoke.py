from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl import SpecConfig, run_particle_system


def test_run_particle_system_log_score_smoke() -> None:
    rng = np.random.default_rng(0)
    x_obs = rng.normal(loc=0.5, scale=1.0, size=(20, 1)).astype(np.float64)
    init_particles = rng.normal(size=(6, 1)).astype(np.float64)

    cfg = SpecConfig(
        p=6,
        dt_t=lambda step: 1e-3,
        B=10,
        K=1000,
        thin=10,
        seed=0,
        lam_n=1.0,
        sqrt2=np.sqrt(2.0),
    )

    out = run_particle_system(
        init_particles=init_particles,
        x_obs=x_obs,
        cfg=cfg,
        prior=GaussianPrior(prior_var=10.0),
        rule="log_score",
        use_fuse=False,
        leave_one_out=True,
        logpdf=lambda theta, xobs: normal_logpdf(theta, xobs, sigma=1.0),
        grad_logpdf_theta=lambda theta, xobs: normal_grad_logpdf_theta(theta, xobs, sigma=1.0),
        rng=rng,
    )

    assert out["final_particles"].shape == (cfg.p, 1)
    assert np.isfinite(out["final_particles"]).all()


def test_run_particle_system_mmd2_smoke() -> None:
    rng = np.random.default_rng(1)
    x_obs = rng.normal(size=(12, 2)).astype(np.float64)
    init_particles = rng.normal(size=(5, 2)).astype(np.float64)

    cfg = SpecConfig(
        p=5,
        dt_t=lambda step: 1e-3,
        B=10,
        K=1000,
        thin=10,
        seed=1,
        lam_n=1.0,
        sqrt2=np.sqrt(2.0),
    )

    def grad_L_mmd(theta, theta2, x):
        return (theta - theta2) + 0.1 * x

    out = run_particle_system(
        init_particles=init_particles,
        x_obs=x_obs,
        cfg=cfg,
        prior=GaussianPrior(prior_var=10.0),
        rule="mmd2",
        use_fuse=False,
        leave_one_out=True,
        grad_L_mmd=grad_L_mmd,
        rng=rng,
    )

    assert out["final_particles"].shape == (cfg.p, 2)
    assert np.isfinite(out["final_particles"]).all()
