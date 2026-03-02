from __future__ import annotations

import math

import numpy as np

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.schedules.fuse import FuseState
from pro_particles.spec_impl import SpecConfig, run_particle_system
from pro_particles.spec_impl.drift_mmd2 import drift_mmd2


def test_fuse_runner_uses_prev_particles_for_movement() -> None:
    rng_init = np.random.default_rng(123)
    p, d, n = 3, 2, 6
    init_particles = rng_init.normal(size=(p, d)).astype(np.float64)
    x_obs = rng_init.normal(size=(n, d)).astype(np.float64)

    cfg = SpecConfig(
        p=p,
        dt_t=lambda step: 1e-2,
        B=1,
        K=2,
        thin=1,
        seed=0,
        lam_n=5.0,
    )

    prior = GaussianPrior(prior_var=2.0)

    def grad_L_mmd(theta: np.ndarray, theta2: np.ndarray, x: np.ndarray) -> np.ndarray:
        return (theta - theta2) + 0.1 * x

    def fuse_grad_fn(particles: np.ndarray, xobs: np.ndarray, lam_n: float, prior_in: GaussianPrior) -> np.ndarray:
        drift = drift_mmd2(
            particles=particles,
            x_obs=xobs,
            lam_n=lam_n,
            prior=prior_in,
            grad_L_mmd=grad_L_mmd,
            leave_one_out=True,
        )
        return -drift  # grad_t = lam_n * wq - prior_grad

    fuse_state = FuseState(r_eps=1e-12)
    rng_expected = np.random.default_rng(999)
    drift0 = drift_mmd2(
        particles=init_particles,
        x_obs=x_obs,
        lam_n=cfg.lam_n,
        prior=prior,
        grad_L_mmd=grad_L_mmd,
        leave_one_out=True,
    )
    noise0 = rng_expected.normal(size=init_particles.shape)
    eta0 = fuse_state.r_eps
    x1 = init_particles + drift0 * eta0 + (SpecConfig.SQRT2 * math.sqrt(eta0)) * noise0
    movement = float(np.mean(np.sum((x1 - init_particles) ** 2, axis=1)))
    rng_run = np.random.default_rng(999)

    run_particle_system(
        init_particles=init_particles,
        x_obs=x_obs,
        cfg=cfg,
        prior=prior,
        rule="mmd2",
        use_fuse=True,
        leave_one_out=True,
        grad_L_mmd=grad_L_mmd,
        fuse_state=fuse_state,
        fuse_grad_fn=fuse_grad_fn,
        rng=rng_run,
    )

    assert fuse_state.x1 is not None
    assert np.allclose(fuse_state.x1, x1)
    assert math.isclose(fuse_state.max_movement, movement, rel_tol=1e-10, abs_tol=0.0)
