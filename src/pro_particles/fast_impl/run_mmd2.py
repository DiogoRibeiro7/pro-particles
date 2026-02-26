from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from pro_particles.fast_impl.mmd2 import (
    drift_mmd2_gaussian_location_fast,
    drift_mmd2_linear_regression_fast,
    wq_mmd2_gaussian_location_fast,
    wq_mmd2_linear_regression_fast,
)
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.schedules.fuse import FuseState, update_eta
from pro_particles.spec_impl.config import SpecConfig


ArrayF = NDArray[np.float64]


def run_particle_system_mmd2_fast(
    *,
    init_particles: ArrayF,
    x_obs: ArrayF,
    y_obs: Optional[ArrayF],
    cfg: SpecConfig,
    prior: GaussianPrior,
    sigma: float,
    lengthscale: float,
    m: int,
    use_fuse: bool,
    leave_one_out: bool,
    fuse_state: Optional[FuseState],
    rng: Optional[np.random.Generator] = None,
    model: str,
) -> Dict[str, ArrayF]:
    """Fast MMD² runner with optional Fuse schedule."""
    cfg.validate()

    particles = init_particles.astype(np.float64, copy=True)
    x_obs = x_obs.astype(np.float64, copy=False)
    if y_obs is not None:
        y_obs = y_obs.astype(np.float64, copy=False).reshape(-1)

    if particles.shape[0] != cfg.p:
        raise ValueError("cfg.p must match init_particles.shape[0].")

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    saved = []
    prev_particles = None

    for step in range(cfg.K):
        particles_t = particles
        if model == "gaussian_location":
            drift = drift_mmd2_gaussian_location_fast(
                particles=particles_t,
                x_obs=x_obs,
                lam_n=cfg.lam_n,
                prior=prior,
                sigma=sigma,
                lengthscale=lengthscale,
                m=m,
                rng=rng,
                leave_one_out=leave_one_out,
            )
        elif model == "linear_regression":
            if y_obs is None:
                raise ValueError("y_obs required for linear_regression.")
            drift = drift_mmd2_linear_regression_fast(
                particles=particles_t,
                x=x_obs,
                y=y_obs,
                lam_n=cfg.lam_n,
                prior=prior,
                sigma=sigma,
                lengthscale=lengthscale,
                m=m,
                rng=rng,
                leave_one_out=leave_one_out,
            )
        else:
            raise ValueError("Unknown model.")

        if use_fuse:
            if fuse_state is None:
                raise ValueError("fuse_state required.")
            if model == "gaussian_location":
                wq = wq_mmd2_gaussian_location_fast(
                    particles=particles_t,
                    x_obs=x_obs,
                    sigma=sigma,
                    lengthscale=lengthscale,
                    m=m,
                    rng=rng,
                    leave_one_out=leave_one_out,
                )
            else:
                assert y_obs is not None  # linear regression requires y
                wq = wq_mmd2_linear_regression_fast(
                    particles=particles_t,
                    x=x_obs,
                    y=y_obs,
                    sigma=sigma,
                    lengthscale=lengthscale,
                    m=m,
                    rng=rng,
                    leave_one_out=leave_one_out,
                )
            prior_grad = np.array([prior.grad_log_pdf(theta) for theta in particles])
            grad_t = cfg.lam_n * wq - prior_grad
            eta_t = update_eta(
                fuse_state,
                t=step,
                particles_t=particles_t,
                particles_prev=prev_particles,
                grad_t=grad_t,
            )
        else:
            eta_t = cfg.dt_t(step)

        noise = rng.normal(size=particles.shape)
        particles = particles + drift * eta_t + (cfg.sqrt2 * np.sqrt(eta_t)) * noise

        if use_fuse and step == 0 and fuse_state is not None and fuse_state.x1 is None:
            fuse_state.x1 = particles.copy()

        if step >= cfg.B and ((step - cfg.B) % cfg.thin == 0):
            saved.append(particles.copy())

        prev_particles = particles_t.copy()

    if saved:
        saved_particles = np.stack(saved, axis=0)
        time_avg_samples = saved_particles.reshape(-1, particles.shape[1])
    else:
        saved_particles = np.empty((0, cfg.p, particles.shape[1]), dtype=np.float64)
        time_avg_samples = np.empty((0, particles.shape[1]), dtype=np.float64)

    return {
        "final_particles": particles,
        "saved_particles": saved_particles,
        "time_avg_samples": time_avg_samples,
    }
