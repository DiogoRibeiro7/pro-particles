"""Vectorized MMD²-specific runner for performance-critical use."""

from __future__ import annotations

from typing import Dict, Optional

import logging
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
logger = logging.getLogger(__name__)


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
    """Fast MMD² runner with optional Fuse schedule.

    Parameters
    ----------
    init_particles:
        Initial particles, shape (p, d).
    x_obs:
        Observations or covariates, shape (n, d).
    y_obs:
        Optional responses, shape (n,).
    cfg:
        Spec configuration.
    prior:
        Prior distribution.
    sigma:
        Observation noise scale.
    lengthscale:
        RBF kernel lengthscale.
    m:
        Monte Carlo samples per particle/observation.
    use_fuse:
        Whether to use the Fuse step size schedule.
    leave_one_out:
        Whether to use leave-one-out interaction.
    fuse_state:
        Fuse state (required if use_fuse is True).
    rng:
        Random number generator.
    model:
        "gaussian_location" or "linear_regression".

    Returns
    -------
    Dict[str, ArrayF]
        Dictionary with final particles and time-averaged samples.

    References
    ----------
    docs/scoring_rules.md (p. 71).
    docs/fuse_spec.md (Algorithm 1).
    """
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

    log_every = max(1, cfg.K // 10)
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

        if not np.all(np.isfinite(drift)):
            logger.error("Non-finite drift at step %d/%d.", step + 1, cfg.K)
            raise ValueError("Non-finite drift encountered.")

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
            # grad_t = ∇U(x) = λ_n W(Q) - ∇log π (potential gradient, not drift)
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

        if step % log_every == 0 or step == cfg.K - 1:
            logger.info(
                "step %d/%d | eta=%.6e | max|drift|=%.6e | mean|particles|=%.6e",
                step + 1,
                cfg.K,
                eta_t,
                float(np.max(np.abs(drift))),
                float(np.mean(np.abs(particles))),
            )

        noise = rng.normal(size=particles.shape)
        particles = particles + drift * eta_t + (SpecConfig.SQRT2 * np.sqrt(eta_t)) * noise

        if not np.all(np.isfinite(particles)):
            logger.error("Non-finite particles at step %d/%d.", step + 1, cfg.K)
            raise ValueError("Non-finite particles encountered.")

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
