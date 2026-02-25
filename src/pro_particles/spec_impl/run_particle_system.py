from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.schedules.fuse import FuseState, update_eta
from pro_particles.spec_impl.config import SpecConfig
from pro_particles.spec_impl.drift_logscore import drift_logscore
from pro_particles.spec_impl.drift_mmd2 import drift_mmd2


ArrayF = NDArray[np.float64]


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def run_particle_system(
    *,
    init_particles: ArrayF,  # (p, d)
    x_obs: ArrayF,           # (n, xdim)
    cfg: SpecConfig,
    prior: GaussianPrior,
    rule: str,
    use_fuse: bool,
    leave_one_out: bool,
    logpdf: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
    grad_logpdf_theta: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
    grad_L_mmd: Optional[Callable[[ArrayF, ArrayF, ArrayF], ArrayF]] = None,
    fuse_state: Optional[FuseState] = None,
    fuse_grad_fn: Optional[Callable[[ArrayF, ArrayF, float, GaussianPrior], ArrayF]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Paper-faithful runner for all scoring rules (docs/spec.md, docs/scoring_rules.md).

    - Discretisation and averaging follow docs/spec.md §2–§3.
    - Rule-specific drift follows docs/scoring_rules.md.
    - If use_fuse is True, step size is updated per docs/fuse_spec.md (Algorithm 1).
    """
    cfg.validate()

    particles = _ensure_2d(init_particles, "init_particles").copy()
    x_obs = _ensure_2d(x_obs, "x_obs")

    if particles.shape[0] != cfg.p:
        raise ValueError("cfg.p must match init_particles.shape[0].")

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    if rule == "log_score":
        if logpdf is None or grad_logpdf_theta is None:
            raise ValueError("logpdf and grad_logpdf_theta are required for log_score.")
    elif rule == "mmd2":
        if grad_L_mmd is None:
            raise ValueError("grad_L_mmd is required for mmd2.")
    else:
        raise ValueError(f"Unsupported rule: {rule}.")

    if use_fuse:
        if fuse_state is None:
            raise ValueError("fuse_state is required when use_fuse=True.")
        if fuse_grad_fn is None:
            raise ValueError("fuse_grad_fn is required when use_fuse=True.")

    saved: list[ArrayF] = []
    prev_particles: Optional[ArrayF] = None

    for step in range(cfg.K):
        if rule == "log_score":
            drift = drift_logscore(
                particles=particles,
                x_obs=x_obs,
                lam_n=cfg.lam_n,
                prior=prior,
                logpdf=logpdf,  # type: ignore[arg-type]
                grad_logpdf_theta=grad_logpdf_theta,  # type: ignore[arg-type]
                leave_one_out=leave_one_out,
            )
        else:
            drift = drift_mmd2(
                particles=particles,
                x_obs=x_obs,
                lam_n=cfg.lam_n,
                prior=prior,
                grad_L_mmd=grad_L_mmd,  # type: ignore[arg-type]
                leave_one_out=leave_one_out,
            )

        if not np.all(np.isfinite(drift)):
            raise ValueError("Non-finite drift encountered.")

        if use_fuse:
            grad_t = fuse_grad_fn(particles, x_obs, cfg.lam_n, prior)
            eta_t = update_eta(
                fuse_state,  # type: ignore[arg-type]
                t=step,
                particles_t=particles,
                particles_prev=prev_particles,
                grad_t=grad_t,
            )
        else:
            eta_t = cfg.dt_t(step)
            if eta_t <= 0.0:
                raise ValueError("dt_t(step) must be > 0.")

        noise = rng.normal(size=particles.shape)
        particles = particles + drift * eta_t + (cfg.sqrt2 * np.sqrt(eta_t)) * noise

        if not np.all(np.isfinite(particles)):
            raise ValueError("Non-finite particles encountered.")

        if use_fuse and step == 0 and fuse_state is not None and fuse_state.x1 is None:
            fuse_state.x1 = particles.copy()

        if step >= cfg.B and ((step - cfg.B) % cfg.thin == 0):
            saved.append(particles.copy())

        prev_particles = particles.copy()

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
