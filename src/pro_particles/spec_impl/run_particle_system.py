"""Full-featured runner supporting all scoring rules and FUSE schedule."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol

import logging
import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.schedules.fuse import FuseState, update_eta
from pro_particles.spec_impl.config import SpecConfig, AdaptiveStepConfig
from pro_particles.spec_impl.drift_logscore import drift_logscore
from pro_particles.spec_impl.drift_mmd2 import drift_mmd2
from pro_particles.spec_impl.drift_energy_score import drift_energy_score
from pro_particles.spec_impl.drift_crps import drift_crps


ArrayF = NDArray[np.float64]
logger = logging.getLogger(__name__)


class FuseGradFn(Protocol):
    def __call__(
        self,
        particles: ArrayF,
        x_obs: ArrayF,
        lam_n: float,
        prior: GaussianPrior,
    ) -> ArrayF:
        """Return ∇U(x) = λ_n W(Q)[θ] - ∇log π(θ), shape (p, d)."""
        ...


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
    grad_L_energy: Optional[Callable[[ArrayF, ArrayF, ArrayF], ArrayF]] = None,
    grad_L_crps: Optional[Callable[[ArrayF, ArrayF, ArrayF], ArrayF]] = None,
    fuse_state: Optional[FuseState] = None,
    fuse_grad_fn: Optional[FuseGradFn] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Paper-faithful runner for all scoring rules (docs/spec.md, docs/scoring_rules.md).

    Parameters
    ----------
    fuse_grad_fn:
        Computes ∇U(x) = λ_n W(Q)[θ] - ∇log π(θ) for all particles.
        This is the POTENTIAL GRADIENT, not the drift. The drift is -∇U(x).
        Must return array of shape (p, d).

    Notes
    -----
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
    elif rule == "energy_score":
        if grad_L_energy is None:
            raise ValueError("grad_L_energy is required for energy_score.")
    elif rule == "crps":
        if grad_L_crps is None:
            raise ValueError("grad_L_crps is required for crps.")
    else:
        raise ValueError(f"Unsupported rule: {rule}.")

    if use_fuse:
        if fuse_state is None:
            raise ValueError("fuse_state is required when use_fuse=True.")
        if fuse_grad_fn is None:
            raise ValueError("fuse_grad_fn is required when use_fuse=True.")

    saved: list[ArrayF] = []
    prev_particles: Optional[ArrayF] = None

    log_every = max(1, cfg.K // 10)
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
        elif rule == "mmd2":
            drift = drift_mmd2(
                particles=particles,
                x_obs=x_obs,
                lam_n=cfg.lam_n,
                prior=prior,
                grad_L_mmd=grad_L_mmd,  # type: ignore[arg-type]
                leave_one_out=leave_one_out,
            )
        elif rule == "energy_score":
            drift = drift_energy_score(
                particles=particles,
                x_obs=x_obs,
                lam_n=cfg.lam_n,
                prior=prior,
                grad_L_energy=grad_L_energy,  # type: ignore[arg-type]
                leave_one_out=leave_one_out,
            )
        else:
            drift = drift_crps(
                particles=particles,
                x_obs=x_obs,
                lam_n=cfg.lam_n,
                prior=prior,
                grad_L_crps=grad_L_crps,  # type: ignore[arg-type]
                leave_one_out=leave_one_out,
            )

        if not np.all(np.isfinite(drift)):
            logger.error("Non-finite drift at step %d/%d.", step + 1, cfg.K)
            raise ValueError("Non-finite drift encountered.")

        particles_t = particles
        if use_fuse:
            grad_t = fuse_grad_fn(particles_t, x_obs, cfg.lam_n, prior)  # type: ignore[misc]
            eta_t = update_eta(
                fuse_state,  # type: ignore[arg-type]
                t=step,
                particles_t=particles_t,
                particles_prev=prev_particles,
                grad_t=grad_t,
            )
            eta_t = _clamp_eta_if_needed(cfg.adaptive_step, eta_t)
        else:
            eta_t = _select_dt(cfg, step, drift)

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


def _select_dt(cfg: SpecConfig, step: int, drift: ArrayF) -> float:
    dt = cfg.dt_t(step)
    if dt <= 0.0:
        raise ValueError("dt_t(step) must be > 0.")
    if cfg.adaptive_step is None or not cfg.adaptive_step.enabled:
        return dt
    max_abs_drift = float(np.max(np.abs(drift)))
    return _adapt_dt(cfg.adaptive_step, dt, max_abs_drift)


def _adapt_dt(adapt: AdaptiveStepConfig, dt: float, max_abs_drift: float) -> float:
    step_cap = adapt.max_drift_step / (max_abs_drift + adapt.eps)
    dt_new = min(dt, step_cap)
    if dt_new < adapt.dt_min:
        dt_new = adapt.dt_min
    if dt_new > adapt.dt_max:
        dt_new = adapt.dt_max
    return dt_new


def _clamp_eta_if_needed(adapt: Optional[AdaptiveStepConfig], eta: float) -> float:
    if adapt is None or not adapt.enabled:
        return eta
    if eta < adapt.dt_min:
        return adapt.dt_min
    if eta > adapt.dt_max:
        return adapt.dt_max
    return eta
