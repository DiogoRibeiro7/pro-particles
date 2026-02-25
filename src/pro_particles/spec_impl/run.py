from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.config import SpecConfig
from pro_particles.schedules.fuse import FuseState, update_eta


ArrayF = NDArray[np.float64]


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def run_em(
    *,
    init_particles: ArrayF,  # (p, d)
    x_obs: ArrayF,           # (n, xdim)
    cfg: SpecConfig,
    prior: GaussianPrior,
    drift_fn: Callable[[ArrayF, ArrayF, float, GaussianPrior], ArrayF],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Euler–Maruyama loop from docs/spec.md §2–§3.

    - Drift sign and sqrt(2) factor match the SDE in docs/spec.md §2.
    - Time-averaging / burn-in follows docs/spec.md §3.
    """
    cfg.validate()

    particles = _ensure_2d(init_particles, "init_particles").copy()
    x_obs = _ensure_2d(x_obs, "x_obs")

    if particles.shape[0] != cfg.p:
        raise ValueError("cfg.p must match init_particles.shape[0].")

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    saved: list[ArrayF] = []

    for step in range(cfg.K):
        dt = cfg.dt_t(step)
        if dt <= 0.0:
            raise ValueError("dt_t(step) must be > 0.")

        drift = drift_fn(particles, x_obs, cfg.lam_n, prior)
        if not np.all(np.isfinite(drift)):
            raise ValueError("Non-finite drift encountered.")

        noise = rng.normal(size=particles.shape)
        particles = particles + drift * dt + (cfg.sqrt2 * np.sqrt(dt)) * noise

        if not np.all(np.isfinite(particles)):
            raise ValueError("Non-finite particles encountered.")

        if step >= cfg.B and ((step - cfg.B) % cfg.thin == 0):
            saved.append(particles.copy())

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


def run_em_fuse(
    *,
    init_particles: ArrayF,  # (p, d)
    x_obs: ArrayF,           # (n, xdim)
    cfg: SpecConfig,
    prior: GaussianPrior,
    drift_and_grad_fn: Callable[[ArrayF, ArrayF, float, GaussianPrior], Tuple[ArrayF, ArrayF]],
    fuse_state: FuseState,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Euler–Maruyama loop with Fuse schedule (docs/fuse_spec.md, Algorithm 1).

    - Step size updated per Fuse Algorithm 1 (docs/fuse_spec.md, Eq. 180).
    - Half-step and noise step match Eq. 181–182.
    """
    cfg.validate()

    particles = _ensure_2d(init_particles, "init_particles").copy()
    x_obs = _ensure_2d(x_obs, "x_obs")

    if particles.shape[0] != cfg.p:
        raise ValueError("cfg.p must match init_particles.shape[0].")

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    saved: list[ArrayF] = []
    prev_particles: Optional[ArrayF] = None

    for step in range(cfg.K):
        particles_before = particles.copy()
        drift, grad_t = drift_and_grad_fn(particles, x_obs, cfg.lam_n, prior)
        if not np.all(np.isfinite(drift)):
            raise ValueError("Non-finite drift encountered.")

        eta_t = update_eta(
            fuse_state,
            t=step,
            particles_t=particles,
            particles_prev=prev_particles,
            grad_t=grad_t,
        )

        noise = rng.normal(size=particles.shape)
        particles = particles + drift * eta_t + (cfg.sqrt2 * np.sqrt(eta_t)) * noise

        if not np.all(np.isfinite(particles)):
            raise ValueError("Non-finite particles encountered.")

        if step == 0 and fuse_state.x1 is None:
            fuse_state.x1 = particles.copy()

        if step >= cfg.B and ((step - cfg.B) % cfg.thin == 0):
            saved.append(particles.copy())

        prev_particles = particles_before

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
