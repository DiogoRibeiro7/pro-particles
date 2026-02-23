from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pro_particles.drift.log_score import pro_drift_log_score
from pro_particles.drift.mmd2 import pro_drift_mmd2_location_gaussian
from pro_particles.priors.gaussian import GaussianPrior


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for Euler–Maruyama particle simulation."""

    n_steps: int = 20_000
    burn_in: int = 5_000
    dt: float = 1e-3
    thin: int = 10
    seed: Optional[int] = 0


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def sample_pro_posterior(
    *,
    init_particles: ArrayF,  # (p, d)
    x_obs: ArrayF,          # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    cfg: SamplerConfig,
    mode: Literal["log_score", "mmd2"],
    logpdf: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
    grad_logpdf_theta: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
    sigma: float = 1.0,
    lengthscale: float = 1.0,
    m: int = 64,
) -> Tuple[ArrayF, ArrayF]:
    """Run the interacting particle system and return samples after burn-in."""
    particles = _ensure_2d(init_particles, "init_particles").copy()
    x_obs2 = _ensure_2d(x_obs, "x_obs")

    if lam_n <= 0.0:
        raise ValueError("lam_n must be > 0.")
    if cfg.n_steps <= 0:
        raise ValueError("cfg.n_steps must be > 0.")
    if cfg.burn_in < 0 or cfg.burn_in >= cfg.n_steps:
        raise ValueError("cfg.burn_in must be in [0, n_steps).")
    if cfg.dt <= 0.0:
        raise ValueError("cfg.dt must be > 0.")
    if cfg.thin <= 0:
        raise ValueError("cfg.thin must be > 0.")
    if mode == "log_score" and (logpdf is None or grad_logpdf_theta is None):
        raise ValueError("logpdf and grad_logpdf_theta are required for mode='log_score'.")

    rng = np.random.default_rng(cfg.seed)
    p, d = particles.shape

    collected: list[ArrayF] = []

    for step in range(cfg.n_steps):
        if mode == "log_score":
            drift = pro_drift_log_score(
                particles=particles,
                x_obs=x_obs2,
                lam_n=lam_n,
                prior=prior,
                logpdf=logpdf,  # type: ignore[arg-type]
                grad_logpdf_theta=grad_logpdf_theta,  # type: ignore[arg-type]
            )
        else:
            drift = pro_drift_mmd2_location_gaussian(
                particles=particles,
                x_obs=x_obs2,
                lam_n=lam_n,
                prior=prior,
                sigma=sigma,
                lengthscale=lengthscale,
                m=m,
                rng=rng,
            )

        noise = rng.normal(size=(p, d))
        particles = particles + drift * cfg.dt + np.sqrt(2.0 * cfg.dt) * noise

        if step >= cfg.burn_in and ((step - cfg.burn_in) % cfg.thin == 0):
            collected.append(particles.copy())

    samples = np.vstack(collected) if collected else np.empty((0, d), dtype=np.float64)
    return particles, samples
