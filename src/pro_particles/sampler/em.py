"""Convenience API for quick experiments. Delegates to spec_impl internally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pro_particles.kernels.rbf import grad_gaussian_kernel_wrt_first_arg
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.config import SpecConfig
from pro_particles.spec_impl.run_particle_system import run_particle_system


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

    spec_cfg = SpecConfig(
        p=p,
        dt_t=lambda _step: cfg.dt,
        B=cfg.burn_in,
        K=cfg.n_steps,
        thin=cfg.thin,
        seed=cfg.seed,
        lam_n=lam_n,
    )

    grad_L_mmd = None
    if mode == "mmd2":
        if x_obs2.shape[1] != d:
            raise ValueError("x_obs second dimension must match particle dimension d.")

        def grad_L_mmd(theta: ArrayF, theta_prime: ArrayF, x_i: ArrayF) -> ArrayF:
            eps = rng.normal(size=(m, d))
            eps2 = rng.normal(size=(m, d))
            y = theta[None, :] + sigma * eps
            y2 = theta_prime[None, :] + sigma * eps2
            grad_k_yy2 = grad_gaussian_kernel_wrt_first_arg(
                y[:, None, :], y2[None, :, :], lengthscale
            ).mean(axis=(0, 1))
            grad_k_yx = grad_gaussian_kernel_wrt_first_arg(
                y, x_i[None, :], lengthscale
            ).mean(axis=0)
            return grad_k_yy2 - grad_k_yx

    out = run_particle_system(
        init_particles=particles,
        x_obs=x_obs2,
        cfg=spec_cfg,
        prior=prior,
        rule=mode,
        use_fuse=False,
        leave_one_out=True,
        logpdf=logpdf,
        grad_logpdf_theta=grad_logpdf_theta,
        grad_L_mmd=grad_L_mmd,
        rng=rng,
    )

    return out["final_particles"], out["time_avg_samples"]
