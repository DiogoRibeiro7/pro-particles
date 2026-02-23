from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray

from pro_particles.kernels.rbf import grad_gaussian_kernel_wrt_first_arg
from pro_particles.priors.gaussian import GaussianPrior


ArrayF = NDArray[np.float64]


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def wq_log_score(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    p: int,
    leave_one_out: bool,
    logpdf: Callable[[ArrayF, ArrayF], ArrayF],
    grad_logpdf_theta: Callable[[ArrayF, ArrayF], ArrayF],
) -> ArrayF:
    """Compute W(Q)[theta] for the logarithmic score (Appendix E.1).

    W(Q)[theta] = (1/n) sum_i ∇_theta dP_theta(x_i) / ∫ dP_{theta'}(x_i) dQ(theta').
    """
    particles = _ensure_2d(particles, "particles")
    x_obs = _ensure_2d(x_obs, "x_obs")

    if particles.shape[0] != p:
        raise ValueError("p must match particles.shape[0].")
    if p < 2 and leave_one_out:
        raise ValueError("leave_one_out requires p >= 2.")
    if x_obs.shape[0] < 1:
        raise ValueError("Need at least one observation.")

    p, d = particles.shape
    n = x_obs.shape[0]

    logp = np.empty((p, n), dtype=np.float64)
    gradlogp = np.empty((p, n, d), dtype=np.float64)
    for j in range(p):
        theta_j = particles[j]
        logp[j] = logpdf(theta_j, x_obs)
        gradlogp[j] = grad_logpdf_theta(theta_j, x_obs)

    wq = np.zeros((p, d), dtype=np.float64)
    if leave_one_out:
        for j in range(p):
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            lse = np.logaddexp.reduce(logp[mask], axis=0)
            log_avg = lse - np.log(float(p - 1))
            ratio = np.exp(logp[j] - log_avg)
            wq[j] = (ratio[:, None] * gradlogp[j]).mean(axis=0)
        return wq

    lse_all = np.logaddexp.reduce(logp, axis=0)
    log_avg_all = lse_all - np.log(float(p))
    for j in range(p):
        ratio = np.exp(logp[j] - log_avg_all)
        wq[j] = (ratio[:, None] * gradlogp[j]).mean(axis=0)
    return wq


def wq_mmd2_location_gaussian(
    *,
    particles: ArrayF,  # (p, d), interpreted as Gaussian mean μ
    x_obs: ArrayF,      # (n, d)
    p: int,
    leave_one_out: bool,
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
) -> ArrayF:
    """Compute W(Q)[theta] for squared MMD, Gaussian location model (Appendix E.1)."""
    particles = _ensure_2d(particles, "particles")
    x_obs = _ensure_2d(x_obs, "x_obs")

    if particles.shape[0] != p:
        raise ValueError("p must match particles.shape[0].")
    if p < 2 and leave_one_out:
        raise ValueError("leave_one_out requires p >= 2.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if m < 2:
        raise ValueError("m must be >= 2.")
    if x_obs.shape[1] != particles.shape[1]:
        raise ValueError("x_obs second dimension must match particle dimension d.")

    p, d = particles.shape
    n = x_obs.shape[0]

    wq = np.zeros((p, d), dtype=np.float64)

    eps_j = rng.normal(size=(p, m, d))
    eps_l = rng.normal(size=(p, m, d))

    for j in range(p):
        mu_j = particles[j]
        interaction_sum = np.zeros(d, dtype=np.float64)

        for l in range(p):
            if leave_one_out and l == j:
                continue
            mu_l = particles[l]

            Y = mu_j[None, :] + sigma * eps_j[j]  # (m, d)
            Yp = mu_l[None, :] + sigma * eps_l[l] # (m, d)

            grad_over_data = np.zeros(d, dtype=np.float64)
            for i in range(n):
                x_i = x_obs[i]

                Y_exp = Y[:, None, :]
                Yp_exp = Yp[None, :, :]
                grad_k_YYp = grad_gaussian_kernel_wrt_first_arg(
                    Y_exp, Yp_exp, lengthscale
                )
                term1 = grad_k_YYp.mean(axis=(0, 1))

                grad_k_Yx = grad_gaussian_kernel_wrt_first_arg(
                    Y, x_i[None, :], lengthscale
                )
                term2 = grad_k_Yx.mean(axis=0)

                grad_over_data += (term1 - term2)

            grad_over_data /= float(n)
            interaction_sum += grad_over_data

        denom = float(p - 1) if leave_one_out else float(p)
        wq[j] = interaction_sum / denom

    return wq


def particle_system_step(
    *,
    particles: ArrayF,  # (p, d)
    x_obs: ArrayF,      # (n, xdim)
    p: int,
    dt: float,
    lam_n: float,
    sqrt2: float,
    prior: GaussianPrior,
    wq_fn: Callable[[ArrayF, ArrayF, int, bool], ArrayF],
    leave_one_out: bool,
    rng: np.random.Generator,
) -> ArrayF:
    """Single Euler–Maruyama step matching the paper's SDE."""
    particles = _ensure_2d(particles, "particles")
    x_obs = _ensure_2d(x_obs, "x_obs")

    if particles.shape[0] != p:
        raise ValueError("p must match particles.shape[0].")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if lam_n <= 0.0:
        raise ValueError("lam_n must be > 0.")
    if sqrt2 <= 0.0:
        raise ValueError("sqrt2 must be > 0.")
    if not np.all(np.isfinite(particles)):
        raise ValueError("particles contains non-finite values.")
    if not np.all(np.isfinite(x_obs)):
        raise ValueError("x_obs contains non-finite values.")

    p, d = particles.shape

    wq = wq_fn(particles=particles, x_obs=x_obs, p=p, leave_one_out=leave_one_out)
    prior_grad = np.empty((p, d), dtype=np.float64)
    for j in range(p):
        prior_grad[j] = prior.grad_log_pdf(particles[j])

    drift = -(lam_n * wq - prior_grad)
    noise = rng.normal(size=(p, d))
    updated = particles + drift * dt + (sqrt2 * np.sqrt(dt)) * noise

    if not np.all(np.isfinite(updated)):
        raise ValueError("updated particles contain non-finite values.")

    return updated


def run_particle_system(
    *,
    init_particles: ArrayF,  # (p, d)
    x_obs: ArrayF,           # (n, xdim)
    p: int,
    dt_t: Callable[[int], float],
    lam_n: float,
    sqrt2: float,
    prior: GaussianPrior,
    wq_fn: Callable[[ArrayF, ArrayF, int, bool], ArrayF],
    leave_one_out: bool,
    B: int,
    K: int,
    thinning: int,
    rng: np.random.Generator,
) -> Dict[str, ArrayF]:
    """Run the particle system for K steps and collect time-averaged samples."""
    particles = _ensure_2d(init_particles, "init_particles").copy()
    x_obs = _ensure_2d(x_obs, "x_obs")

    if particles.shape[0] != p:
        raise ValueError("p must match init_particles.shape[0].")
    if K <= 0:
        raise ValueError("K must be > 0.")
    if B < 0 or B >= K:
        raise ValueError("B must be in [0, K).")
    if thinning <= 0:
        raise ValueError("thinning must be > 0.")

    saved: list[ArrayF] = []

    for step in range(K):
        dt = dt_t(step)
        particles = particle_system_step(
            particles=particles,
            x_obs=x_obs,
            p=p,
            dt=dt,
            lam_n=lam_n,
            sqrt2=sqrt2,
            prior=prior,
            wq_fn=wq_fn,
            leave_one_out=leave_one_out,
            rng=rng,
        )

        if step >= B and ((step - B) % thinning == 0):
            saved.append(particles.copy())

    if saved:
        saved_particles = np.stack(saved, axis=0)
        time_avg_samples = saved_particles.reshape(-1, particles.shape[1])
    else:
        saved_particles = np.empty((0, p, particles.shape[1]), dtype=np.float64)
        time_avg_samples = np.empty((0, particles.shape[1]), dtype=np.float64)

    return {
        "final_particles": particles,
        "saved_particles": saved_particles,
        "time_avg_samples": time_avg_samples,
    }
