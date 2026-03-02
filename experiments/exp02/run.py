from __future__ import annotations

# ruff: noqa: E402

import json
from pathlib import Path
import sys

# Path setup for standalone execution (not needed under `poetry run`)
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

import numpy as np

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl import SpecConfig, run_particle_system
from pro_particles.fast_impl.run_mmd2 import run_particle_system_mmd2_fast
from pro_particles.metrics.metrics import (
    crps_ensemble,
    elpd_from_lppd,
    lppd_from_logpdf,
    mmd2_to_dirac,
)
from pro_particles.experiments.data import make_d4_linear_regression_data
from pro_particles.schedules.fuse import FuseState

from experiments.utils import (
    env_info,
    git_commit,
    is_dirty,
    median_heuristic_lengthscale,
    save_json,
    sha256_array,
    sha256_file,
    utc_now,
)


def grad_L_mmd_linear_regression(
    theta: np.ndarray,
    theta2: np.ndarray,
    x: np.ndarray,
    y: float,
    *,
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    eps1 = rng.normal(size=m)
    eps2 = rng.normal(size=m)
    y1 = x @ theta + sigma * eps1
    y2 = x @ theta2 + sigma * eps2

    k_y1y2 = np.exp(-0.5 * (y1[:, None] - y2[None, :]) ** 2 / (lengthscale**2))
    dkyy = (k_y1y2 * (y2[None, :] - y1[:, None]) / (lengthscale**2)).mean()

    k_y1y = np.exp(-0.5 * (y1 - y) ** 2 / (lengthscale**2))
    dky = (k_y1y * (y - y1) / (lengthscale**2)).mean()

    return (dkyy - dky) * x


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    cfg_path = exp_dir / "config.json"
    if len(sys.argv) > 1:
        cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())

    seed = int(cfg["data"]["seed"])
    n = int(cfg["data"]["n"])
    n_test = int(cfg["data"]["n_test"])
    sigma = float(cfg["data"]["sigma"])
    theta = np.array(cfg["data"]["theta"], dtype=np.float64)

    rng = np.random.default_rng(seed)

    def z_sampler(rng_: np.random.Generator, n_: int) -> np.ndarray:
        return rng_.normal(size=(n_, 2))

    z_train, y_train = make_d4_linear_regression_data(
        seed=seed,
        n=n,
        sigma=sigma,
        theta=theta,
        z_sampler=z_sampler,
        regime=cfg["data"]["regime"],
    )
    z_test, y_test = make_d4_linear_regression_data(
        seed=seed + 1,
        n=n_test,
        sigma=sigma,
        theta=theta,
        z_sampler=z_sampler,
        regime=cfg["data"]["regime"],
    )

    lengthscale = median_heuristic_lengthscale(y_train.reshape(-1, 1))

    x_obs = np.hstack([z_train, y_train])  # pack (x_i, y_i)

    lam_n = float(np.sqrt(n))
    init_particles = rng.normal(size=(cfg["method"]["p"], 2)).astype(np.float64)

    spec_cfg = SpecConfig(
        p=cfg["method"]["p"],
        dt_t=lambda step: float(cfg["method"].get("dt", 1.0)),
        B=cfg["method"]["B"],
        K=cfg["method"]["K"],
        thin=cfg["method"]["thin"],
        seed=seed,
        lam_n=lam_n,
    )

    fuse_state = FuseState(r_eps=cfg["method"]["r_eps"]) if cfg["method"]["use_fuse"] else None

    def grad_L(theta1, theta2, x):
        x_feat = x[:2]
        y_val = float(x[2])
        return grad_L_mmd_linear_regression(
            theta1,
            theta2,
            x_feat,
            y_val,
            sigma=sigma,
            lengthscale=lengthscale,
            m=cfg["method"]["m_mmd"],
            rng=rng,
        )

    def fuse_grad_fn(particles, x_obs, lam_n_val, prior):
        p = particles.shape[0]
        n_obs = x_obs.shape[0]
        wq = np.zeros_like(particles)
        for j in range(p):
            acc = np.zeros((n_obs, particles.shape[1]), dtype=np.float64)
            for ell in range(p):
                if ell == j:
                    continue
                for i in range(n_obs):
                    acc[i] += grad_L(particles[j], particles[ell], x_obs[i])
            acc /= float(p - 1)
            wq[j] = acc.mean(axis=0)
        prior_grad = np.array([prior.grad_log_pdf(theta_) for theta_ in particles])
        return lam_n_val * wq - prior_grad

    if cfg["method"].get("impl", "fast") == "fast":
        out = run_particle_system_mmd2_fast(
            init_particles=init_particles,
            x_obs=z_train,
            y_obs=y_train[:, 0],
            cfg=spec_cfg,
            prior=GaussianPrior(prior_var=10.0),
            sigma=sigma,
            lengthscale=lengthscale,
            m=cfg["method"]["m_mmd"],
            use_fuse=cfg["method"]["use_fuse"],
            leave_one_out=True,
            fuse_state=fuse_state,
            rng=rng,
            model="linear_regression",
        )
    else:
        out = run_particle_system(
            init_particles=init_particles,
            x_obs=x_obs,
            cfg=spec_cfg,
            prior=GaussianPrior(prior_var=10.0),
            rule="mmd2",
            use_fuse=cfg["method"]["use_fuse"],
            leave_one_out=True,
            grad_L_mmd=grad_L,
            fuse_state=fuse_state,
            fuse_grad_fn=fuse_grad_fn if cfg["method"]["use_fuse"] else None,
            rng=rng,
        )

    samples = out["time_avg_samples"]
    samples_path = exp_dir / "samples_method.npz"
    np.savez(samples_path, samples=samples)

    # Predictive log scores on test data
    logpdf = np.empty((samples.shape[0], n_test), dtype=np.float64)
    mu_mat = (z_test @ samples.T).T  # (S, n_test)
    for s in range(samples.shape[0]):
        mu = mu_mat[s]
        logpdf[s] = -0.5 * ((y_test[:, 0] - mu) ** 2) / (sigma**2) - np.log(sigma) - 0.5 * np.log(
            2.0 * np.pi
        )

    lppd = lppd_from_logpdf(logpdf)
    elpd = elpd_from_lppd(lppd)

    # MMD2 to dirac for each test point using predictive samples
    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = a.reshape(a.shape[0], -1)
        b2 = b.reshape(b.shape[0], -1)
        diff = a2[:, None, :] - b2[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        return np.exp(-0.5 * dist2 / (lengthscale**2))

    eps = rng.normal(size=mu_mat.shape)
    pred_samples = mu_mat + sigma * eps  # (S, n_test)

    mmd2_vals = []
    for i in range(n_test):
        y_samples = pred_samples[:, i]
        mmd2_vals.append(
            mmd2_to_dirac(samples=y_samples.reshape(-1, 1), x=np.array([y_test[i, 0]]), kernel=kernel)
        )

    crps_vals = crps_ensemble(
        y_true=y_test[:, 0],
        samples=pred_samples,
    )

    result = {
        "schema_version": "1.0.0",
        "paper_id": cfg["paper_id"],
        "exp_id": cfg["exp_id"],
        "exp_name": cfg["exp_name"],
        "created_at_utc": utc_now(),
        "code": {"repo": "pro-particles", "git_commit": git_commit(), "dirty": is_dirty()},
        "environment": env_info(),
        "data": {
            "generator": "make_d4_linear_regression_data",
            "seed": seed,
            "n": n,
            "d_x": 2,
            "notes": cfg["data"]["regime"],
            "params": {"sigma": sigma, "theta": cfg["data"]["theta"], "z_sampler": cfg["data"]["z_sampler"]},
            "hash": sha256_array(z_train),
        },
        "configs": {
            "method": {
                "name": "pro_particles",
                "rule": cfg["method"]["rule"],
                "lam_n": {"type": "paper_default", "value": lam_n},
                "prior": {"type": "gaussian_isotropic", "prior_var": 10.0},
                "schedule": {"type": "fuse", "params": {"r_eps": cfg["method"]["r_eps"]}},
                "sampler": {
                    "p": cfg["method"]["p"],
                    "burn_in": cfg["method"]["B"],
                    "n_steps": cfg["method"]["K"],
                    "thin": cfg["method"]["thin"],
                },
                "model": {"family": "linear_regression", "sigma": sigma},
            },
            "baselines": [],
        },
        "runs": [
            {
                "run_id": "method",
                "kind": "method",
                "name": "pro_particles",
                "seed": seed,
                "status": "ok",
                "timing": {},
                "samples": {
                    "shape": list(samples.shape),
                    "n_saved": int(samples.shape[0]),
                    "path": str(samples_path.name),
                    "hash": sha256_file(samples_path),
                },
                "diagnostics": {},
                "metrics": {
                    "elpd": elpd,
                    "mmd2_test_mean": float(np.mean(mmd2_vals)),
                    "crps_mean": float(np.mean(crps_vals)),
                },
            }
        ],
        "summary": {},
    }

    save_json(exp_dir / "results.json", result)


if __name__ == "__main__":
    main()
