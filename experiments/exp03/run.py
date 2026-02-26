from __future__ import annotations

import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

import numpy as np

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl import SpecConfig, run_particle_system
from pro_particles.metrics.metrics import elpd_from_lppd, lppd_from_logpdf
from pro_particles.experiments.data import make_d5_binary_classification_data
from pro_particles.schedules.fuse import FuseState

from experiments.utils import (
    env_info,
    git_commit,
    is_dirty,
    save_json,
    sha256_array,
    sha256_file,
    utc_now,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    cfg_path = exp_dir / "config.json"
    if len(sys.argv) > 1:
        cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())

    seed = int(cfg["data"]["seed"])
    n = int(cfg["data"]["n"])
    rng = np.random.default_rng(seed)

    x, y = make_d5_binary_classification_data(seed=seed, n=n)

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
        sqrt2=np.sqrt(2.0),
    )

    fuse_state = FuseState(r_eps=cfg["method"]["r_eps"]) if cfg["method"]["use_fuse"] else None

    def logpdf(theta, x_obs):
        logits = x_obs @ theta
        p = sigmoid(logits)
        return np.log(p) * y[:, 0] + np.log(1.0 - p) * (1.0 - y[:, 0])

    def grad_logpdf_theta(theta, x_obs):
        logits = x_obs @ theta
        p = sigmoid(logits)
        grad = (y[:, 0] - p)[:, None] * x_obs
        return grad

    def fuse_grad_fn(particles, x_obs, lam_n_val, prior):
        # gradU = lam_n * W(Q) - grad log prior; use log-score W(Q)
        p = particles.shape[0]
        n_obs = x_obs.shape[0]
        logp = np.empty((p, n_obs), dtype=np.float64)
        gradlogp = np.empty((p, n_obs, 2), dtype=np.float64)
        for j in range(p):
            logp[j] = logpdf(particles[j], x_obs)
            gradlogp[j] = grad_logpdf_theta(particles[j], x_obs)
        dP = np.exp(logp)
        grad_dP = dP[:, :, None] * gradlogp
        wq = np.zeros((p, 2), dtype=np.float64)
        for j in range(p):
            denom = dP[np.arange(p) != j].mean(axis=0)
            wq[j] = (grad_dP[j] / denom[:, None]).mean(axis=0)
        prior_grad = np.array([prior.grad_log_pdf(theta) for theta in particles])
        return lam_n_val * wq - prior_grad

    out = run_particle_system(
        init_particles=init_particles,
        x_obs=x,
        cfg=spec_cfg,
        prior=GaussianPrior(prior_var=10.0),
        rule="log_score",
        use_fuse=cfg["method"]["use_fuse"],
        leave_one_out=True,
        logpdf=logpdf,
        grad_logpdf_theta=grad_logpdf_theta,
        fuse_state=fuse_state,
        fuse_grad_fn=fuse_grad_fn if cfg["method"]["use_fuse"] else None,
        rng=rng,
    )

    samples = out["time_avg_samples"]
    samples_path = exp_dir / "samples_method.npz"
    np.savez(samples_path, samples=samples)

    logpdf_vals = np.vstack([logpdf(theta, x) for theta in samples])
    lppd = lppd_from_logpdf(logpdf_vals)
    elpd = elpd_from_lppd(lppd)

    result = {
        "schema_version": "1.0.0",
        "paper_id": cfg["paper_id"],
        "exp_id": cfg["exp_id"],
        "exp_name": cfg["exp_name"],
        "created_at_utc": utc_now(),
        "code": {"repo": "pro-particles", "git_commit": git_commit(), "dirty": is_dirty()},
        "environment": env_info(),
        "data": {
            "generator": "make_d5_binary_classification_data",
            "seed": seed,
            "n": n,
            "d_x": 2,
            "notes": "quadrant rule",
            "params": {},
            "hash": sha256_array(x),
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
                "model": {"family": "logistic_regression"},
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
                },
            }
        ],
        "summary": {},
    }

    save_json(exp_dir / "results.json", result)


if __name__ == "__main__":
    main()
