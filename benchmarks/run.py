from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler.em import SamplerConfig, sample_pro_posterior


@dataclass(frozen=True)
class BenchResult:
    name: str
    seconds: float
    samples_shape: List[int]
    notes: str


def _env_info() -> Dict[str, Any]:
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def bench_log_score_normal_location(seed: int = 0) -> BenchResult:
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.normal(0.0, 1.0, size=n)
    x_obs = x.reshape(-1, 1).astype(np.float64)

    p = 24
    d = 1
    prior = GaussianPrior(prior_var=16.0)
    init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(p, d)).astype(np.float64)

    cfg = SamplerConfig(n_steps=2000, burn_in=500, dt=1e-3, thin=10, seed=seed)
    lam_n = float(np.sqrt(n))

    start = time.perf_counter()
    _, samples = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="log_score",
        logpdf=lambda th, xx: normal_logpdf(th, xx, sigma=1.0),
        grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma=1.0),
    )
    end = time.perf_counter()
    return BenchResult(
        name="log_score_normal_location",
        seconds=end - start,
        samples_shape=list(samples.shape),
        notes="n=300, p=24, K=2000",
    )


def main() -> None:
    results = [
        bench_log_score_normal_location(seed=0),
    ]

    payload = {
        "git_commit": _git_commit(),
        "env": _env_info(),
        "results": [asdict(r) for r in results],
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmarks.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
