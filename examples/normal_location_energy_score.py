from __future__ import annotations

import numpy as np

from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler.em import SamplerConfig, sample_pro_posterior


def main() -> None:
    rng = np.random.default_rng(123)

    # Data in R^1 (misspecified for single Gaussian mean)
    n = 250
    x = np.where(rng.random(n) < 0.4, rng.normal(-1.0, 0.6, n), rng.normal(1.4, 0.7, n))
    x_obs = x.reshape(-1, 1).astype(np.float64)

    p = 24
    d = 1
    prior = GaussianPrior(prior_var=16.0)
    init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(p, d)).astype(np.float64)

    cfg = SamplerConfig(n_steps=6_000, burn_in=1_500, dt=8e-4, thin=10, seed=21)

    # Energy score-specific hyperparameters (example values)
    lam_n = float(np.sqrt(n))
    sigma = 0.6
    m = 48

    _, samples = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="energy_score",
        sigma=sigma,
        m=m,
    )

    print("Collected samples shape:", samples.shape)
    print("Posterior mean (approx):", float(samples.mean()))
    print("Posterior std (approx):", float(samples.std(ddof=1)))


if __name__ == "__main__":
    main()
