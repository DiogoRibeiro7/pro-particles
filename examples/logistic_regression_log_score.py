from __future__ import annotations

import numpy as np

from pro_particles.models.logistic_regression import (
    logistic_grad_logpdf_theta,
    logistic_logpdf,
)
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler.em import SamplerConfig, sample_pro_posterior


def main() -> None:
    rng = np.random.default_rng(123)

    # Synthetic logistic regression data
    n = 500
    d = 3
    x = rng.normal(size=(n, d)).astype(np.float64)
    true_beta = np.array([1.2, -0.8, 0.5], dtype=np.float64)
    logits = x @ true_beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.float64)
    x_obs = np.concatenate([x, y[:, None]], axis=1)

    p = 32
    prior = GaussianPrior(prior_var=9.0)
    init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(p, d)).astype(np.float64)

    cfg = SamplerConfig(n_steps=8_000, burn_in=2_000, dt=8e-4, thin=10, seed=7)
    lam_n = float(np.sqrt(n))

    _, samples = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="log_score",
        logpdf=logistic_logpdf,
        grad_logpdf_theta=logistic_grad_logpdf_theta,
    )

    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    print("Collected samples shape:", samples.shape)
    print("Posterior mean (approx):", mean)
    print("Posterior std (approx):", std)


if __name__ == "__main__":
    main()
