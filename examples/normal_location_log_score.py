from __future__ import annotations

import numpy as np

from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler.em import SamplerConfig, sample_pro_posterior


def main() -> None:
    rng = np.random.default_rng(123)

    # Synthetic data: mixture (misspecified for single Gaussian mean)
    n = 500
    x = np.where(
        rng.random(n) < 0.2,
        rng.normal(-2.0, 0.5, n),
        rng.normal(2.0, 0.5, n),
    )
    x_obs = x.reshape(-1, 1).astype(np.float64)

    p = 32
    d = 1
    prior = GaussianPrior(prior_var=25.0)
    init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(p, d)).astype(np.float64)

    cfg = SamplerConfig(n_steps=10_000, burn_in=2_000, dt=5e-4, thin=10, seed=7)
    sigma_model = 0.5
    lam_n = float(np.sqrt(n))

    _, samples = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,
        cfg=cfg,
        mode="log_score",
        logpdf=lambda th, xx: normal_logpdf(th, xx, sigma_model),
        grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma_model),
    )

    print("Collected samples shape:", samples.shape)
    print("Posterior mean (approx):", float(samples.mean()))
    print("Posterior std (approx):", float(samples.std(ddof=1)))


if __name__ == "__main__":
    main()
