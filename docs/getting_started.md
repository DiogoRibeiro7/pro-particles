# Getting Started

This guide covers installation, a minimal run, and where to look next.

## Install

```bash
poetry install
```

## Minimal Run (Log Score)

```python
import numpy as np

from pro_particles.models.normal_location import normal_logpdf, normal_grad_logpdf_theta
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler import SamplerConfig, sample_pro_posterior

rng = np.random.default_rng(123)
x_obs = rng.normal(size=(200, 1)).astype(np.float64)

prior = GaussianPrior(prior_var=9.0)
init_particles = rng.normal(0.0, np.sqrt(prior.prior_var), size=(16, 1)).astype(np.float64)
cfg = SamplerConfig(n_steps=5000, burn_in=1000, dt=1e-3, thin=10, seed=0)

_, samples = sample_pro_posterior(
    init_particles=init_particles,
    x_obs=x_obs,
    lam_n=float(np.sqrt(x_obs.shape[0])),
    prior=prior,
    cfg=cfg,
    mode="log_score",
    logpdf=lambda th, xx: normal_logpdf(th, xx, sigma=1.0),
    grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma=1.0),
)

print(samples.shape)
```

## Minimal Run (Wrapper + Diagnostics)

```python
import numpy as np

from pro_particles.models.normal_location import normal_logpdf, normal_grad_logpdf_theta
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler import ProSampler, SamplerConfig

rng = np.random.default_rng(123)
x_obs = rng.normal(size=(200, 1)).astype(np.float64)
init_particles = rng.normal(size=(16, 1)).astype(np.float64)

sampler = ProSampler(
    mode="log_score",
    lam_n=float(np.sqrt(x_obs.shape[0])),
    prior=GaussianPrior(prior_var=9.0),
    cfg=SamplerConfig(n_steps=3000, burn_in=500, dt=1e-3, thin=10, seed=0),
    logpdf=lambda th, xx: normal_logpdf(th, xx, sigma=1.0),
    grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma=1.0),
)

result = sampler.sample(init_particles=init_particles, x_obs=x_obs)
print(result.diagnostics.ess)
```

## Where to Look Next

- `examples/normal_location_log_score.py`
- `examples/normal_location_mmd2.py`
- `examples/normal_location_energy_score.py`
- `examples/normal_location_crps.py`
- `docs/scoring_rules.md`
- `docs/spec.md`
