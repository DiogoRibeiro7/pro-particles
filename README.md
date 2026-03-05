# pro-particles

NumPy-first implementation of an interacting-particle sampler for Proper-Scoring-Rule (PrO) posteriors.

## What is implemented

- Interacting particle system (leave-one-out interaction).
- Euler–Maruyama discretisation.
- Burn-in + thinning + time-averaged empirical sampling.
- PrO drift for:
  - Log-score (generic: plug in `logpdf` + `grad_logpdf_theta`).
  - MMD² (example implementation for Gaussian location model).

## Install (Poetry)

```bash
poetry install
```

## Run examples

```bash
poetry run python examples/normal_location_log_score.py
poetry run python examples/normal_location_mmd2.py
poetry run python examples/normal_location_energy_score.py
poetry run python examples/normal_location_crps.py
poetry run python examples/logistic_regression_log_score.py
poetry run python examples/hierarchical_gaussian_log_score.py
poetry run python examples/svgd_logistic_regression.py
```

## Run tests

```bash
poetry run pytest -q
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Documentation

- Getting started: `docs/getting_started.md`
- API reference (summary): `docs/api_reference.md`

## Benchmarks

```bash
python benchmarks/run.py
```

## Paper-Ready Experiments

```bash
python experiments/paper/run_all_paper.py
```

## Sampler API

The higher-level sampler wrapper provides configuration, seeding, and diagnostics:

```python
from pro_particles.models.normal_location import normal_logpdf, normal_grad_logpdf_theta
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.sampler import ProSampler, SamplerConfig

prior = GaussianPrior(prior_var=9.0)
cfg = SamplerConfig(n_steps=1000, burn_in=200, dt=1e-3, thin=10, seed=0)

sampler = ProSampler(
    mode="log_score",
    lam_n=2.0,
    prior=prior,
    cfg=cfg,
    logpdf=lambda th, xx: normal_logpdf(th, xx, sigma=1.0),
    grad_logpdf_theta=lambda th, xx: normal_grad_logpdf_theta(th, xx, sigma=1.0),
)

result = sampler.sample(init_particles=init_particles, x_obs=x_obs)
ess = result.diagnostics.ess
```

Diagnostics now also include split R-hat and trajectory summaries:

```python
rhat = result.diagnostics.rhat
summary = result.diagnostics.trajectory
```

### Adaptive Step Size

Enable adaptive step-size in the sampler config:

```python
cfg = SamplerConfig(
    n_steps=3000,
    burn_in=500,
    dt=1e-3,
    thin=10,
    seed=0,
    adaptive_step=True,
    max_drift_step=0.5,
    dt_min=1e-6,
    dt_max=1e-1,
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pro_particles,
  title = {pro-particles: Interacting-particle sampler for Proper-Scoring-Rule posteriors},
  author = {Ribeiro, Diogo},
  year = {2026},
  url = {https://github.com/diogoribeiro7/pro-particles}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
