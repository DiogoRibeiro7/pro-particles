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
```

## Run tests

```bash
poetry run pytest -q
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pro_particles,
  title = {pro-particles: Interacting-particle sampler for Proper-Scoring-Rule posteriors},
  author = {Ribeiro, Diogo},
  year = {2026},
  url = {https://github.com/yourusername/pro-particles}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
