# Roadmap

This roadmap captures planned feature development and priorities. It is a living document and will evolve as the project grows.

## Next (1–2 months)

- TBD

## Done

- Add example: Bayesian logistic regression.
- Add convergence diagnostics (ESS, R-hat-like proxies) and trajectory summaries.
- Implement adaptive step-size and stability checks for Euler–Maruyama.
- Expand documentation with a "Getting Started" tutorial and API reference pages.
- Add a higher-level `Sampler` wrapper API with clear configuration, seeding, and diagnostics.
- Improve MMD^2 support by adding kernels (RBF, Matérn) and bandwidth selection utilities.
- Add PrO drift implementations for additional scoring rules (e.g., energy score, CRPS) with reference examples.

## Near-Term (3–6 months)

- Provide GPU-friendly paths via JAX (optional backend) while keeping NumPy-first default.
- Add more examples: hierarchical Gaussian models.

## Longer-Term

- Support other interacting-particle variants (e.g., SVGD-like updates).
- Benchmark suite with reproducible performance comparisons.
- Paper-ready experiment scripts and dataset loaders.

## How to Contribute

If you'd like to propose or prioritize items, open an issue or PR with your suggestion.
