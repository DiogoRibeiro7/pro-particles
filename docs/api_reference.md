# API Reference (Summary)

This is a brief, stable summary of the public APIs used in examples and tests.

## Sampler

### `sample_pro_posterior`

Module: `pro_particles.sampler.em`

Key parameters:
- `init_particles`: `(p, d)` array
- `x_obs`: `(n, xdim)` array
- `lam_n`: positive float
- `prior`: `GaussianPrior`
- `cfg`: `SamplerConfig`
- `mode`: `"log_score"`, `"mmd2"`, `"energy_score"`, `"crps"`
- `logpdf`, `grad_logpdf_theta`: required for `mode="log_score"`
- `sigma`, `lengthscale`, `m`, `kernel`: used for `"mmd2"`, `"energy_score"`, `"crps"`

Returns:
- `final_particles`: `(p, d)`
- `time_avg_samples`: `(S, d)`

### `ProSampler`

Module: `pro_particles.sampler.wrapper`

High-level wrapper that runs the particle system and returns diagnostics.

Returns `SamplerResult`:
- `final_particles`
- `saved_particles`
- `time_avg_samples`
- `diagnostics`

## Kernels

Module: `pro_particles.kernels`

- `gaussian_kernel`, `grad_gaussian_kernel_wrt_first_arg`
- `matern_kernel`, `grad_matern_kernel_wrt_first_arg`

Bandwidth utilities:
- `median_heuristic_lengthscale`
- `scott_bandwidth`
- `silverman_bandwidth`

## Diagnostics

Module: `pro_particles.diagnostics.convergence`

- `trace_plot_data`
- `running_mean`
- `effective_sample_size`
- `split_rhat`
- `trajectory_summary`

## Algorithms

Module: `pro_particles.algorithms`

- `SVGDConfig`
- `run_svgd`
- `svgd_update`

## Drift (Spec Implementations)

Module: `pro_particles.spec_impl`

- `drift_logscore`
- `drift_mmd2`
- `drift_energy_score`
- `drift_crps`

## Priors

Module: `pro_particles.priors.gaussian`

- `GaussianPrior`
