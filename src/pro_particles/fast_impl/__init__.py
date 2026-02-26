"""Vectorized (paper-faithful) implementations for speed."""

from pro_particles.fast_impl.mmd2 import (
    drift_mmd2_gaussian_location_fast,
    drift_mmd2_linear_regression_fast,
    wq_mmd2_gaussian_location_fast,
    wq_mmd2_linear_regression_fast,
)
from pro_particles.fast_impl.run_mmd2 import run_particle_system_mmd2_fast

__all__ = [
    "wq_mmd2_gaussian_location_fast",
    "drift_mmd2_gaussian_location_fast",
    "wq_mmd2_linear_regression_fast",
    "drift_mmd2_linear_regression_fast",
    "run_particle_system_mmd2_fast",
]
