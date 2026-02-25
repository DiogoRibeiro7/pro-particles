"""Spec-accurate particle system implementation for the PrO posterior."""

from pro_particles.spec_impl.config import SpecConfig
from pro_particles.spec_impl.drift_logscore import drift_logscore
from pro_particles.spec_impl.drift_mmd2 import drift_mmd2
from pro_particles.spec_impl.run import run_em

__all__ = [
    "SpecConfig",
    "drift_logscore",
    "drift_mmd2",
    "run_em",
]
