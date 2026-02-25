from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pro_particles.experiments.data import (
    make_d1_normal_location_data,
    make_d2_palmer_penguins_data,
    make_d4_linear_regression_data,
    make_d5_binary_classification_data,
)


def test_d1_deterministic() -> None:
    a = make_d1_normal_location_data(seed=0, regime="mixture")
    b = make_d1_normal_location_data(seed=0, regime="mixture")
    assert np.array_equal(a, b)


def test_d5_deterministic() -> None:
    x1, y1 = make_d5_binary_classification_data(seed=1, n=1000)
    x2, y2 = make_d5_binary_classification_data(seed=1, n=1000)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_d4_deterministic() -> None:
    def z_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(n, 2))

    z1, y1 = make_d4_linear_regression_data(
        seed=2,
        n=500,
        sigma=1.0,
        theta=np.array([1.0, 1.0]),
        z_sampler=z_sampler,
        regime="T",
    )
    z2, y2 = make_d4_linear_regression_data(
        seed=2,
        n=500,
        sigma=1.0,
        theta=np.array([1.0, 1.0]),
        z_sampler=z_sampler,
        regime="T",
    )
    assert np.array_equal(z1, z2)
    assert np.array_equal(y1, y2)


def test_d2_optional() -> None:
    try:
        data1 = make_d2_palmer_penguins_data(seed=0)
        data2 = make_d2_palmer_penguins_data(seed=0)
    except ImportError:
        return
    assert np.array_equal(data1, data2)
