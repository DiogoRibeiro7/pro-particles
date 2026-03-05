from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def make_d1_normal_location_data(
    *,
    seed: int,
    regime: Literal["well_specified", "mixture", "claw", "heavy_tails"],
    n: int = 1000,
    sigma: float = 1.0,
) -> ArrayF:
    """D.1 Normal location illustrations (Appendix D.1, p. 63).

    Generates n=1,000 observations from:
    - well_specified: N(0, 1^2)
    - mixture: N(-2, 1^2) with prob 0.2; N(2, 1^2) with prob 0.8
    - claw: N(-2,1^2), N(0,1^2), N(2,1^2) each with prob 1/3
    - heavy_tails: Student-t_1.5

    Returns
    -------
    ArrayF
        Observations, shape (n, 1).
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")

    rng = np.random.default_rng(seed)

    if regime == "well_specified":
        y = rng.normal(loc=0.0, scale=sigma, size=n)
    elif regime == "mixture":
        mix = rng.uniform(size=n) < 0.2
        y = np.empty(n, dtype=np.float64)
        y[mix] = rng.normal(loc=-2.0, scale=sigma, size=mix.sum())
        y[~mix] = rng.normal(loc=2.0, scale=sigma, size=(~mix).sum())
    elif regime == "claw":
        u = rng.uniform(size=n)
        y = np.empty(n, dtype=np.float64)
        y[u < 1.0 / 3.0] = rng.normal(loc=-2.0, scale=sigma, size=(u < 1.0 / 3.0).sum())
        mid = (u >= 1.0 / 3.0) & (u < 2.0 / 3.0)
        y[mid] = rng.normal(loc=0.0, scale=sigma, size=mid.sum())
        y[u >= 2.0 / 3.0] = rng.normal(loc=2.0, scale=sigma, size=(u >= 2.0 / 3.0).sum())
    elif regime == "heavy_tails":
        y = rng.standard_t(df=1.5, size=n) * sigma
    else:
        raise ValueError(f"Unknown regime: {regime}.")

    return y.reshape(-1, 1).astype(np.float64, copy=False)


def make_d2_palmer_penguins_data(
    *,
    seed: int,
) -> ArrayF:
    """D.2 Palmer penguins (Appendix D.2, p. 64).

    Loads bill length/depth (mm), centers to zero mean and scales to unit variance jointly.

    Returns
    -------
    ArrayF
        Standardized data, shape (n, 2).
    """
    _ = seed  # deterministic; seed included for API consistency
    from pro_particles.experiments.datasets import load_palmer_penguins

    data = load_palmer_penguins()

    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, ddof=0, keepdims=True)
    if np.any(std == 0.0):
        raise ValueError("Zero variance in penguin features.")

    data = (data - mean) / std
    return data


def make_d4_linear_regression_data(
    *,
    seed: int,
    n: int,
    sigma: float,
    theta: ArrayF,
    z_sampler: Callable[[np.random.Generator, int], ArrayF],
    regime: Literal["T", "CR", "NT"],
) -> Tuple[ArrayF, ArrayF]:
    """Section 2 / Appendix D.4 synthetic linear regression (pp. 9–11, 67–68).

    Model: x_i = z_i^T theta + eps_i.
    Misspecification regimes:
    - T: eps_i ~ t_3(0, sigma^2)
    - CR: random coefficients: theta2 = 2 * (-1)^{xi_i}, xi_i ~ Ber(1/2)
    - NT: combined T + CR

    Returns
    -------
    Tuple[ArrayF, ArrayF]
        Covariates z (n, 2) and responses y (n, 1).
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if theta.ndim != 1 or theta.shape[0] != 2:
        raise ValueError("theta must be shape (2,) for Section 2 model.")

    rng = np.random.default_rng(seed)
    z = z_sampler(rng, n).astype(np.float64, copy=False)
    if z.shape != (n, 2):
        raise ValueError("z_sampler must return shape (n, 2).")

    theta_use = theta.copy()
    if regime in ("CR", "NT"):
        xi = rng.integers(low=0, high=2, size=n)
        theta2 = 2.0 * ((-1.0) ** xi)
        x = z[:, 0] * theta_use[0] + z[:, 1] * theta2
    else:
        x = z @ theta_use

    if regime in ("T", "NT"):
        eps = rng.standard_t(df=3.0, size=n) * sigma
    else:
        eps = rng.normal(loc=0.0, scale=sigma, size=n)

    y = x + eps
    return z, y.reshape(-1, 1).astype(np.float64, copy=False)


def make_d5_binary_classification_data(
    *,
    seed: int,
    n: int = 1000,
) -> Tuple[ArrayF, ArrayF]:
    """Section 5.1 binary classification (p. 26).

    Covariates uniformly on [-2,2]^2. Labels:
    - top-left (x0 < 0, x1 > 0): y = 0
    - bottom-left (x0 > 0, x1 < 0): y = 1
    - other quadrants: y ~ Ber(0.5)

    Returns
    -------
    Tuple[ArrayF, ArrayF]
        Covariates x (n, 2) and labels y (n, 1).
    """
    if n <= 0:
        raise ValueError("n must be > 0.")

    rng = np.random.default_rng(seed)
    x = rng.uniform(low=-2.0, high=2.0, size=(n, 2)).astype(np.float64)

    y = np.empty(n, dtype=np.float64)
    top_left = (x[:, 0] < 0.0) & (x[:, 1] > 0.0)
    bottom_left = (x[:, 0] > 0.0) & (x[:, 1] < 0.0)

    y[top_left] = 0.0
    y[bottom_left] = 1.0

    remaining = ~(top_left | bottom_left)
    y[remaining] = rng.integers(low=0, high=2, size=remaining.sum())

    return x, y.reshape(-1, 1)
