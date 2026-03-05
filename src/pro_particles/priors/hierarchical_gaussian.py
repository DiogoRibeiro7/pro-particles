from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class HierarchicalGaussianPrior:
    """Gaussian prior for hierarchical mean model.

    Parameters
    ----------
    n_groups:
        Number of group means in the parameter vector.
    sigma_alpha:
        Std dev for global mean alpha.
    sigma_group:
        Std dev for group means around alpha.

    Parameter vector:
        theta = [alpha, mu_0, ..., mu_{G-1}]
    """

    n_groups: int
    sigma_alpha: float = 2.0
    sigma_group: float = 1.0

    def grad_log_pdf(self, theta: ArrayF) -> ArrayF:
        if self.n_groups <= 0:
            raise ValueError("n_groups must be > 0.")
        if self.sigma_alpha <= 0.0:
            raise ValueError("sigma_alpha must be > 0.")
        if self.sigma_group <= 0.0:
            raise ValueError("sigma_group must be > 0.")
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D (d,), got shape {theta.shape}.")
        if theta.shape[0] != self.n_groups + 1:
            raise ValueError("theta must have length n_groups + 1.")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains non-finite values.")

        alpha = float(theta[0])
        mus = theta[1:]

        grad = np.zeros_like(theta, dtype=np.float64)
        grad[0] = -alpha / (self.sigma_alpha * self.sigma_alpha)
        grad[0] += np.sum(mus - alpha) / (self.sigma_group * self.sigma_group)
        grad[1:] = -(mus - alpha) / (self.sigma_group * self.sigma_group)
        return grad
