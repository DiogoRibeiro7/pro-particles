from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianPrior:
    """Isotropic Gaussian prior: Pi(theta) = N(0, prior_var I).

    Notes
    -----
    For the particle SDE drift we use the score of the prior:
        ∇_theta log dPi(theta) = -theta / prior_var
    """

    prior_var: float = 10.0

    def grad_log_pdf(self, theta: ArrayF) -> ArrayF:
        """Compute ∇_theta log dPi(theta) for an isotropic Gaussian prior."""
        if self.prior_var <= 0.0:
            raise ValueError("prior_var must be > 0.")
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D (d,), got shape {theta.shape}.")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains non-finite values.")
        return -theta / self.prior_var
