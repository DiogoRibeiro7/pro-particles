from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Optional


@dataclass(frozen=True)
class SpecConfig:
    """Exact-spec configuration.

    See docs/spec.md §2–§4 for the parameters and their roles.
    """

    SQRT2: ClassVar[float] = 1.4142135623730951

    p: int
    dt_t: Callable[[int], float]
    B: int
    K: int
    thin: int
    seed: Optional[int]
    lam_n: float

    def validate(self) -> None:
        """Validate config values (spec requires explicit choices).

        See docs/spec.md §2–§4.
        """
        if self.p <= 0:
            raise ValueError("p must be > 0.")
        if self.K <= 0:
            raise ValueError("K must be > 0.")
        if self.B < 0 or self.B >= self.K:
            raise ValueError("B must be in [0, K).")
        if self.thin <= 0:
            raise ValueError("thin must be > 0.")
        if self.lam_n <= 0.0:
            raise ValueError("lam_n must be > 0.")
