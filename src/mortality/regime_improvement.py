from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RegimeMortalityModel:
    """
    Mortality improvement with regime-dependent drift and volatility.
    """
    mu: dict        # mu[state]
    sigma: dict     # sigma[state]

    def simulate(
        self,
        regimes: np.ndarray,
        kappa0: float = 0.0,
        seed: int = 123,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        T = len(regimes) - 1
        kappa = np.zeros(T + 1)
        kappa[0] = kappa0

        for t in range(T):
            s = regimes[t]
            eps = rng.normal()
            kappa[t + 1] = kappa[t] + self.mu[s] + self.sigma[s] * eps

        return kappa
