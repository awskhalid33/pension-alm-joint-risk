from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MortalityImprovementRW:
    """
    Random-walk mortality improvement index:

      kappa_{t+1} = kappa_t + mu + sigma * eps

    Interpretation: log m_x shifts by kappa_t.
    If mu < 0, mortality improves over time.
    """
    mu: float = -0.01     # about -1% per year in log terms
    sigma: float = 0.01   # volatility

    def simulate(
        self,
        n_years: int,
        seed: int = 123,
        kappa0: float = 0.0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        kappas = np.zeros(n_years + 1, dtype=float)
        kappas[0] = float(kappa0)

        for t in range(n_years):
            eps = rng.normal(0.0, 1.0)
            kappas[t + 1] = kappas[t] + self.mu + self.sigma * eps

        return kappas
