from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NSFactors:
    level: float
    slope: float
    curvature: float


@dataclass(frozen=True)
class StochasticNelsonSiegelModel:
    """
    Stochastic Nelson–Siegel in factor form.

    Factors x_t = [level, slope, curvature]^T evolve as VAR(1):
      x_{t+1} = A x_t + eps_{t+1},  eps ~ N(0, Sigma)

    Mapping to zero rates uses standard NS loadings with fixed tau.
    """
    A: np.ndarray              # 3x3
    Sigma: np.ndarray          # 3x3 covariance
    tau: float = 2.5
    dt_years: float = 0.25     # quarterly steps

    def zero_rate(self, factors: NSFactors, ttm: float) -> float:
        """
        Zero rate for time-to-maturity (ttm) in years, continuous compounding.
        """
        ttm = float(ttm)
        if ttm <= 0.0:
            # limit t->0: level + slope
            return float(factors.level + factors.slope)

        x = ttm / float(self.tau)
        a = (1.0 - np.exp(-x)) / x
        b = a - np.exp(-x)

        z = factors.level + factors.slope * a + factors.curvature * b
        return float(z)

    def discount_factor(self, factors: NSFactors, ttm: float) -> float:
        z = self.zero_rate(factors, ttm)
        return float(np.exp(-z * float(ttm)))

    def step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        One VAR(1) step.
        """
        eps = rng.multivariate_normal(mean=np.zeros(3), cov=self.Sigma)
        x_next = self.A @ x + eps
        return x_next

    def simulate_factors(
        self,
        x0: np.ndarray,
        n_steps: int,
        seed: int = 123,
    ) -> np.ndarray:
        """
        Returns array shape (n_steps+1, 3)
        """
        rng = np.random.default_rng(seed)
        x_path = np.zeros((n_steps + 1, 3), dtype=float)
        x_path[0] = x0

        for t in range(n_steps):
            x_path[t + 1] = self.step(x_path[t], rng)

        return x_path

    @staticmethod
    def factors_from_array(x: np.ndarray) -> NSFactors:
        return NSFactors(level=float(x[0]), slope=float(x[1]), curvature=float(x[2]))
