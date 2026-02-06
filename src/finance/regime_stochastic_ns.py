from __future__ import annotations

import numpy as np
from src.finance.stochastic_ns import StochasticNelsonSiegelModel


class RegimeStochasticNS:
    """
    Nelson–Siegel model with regime-dependent VAR parameters.
    """

    def __init__(self, models: dict):
        self.models = models

    def simulate(
        self,
        regimes: np.ndarray,
        x0: np.ndarray,
        seed: int = 123,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        T = len(regimes) - 1
        x_path = np.zeros((T + 1, 3))
        x_path[0] = x0

        for t in range(T):
            model = self.models[regimes[t]]
            eps = rng.multivariate_normal(
                mean=np.zeros(3),
                cov=model.Sigma,
            )
            x_path[t + 1] = model.A @ x_path[t] + eps

        return x_path
