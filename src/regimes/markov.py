from __future__ import annotations

import numpy as np


class MarkovRegime:
    """
    Discrete-time Markov regime process.
    """

    def __init__(self, transition_matrix: np.ndarray):
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")
        self.P = transition_matrix
        self.n_states = transition_matrix.shape[0]

    def simulate(self, n_steps: int, s0: int, seed: int = 123) -> np.ndarray:
        rng = np.random.default_rng(seed)
        states = np.zeros(n_steps + 1, dtype=int)
        states[0] = s0

        for t in range(n_steps):
            states[t + 1] = rng.choice(
                self.n_states,
                p=self.P[states[t]],
            )
        return states
