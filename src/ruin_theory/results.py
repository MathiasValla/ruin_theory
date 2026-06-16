"""Result containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SimulationPath:
    """A piecewise-linear reserve trajectory with jumps at claim times."""

    times: np.ndarray
    reserves: np.ndarray
    claim_times: np.ndarray
    claim_sizes: np.ndarray
    ruin_time: float | None
    horizon: float
    initial_capital: float
    premium_rate: float
    injection_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    injection_sizes: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def ruined(self) -> bool:
        return self.ruin_time is not None

    @property
    def terminal_reserve(self) -> float:
        return float(self.reserves[-1])

    @property
    def minimum_reserve(self) -> float:
        return float(np.min(self.reserves))


@dataclass(frozen=True)
class RuinEstimate:
    """Monte Carlo estimate with a normal-approximation confidence interval."""

    probability: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_simulations: int
    horizon: float | None
    ruin_times: np.ndarray

    @property
    def expected_time_to_ruin(self) -> float:
        finite = self.ruin_times[np.isfinite(self.ruin_times)]
        if finite.size == 0:
            return float("inf")
        return float(np.mean(finite))
