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

    def _ruin_reserve_index(self) -> int | None:
        if self.ruin_time is None:
            return None
        times = np.asarray(self.times, dtype=float)
        reserves = np.asarray(self.reserves, dtype=float)
        at_ruin = np.isclose(times, float(self.ruin_time)) & (reserves < 0.0)
        candidates = np.flatnonzero(at_ruin)
        if candidates.size:
            return int(candidates[0])
        candidates = np.flatnonzero(reserves < 0.0)
        if candidates.size:
            return int(candidates[0])
        return None

    @property
    def surplus_before_ruin(self) -> float | None:
        index = self._ruin_reserve_index()
        if index is None or index == 0:
            return None
        return float(self.reserves[index - 1])

    @property
    def deficit_at_ruin(self) -> float | None:
        index = self._ruin_reserve_index()
        if index is None:
            return None
        return float(max(-self.reserves[index], 0.0))

    @property
    def claim_causing_ruin(self) -> float | None:
        if self.ruin_time is None:
            return None
        claim_times = np.asarray(self.claim_times, dtype=float)
        claim_sizes = np.asarray(self.claim_sizes, dtype=float)
        candidates = np.flatnonzero(np.isclose(claim_times, float(self.ruin_time)))
        if candidates.size:
            return float(claim_sizes[candidates[-1]])
        surplus = self.surplus_before_ruin
        deficit = self.deficit_at_ruin
        if surplus is None or deficit is None:
            return None
        return float(surplus + deficit)


@dataclass(frozen=True)
class RuinEstimate:
    """Monte Carlo ruin estimate with confidence interval metadata."""

    probability: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_simulations: int
    horizon: float | None
    ruin_times: np.ndarray
    ci_method: str = "wilson"

    @property
    def expected_time_to_ruin(self) -> float:
        finite = self.ruin_times[np.isfinite(self.ruin_times)]
        if finite.size == 0:
            return float("inf")
        return float(np.mean(finite))


@dataclass(frozen=True)
class GerberShiuResult:
    """Monte Carlo estimate of a discounted penalty at ruin."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_simulations: int
    horizon: float | None
    discount_rate: float
    penalty_values: np.ndarray
    discounted_penalties: np.ndarray
    ruin_times: np.ndarray
    surplus_before_ruin: np.ndarray
    deficits_at_ruin: np.ndarray
    claim_causing_ruin: np.ndarray
    ci_method: str = "normal"

    @property
    def ruined(self) -> np.ndarray:
        return np.isfinite(self.ruin_times)

    @property
    def ruin_probability(self) -> float:
        return float(np.mean(self.ruined))

    @property
    def mean_surplus_before_ruin(self) -> float:
        values = self.surplus_before_ruin[self.ruined]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")

    @property
    def mean_deficit_at_ruin(self) -> float:
        values = self.deficits_at_ruin[self.ruined]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")
