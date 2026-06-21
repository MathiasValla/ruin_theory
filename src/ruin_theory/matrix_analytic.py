"""Matrix-analytic helpers for phase-type renewal risk models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .distributions import ClaimDistribution, phase_type


@dataclass(frozen=True)
class PhaseTypeRenewalCountResult:
    """Finite-horizon count law for phase-type renewal arrivals."""

    horizon: float
    probabilities: np.ndarray
    tail_probability: float
    max_count: int
    interarrival_distribution: ClaimDistribution

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.probabilities) + self.tail_probability)

    @property
    def expected_count_lower_bound(self) -> float:
        counts = np.arange(self.probabilities.size, dtype=float)
        return float(np.dot(counts, self.probabilities))


def phase_type_convolution(
    interarrival_distribution: ClaimDistribution,
    count: int,
) -> ClaimDistribution:
    """Return the PH law of a sum of `count` iid PH interarrival times."""

    if interarrival_distribution.name != "phase_type":
        raise ValueError("interarrival_distribution must be phase_type")
    n = int(count)
    if n != count or n <= 0:
        raise ValueError("count must be a positive integer")
    initial = np.asarray(
        interarrival_distribution.metadata["initial_probabilities"],
        dtype=float,
    )
    matrix = np.asarray(interarrival_distribution.metadata["subgenerator"], dtype=float)
    exits = np.asarray(interarrival_distribution.metadata["exit_rates"], dtype=float)
    phases = initial.size
    dimension = n * phases
    block = np.zeros((dimension, dimension), dtype=float)
    for index in range(n):
        start = index * phases
        end = start + phases
        block[start:end, start:end] = matrix
        if index < n - 1:
            next_start = end
            next_end = next_start + phases
            block[start:end, next_start:next_end] = np.outer(exits, initial)
    alpha = np.zeros(dimension, dtype=float)
    alpha[:phases] = initial
    return phase_type(alpha, block)


def phase_type_renewal_count_pmf(
    interarrival_distribution: ClaimDistribution,
    horizon: float,
    *,
    max_count: int,
) -> PhaseTypeRenewalCountResult:
    """Compute `P(N(t)=n)` for a Sparre-Andersen process with PH waits."""

    if interarrival_distribution.name != "phase_type":
        raise ValueError("interarrival_distribution must be phase_type")
    t = _positive_float(horizon, "horizon")
    maximum = _nonnegative_int(max_count, "max_count")
    cdfs = np.zeros(maximum + 2, dtype=float)
    cdfs[0] = 1.0
    for count in range(1, maximum + 2):
        cdfs[count] = float(phase_type_convolution(interarrival_distribution, count).cdf(t))
    probabilities = np.maximum(cdfs[: maximum + 1] - cdfs[1 : maximum + 2], 0.0)
    tail = float(np.clip(cdfs[maximum + 1], 0.0, 1.0))
    total = float(probabilities.sum() + tail)
    if total > 0.0:
        probabilities = probabilities / total
        tail /= total
    return PhaseTypeRenewalCountResult(
        horizon=t,
        probabilities=probabilities,
        tail_probability=tail,
        max_count=maximum,
        interarrival_distribution=interarrival_distribution,
    )


def sparre_andersen_phase_type_ruin_probability_by_count(
    count_ruin_probabilities: ArrayLike,
    interarrival_distribution: ClaimDistribution,
    horizon: float,
) -> float:
    """Mix claim-count ruin probabilities with a PH renewal count law."""

    ruin_by_count = np.asarray(count_ruin_probabilities, dtype=float)
    if ruin_by_count.ndim != 1 or ruin_by_count.size == 0:
        raise ValueError("count_ruin_probabilities must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(ruin_by_count)) or np.any((ruin_by_count < 0.0) | (ruin_by_count > 1.0)):
        raise ValueError("count_ruin_probabilities must lie in [0, 1]")
    count_law = phase_type_renewal_count_pmf(
        interarrival_distribution,
        horizon,
        max_count=ruin_by_count.size - 1,
    )
    value = float(np.dot(count_law.probabilities, ruin_by_count))
    value += count_law.tail_probability * float(ruin_by_count[-1])
    return float(np.clip(value, 0.0, 1.0))


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_int(value: int, name: str) -> int:
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result
