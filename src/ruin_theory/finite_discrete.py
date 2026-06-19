"""Exact finite-time ruin formulas for integer-valued claims."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


FiniteTimeDiscreteMethod = Literal["seal", "takacs", "picard-lefevre", "inventory"]


@dataclass(frozen=True)
class FiniteTimeDiscreteRuinResult:
    """Exact finite-time ruin result for a lattice Cramer-Lundberg model."""

    initial_capital: int
    horizon: float
    premium_rate: float
    claim_arrival_rate: float
    claim_pmf: np.ndarray
    method: str
    survival_probability: float
    ruin_probability: float
    premium_units: float
    inventory_times: np.ndarray
    survival_probabilities: np.ndarray
    state_probabilities: np.ndarray
    convention: str


def _nonnegative_int(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _claim_pmf(values: ArrayLike) -> np.ndarray:
    pmf = np.asarray(values, dtype=float)
    if pmf.ndim != 1 or pmf.size == 0:
        raise ValueError("claim_pmf must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(pmf)) or np.any(pmf < 0.0):
        raise ValueError("claim_pmf must contain finite non-negative probabilities")
    total = float(np.sum(pmf))
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("claim_pmf must sum to one for exact finite-time formulas")
    return pmf.copy()


def _method(value: str) -> str:
    method = value.lower().replace("_", "-")
    aliases = {
        "direct": "inventory",
        "recursive": "inventory",
        "picard-lefevre-recursion": "inventory",
        "pl": "picard-lefevre",
        "seal-takacs": "seal",
    }
    method = aliases.get(method, method)
    if method not in {"seal", "takacs", "picard-lefevre", "inventory"}:
        raise ValueError("method must be 'seal', 'takacs', 'picard-lefevre' or 'inventory'")
    return method


def _floor_nonnegative(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return int(nearest)
    return int(math.floor(value))


def compound_poisson_lattice_pmf(
    claim_pmf: ArrayLike,
    *,
    mean: float,
    max_aggregate: int,
) -> np.ndarray:
    """Return ``P(S=j)`` for a compound Poisson sum with integer severities.

    ``claim_pmf[k]`` is the probability of a claim of amount ``k``. The
    returned array is exact on indices ``0, ..., max_aggregate``; probability
    mass above ``max_aggregate`` is intentionally not returned.
    """

    pmf = _claim_pmf(claim_pmf)
    poisson_mean = _finite_nonnegative(mean, "mean")
    max_index = _nonnegative_int(max_aggregate, "max_aggregate")
    return _compound_poisson_lattice_pmf(pmf, poisson_mean, max_index)


def _compound_poisson_lattice_pmf(
    claim_pmf: np.ndarray,
    mean: float,
    max_aggregate: int,
) -> np.ndarray:
    aggregate = np.zeros(max_aggregate + 1, dtype=float)
    aggregate[0] = math.exp(-mean * (1.0 - float(claim_pmf[0])))
    if max_aggregate == 0:
        return aggregate
    support_max = min(claim_pmf.size - 1, max_aggregate)
    for j in range(1, max_aggregate + 1):
        upper = min(j, support_max)
        if upper == 0:
            continue
        indices = np.arange(1, upper + 1)
        weighted_previous = indices * claim_pmf[1 : upper + 1] * aggregate[j - indices]
        aggregate[j] = mean * float(np.sum(weighted_previous)) / j
    return aggregate


def _h_values(
    claim_pmf: np.ndarray,
    premium_units: float,
    claim_arrival_rate: float,
    premium_rate: float,
    max_index: int,
) -> np.ndarray:
    mean = claim_arrival_rate * premium_units / premium_rate
    return _compound_poisson_lattice_pmf(claim_pmf, mean, max_index)


def _h_tilde(values: np.ndarray, premium_units: float, max_index: int) -> float:
    if max_index < 0:
        return 0.0
    if math.isclose(premium_units, 0.0, abs_tol=1e-14):
        return 1.0
    indices = np.arange(max_index + 1, dtype=float)
    return float(math.fsum((1.0 - indices / premium_units) * values[: max_index + 1]))


def _seal_survival(
    claim_pmf: np.ndarray,
    *,
    initial_capital: int,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> float:
    x = premium_rate * horizon
    n = _floor_nonnegative(x)
    final_index = initial_capital + n
    final = _h_values(claim_pmf, x, claim_arrival_rate, premium_rate, final_index)
    survival = math.fsum(final[: final_index + 1])
    correction = 0.0
    for k in range(1, n + 1):
        hit = _h_values(
            claim_pmf,
            float(k),
            claim_arrival_rate,
            premium_rate,
            initial_capital + k,
        )[initial_capital + k]
        tail = _h_values(claim_pmf, x - k, claim_arrival_rate, premium_rate, n - k)
        correction += hit * _h_tilde(tail, x - k, n - k)
    return survival - correction


def _takacs_survival(
    claim_pmf: np.ndarray,
    *,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> float:
    x = premium_rate * horizon
    if math.isclose(x, 0.0, abs_tol=1e-14):
        return 1.0
    n = _floor_nonnegative(x)
    values = _h_values(claim_pmf, x, claim_arrival_rate, premium_rate, n)
    return _h_tilde(values, x, n)


def _picard_lefevre_survival(
    claim_pmf: np.ndarray,
    *,
    initial_capital: int,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> float:
    x = premium_rate * horizon
    if math.isclose(x, 0.0, abs_tol=1e-14):
        return 1.0
    upper = _floor_nonnegative(initial_capital + x)
    base = _h_values(claim_pmf, x, claim_arrival_rate, premium_rate, initial_capital)
    survival = math.fsum(base[: initial_capital + 1])
    if upper <= initial_capital:
        return survival

    for j in range(initial_capital + 1):
        formal = _h_values(
            claim_pmf,
            float(j - initial_capital),
            claim_arrival_rate,
            premium_rate,
            j,
        )[j]
        tau = initial_capital + x - j
        values = _h_values(claim_pmf, tau, claim_arrival_rate, premium_rate, upper - j)
        inner = math.fsum(
            values[i - j] * (initial_capital + x - i) / tau
            for i in range(initial_capital + 1, upper + 1)
        )
        survival += formal * inner
    return survival


def _inventory_result(
    claim_pmf: np.ndarray,
    *,
    initial_capital: int,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    final_boundary = _floor_nonnegative(initial_capital + premium_rate * horizon) + 1
    first_boundary = initial_capital + 1
    state = np.zeros(final_boundary, dtype=float)
    state[0] = 1.0
    previous_time = 0.0
    times: list[float] = []
    survivals: list[float] = []

    for boundary in range(first_boundary, final_boundary + 1):
        if boundary == final_boundary:
            inventory_time = horizon
        else:
            inventory_time = (boundary - initial_capital) / premium_rate
        elapsed = max(0.0, inventory_time - previous_time)
        increment = _h_values(
            claim_pmf,
            premium_rate * elapsed,
            claim_arrival_rate,
            premium_rate,
            final_boundary - 1,
        )
        convolved = np.convolve(state, increment)[:final_boundary]
        next_state = np.zeros_like(state)
        retained = min(boundary, final_boundary)
        next_state[:retained] = convolved[:retained]
        state = next_state
        previous_time = inventory_time
        times.append(inventory_time)
        survivals.append(float(np.sum(state)))

    return float(np.sum(state)), np.asarray(times), np.asarray(survivals), state.copy()


def finite_time_ruin_discrete(
    claim_pmf: ArrayLike,
    *,
    initial_capital: int,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
    method: FiniteTimeDiscreteMethod = "seal",
    return_result: bool = False,
) -> float | FiniteTimeDiscreteRuinResult:
    """Exact finite-time ruin probability for integer-valued claim sizes.

    The model is ``R_t = u + c t - S_t`` with a homogeneous compound Poisson
    aggregate claim process and integer claim sizes. The initial capital must
    be an integer measured on the same lattice as ``claim_pmf``.
    """

    pmf = _claim_pmf(claim_pmf)
    u = _nonnegative_int(initial_capital, "initial_capital")
    c = _finite_positive(premium_rate, "premium_rate")
    lam = _finite_nonnegative(claim_arrival_rate, "claim_arrival_rate")
    time = _finite_nonnegative(horizon, "horizon")
    selected = _method(method)

    inventory_times = np.array([], dtype=float)
    survival_grid = np.array([], dtype=float)
    state = np.array([], dtype=float)

    if time == 0.0 or lam == 0.0 or float(np.sum(pmf[1:])) == 0.0:
        survival = 1.0
    elif selected == "inventory":
        survival, inventory_times, survival_grid, state = _inventory_result(
            pmf,
            initial_capital=u,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )
    elif selected == "takacs":
        if u != 0:
            raise ValueError("method='takacs' is the zero-initial-capital Takacs formula")
        survival = _takacs_survival(
            pmf,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )
    elif selected == "picard-lefevre":
        survival = _picard_lefevre_survival(
            pmf,
            initial_capital=u,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )
    else:
        survival = _seal_survival(
            pmf,
            initial_capital=u,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )

    survival = float(np.clip(survival, 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDiscreteRuinResult(
        initial_capital=u,
        horizon=time,
        premium_rate=c,
        claim_arrival_rate=lam,
        claim_pmf=pmf,
        method=selected,
        survival_probability=survival,
        ruin_probability=ruin,
        premium_units=c * time,
        inventory_times=inventory_times,
        survival_probabilities=survival_grid,
        state_probabilities=state,
        convention=(
            "integer claim amounts; ruin occurs when S_t > u + c t; "
            "inventory states keep aggregate claims strictly below the active integer boundary"
        ),
    )
