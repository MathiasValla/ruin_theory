"""Multirisk dividend and insolvency-penalty CTMC approximations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .markov_modulated import CommonShock, Vector, VectorPmf, _clean_vector_pmf


PremiumRateFunction = Callable[[int, int, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class MultiriskDividendCTMCResult:
    """Expected multirisk dividends and insolvency penalties until ruin."""

    expected_time_to_ruin: float
    ruin_probability: float
    expected_dividends: np.ndarray
    expected_penalties: np.ndarray
    ruin_state_probabilities: dict[tuple[float, ...], float]
    expected_surplus_at_ruin: np.ndarray
    expected_deficit_at_ruin: np.ndarray
    state_count: int
    grid_step: float
    initial_reserves: np.ndarray
    barriers: np.ndarray
    lower_bounds: np.ndarray
    ruin_lines: tuple[int, ...]


@dataclass(frozen=True)
class MultiriskDividendConvergenceResult:
    """Convergence diagnostics across CTMC discretizations."""

    grid_steps: np.ndarray
    state_counts: np.ndarray
    expected_time_to_ruin: np.ndarray
    ruin_probabilities: np.ndarray
    expected_dividends: np.ndarray
    expected_penalties: np.ndarray

    @property
    def last_time_change(self) -> float:
        if self.expected_time_to_ruin.size < 2:
            return 0.0
        return float(abs(self.expected_time_to_ruin[-1] - self.expected_time_to_ruin[-2]))


def linear_status_premium_function(
    base_premium_rates: ArrayLike,
    interaction_matrix: ArrayLike | None = None,
    *,
    min_rate: float = 0.0,
) -> PremiumRateFunction:
    """Build a positive status-dependent premium-rate function.

    Statuses are `-1` for an insolvent line, `0` for an interior line and `1`
    for a line at its dividend barrier.
    """

    base = _as_1d_float(base_premium_rates, "base_premium_rates")
    floor = _nonnegative_float(min_rate, "min_rate")
    if interaction_matrix is None:
        interactions = np.zeros((base.size, base.size), dtype=float)
    else:
        interactions = np.asarray(interaction_matrix, dtype=float)
        if interactions.shape != (base.size, base.size):
            raise ValueError("interaction_matrix must have shape (n_lines, n_lines)")
        if np.any(~np.isfinite(interactions)):
            raise ValueError("interaction_matrix must contain finite values")

    def premium(
        line: int,
        environment_state: int,
        reserves: np.ndarray,
        statuses: np.ndarray,
    ) -> float:
        del environment_state, reserves
        rate = base[line] + float(np.dot(interactions[line], statuses))
        return max(floor, rate)

    return premium


def estimate_multirisk_dividend_penalties_ctmc(
    *,
    initial_reserves: ArrayLike,
    barriers: ArrayLike,
    lower_bounds: ArrayLike,
    grid_step: float,
    environment_generator: ArrayLike,
    environment_initial: ArrayLike,
    shocks: Sequence[CommonShock],
    base_premium_rates: ArrayLike | None = None,
    premium_rate_function: PremiumRateFunction | None = None,
    transition_claim_pmfs: Mapping[tuple[int, int], Mapping[Sequence[int], float]] | None = None,
    ruin_lines: Sequence[int] = (0,),
    max_states: int = 20000,
) -> MultiriskDividendCTMCResult:
    """Approximate multirisk dividends and insolvency penalties by a finite CTMC."""

    step = _positive_float(grid_step, "grid_step")
    initial = _as_1d_float(initial_reserves, "initial_reserves")
    barriers_array = _as_1d_float(barriers, "barriers")
    lower = _as_1d_float(lower_bounds, "lower_bounds")
    if initial.shape != barriers_array.shape or lower.shape != initial.shape:
        raise ValueError("initial_reserves, barriers and lower_bounds must have matching shapes")
    if np.any(lower > initial) or np.any(initial > barriers_array):
        raise ValueError("initial_reserves must lie between lower_bounds and barriers")
    if np.any(barriers_array <= 0.0):
        raise ValueError("barriers must be positive")

    n_lines = initial.size
    generator = _clean_generator(environment_generator)
    env_initial = _as_probability_vector(environment_initial, "environment_initial")
    if generator.shape[0] != env_initial.size:
        raise ValueError("environment_generator and environment_initial dimensions must match")
    n_env = env_initial.size

    shock_tuple = tuple(shocks)
    if not shock_tuple:
        raise ValueError("shocks must contain at least one CommonShock")
    for shock in shock_tuple:
        if shock.n_states != n_env or shock.n_lines != n_lines:
            raise ValueError("all shocks must match environment states and reserve dimension")

    premium_function = _premium_function(base_premium_rates, premium_rate_function, n_lines)
    ruin_tuple = _clean_ruin_lines(ruin_lines, n_lines)
    transition_pmfs = _clean_transition_claim_pmfs(transition_claim_pmfs, n_env, n_lines)

    initial_units = _to_units(initial, step, "initial_reserves")
    barrier_units = _to_units(barriers_array, step, "barriers")
    lower_units = _to_units(lower, step, "lower_bounds")
    state_lower = np.array(lower_units, copy=True)
    for line in ruin_tuple:
        state_lower[line] = max(0, state_lower[line])
    if np.any(initial_units < state_lower) or np.any(initial_units > barrier_units):
        raise ValueError("initial_reserves must be feasible after applying ruin_lines")

    coordinate_ranges = [range(int(lo), int(hi) + 1) for lo, hi in zip(state_lower, barrier_units)]
    reserve_states = [tuple(values) for values in _cartesian_product(coordinate_ranges)]
    state_count = len(reserve_states) * n_env
    full_state_count = state_count
    maximum = int(max_states)
    if maximum <= 0:
        raise ValueError("max_states must be positive")
    if state_count > maximum:
        raise ValueError("finite CTMC state count exceeds max_states")

    states: list[tuple[int, Vector]] = []
    state_index: dict[tuple[int, Vector], int] = {}
    for env in range(n_env):
        for reserves in reserve_states:
            state_index[(env, reserves)] = len(states)
            states.append((env, reserves))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    dividends = np.zeros((state_count, n_lines), dtype=float)
    penalties = np.zeros((state_count, n_lines), dtype=float)
    absorption = np.zeros(state_count, dtype=float)
    terminal_rates: dict[Vector, np.ndarray] = {}

    zero_claim = tuple([0] * n_lines)
    for row, (env, units) in enumerate(states):
        reserves = step * np.asarray(units, dtype=float)
        statuses = _statuses(units, barrier_units)
        total_rate = 0.0

        for line in range(n_lines):
            rate = _premium_rate(premium_function, line, env, reserves, statuses)
            if units[line] >= barrier_units[line]:
                if np.any(np.delete(statuses, line) < 0):
                    penalties[row, line] += rate
                else:
                    dividends[row, line] += rate
                continue
            next_units = list(units)
            next_units[line] += 1
            next_key = (env, tuple(next_units))
            total_rate += _add_transition(row, state_index[next_key], rate, rows, cols, data)

        for shock in shock_tuple:
            intensity = float(shock.intensities[env])
            if intensity <= 0.0:
                continue
            for claim, probability in shock.claim_pmfs[env].items():
                if claim == zero_claim or probability <= 0.0:
                    continue
                total_rate += _add_claim_transition(
                    row,
                    env,
                    units,
                    claim,
                    intensity * probability,
                    state_index,
                    state_lower,
                    barrier_units,
                    ruin_tuple,
                    absorption,
                    terminal_rates,
                    rows,
                    cols,
                    data,
                )

        for next_env, rate in enumerate(generator[env]):
            if next_env == env or rate <= 0.0:
                continue
            pmf = transition_pmfs.get((env, next_env), {zero_claim: 1.0})
            for claim, probability in pmf.items():
                total_rate += _add_claim_transition(
                    row,
                    next_env,
                    units,
                    claim,
                    rate * probability,
                    state_index,
                    state_lower,
                    barrier_units,
                    ruin_tuple,
                    absorption,
                    terminal_rates,
                    rows,
                    cols,
                    data,
                )

        rows.append(row)
        cols.append(row)
        data.append(-total_rate)

    q_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(state_count, state_count)).tocsr()
    alpha = np.zeros(state_count, dtype=float)
    initial_key = tuple(int(value) for value in initial_units)
    for env, probability in enumerate(env_initial):
        alpha[state_index[(env, initial_key)]] = probability

    reachable = _reachable_indices(q_matrix, alpha)
    if reachable.size < state_count:
        q_matrix = q_matrix[reachable][:, reachable]
        alpha = alpha[reachable]
        dividends = dividends[reachable]
        penalties = penalties[reachable]
        absorption = absorption[reachable]
        terminal_rates = {key: rates[reachable] for key, rates in terminal_rates.items()}
        state_count = int(reachable.size)

    operator = -q_matrix
    rhs = np.column_stack([np.ones(state_count), absorption, dividends, penalties])
    values = _solve_transient(operator, rhs)
    expected_time = float(alpha @ values[:, 0])
    ruin_probability = float(alpha @ values[:, 1])
    expected_dividends = alpha @ values[:, 2 : 2 + n_lines]
    expected_penalties = alpha @ values[:, 2 + n_lines :]

    terminal_probabilities = _terminal_distribution(operator, alpha, terminal_rates, step)
    surplus_at_ruin = np.zeros(n_lines, dtype=float)
    deficit_at_ruin = np.zeros(n_lines, dtype=float)
    for terminal, probability in terminal_probabilities.items():
        vector = np.asarray(terminal, dtype=float)
        surplus_at_ruin += probability * vector
        deficit_at_ruin += probability * np.maximum(-vector, 0.0)

    return MultiriskDividendCTMCResult(
        expected_time_to_ruin=expected_time,
        ruin_probability=ruin_probability,
        expected_dividends=np.asarray(expected_dividends, dtype=float),
        expected_penalties=np.asarray(expected_penalties, dtype=float),
        ruin_state_probabilities=terminal_probabilities,
        expected_surplus_at_ruin=surplus_at_ruin,
        expected_deficit_at_ruin=deficit_at_ruin,
        state_count=full_state_count,
        grid_step=step,
        initial_reserves=initial,
        barriers=barriers_array,
        lower_bounds=lower,
        ruin_lines=ruin_tuple,
    )


def multirisk_dividend_convergence(
    results: Sequence[MultiriskDividendCTMCResult],
) -> MultiriskDividendConvergenceResult:
    """Collect convergence diagnostics from several CTMC discretizations."""

    result_tuple = tuple(results)
    if not result_tuple:
        raise ValueError("results must contain at least one result")
    n_lines = result_tuple[0].expected_dividends.size
    if any(result.expected_dividends.size != n_lines for result in result_tuple):
        raise ValueError("all results must have the same number of lines")
    order = np.argsort([result.grid_step for result in result_tuple])[::-1]
    ordered = tuple(result_tuple[int(index)] for index in order)
    return MultiriskDividendConvergenceResult(
        grid_steps=np.asarray([result.grid_step for result in ordered], dtype=float),
        state_counts=np.asarray([result.state_count for result in ordered], dtype=int),
        expected_time_to_ruin=np.asarray(
            [result.expected_time_to_ruin for result in ordered],
            dtype=float,
        ),
        ruin_probabilities=np.asarray([result.ruin_probability for result in ordered], dtype=float),
        expected_dividends=np.vstack([result.expected_dividends for result in ordered]),
        expected_penalties=np.vstack([result.expected_penalties for result in ordered]),
    )


def _add_claim_transition(
    row: int,
    next_env: int,
    units: Vector,
    claim: Vector,
    rate: float,
    state_index: Mapping[tuple[int, Vector], int],
    lower_units: np.ndarray,
    barrier_units: np.ndarray,
    ruin_lines: tuple[int, ...],
    absorption: np.ndarray,
    terminal_rates: dict[Vector, np.ndarray],
    rows: list[int],
    cols: list[int],
    data: list[float],
) -> float:
    if rate <= 0.0:
        return 0.0
    post_claim = tuple(int(unit) - int(amount) for unit, amount in zip(units, claim))
    if any(post_claim[line] < 0 for line in ruin_lines):
        absorption[row] += rate
        terminal_rates.setdefault(post_claim, np.zeros_like(absorption))[row] += rate
        return rate
    clipped = tuple(
        int(min(max(value, lower_units[line]), barrier_units[line]))
        for line, value in enumerate(post_claim)
    )
    return _add_transition(row, state_index[(next_env, clipped)], rate, rows, cols, data)


def _add_transition(
    row: int,
    col: int,
    rate: float,
    rows: list[int],
    cols: list[int],
    data: list[float],
) -> float:
    if rate <= 0.0 or row == col:
        return 0.0
    rows.append(row)
    cols.append(col)
    data.append(float(rate))
    return float(rate)


def _terminal_distribution(
    operator: sparse.csr_matrix,
    alpha: np.ndarray,
    terminal_rates: Mapping[Vector, np.ndarray],
    step: float,
) -> dict[tuple[float, ...], float]:
    if not terminal_rates:
        return {}
    keys = list(terminal_rates)
    rhs = np.column_stack([terminal_rates[key] for key in keys])
    values = _solve_transient(operator, rhs)
    probabilities = alpha @ values
    return {
        tuple(float(step * value) for value in key): float(probability)
        for key, probability in zip(keys, np.ravel(probabilities))
        if probability > 1e-14
    }


def _solve_transient(operator: sparse.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    try:
        solution = np.asarray(sparse_linalg.spsolve(operator, rhs), dtype=float)
    except Exception as exc:  # pragma: no cover - SciPy exposes several subclasses here.
        raise ValueError(
            "transient CTMC generator is singular; check absorbing ruin rates",
        ) from exc
    if rhs.ndim == 2 and solution.ndim == 1:
        return solution[:, None]
    return solution


def _reachable_indices(q_matrix: sparse.csr_matrix, alpha: np.ndarray) -> np.ndarray:
    starts = list(np.flatnonzero(alpha > 0.0))
    seen = np.zeros(alpha.size, dtype=bool)
    stack = starts[:]
    for index in starts:
        seen[index] = True
    while stack:
        row = stack.pop()
        row_start, row_end = q_matrix.indptr[row], q_matrix.indptr[row + 1]
        row_indices = q_matrix.indices[row_start:row_end]
        row_values = q_matrix.data[row_start:row_end]
        for col, value in zip(row_indices, row_values):
            if col != row and value > 0.0 and not seen[col]:
                seen[col] = True
                stack.append(int(col))
    return np.flatnonzero(seen)


def _premium_function(
    base_premium_rates: ArrayLike | None,
    premium_rate_function: PremiumRateFunction | None,
    n_lines: int,
) -> PremiumRateFunction:
    if premium_rate_function is not None:
        if not callable(premium_rate_function):
            raise TypeError("premium_rate_function must be callable")
        return premium_rate_function
    if base_premium_rates is None:
        raise ValueError("base_premium_rates is required without premium_rate_function")
    base = _as_1d_float(base_premium_rates, "base_premium_rates")
    if base.size != n_lines:
        raise ValueError("base_premium_rates must match reserve dimension")
    return linear_status_premium_function(base)


def _premium_rate(
    premium_rate_function: PremiumRateFunction,
    line: int,
    env: int,
    reserves: np.ndarray,
    statuses: np.ndarray,
) -> float:
    rate = float(premium_rate_function(line, env, reserves.copy(), statuses.copy()))
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("premium_rate_function must return finite non-negative rates")
    return rate


def _clean_transition_claim_pmfs(
    pmfs: Mapping[tuple[int, int], Mapping[Sequence[int], float]] | None,
    n_env: int,
    n_lines: int,
) -> dict[tuple[int, int], VectorPmf]:
    if pmfs is None:
        return {}
    cleaned: dict[tuple[int, int], VectorPmf] = {}
    for raw_key, raw_pmf in pmfs.items():
        if len(raw_key) != 2:
            raise ValueError("transition_claim_pmfs keys must be environment pairs")
        start, end = (int(raw_key[0]), int(raw_key[1]))
        if start < 0 or start >= n_env or end < 0 or end >= n_env or start == end:
            raise ValueError("transition_claim_pmfs keys must be valid off-diagonal pairs")
        pmf = _clean_vector_pmf(raw_pmf, name="transition_claim_pmf")
        if len(next(iter(pmf))) != n_lines:
            raise ValueError("transition claim vectors must match reserve dimension")
        cleaned[(start, end)] = pmf
    return cleaned


def _statuses(units: Vector, barrier_units: np.ndarray) -> np.ndarray:
    status = np.zeros(len(units), dtype=int)
    for line, value in enumerate(units):
        if value < 0:
            status[line] = -1
        elif value >= barrier_units[line]:
            status[line] = 1
    return status


def _clean_generator(generator: ArrayLike) -> np.ndarray:
    matrix = np.asarray(generator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("environment_generator must be a non-empty square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("environment_generator must contain finite values")
    off_diagonal = matrix - np.diag(np.diag(matrix))
    if np.any(off_diagonal < 0.0):
        raise ValueError("environment_generator off-diagonal entries must be non-negative")
    if not np.allclose(np.sum(matrix, axis=1), 0.0):
        raise ValueError("environment_generator rows must sum to zero")
    return matrix


def _clean_ruin_lines(lines: Sequence[int], n_lines: int) -> tuple[int, ...]:
    if not lines:
        raise ValueError("ruin_lines must not be empty")
    cleaned = tuple(sorted({int(line) for line in lines}))
    if any(line < 0 or line >= n_lines for line in cleaned):
        raise ValueError("ruin_lines entries must be valid line indices")
    if any(int(raw) != raw for raw in lines):
        raise ValueError("ruin_lines entries must be integers")
    return cleaned


def _as_probability_vector(values: ArrayLike, name: str) -> np.ndarray:
    vector = _as_1d_float(values, name)
    if np.any(vector < 0.0) or not np.isclose(np.sum(vector), 1.0):
        raise ValueError(f"{name} must be a probability vector")
    return vector


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _to_units(values: np.ndarray, step: float, name: str) -> np.ndarray:
    scaled = values / step
    units = np.rint(scaled).astype(int)
    if not np.allclose(scaled, units):
        raise ValueError(f"{name} must lie on the grid defined by grid_step")
    return units


def _cartesian_product(ranges: Sequence[range]) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = [()]
    for values in ranges:
        result = [prefix + (value,) for prefix in result for value in values]
    return result
