"""Seasonal allocation and exact step-calendar Monte Carlo (Minier et al., 2026)."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
import operator

import numpy as np
from scipy import integrate, optimize, stats

from .distributions import ClaimDistribution
from .prevention import periodic_lundberg_coefficient


def _vector(values, name, *, positive=False):
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or not result.size
        or not np.all(np.isfinite(result))
        or np.any(result <= 0 if positive else result < 0)
    ):
        raise ValueError(f"{name} must be a finite nonempty nonnegative vector")
    return result.copy()


@dataclass(frozen=True)
class PreventionResponseCurve:
    """Response with an analytic inverse marginal benefit.

    Exponential: floor + scale*exp(-shape*p); quadratic: floor +
    scale*(offset-p)**2 on [0,offset]; reciprocal: floor +
    scale/(offset+p)**shape. Parameters need not enforce f(0)=1.
    """

    family: str = "exponential"
    scale: float = 1.0
    shape: float = 1.0
    offset: float = 1.0
    floor: float = 0.0

    def __post_init__(self):
        if self.family not in {"exponential", "quadratic", "reciprocal"}:
            raise ValueError("unknown response family")
        if not all(np.isfinite(x) and x > 0 for x in (self.scale, self.shape, self.offset)):
            raise ValueError("scale, shape and offset must be finite and positive")
        if not np.isfinite(self.floor) or self.floor < 0:
            raise ValueError("floor must be finite and nonnegative")

    def __call__(self, prevention):
        p = np.asarray(prevention, dtype=float)
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("prevention must be finite and nonnegative")
        if self.family == "exponential":
            return self.floor + self.scale * np.exp(-self.shape * p)
        if self.family == "quadratic":
            if np.any(p > self.offset):
                raise ValueError("quadratic prevention exceeds its decreasing domain")
            return self.floor + self.scale * (self.offset - p) ** 2
        return self.floor + self.scale / (self.offset + p) ** self.shape

    def inverse_marginal(self, marginal):
        target = np.asarray(marginal, dtype=float)
        with np.errstate(divide="ignore", over="ignore"):
            if self.family == "exponential":
                return (np.log(self.scale * self.shape) - np.log(target)) / self.shape
            if self.family == "quadratic":
                return self.offset - target / (2 * self.scale)
            return (self.shape * self.scale / target) ** (1 / (self.shape + 1)) - self.offset


@dataclass(frozen=True)
class SeasonalAllocationResult:
    amounts: np.ndarray
    effective_amounts: np.ndarray
    cost_weights: np.ndarray
    budget_spent: float
    controlled_loss: float
    total_outflow: float
    marginal_cost: float | None
    budget_convention: str
    method: str


def optimize_seasonal_prevention(
    weights,
    *,
    response: PreventionResponseCurve | Callable,
    max_prevention: float,
    min_prevention: float = 0.0,
    durations=None,
    budget: float | None = None,
    budget_ceiling: float | None = None,
    budget_convention: str = "rate",
    lag_steps: int = 0,
) -> SeasonalAllocationResult:
    """Minimize expected losses at fixed budget, or cost+loss at free budget.

    weights are integrated expected-loss weights Lambda_i*mu_i*duration_i.
    rate convention charges duration_i*p_i; amount convention charges p_i
    (Section 4.2 of the source). budget=None selects free budget (Section 4.2.2).
    A ceiling is an inequality; max_prevention is a separate pointwise cap.
    min_prevention enforces investment in every interval (the source's simple
    threshold variant); it is not an optional on/off activation model.
    Custom responses use numerical optimization under the convexity assumption.
    Integer lag shifts require equal interval lengths.
    """
    w = _vector(weights, "weights")
    dt = (
        np.full(w.size, 1 / w.size)
        if durations is None
        else _vector(
            durations,
            "durations",
            positive=True,
        )
    )
    if dt.shape != w.shape or not np.isclose(dt.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("durations must match weights and sum to one")
    if budget_convention not in {"rate", "amount"}:
        raise ValueError("budget_convention must be rate or amount")
    cost = dt if budget_convention == "rate" else np.ones_like(dt)
    cap = float(max_prevention)
    if not np.isfinite(cap) or cap < 0:
        raise ValueError("max_prevention must be finite and nonnegative")
    minimum = float(min_prevention)
    if not np.isfinite(minimum) or not 0 <= minimum <= cap:
        raise ValueError("min_prevention must lie between zero and max_prevention")
    lag = operator.index(lag_steps)
    if lag and not np.allclose(dt, dt[0], rtol=0, atol=1e-12):
        raise ValueError("integer lag requires uniform durations")
    shifted = np.roll(w, -lag)
    maximum_budget = float(cost.sum() * cap)
    minimum_budget = float(cost.sum() * minimum)
    for value in (budget, budget_ceiling):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("budget and budget_ceiling must be finite and nonnegative")
    if (budget is not None and budget < minimum_budget) or (
        budget_ceiling is not None and budget_ceiling < minimum_budget
    ):
        raise ValueError("budget or ceiling cannot fund min_prevention in every interval")
    if budget is not None and (
        budget > maximum_budget or (budget_ceiling is not None and budget > budget_ceiling)
    ):
        raise ValueError("fixed budget exceeds the feasible cap or ceiling")
    if not callable(response):
        raise TypeError("response must be callable")
    if isinstance(response, PreventionResponseCurve):
        response(np.array([0.0, cap]))

        def allocation(multiplier, log_multiplier=None):
            p = np.full_like(w, minimum)
            active = shifted > 0
            if response.family == "exponential" and log_multiplier is not None:
                values = (
                    np.log(response.scale * response.shape)
                    - log_multiplier
                    - np.log(cost[active])
                    + np.log(shifted[active])
                ) / response.shape
            else:
                values = response.inverse_marginal(multiplier * cost[active] / shifted[active])
            p[active] = np.clip(values, minimum, cap)
            return p

        if cap == minimum or budget == minimum_budget or budget_ceiling == minimum_budget:
            p, multiplier = np.full_like(w, minimum), None
        elif budget is None:
            p, multiplier = allocation(1.0), 1.0
            if budget_ceiling is not None and cost @ p > budget_ceiling:
                high = 2.0
                while cost @ allocation(high) > budget_ceiling:
                    high *= 2
                multiplier = optimize.brentq(
                    lambda x: cost @ allocation(x) - budget_ceiling,
                    1,
                    high,
                )
                p = allocation(multiplier)
        else:
            active_capacity = float(
                cost[shifted > 0].sum() * cap + cost[shifted == 0].sum() * minimum
            )
            if budget >= active_capacity:
                p = np.where(shifted > 0, cap, minimum)
                if budget > active_capacity:
                    p[shifted == 0] += (budget - active_capacity) / cost[shifted == 0].sum()
                multiplier = 0.0
            else:
                # Logarithmic multiplier search also handles strong exponential response.
                low, high = -1.0, 1.0

                def residual(log_value):
                    multiplier = np.exp(np.clip(log_value, -700, 700))
                    return float(cost @ allocation(multiplier, log_value) - budget)

                while residual(low) < 0:
                    low *= 2
                while residual(high) > 0:
                    high *= 2
                log_multiplier = optimize.brentq(residual, low, high)
                multiplier = float(np.exp(np.clip(log_multiplier, -700, 700)))
                p = allocation(multiplier, log_multiplier)
        method = "analytic marginal/KKT"
    elif cap == minimum or budget == minimum_budget or budget_ceiling == minimum_budget:
        p, multiplier, method = np.full_like(w, minimum), None, "minimum allocation"
    else:

        def values(p):
            out = np.asarray([response(float(x)) for x in p], dtype=float)
            if not np.all(np.isfinite(out)) or np.any(out < 0):
                raise ValueError("response must return finite nonnegative values")
            return out

        scale = max(1.0, float(shifted.sum()), maximum_budget)
        objective = lambda p: (
            float(shifted @ values(p) + (cost @ p if budget is None else 0)) / scale
        )
        constraints = []
        if budget is not None:
            constraints.append({"type": "eq", "fun": lambda p: cost @ p - budget})
        elif budget_ceiling is not None:
            constraints.append({"type": "ineq", "fun": lambda p: budget_ceiling - cost @ p})
        initial = np.full(w.size, max(minimum, (budget or 0) / cost.sum()))
        fitted = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(minimum, cap)] * w.size,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not fitted.success:
            raise RuntimeError(f"seasonal optimization failed: {fitted.message}")
        p, multiplier, method = fitted.x, None, "numerical SLSQP"
    effective = np.roll(p, lag)
    values = np.asarray([response(float(x)) for x in effective], dtype=float)
    if values.shape != w.shape or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("response must return finite nonnegative scalar values")
    loss = float(w @ values)
    spent = float(cost @ p)
    return SeasonalAllocationResult(
        p, effective, cost.copy(), spent, loss, spent + loss, multiplier, budget_convention, method
    )


def integrate_seasonal_pressure(frequency: Callable, *, breaks, severity_mean=None):
    """Quadrature weights for affine, sinusoidal or general continuous profiles."""
    grid = np.asarray(breaks, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or grid[0] != 0
        or grid[-1] != 1
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("breaks must increase strictly from zero to one")

    def integrand(t):
        rate = float(frequency(t))
        mean = 1 if severity_mean is None else float(severity_mean(t))
        if not np.isfinite(rate) or rate < 0 or not np.isfinite(mean) or mean < 0:
            raise ValueError("frequency and severity mean must be finite and nonnegative")
        value = rate * mean
        if not np.isfinite(value) or value < 0:
            raise ValueError("seasonal intensity/mean product must be finite and nonnegative")
        return value

    return np.array(
        [integrate.quad(integrand, a, b, epsabs=1e-10)[0] for a, b in zip(grid[:-1], grid[1:])]
    )


def seasonal_lundberg_coefficient(
    frequencies,
    distributions: Sequence[ClaimDistribution],
    *,
    prevention,
    response: Callable,
    premium_rate: float,
    durations=None,
):
    """Averaged Lundberg root for season-dependent light-tailed severities."""
    rates = _vector(frequencies, "frequencies")
    p = _vector(prevention, "prevention")
    dt = (
        np.full(rates.size, 1 / rates.size)
        if durations is None
        else _vector(
            durations,
            "durations",
            positive=True,
        )
    )
    if (
        p.shape != rates.shape
        or dt.shape != rates.shape
        or len(distributions) != rates.size
        or not np.isclose(dt.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("seasonal inputs must match and durations sum to one")
    intensities = rates * dt * np.array([response(x) for x in p])
    if not np.all(np.isfinite(intensities)) or np.any(intensities < 0) or intensities.sum() <= 0:
        raise ValueError("controlled intensities must be finite and nonnegative with positive sum")
    active = np.flatnonzero(intensities > 0)
    weights = intensities[active] / intensities.sum()
    laws = [distributions[i] for i in active]

    def sample(rng, n):
        chosen = rng.choice(len(laws), size=n, p=weights)
        draws = np.empty(n)
        for i, law in enumerate(laws):
            mask = chosen == i
            draws[mask] = law.sample(int(mask.sum()), rng)
        return draws

    mixture = ClaimDistribution(
        "seasonal_light_tail_mixture",
        float(weights @ np.array([law.mean() for law in laws])),
        None,
        sample,
        mgf_function=lambda r: sum(w * law.mgf(r) for w, law in zip(weights, laws)),
    )
    return periodic_lundberg_coefficient(
        mixture,
        premium_rate=premium_rate,
        annual_budget=float(dt @ p),
        controlled_frequency=float(intensities.sum()),
    )


@dataclass(frozen=True)
class SeasonalLundbergAllocation:
    allocation: SeasonalAllocationResult
    coefficient: float
    equation_residual: float


def optimize_seasonal_lundberg(
    frequencies,
    distributions: Sequence[ClaimDistribution],
    *,
    response: PreventionResponseCurve,
    premium_rate: float,
    max_prevention: float,
    min_prevention: float = 0.0,
    durations=None,
    budget: float | None = None,
    budget_ceiling: float | None = None,
    lag_steps: int = 0,
) -> SeasonalLundbergAllocation:
    """Maximize the averaged adjustment coefficient by nested convex allocation.

    For a trial R, minimize sum(lambda_i*dt_i*f(p_i)*(M_i(R)-1))/R
    plus annual prevention cost. Fixed budget and free budget with an optional
    ceiling are supported. The three analytic convex response families are
    required; this is not a nonconvex activation-threshold optimizer.
    Expenditure uses the rate convention. The returned root alone does not
    establish a uniform-in-phase Lundberg bound for arbitrary seasonal models.
    """
    rates = _vector(frequencies, "frequencies")
    dt = (
        np.full(rates.size, 1 / rates.size)
        if durations is None
        else _vector(durations, "durations", positive=True)
    )
    if dt.shape != rates.shape or len(distributions) != rates.size:
        raise ValueError("frequencies, distributions and durations must match")
    if not isinstance(response, PreventionResponseCurve):
        raise TypeError("response must be a convex PreventionResponseCurve")
    premium = float(premium_rate)
    if not np.isfinite(premium) or premium <= 0:
        raise ValueError("premium_rate must be finite and positive")
    means = np.array([law.mean() for law in distributions])
    if np.any(~np.isfinite(means)) or np.any(means < 0):
        raise ValueError("claim means must be finite and nonnegative")
    common = dict(
        response=response,
        max_prevention=max_prevention,
        min_prevention=min_prevention,
        durations=dt,
        budget=budget,
        budget_ceiling=budget_ceiling,
        lag_steps=lag_steps,
    )
    initial = optimize_seasonal_prevention(rates * dt * means, **common)
    if initial.total_outflow >= premium:
        raise ValueError("no admissible policy satisfies the net profit condition")
    if initial.controlled_loss == 0:
        raise ValueError("zero controlled claim pressure has no finite adjustment coefficient")
    if not np.any((rates > 0) & (means > 0)):
        raise ValueError("positive claim pressure is required")

    def at_root(r):
        if r == 0:
            return initial, initial.total_outflow - premium
        active = rates > 0
        moments = np.ones(rates.size)
        with np.errstate(over="ignore"):
            moments[active] = [law.mgf(r) for law, use in zip(distributions, active) if use]
        if np.any(~np.isfinite(moments)):
            return None, np.inf
        with np.errstate(over="ignore"):
            weights = rates * dt * np.maximum(moments - 1, 0) / r
        if np.any(~np.isfinite(weights)):
            return None, np.inf
        allocation = optimize_seasonal_prevention(weights, **common)
        return allocation, allocation.total_outflow - premium

    high = 1.0
    for _ in range(80):
        if at_root(high)[1] > 0:
            break
        high *= 2
    else:
        raise ValueError("could not bracket an optimal finite adjustment coefficient")
    root = optimize.brentq(lambda r: at_root(r)[1], 0, high, xtol=1e-12)
    weighted_allocation, residual = at_root(root)
    p = weighted_allocation.amounts
    effective = np.roll(p, lag_steps)
    loss = float((rates * dt * means) @ response(effective))
    spent = float(dt @ p)
    allocation = SeasonalAllocationResult(
        p,
        effective,
        dt.copy(),
        spent,
        loss,
        spent + loss,
        weighted_allocation.marginal_cost,
        "rate",
        "nested convex Lundberg/KKT",
    )
    return SeasonalLundbergAllocation(allocation, float(root), float(root * residual))


@dataclass(frozen=True)
class SeasonalRuinComparison:
    labels: tuple[str, ...]
    estimates: tuple
    ruined: np.ndarray
    terminal_reserves: np.ndarray
    absolute_gains: np.ndarray
    gain_standard_errors: np.ndarray
    relative_gains: np.ndarray
    relative_gain_standard_errors: np.ndarray
    trajectories: Mapping[str, tuple] | None = None


def compare_seasonal_prevention(
    frequencies,
    distributions: Sequence[ClaimDistribution],
    policies: Mapping[str, Sequence],
    *,
    response: Callable,
    premium_rate: float,
    initial_capital: float,
    horizon: float,
    durations=None,
    starting_phase: float = 0.0,
    n_simulations: int = 1000,
    seed: int | None = None,
    ci_level: float = 0.95,
    lag_steps: int = 0,
    record_paths: int = 0,
) -> SeasonalRuinComparison:
    """Exact event-time simulation for step intensities and spending rates.

    Shared dominating Poisson events, severities and uniforms couple all policies.
    The first policy is the reference. Gain errors use paired observations;
    relative gain is NaN when the reference has no observed ruin. Negative
    within-period drift is monitored as well as accepted claim jumps. Durations
    need not sum to one (finite dynamic calendars can span several years).
    """
    from .results import RuinEstimate

    rates = _vector(frequencies, "frequencies")
    dt = (
        np.full(rates.size, 1 / rates.size)
        if durations is None
        else _vector(
            durations,
            "durations",
            positive=True,
        )
    )
    labels = tuple(policies)
    calendars = np.array([_vector(policies[x], "policy") for x in labels])
    n = operator.index(n_simulations)
    recorded = operator.index(record_paths)
    if recorded < 0 or recorded > n:
        raise ValueError("record_paths must be between zero and n_simulations")
    if (
        not labels
        or calendars.shape != (len(labels), rates.size)
        or dt.shape != rates.shape
        or len(distributions) != rates.size
        or n < 2
    ):
        raise ValueError(
            "policies, laws and durations must match; n_simulations must be at least two"
        )
    if (
        not np.isfinite(horizon)
        or horizon <= 0
        or not np.isfinite(initial_capital)
        or initial_capital < 0
        or not np.isfinite(premium_rate)
        or premium_rate < 0
        or not np.isfinite(starting_phase)
        or not 0 < ci_level < 1
    ):
        raise ValueError("invalid horizon, capital, premium, phase or confidence level")
    factors = np.array([[response(x) for x in p] for p in calendars], dtype=float)
    lag = operator.index(lag_steps)
    if lag and not np.allclose(dt, dt[0], rtol=0, atol=1e-12):
        raise ValueError("integer lag requires uniform durations")
    factors = np.roll(factors, lag, axis=1)
    if not np.all(np.isfinite(factors)) or np.any(factors < 0):
        raise ValueError("response values must be finite and nonnegative")
    dominating = rates * factors.max(axis=0)
    edges = np.r_[0.0, np.cumsum(dt)]
    cycle = float(dt.sum())
    segments = []
    time = 0.0
    phase = starting_phase % cycle
    while time < horizon:
        period = min(int(np.searchsorted(edges, phase, side="right") - 1), rates.size - 1)
        length = min(float(edges[period + 1] - phase), horizon - time)
        if length <= 0:
            phase = 0.0
            continue
        segments.append((time, length, period))
        time += length
        phase = 0.0 if period == rates.size - 1 else float(edges[period + 1])
    rng = np.random.default_rng(seed)
    reserves = np.full((len(labels), n), initial_capital, dtype=float)
    ruin_times = np.full_like(reserves, np.inf)
    path_times = [[0.0] for _ in range(recorded)]
    path_values = [[reserves[:, i].copy()] for i in range(recorded)]
    for begin, length, period in segments:
        drift = premium_rate - calendars[:, period]
        counts = rng.poisson(dominating[period] * length, size=n)
        for path, count in enumerate(counts):
            times = np.sort(rng.uniform(0, length, int(count)))
            claims = distributions[period].sample(int(count), rng)
            uniforms = rng.random(int(count))
            previous = 0.0
            for event, claim, uniform in zip(
                np.r_[times, length], np.r_[claims, 0], np.r_[uniforms, np.inf]
            ):
                before = reserves[:, path].copy()
                reserves[:, path] += drift * (event - previous)
                crossing = (reserves[:, path] < 0) & np.isinf(ruin_times[:, path])
                ruin_times[crossing, path] = begin + previous + before[crossing] / -drift[crossing]
                if path < recorded:
                    path_times[path].append(begin + event)
                    path_values[path].append(reserves[:, path].copy())
                if dominating[period] > 0:
                    accepted = uniform < rates[period] * factors[:, period] / dominating[period]
                    reserves[:, path] -= claim * accepted
                    jumped = (reserves[:, path] < 0) & np.isinf(ruin_times[:, path])
                    ruin_times[jumped, path] = begin + event
                    if path < recorded and np.any(accepted):
                        path_times[path].append(begin + event)
                        path_values[path].append(reserves[:, path].copy())
                previous = event
    ruined = np.isfinite(ruin_times)
    probabilities = ruined.mean(axis=1)
    z = stats.norm.ppf((1 + ci_level) / 2)
    denominator = 1 + z * z / n
    center = (probabilities + z * z / (2 * n)) / denominator
    radius = (
        z * np.sqrt(probabilities * (1 - probabilities) / n + z * z / (4 * n * n)) / denominator
    )
    estimates = tuple(
        RuinEstimate(
            float(p), float(np.sqrt(p * (1 - p) / n)), float(low), float(high), n, horizon, t.copy()
        )
        for p, low, high, t in zip(probabilities, center - radius, center + radius, ruin_times)
    )
    differences = ruined[0].astype(float) - ruined.astype(float)
    absolute = differences.mean(axis=1)
    errors = differences.std(axis=1, ddof=1) / np.sqrt(n)
    relative, relative_errors = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
    if probabilities[0] > 0:
        relative = absolute / probabilities[0]
        influences = (
            -ruined.astype(float) / probabilities[0]
            + probabilities[:, None] * ruined[0] / probabilities[0] ** 2
        )
        relative_errors = influences.std(axis=1, ddof=1) / np.sqrt(n)
    trajectories = (
        None
        if not recorded
        else {
            label: tuple(
                (np.asarray(path_times[i]), np.asarray(path_values[i])[:, row])
                for i in range(recorded)
            )
            for row, label in enumerate(labels)
        }
    )
    return SeasonalRuinComparison(
        labels,
        estimates,
        ruined,
        reserves,
        absolute,
        errors,
        relative,
        relative_errors,
        trajectories,
    )


def seasonal_exponential_profile_root(
    *, base_rate, variation, frequency, premium_rate, profile="cosine"
):
    """Closed-form unprevented roots for the paper's exponential-rate profiles.

    cosine: eta(t)=base_rate+variation*cos(2*pi*t).
    triangular: eta(t)=base_rate+variation*min(t,1-t).
    affine: eta(t)=base_rate+variation*t on one repeated period.
    The zero-variation limit is the homogeneous exponential root.
    """
    for value in (base_rate, frequency, premium_rate):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("base_rate, frequency and premium_rate must be positive")
    if not np.isfinite(variation):
        raise ValueError("variation must be finite")
    if profile == "cosine":
        if base_rate <= abs(variation):
            raise ValueError("all exponential rates must be positive")
        root = base_rate - np.hypot(frequency / premium_rate, variation)
    elif profile in {"triangular", "affine"}:
        if variation < 0:
            raise ValueError("triangular/affine variation must be nonnegative")
        if variation == 0:
            root = base_rate - frequency / premium_rate
        else:
            delta = variation / 2 if profile == "triangular" else variation
            z = premium_rate * delta / frequency
            correction = delta * np.exp(-z) / (-np.expm1(-z))
            root = base_rate - correction
    else:
        raise ValueError("profile must be cosine, triangular or affine")
    if root <= 0:
        raise ValueError("the profile must satisfy the net profit condition")
    return float(root)


@dataclass(frozen=True)
class StormLossDayCalibration:
    frequencies: np.ndarray
    distributions: tuple[ClaimDistribution, ...]
    loss_weights: np.ndarray
    n_years: int


def calibrate_storm_loss_days(losses, months, years, *, retained_loss_cap, monetary_unit=1e9):
    """Monthly Poisson--empirical calibration from all daily observations.

    Include zero-loss days so years and exposure are represented. Only positive
    retained losses define jumps; the severe-loss diagnostic threshold is NOT
    a claim filter. Provide complete years, and bootstrap year blocks outside
    this function when quantifying calibration (rather than simulation) error.
    """
    from .distributions import deterministic, empirical

    loss = _vector(losses, "losses")
    month, year = np.asarray(months), np.asarray(years)
    if (
        month.shape != loss.shape
        or year.shape != loss.shape
        or not np.issubdtype(month.dtype, np.integer)
        or not np.issubdtype(year.dtype, np.integer)
        or np.any(month < 1)
        or np.any(month > 12)
    ):
        raise ValueError("matching integer months (1..12) and years are required")
    if not np.isfinite(retained_loss_cap) or retained_loss_cap <= 0:
        raise ValueError("retained_loss_cap must be positive")
    if not np.isfinite(monetary_unit) or monetary_unit <= 0:
        raise ValueError("monetary_unit must be positive")
    n_years = np.unique(year).size
    for y in np.unique(year):
        if np.unique(month[year == y]).size != 12:
            raise ValueError("each calibration year must cover all twelve months")
    retained = np.minimum(loss, retained_loss_cap) / monetary_unit
    laws, rates = [], []
    for m in range(1, 13):
        positive = retained[(month == m) & (retained > 0)]
        laws.append(empirical(positive) if positive.size else deterministic(0))
        rates.append(12 * positive.size / n_years)
    rates = np.asarray(rates)
    weights = rates * np.array([law.mean() for law in laws]) / 12
    return StormLossDayCalibration(rates, tuple(laws), weights, int(n_years))
