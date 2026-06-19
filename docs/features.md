# Feature Documentation

This page documents the current public API by feature. The package favors
small composable objects: define a severity law, plug it into a risk process,
then choose either a closed-form formula, a Monte Carlo estimator, or a plot.

## Severity Distributions

Severity factories return a `ClaimDistribution`. A distribution stores a name,
moments, a sampler, and optional CDF, survival, density, MGF and Laplace
methods.

Available factories:

- `exponential(rate)`: claim size with mean `1 / rate`.
- `gamma(shape, rate=...)` or `gamma(shape, scale=...)`.
- `erlang(shape, rate)`: integer-shape gamma.
- `deterministic(value)`: degenerate claim size.
- `mixture_exponential(rates, weights=None)`: hyperexponential mixture.
- `phase_type(initial_probabilities, subgenerator)`: continuous phase-type law
  `PH(alpha, T)`, where `subgenerator` is the transient generator matrix.
- `pareto(shape, scale)`: Pareto type I on `[scale, infinity)`.
- `lognormal(meanlog, sdlog)`.
- `weibull(shape, scale)`.
- `empirical(data)`: bootstrap distribution from observed non-negative losses.
- `scipy_distribution(name, **params)`: wrapper for a non-negative SciPy law.

Useful methods:

- `sample(n, rng=None)`: draw non-negative claim sizes.
- `mean()`, `variance()`, `second_moment()`.
- `cdf(x)`, `survival(x)`, `pdf(x)` when available.
- `mgf(t)`, `laplace(s)` when available.

Minimal example:

```python
import numpy as np
from ruin_theory import exponential, mixture_exponential

claims = exponential(rate=5.0)
print(claims.mean())
print(claims.survival(np.array([0.0, 1.0, 2.0])))

hyper = mixture_exponential(rates=[3.0, 7.0], weights=[0.5, 0.5])
print(hyper.mgf(0.5))
```

Phase-type example:

```python
import numpy as np
from ruin_theory import phase_type

# Erlang(2, rate=2) as a two-phase PH law.
claims = phase_type(
    initial_probabilities=[1.0, 0.0],
    subgenerator=np.array([[-2.0, 2.0], [0.0, -2.0]]),
)
print(claims.mean(), claims.variance())
print(claims.survival([0.0, 1.0, 2.0]))
```

## Loss Utilities

These helpers operate on severity laws and observed losses before they enter a
risk process or an aggregate-loss recursion.

Available functions:

- `raw_moment(distribution, order=1)`: returns `E[X**order]`.
- `limited_moment(distribution, limit, order=1, epsabs=1e-10)`: returns
  `E[min(X, limit)**order]`; `limit` may be scalar or array-like.
- `empirical_moment(data, order=1)` and
  `empirical_limited_moment(data, limit, order=1)`: empirical counterparts for
  one-dimensional non-negative observations.
- `coverage_transform(distribution_or_losses, deductible=0, limit=None,
  coinsurance=1, inflation=1, franchise=False, franchise_deductible=None)`:
  applies ordinary or franchise coverage. It returns transformed payments for
  arrays and a transformed `ClaimDistribution` for distributions.
- `discretize(distribution, from_=0, to=..., step=..., method="upper")`:
  returns a `DiscretizedDistribution` with `support`, `pmf`, `step`,
  `total_mass`, `mean`, and `cdf(x)`.

`discretize` supports four endpoint/moment-matching methods:

- `"upper"`: interval mass `(x, x+h]` is allocated to the left endpoint.
- `"lower"`: interval mass `(x-h, x]` is allocated to the right endpoint.
- `"rounding"`: interval mass around the nearest grid point.
- `"unbiased"`: local first-moment matching on the grid.

Minimal example:

```python
import numpy as np
from ruin_theory import coverage_transform, discretize, exponential, limited_moment

claims = exponential(rate=2.0)
print(limited_moment(claims, np.array([1.0, 2.0])))

covered = coverage_transform(claims, deductible=0.5, limit=2.0, coinsurance=0.8)
grid = discretize(covered, from_=0.0, to=5.0, step=0.25, method="unbiased")
print(grid.total_mass, grid.mean)
```

## Aggregate Loss Distributions

`AggregateDistribution` represents a finite lattice approximation of an
aggregate loss `S = X_1 + ... + X_N`. Its probability mass can sum below one
when the infinite tail was intentionally truncated.

Constructor arguments:

- `grid`: strictly increasing support values.
- `pmf`: non-negative probabilities with total mass at most one.
- `name`: optional label.
- `metadata`: optional dictionary of construction details.

Useful methods:

- `cdf(x)` and `survival(x)`.
- `ppf(q)`, `quantile(q)`, `value_at_risk(q)`.
- `mean()` and `variance()`.
- `tail_value_at_risk(level, allow_truncated=False)`.

Panjer recursion:

- `panjer_recursion(severity_pmf, frequency, frequency_params=None,
  max_aggregate=None, support=None, grid_step=1, normalize_severity=False,
  name=None)`.
- `compound_poisson_distribution(severity_pmf, rate=None, mean=None, ...)` is a
  convenience wrapper.

Supported frequency laws are Poisson, binomial, geometric, and negative
binomial. `severity_pmf[j]` is the mass at amount `j * grid_step`, unless an
explicit equally spaced `support` starting at zero is supplied.

Minimal example:

```python
from ruin_theory import compound_poisson_distribution

aggregate = compound_poisson_distribution(
    severity_pmf=[0.7, 0.2, 0.1],
    rate=3.0,
    max_aggregate=20,
)
print(aggregate.mean(), aggregate.value_at_risk(0.95))
```

## Risk Processes

### `CramerLundbergProcess`

Classical reserve process

```text
R_t = u + c t - sum_{i <= N_t} X_i
```

with Poisson arrivals.

Arguments:

- `initial_capital`: initial reserve `u`, non-negative.
- `premium_rate`: deterministic premium inflow rate `c`, non-negative.
- `claim_arrival_rate`: baseline Poisson intensity.
- `claim_distribution`: a `ClaimDistribution`.
- `prevention`: optional `PreventionProgram`.
- `by_claims`: tuple of `ByClaimModel` objects.
- `capital_injections`: tuple of `CapitalInjectionModel` objects.
- `name`: metadata label.

Derived properties:

- `claim_arrival_rate`: prevention-adjusted mean frequency.
- `expected_claim_amount`: primary severity after linear prevention plus
  expected by-claims.
- `claim_intensity`: `claim_arrival_rate * expected_claim_amount`.
- `safety_loading`: `premium_rate / claim_intensity - 1`.

Minimal example:

```python
from ruin_theory import CramerLundbergProcess, exponential

model = CramerLundbergProcess(
    initial_capital=2.0,
    premium_rate=1.0,
    claim_arrival_rate=3.0,
    claim_distribution=exponential(rate=5.0),
)
print(model.safety_loading)
```

### `SparreAndersenProcess`

Renewal-arrival reserve process. Use this when interarrival times are not
exponential.

Arguments:

- `initial_capital`, `premium_rate`, `claim_distribution`: as above.
- `interarrival_distribution`: positive `ClaimDistribution` for renewal waits.
- `prevention`, `by_claims`, `capital_injections`, `name`: as above.

Minimal example:

```python
from ruin_theory import SparreAndersenProcess, deterministic, exponential

model = SparreAndersenProcess(
    initial_capital=10.0,
    premium_rate=1.0,
    interarrival_distribution=deterministic(1.0),
    claim_distribution=exponential(rate=2.0),
)
```

## Frequency Models

Most users can construct processes directly. For advanced composition,
`FrequencyModel.poisson(rate)` and `FrequencyModel.renewal(interarrival_distribution)`
provide the frequency layer used by `RiskProcess`.

```python
from ruin_theory import FrequencyModel, deterministic

poisson = FrequencyModel.poisson(rate=3.0)
renewal = FrequencyModel.renewal(deterministic(1.0))
```

## Prevention

`PreventionProgram` acts on frequency and/or severity.

Arguments:

- `frequency_multiplier`: non-negative multiplier for the arrival rate.
  Values below one reduce frequency.
- `severity_multiplier`: non-negative multiplier for primary claim sizes.
- `severity_transform`: optional callable preserving the claim array shape.
  Use it for caps, deductibles, nonlinear mitigation, or engineering controls.
- `frequency_windows`: optional tuple of `(start, end, multiplier)` intervals.
  Intervals are interpreted as `[start, end)`. They let simulations encode
  periodic or temporary frequency controls.
- `name`: metadata label.

Closed-form formulas require stationary frequency prevention, so they reject
models with `frequency_windows`. Simulations support these windows by advancing
the claim clock through periods with different multipliers.

Minimal example:

```python
from ruin_theory import CramerLundbergProcess, PreventionProgram, exponential

prevention = PreventionProgram(
    frequency_multiplier=0.8,
    severity_multiplier=0.9,
    frequency_windows=((2.0, 4.0, 0.2),),
)
model = CramerLundbergProcess(
    initial_capital=5.0,
    premium_rate=1.0,
    claim_arrival_rate=3.0,
    claim_distribution=exponential(rate=5.0),
    prevention=prevention,
)
print(prevention.apply_frequency(3.0, time=3.0))
```

### `optimize_constant_prevention`

Optimizes the constant prevention spend `p` in the Gauchon et al. (2020)
classical model

```text
U(t, p) = u + (c - p)t - sum_{i <= N_p(t)} X_i,
```

where `N_p` has intensity `lambda(p)`. In that model, the prevention amount
maximizing the infinite-time non-ruin probability also maximizes the adjustment
coefficient. The implementation minimizes the loss ratio
`lambda(p) * E[X] / (c - p)`.

Arguments:

- `claim_distribution`: severity law with finite positive mean.
- `premium_rate`: gross premium rate `c`.
- `frequency_function`: callable `lambda(p)` returning the claim arrival
  intensity after spending `p` per unit time on prevention.
- `max_prevention`: optional upper bound for admissible `p`; defaults to just
  below `premium_rate`.
- `activation_threshold`: optional threshold `P` for models where prevention is
  inactive before `P`; the optimizer compares the active optimum with `p=0`.
- `initial_capital`: stored in the returned model.
- `compute_adjustment`: whether to compute the Lundberg coefficient when the
  optimized model has positive safety loading.
- `tol`: scalar optimizer tolerance.

Returns a `ConstantPreventionResult` with the optimal `amount`, net premium
rate, optimized frequency, loss ratio, safety loading, non-ruin probability at
zero, optional adjustment coefficient, induced `PreventionProgram`, and induced
`CramerLundbergProcess`.

Minimal example:

```python
import math
from ruin_theory import exponential, optimize_constant_prevention

result = optimize_constant_prevention(
    exponential(rate=1.0),
    premium_rate=10.0,
    frequency_function=lambda p: math.exp(-0.2 * p),
)
print(result.amount)
print(result.safety_loading)
print(result.adjustment_coefficient)
```

### `optimize_expected_surplus_prevention`

Optimizes the same constant prevention spend `p` for the expected surplus at a
fixed horizon. In the Gauchon et al. (2020) model,

```text
E[U(t, p)] = u + (c - p - lambda(p) E[X]) t.
```

The function therefore maximizes the net drift
`c - p - lambda(p) * E[X]`. This criterion is not the same as the infinite-time
ruin-probability criterion; for example, with `lambda(p) = exp(-a p)` and
`E[X] = 1`, an interior expected-surplus optimum exists only when `a > 1`.

Arguments:

- `claim_distribution`: severity law with finite positive mean.
- `premium_rate`: gross premium rate `c`.
- `frequency_function`: callable `lambda(p)` returning the claim arrival
  intensity after spending `p` per unit time on prevention.
- `horizon`: positive time horizon used in `E[U(t, p)]`.
- `max_prevention`: optional upper bound for admissible `p`; defaults to just
  below `premium_rate`.
- `activation_threshold`: optional threshold `P` for inactive prevention before
  `P`; the optimizer compares the active optimum with `p=0`.
- `initial_capital`: initial surplus `u`.
- `tol`: scalar optimizer tolerance.

Returns an `ExpectedSurplusPreventionResult` with the optimal `amount`, net
premium rate, optimized frequency, expected claim amount, net drift, expected
surplus at the horizon, induced `PreventionProgram`, and induced
`CramerLundbergProcess`.

Minimal example:

```python
import math
from ruin_theory import exponential, optimize_expected_surplus_prevention

result = optimize_expected_surplus_prevention(
    exponential(rate=1.0),
    premium_rate=10.0,
    frequency_function=lambda p: math.exp(-2.0 * p),
    horizon=3.0,
    initial_capital=5.0,
)
print(result.amount)
print(result.net_drift)
print(result.expected_surplus)
```

### `optimize_periodic_prevention_calendar`

Optimizes a discrete periodic prevention calendar under the Minier-Valla-Lefevre
seasonal-prevention KKT rule. The implemented finite-dimensional problem is

```text
min sum_i W_i exp(-a p_i)
subject to sum_i d_i p_i = pbar, 0 <= p_i <= pmax,
```

where `W_i` is the integrated seasonal pressure in period `i`, `d_i` is the
period duration as a fraction of the year, and `a` is the exponential prevention
effectiveness. For a light-tailed Lundberg pressure with season-dependent
severity, use `W_i` proportional to `Lambda_i * (M_i(rho) - 1)`; for an
expected-loss calendar, use `W_i` proportional to frequency times retained mean
severity.

Arguments:

- `weights`: non-negative period pressures `W_i`. They may already include the
  period duration.
- `annual_budget`: annual prevention budget `pbar`.
- `max_prevention`: instantaneous annualized spending cap `pmax`.
- `effectiveness`: exponential response parameter `a`.
- `durations`: optional positive period durations summing to one; defaults to
  equal periods.
- `lag_steps`: integer implementation lag. With `lag_steps=1`, spending in one
  period affects the next period.
- `tol`: numerical tolerance for budget feasibility and bisection.

Returns a `PeriodicPreventionResult` with spending `amounts`,
`effective_amounts`, pressure values, durations, the Lagrange threshold `tau`,
baseline/controlled/constant pressures, and a `frequency_windows()` method that
can feed `PreventionProgram`.

Related helpers:

- `periodic_pressure_weights(frequency_rates, severity_weights=None,
  durations=None)`: integrates period rates into annual weights. Use
  `severity_weights=None` for frequency weights, retained mean severities for
  expected-loss pressure, `M_i(rho)-1` for Lundberg pressure, and tail constants
  for heavy-tail pressure.
- `periodic_controlled_pressure(weights, amounts, effectiveness,
  lag_steps=0)`: evaluates the controlled annual pressure for a fixed calendar.
- `periodic_net_profit(premium_rate, annual_budget, claim_mean,
  controlled_frequency)`: returns `c - B(p) - m A(p)`.
- `periodic_lundberg_coefficient(claim_distribution, premium_rate,
  annual_budget, controlled_frequency, upper=None, tol=1e-12)`: solves
  `rho * (c - B(p)) = A(p) * (M_X(rho) - 1)`.

Minimal example:

```python
import numpy as np
from ruin_theory import (
    exponential,
    optimize_periodic_prevention_calendar,
    periodic_lundberg_coefficient,
    periodic_pressure_weights,
)

monthly_pressure = np.array([0.09, 0.07, 0.04, 0.02, 0.01, 0.01,
                             0.01, 0.02, 0.03, 0.05, 0.08, 0.11])
calendar = optimize_periodic_prevention_calendar(
    monthly_pressure,
    annual_budget=0.08,
    max_prevention=0.25,
    effectiveness=5.0,
)
print(calendar.amounts)
print(calendar.controlled_pressure, calendar.constant_pressure)

frequency_weights = periodic_pressure_weights([3.0] * 12)
rho = periodic_lundberg_coefficient(
    exponential(rate=5.0),
    premium_rate=1.0,
    annual_budget=0.0,
    controlled_frequency=frequency_weights.sum(),
)
print(rho)
```

### Heavy-Tail Periodic Prevention

For regularly varying annual/event losses with tail index `alpha in (0, 1)`,
the heavy-tail follow-up replaces the Lundberg coefficient by a controlled
annual tail constant. If frequency prevention is `exp(-a p)` and multiplicative
severity prevention is `exp(-b p)`, the effective exponential response is
`a + alpha * b`.

Available functions:

- `optimize_heavy_tail_prevention_calendar(...)`: optimizes the periodic
  tail-pressure calendar and optionally computes the large-budget expected
  ruin-time asymptotic when `annual_capacity` is supplied.
- `heavy_tail_expected_ruin_time_asymptotic(tail_index, annual_capacity,
  tail_constant, annual_budget=0)`: returns
  `c_p**(alpha/(1-alpha)) * (C Gamma(1-alpha))**(-1/(1-alpha))`.
- `heavy_tail_one_big_jump_ruin_probability(calendar, tail_index,
  initial_capital, annual_capacity, horizon, ...)`: finite-horizon
  one-big-jump heuristic for a stepwise periodic calendar.

Minimal example:

```python
from ruin_theory import optimize_heavy_tail_prevention_calendar

result = optimize_heavy_tail_prevention_calendar(
    [0.02, 0.07, 0.03, 0.01],
    tail_index=0.5,
    annual_budget=0.12,
    max_prevention=0.36,
    frequency_effectiveness=5.0,
    severity_effectiveness=0.0,
    annual_capacity=1.0,
)
print(result.amounts)
print(result.controlled_tail_pressure)
print(result.expected_time_to_ruin_asymptotic)
```

## By-Claims And Capital Injections

### `ByClaimModel`

Secondary claims triggered by primary claims.

Arguments:

- `probability`: probability that a primary claim triggers a secondary count.
- `distribution`: secondary severity distribution.
- `count_mean`: mean number of secondary claims once triggered.
- `count_distribution`: `"poisson"` or `"geometric"` on `{0, 1, ...}`.
- `name`: metadata label.

The model exposes `sample_total(n_primary, rng)` and `expected_amount_per_primary()`.
For light-tail formulas with by-claims, the count PGF is used in the aggregate
claim MGF.

Minimal example:

```python
from ruin_theory import ByClaimModel, CramerLundbergProcess, deterministic, exponential

by_claim = ByClaimModel(
    probability=0.25,
    distribution=deterministic(2.0),
    count_mean=1.5,
    count_distribution="poisson",
)
model = CramerLundbergProcess(
    premium_rate=5.0,
    claim_arrival_rate=1.0,
    claim_distribution=exponential(rate=2.0),
    by_claims=(by_claim,),
)
print(model.expected_claim_amount)
```

### `CapitalInjectionModel`

Independent positive jumps in reserve trajectories.

Arguments:

- `rate`: Poisson rate of injection arrivals.
- `distribution`: positive injection-size distribution.
- `name`: metadata label.

These injections are simulation-only; closed-form classical formulas reject
them.

```python
from ruin_theory import CapitalInjectionModel, deterministic

injection = CapitalInjectionModel(rate=0.2, distribution=deterministic(5.0))
```

## Simulation And Monte Carlo

### `simulate_path`

Simulates one reserve path.

Arguments:

- `model`: `RiskProcess`.
- `horizon`: positive finite time horizon.
- `seed`: integer seed or `np.random.Generator`.
- `max_events`: guard against infinite loops.
- `stop_at_ruin`: if true, stop when reserve first becomes negative.

Returns a `SimulationPath` with arrays for times, reserves, claim times,
claim sizes, injection times and injection sizes.

```python
from ruin_theory import CramerLundbergProcess, exponential, simulate_path

model = CramerLundbergProcess(claim_distribution=exponential(rate=2.0))
path = simulate_path(model, horizon=5.0, seed=123)
print(path.ruined, path.terminal_reserve, path.minimum_reserve)
```

### `estimate_ruin_probability`

Crude Monte Carlo finite-horizon ruin estimator.

Arguments:

- `model`, `horizon`: as above.
- `n_simulations`: number of independent paths.
- `ci_level`: confidence level in `(0, 1)`.
- `ci_method`: `"wilson"` (default) or `"normal"`.
- `seed`: integer seed.
- `return_paths`: if true, also return simulated paths.

Returns a `RuinEstimate` with `probability`, `standard_error`, `ci_low`,
`ci_high`, `n_simulations`, `horizon`, `ruin_times`, and `ci_method`.

```python
from ruin_theory import estimate_ruin_probability

estimate = estimate_ruin_probability(model, horizon=10.0, n_simulations=5000, seed=7)
print(estimate.probability, estimate.ci_low, estimate.ci_high)
```

### `simulate_terminal_reserves`

Returns terminal reserves from repeated full-horizon paths.

```python
from ruin_theory import simulate_terminal_reserves

terminal = simulate_terminal_reserves(model, horizon=10.0, n_simulations=1000, seed=7)
print(terminal.mean())
```

## Closed Forms And Approximations

Closed-form functions are intentionally strict. They require a stationary
`CramerLundbergProcess`, no capital injections, and no nonlinear severity
transform. Primary-claim formulas also reject by-claims.

Available functions:

- `ultimate_ruin_exponential(model, u=None)`: exact ultimate ruin probability
  for exponential primary claims.
- `ultimate_ruin_hyperexponential(model, u)`: exact ultimate ruin probability
  for mixtures of exponentials.
- `ultimate_ruin_phase_type(model, u=None)`: exact ultimate ruin probability
  for phase-type claims in the Cramer-Lundberg model. It uses
  `psi(u) = rho * beta exp((T + rho t beta) u) 1`, with `beta` the equilibrium
  PH initial vector and `t = -T 1`.
- `finite_time_ruin_exponential(model, u, horizon)`: finite-time formula for
  exponential primary claims.
- `expected_time_to_ruin_exponential(model, u=None)`: conditional mean time to
  ruin for exponential claims under the net profit condition.
- `adjustment_coefficient(model, upper=None, tol=1e-12)`: Lundberg root.
- `lundberg_bound(model, u, gamma=None)`: `exp(-gamma u)`.
- `cramer_lundberg_asymptotic(model, u, gamma=None)`: light-tail asymptotic
  `C exp(-gamma u)`.
- `pollaczek_khinchine_monte_carlo(model, u, n_simulations=50000, seed=None)`:
  ultimate ruin estimator via the geometric-sum representation.
- `ultimate_ruin_panjer(model, u=None, step=..., max_value=...,
  discretization="upper")`: deterministic lattice approximation of ultimate
  ruin using the Pollaczek-Khinchine compound-geometric representation.
- `discrete_pollaczek_khinchine_ultimate_ruin(ladder_height_pmf, surplus,
  step=1, rho=..., max_aggregate=None)`: lower-level lattice ruin probability
  from a discretized equilibrium severity law.
- `equilibrium_severity_pmf(distribution, step=..., max_value=...,
  method="upper")`: discretized integrated-tail severity distribution.
- `de_vylder_approximation(model, u)`: three-moment exponential approximation.
- `integrated_tail_survival(distribution, u, scale=1.0)`: equilibrium tail
  `E[(scale X - u)_+] / E[scale X]` for supported severity families.
- `heavy_tail_integrated_tail_asymptotic(model, u, integrated_tail_survival=None)`:
  subexponential approximation `rho / (1-rho) * tail(u)`.

Minimal example:

```python
import numpy as np
from ruin_theory import (
    CramerLundbergProcess,
    adjustment_coefficient,
    exponential,
    lundberg_bound,
    ultimate_ruin_exponential,
)

model = CramerLundbergProcess(
    premium_rate=1.0,
    claim_arrival_rate=3.0,
    claim_distribution=exponential(rate=5.0),
)
u = np.array([0.0, 1.0, 2.0])
gamma = adjustment_coefficient(model)
print(ultimate_ruin_exponential(model, u))
print(lundberg_bound(model, u, gamma=gamma))
```

Phase-type ruin example:

```python
import numpy as np
from ruin_theory import CramerLundbergProcess, phase_type, ultimate_ruin_phase_type

claims = phase_type([1.0, 0.0], [[-2.0, 2.0], [0.0, -2.0]])
model = CramerLundbergProcess(
    premium_rate=2.0,
    claim_arrival_rate=1.0,
    claim_distribution=claims,
)
print(ultimate_ruin_phase_type(model, np.array([0.0, 1.0, 2.0])))
```

Panjer/Pollaczek-Khinchine example:

```python
import numpy as np
from ruin_theory import CramerLundbergProcess, exponential, ultimate_ruin_panjer

model = CramerLundbergProcess(
    premium_rate=1.0,
    claim_arrival_rate=0.5,
    claim_distribution=exponential(rate=1.0),
)
u = np.array([0.0, 1.0, 2.0, 4.0])
print(ultimate_ruin_panjer(model, u, step=0.05, max_value=30.0))
```

## Plotting

Plotting functions accept an optional Matplotlib `Axes` and return the axis.

Available diagnostics:

- `plot_path(path, ax=None, show_ruin=True)`: one reserve trajectory.
- `plot_paths(paths, ax=None, alpha=0.25)`: overlay several trajectories.
- `plot_ruin_curve(u, probabilities, ax=None, label=None, ci_low=None,
  ci_high=None, band_alpha=0.18)`: probability curve with optional band.
- `plot_ruin_time_histogram(estimate, ax=None, bins=30)`: conditional ruin-time
  histogram from a Monte Carlo estimate.
- `plot_terminal_reserve_distribution(terminal_reserves, ax=None, bins=30,
  quantiles=(0.05, 0.5, 0.95), show_zero=True)`: terminal reserve histogram
  with zero and quantile markers.
- `plot_prevention_calendar(calendar, ax=None, labels=None,
  show_effective=True)`: bar plot of a periodic prevention calendar, with the
  lagged effective calendar overlaid when relevant.
- `plot_periodic_pressure(calendar, ax=None, labels=None,
  show_controlled=True)`: baseline and controlled periodic pressure weights.

Minimal example:

```python
import numpy as np
from matplotlib import pyplot as plt
from ruin_theory import (
    estimate_ruin_probability,
    optimize_periodic_prevention_calendar,
    plot_periodic_pressure,
    plot_prevention_calendar,
    plot_ruin_curve,
    plot_terminal_reserve_distribution,
    simulate_terminal_reserves,
    ultimate_ruin_exponential,
)

u = np.linspace(0.0, 8.0, 100)
probabilities = ultimate_ruin_exponential(model, u)
estimate = estimate_ruin_probability(model, horizon=10.0, n_simulations=2000, seed=123)
terminal = simulate_terminal_reserves(model, horizon=10.0, n_simulations=2000, seed=123)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
plot_ruin_curve(u, probabilities, ax=axes[0], label="ultimate")
plot_terminal_reserve_distribution(terminal, ax=axes[1])
calendar = optimize_periodic_prevention_calendar(
    [0.09, 0.07, 0.04, 0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.11],
    annual_budget=0.08,
    max_prevention=0.25,
    effectiveness=5.0,
)
plot_prevention_calendar(calendar)
plot_periodic_pressure(calendar)
plt.show()
```

## Current Limits And Roadmap

Implemented now:

- Classical Cramer-Lundberg exact formulas for exponential and hyperexponential
  primary claims.
- Phase-type severity distributions and exact Cramer-Lundberg ultimate ruin
  probabilities for phase-type primary claims.
- Loss moments, coverage transformations and lattice discretization.
- Aggregate-loss distributions by Panjer recursion for common counting laws.
- Deterministic Pollaczek-Khinchine/Panjer ruin approximations.
- Constant single-risk prevention optimization for ruin probability,
  adjustment coefficient and expected surplus following Gauchon et al. (2020).
- Periodic prevention calendars with projected-log KKT allocation, lagged
  calendars, and heavy-tail tail-pressure optimization.
- Renewal and prevention-rich models in simulation.
- By-claims with Poisson or geometric secondary counts.
- Equilibrium-tail helper and heavy-tail asymptotic path.
- Diagnostics for trajectories, ruin curves, ruin times and terminal reserves.

Planned extensions:

- Matrix-exponential extensions beyond standard phase-type severities.
- Phase-type renewal waits and matrix-valued finite-time ruin solvers.
- Gerber-Shiu penalties with surplus-before-ruin and deficit-at-ruin records.
- Discrete-time INAR/BINAR by-claim processes.
- Finite-horizon dynamic seasonal prevention beyond fixed annual calendars.
- Two-claim-type prevention from Gauchon et al. (2021).
