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
- `lomax(shape, scale)`: shifted Pareto/Pareto II on `[0, infinity)`, useful
  for the local INAR/BINAR scripts that simulate `(Pareto - 1) * scale`.
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

## INAR/BINAR By-Claim Processes

This layer implements the discrete by-claim models used in the local INAR/BINAR
reference scripts. It is separate from `CramerLundbergProcess` because the
secondary claim counts are temporally dependent instead of being attached
independently to each primary claim.

### `INARByClaimModel`

Univariate INAR(1) by-claim model:

```text
N_k ~ Poisson(lambda),
M_0 = rho o Q_0 + N_0,   Q_0 ~ Poisson(gamma_0),
M_k = rho o M_{k-1} + N_k.
```

Arguments:

- `initial_capital`: initial reserve.
- `premium_per_period`: deterministic premium income per discrete period.
- `primary_count_mean`: Poisson mean `lambda` for primary claim counts.
- `initial_byclaim_mean`: Poisson mean `gamma_0` for initial by-claim stock.
- `reproduction`: binomial thinning coefficient `rho`.
- `primary_distribution`: severity distribution for primary claims.
- `byclaim_distribution`: severity distribution for by-claims.
- `name`: optional metadata label.

Useful methods:

- `expected_byclaim_counts(periods)`: theoretical by-claim count means.
- `expected_terminal_reserve(periods)`: theoretical terminal-reserve mean.

The local scripts use `T` as an array length and then report/check quantities
up to `reserve[T - 1]`. To reproduce those printed quantities directly, pass
`periods=T-1`. Passing `periods=T` uses the package convention and evaluates
the full finite horizon after `T` period updates. The scripts' shifted Pareto
sampling convention is represented by `lomax(shape, scale)`.

Simulation functions:

- `simulate_inar_byclaim_path(model, periods, seed=None, stop_at_ruin=True,
  ruin_threshold=0, ruin_inclusive=True)`.
- `simulate_inar_byclaim_terminal_reserves(model, periods, n_simulations,
  seed=None)`.
- `estimate_inar_byclaim_ruin_probability(model, periods, n_simulations=10000,
  ci_level=0.95, ci_method="wilson", seed=None, ...)`.

Minimal example:

```python
from ruin_theory import (
    INARByClaimModel,
    deterministic,
    estimate_inar_byclaim_ruin_probability,
    simulate_inar_byclaim_path,
)

model = INARByClaimModel(
    initial_capital=0.0,
    premium_per_period=36.0,
    primary_count_mean=10.0,
    initial_byclaim_mean=10.0,
    reproduction=0.9,
    primary_distribution=deterministic(2.0),
    byclaim_distribution=deterministic(1.0),
)
path = simulate_inar_byclaim_path(model, periods=11, seed=123)
estimate = estimate_inar_byclaim_ruin_probability(model, periods=11, seed=123)
print(path.ruined, estimate.probability)
```

### `BINARByClaimModel`

Bivariate BINAR(1) by-claim model with primary innovations
`N_k = (N_{1,k}, N_{2,k})` and thinning matrix `A`:

```text
M_0 = A o Q_0 + N_0,
M_k = A o M_{k-1} + N_k,
```

where each `A[i, j] o M_j` is an independent binomial thinning from previous
type `j` into current type `i`.

Arguments:

- `initial_capital`, `premium_per_period`: as above.
- `primary_count_means`: two Poisson means for primary claim types.
- `initial_byclaim_means`: two Poisson means for the initial by-claim stock.
- `reproduction_matrix`: 2-by-2 binomial thinning matrix.
- `primary_distributions`: two primary severity laws.
- `byclaim_distributions`: two by-claim severity laws.
- `name`: optional metadata label.

Simulation functions mirror the INAR names:
`simulate_binar_byclaim_path`, `simulate_binar_byclaim_terminal_reserves`, and
`estimate_binar_byclaim_ruin_probability`.

Minimal example:

```python
from ruin_theory import BINARByClaimModel, deterministic, simulate_binar_byclaim_path

model = BINARByClaimModel(
    initial_capital=1000.0,
    premium_per_period=15000.0,
    primary_count_means=(5.0, 7.0),
    initial_byclaim_means=(1.0, 1.0),
    reproduction_matrix=((0.41, 0.10), (0.05, 0.30)),
    primary_distributions=(deterministic(10.0), deterministic(1.0)),
    byclaim_distributions=(deterministic(0.5), deterministic(0.5)),
)
path = simulate_binar_byclaim_path(model, periods=10, seed=123)
print(path.terminal_reserve)
```

The generic `IntegerByClaimPath` stores reserves, primary counts, by-claim
counts, primary losses, by-claim losses, and the first ruin period. Generic
helpers `simulate_integer_byclaim_path`,
`simulate_integer_byclaim_terminal_reserves`, and
`estimate_integer_byclaim_ruin_probability` accept either model type.

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
- `finite_time_ruin_discrete(claim_pmf, initial_capital, premium_rate,
  claim_arrival_rate, horizon, method="seal")`: exact finite-time ruin
  probability for integer-valued claim sizes in the Cramer-Lundberg model.
  `initial_capital` may be any non-negative real value measured on the same
  unit lattice as the integer claims.
  Available methods are `"seal"` for the stable probability-only
  Seal/Takacs formula, `"takacs"` for the zero-initial-capital Takacs formula,
  `"picard-lefevre"` for the original Picard-Lefevre pseudo-probability
  formula, and `"inventory"` for the direct inventory-date recursion.
- `finite_time_ruin_discrete_inventory(claim_pmf, inventory_times,
  retained_counts, claim_arrival_rate=None, arrival_means=None)`: exact
  Rulliere-Loisel inventory recursion. `retained_counts[i]` is the number of
  aggregate-claim lattice states retained as safe at `inventory_times[i]`.
  Provide either one homogeneous `claim_arrival_rate` or interval-specific
  non-homogeneous Poisson means `arrival_means`.
- `finite_time_ruin_discrete_boundary(claim_pmf, inventory_times,
  boundary_values, claim_arrival_rate=None, arrival_means=None,
  convention="negative", boundary_kind="value")`: exact finite-time ruin for
  an increasing deterministic boundary `h(t)`. `convention="negative"` means
  ruin occurs when reserve is strictly negative, while
  `convention="nonpositive"` treats zero reserve as ruin. Use
  `boundary_kind="crossing"` when `inventory_times` are inverse crossing dates
  `v_n`, as in Picard-Lefevre and Rulliere-Loisel formulas.
- `finite_time_discrete_computation_set(initial_capital, premium_units,
  method="seal")`: returns the `(tau, j)` points used by the selected formula,
  for reproducing Picard-Lefevre vs Seal/Takacs computation-set figures.
- `compound_poisson_lattice_pmf(claim_pmf, mean, max_aggregate)`: exact
  compound-Poisson lattice masses `P(S=j)` for `j <= max_aggregate`; the
  unreturned tail is intentionally truncated.
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

Exact finite-time lattice example:

```python
from ruin_theory import finite_time_ruin_discrete

# Deterministic unit claims: P(W = 1) = 1.
ruin = finite_time_ruin_discrete(
    claim_pmf=[0.0, 1.0],
    initial_capital=5,
    premium_rate=1.25,
    claim_arrival_rate=1.0,
    horizon=10.0,
    method="seal",
)
print(ruin)

details = finite_time_ruin_discrete(
    claim_pmf=[0.0, 0.25, 0.50, 0.25],
    initial_capital=4,
    premium_rate=1.3,
    claim_arrival_rate=0.8,
    horizon=4.2,
    method="inventory",
    return_result=True,
)
print(details.inventory_times)
print(details.survival_probabilities)
```

## Gerber-Shiu Diagnostics

The Gerber-Shiu diagnostic layer estimates the finite-horizon discounted
penalty

```text
E[exp(-delta tau) w(R_{tau-}, |R_tau|); tau <= horizon],
```

where `R_{tau-}` is the surplus immediately before ruin and `|R_tau|` is the
deficit at ruin. This simulation layer follows the Gerber-Shiu definition from
Asmussen and Albrecher, Chapter XII; matrix-valued closed-form solvers remain a
separate planned analytical extension.

Functions:

- `estimate_gerber_shiu(model, horizon, n_simulations=10000, penalty=None,
  discount_rate=0, ci_level=0.95, seed=None, max_events=1000000,
  return_paths=False)`: simulate paths and estimate the discounted penalty.
- `gerber_shiu_from_paths(paths, penalty=None, discount_rate=0, ci_level=0.95,
  horizon=None)`: compute the same diagnostic from pre-simulated
  `SimulationPath` objects.

Arguments:

- `model`: any `RiskProcess` accepted by `simulate_path`.
- `horizon`: finite simulation horizon.
- `n_simulations`: positive integer number of Monte Carlo paths.
- `penalty`: callable `penalty(surplus_before_ruin, deficit_at_ruin)`. If
  omitted, `w == 1`; with `discount_rate=0`, the estimate is the finite-horizon
  ruin probability.
- `discount_rate`: non-negative `delta` in the Gerber-Shiu transform.
- `ci_level`: normal confidence interval level for the sample mean of the
  discounted penalty.
- `max_events`: positive integer event cap passed to `simulate_path`.
- `return_paths`: return the simulated paths together with the result.

Returned `GerberShiuResult` fields:

- `estimate`, `standard_error`, `ci_low`, `ci_high`, `n_simulations`,
  `horizon`, `discount_rate`.
- `ruin_times`: finite ruin times and `inf` for non-ruined paths.
- `surplus_before_ruin`, `deficits_at_ruin`, `claim_causing_ruin`: ruin-state
  diagnostics, with `nan` for non-ruined paths.
- `penalty_values`, `discounted_penalties`: raw and discounted simulated
  penalty values.
- Convenience properties: `ruined`, `ruin_probability`,
  `mean_surplus_before_ruin`, `mean_deficit_at_ruin`.

Minimal example:

```python
from ruin_theory import CramerLundbergProcess, deterministic, estimate_gerber_shiu

model = CramerLundbergProcess(
    initial_capital=1.0,
    premium_rate=0.8,
    claim_arrival_rate=1.2,
    claim_distribution=deterministic(2.0),
)
result = estimate_gerber_shiu(
    model,
    horizon=5.0,
    n_simulations=5000,
    penalty=lambda surplus, deficit: deficit,
    discount_rate=0.03,
    seed=123,
)
print(result.estimate, result.mean_deficit_at_ruin)
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
- `plot_deficit_at_ruin(result, ax=None, bins=30)`: conditional deficit-at-ruin
  histogram from a `GerberShiuResult`.
- `plot_surplus_before_ruin(result, ax=None, bins=30)`: conditional
  surplus-before-ruin histogram from a `GerberShiuResult`.
- `plot_gerber_shiu_scatter(result, ax=None, alpha=0.7)`: surplus/deficit
  scatter plot colored by ruin time.
- `plot_finite_time_discrete_survival(result, ax=None, label=None)`: exact
  survival curve at inventory dates from an inventory-style finite-time result.
- `plot_finite_time_discrete_boundary(result, ax=None, label=None)`: plot the
  deterministic boundary values stored in a `FiniteTimeDiscreteBoundaryResult`.
- `plot_finite_time_discrete_computation_set(initial_capital, premium_units,
  method="seal", ax=None)`: computation-set scatter plot for Picard-Lefevre,
  Seal/Takacs or inventory formulas.
- `plot_terminal_reserve_distribution(terminal_reserves, ax=None, bins=30,
  quantiles=(0.05, 0.5, 0.95), show_zero=True)`: terminal reserve histogram
  with zero and quantile markers.
- `plot_prevention_calendar(calendar, ax=None, labels=None,
  show_effective=True)`: bar plot of a periodic prevention calendar, with the
  lagged effective calendar overlaid when relevant.
- `plot_periodic_pressure(calendar, ax=None, labels=None,
  show_controlled=True)`: baseline and controlled periodic pressure weights.
- `plot_integer_byclaim_path(path, ax=None, show_ruin=True)`: discrete
  INAR/BINAR reserve trajectory.
- `plot_integer_byclaim_counts(path, ax=None, kind="byclaim")`: primary or
  by-claim count bars by period.

Minimal example:

```python
import numpy as np
from matplotlib import pyplot as plt
from ruin_theory import (
    CramerLundbergProcess,
    INARByClaimModel,
    deterministic,
    exponential,
    estimate_gerber_shiu,
    estimate_ruin_probability,
    optimize_periodic_prevention_calendar,
    plot_deficit_at_ruin,
    plot_gerber_shiu_scatter,
    plot_integer_byclaim_counts,
    plot_integer_byclaim_path,
    plot_periodic_pressure,
    plot_prevention_calendar,
    plot_ruin_curve,
    plot_terminal_reserve_distribution,
    plot_surplus_before_ruin,
    simulate_inar_byclaim_path,
    simulate_terminal_reserves,
    ultimate_ruin_exponential,
)

model = CramerLundbergProcess(
    initial_capital=0.0,
    premium_rate=1.4,
    claim_arrival_rate=0.5,
    claim_distribution=exponential(1.0),
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

byclaim_model = INARByClaimModel(
    initial_capital=20.0,
    premium_per_period=6.0,
    primary_count_mean=1.5,
    initial_byclaim_mean=1.0,
    reproduction=0.35,
    primary_distribution=deterministic(1.0),
    byclaim_distribution=deterministic(0.75),
)
byclaim_path = simulate_inar_byclaim_path(byclaim_model, periods=8, seed=123)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
plot_integer_byclaim_path(byclaim_path, ax=axes[0])
plot_integer_byclaim_counts(byclaim_path, ax=axes[1], kind="byclaim")

gs = estimate_gerber_shiu(model, horizon=10.0, n_simulations=2000, seed=123)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
plot_deficit_at_ruin(gs, ax=axes[0])
plot_surplus_before_ruin(gs, ax=axes[1])
plot_gerber_shiu_scatter(gs, ax=axes[2])
plt.show()
```

## Current Limits And Roadmap

Implemented now:

- Classical Cramer-Lundberg exact formulas for exponential and hyperexponential
  primary claims.
- Exact finite-time ruin probabilities for integer-valued claim sizes using
  Seal/Takacs, Picard-Lefevre and direct inventory-recursion formulas.
- Exact finite-time inventory recursions for increasing deterministic
  boundaries, inverse crossing dates and interval-specific non-homogeneous
  Poisson arrival means.
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
- Discrete-time INAR/BINAR by-claim simulation layers.
- Gerber-Shiu discounted penalty simulation diagnostics with deficit-at-ruin and
  surplus-before-ruin plots.
- Equilibrium-tail helper and heavy-tail asymptotic path.
- Diagnostics for trajectories, ruin curves, ruin times and terminal reserves.

Planned extensions:

- Matrix-exponential extensions beyond standard phase-type severities.
- Phase-type renewal waits and matrix-valued finite-time ruin solvers.
- Matrix-valued/closed-form Gerber-Shiu solvers beyond simulation diagnostics.
- Finite-time exact discrete extensions from Picard-Lefevre,
  Rulliere-Loisel, Lefevre-Loisel and Castaner et al.:
  - Full non-constant premium-function builders `h(t)` with automatic inverse
    boundary-date extraction beyond the current explicit-inventory API.
  - Generalized Appell and Sheffer coefficient engines for arbitrary
    increasing boundaries, including base polynomials, coefficient triangular
    systems and Picard-Lefevre polynomial diagnostics.
  - Non-homogeneous compound-Poisson arrivals with time-varying intensity
    `lambda(t)` and time-varying severity laws, including aggregate-increment
    probabilities, exact finite-horizon recursions and computation-set plots.
  - Dependent and exchangeable claim-severity finite-horizon formulas based on
    joint partial sums and order-statistic tail probabilities, plus
    comonotonic/common-factor dependence examples and sensitivity plots.
  - Discrete-time non-homogeneous risk models with interest, inflation,
    beginning/end/mid-period premium timing, lower/upper discretization bounds,
    finite-time ruin probabilities and conditional surplus/deficit
    distributions at ruin.
  - Ruin-severity distribution `chi(t, x)`, conditional surplus given
    non-ruin, conditional deficit given ruin, quantile tables and the figures
    reported by Castaner et al.
  - Non-homogeneous Lundberg-type bounds, periodwise adjustment roots and the
    explicit compound-Poisson/exponential and normal-approximation premium
    principle formulas.
  - Reproduction examples for Rulliere-Loisel computation-set figures and
    tables, Picard-Lefevre polynomial identities, Lefevre-Loisel dependence
    examples and Castaner et al. discretization, severity, quantile and
    Lundberg-bound tables.
- Finite-horizon dynamic seasonal prevention beyond fixed annual calendars.
- Two-claim-type prevention from Gauchon et al. (2021).
