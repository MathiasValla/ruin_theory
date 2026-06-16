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

Minimal example:

```python
import numpy as np
from matplotlib import pyplot as plt
from ruin_theory import (
    estimate_ruin_probability,
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
plt.show()
```

## Current Limits And Roadmap

Implemented now:

- Classical Cramer-Lundberg exact formulas for exponential and hyperexponential
  primary claims.
- Renewal and prevention-rich models in simulation.
- By-claims with Poisson or geometric secondary counts.
- Equilibrium-tail helper and heavy-tail asymptotic path.
- Diagnostics for trajectories, ruin curves, ruin times and terminal reserves.

Planned extensions:

- Phase-type and matrix-exponential severity/wait models.
- Panjer recursion and aggregate-claim distribution objects.
- Actuar-style limited moments and coverage transformations.
- Gerber-Shiu penalties with surplus-before-ruin and deficit-at-ruin records.
- Discrete-time INAR/BINAR by-claim processes.
- Seasonal/periodic prevention optimization beyond simulation windows.
