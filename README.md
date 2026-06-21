# ruin-theory

`ruin-theory` is a Python package for classical and renewal risk processes.
The first release focuses on a reliable computational core:

- Cramer-Lundberg and Sparre-Andersen reserve processes.
- Severity distributions with moment, survival, Laplace and MGF helpers.
- Prevention programs acting on claim frequency and/or claim severity.
- By-claims and independent capital injections in simulation.
- Loss moments, coverage transformations and lattice discretization.
- Aggregate-loss distributions by Panjer recursion, with VaR and TVaR helpers.
- Exact Cramer-Lundberg formulas where implemented; simulation otherwise.
- Monte Carlo estimators, trajectory simulation and plotting diagnostics.

The package is being built from the notation and computational priorities in
Asmussen and Albrecher, *Ruin Probabilities*, Lefevre's ruin-theory notes, and
published actuarial software examples used as numerical reference checks.

## Documentation

- [Feature documentation](docs/features.md): models, distributions, prevention,
  by-claims, formulas, simulation and plots, with arguments and minimal code
  examples.
- [Reproducing R actuar examples](docs/reproduce_r_actuar_package.md): Python
  reproductions of the ruin-theory examples in `R_actuar_package.pdf`.
- [Scientific references](docs/references.md): full citations and
  acknowledgments.
- [INAR/BINAR by-claim examples](examples/inar_binar_byclaim_examples.py):
  reproducible discrete by-claim simulations and plots.
- [Gerber-Shiu diagnostics example](examples/gerber_shiu_diagnostics.py):
  finite-horizon discounted penalties with deficit/surplus plots.
- [Dividend-barrier diagnostics example](examples/barrier_dividends.py):
  barrier hitting probabilities, cumulative dividends and ruin-time plots.
- [Time-in-red allocation example](examples/red_time_allocation.py):
  Loisel red-time curves, negative-area criteria and reserve-allocation plots.
- [Markov-modulated common shocks example](examples/markov_modulated_common_shocks.py):
  finite-time multirisk recursion, solvency regions and dependence-impact plots.
- [Multirisk dividend-penalty example](examples/multirisk_dividend_penalties.py):
  CTMC approximation of dividends, insolvency penalties and ruin-state
  diagnostics.

## Quick start

```python
import numpy as np
from matplotlib import pyplot as plt
from ruin_theory import (
    CramerLundbergProcess,
    estimate_ruin_probability,
    exponential,
    simulate_terminal_reserves,
    ultimate_ruin_exponential,
)

claims = exponential(rate=5.0)
model = CramerLundbergProcess(
    initial_capital=2.0,
    premium_rate=1.0,
    claim_arrival_rate=3.0,
    claim_distribution=claims,
)

surplus = np.array([0.0, 1.0, 2.0])
estimate = estimate_ruin_probability(model, horizon=10.0, n_simulations=5_000, seed=123)
terminal_reserves = simulate_terminal_reserves(model, horizon=10.0, n_simulations=5_000, seed=123)
print(ultimate_ruin_exponential(model, u=surplus))
print(f"Estimated P(ruin by 10): {estimate.probability:.3f}")
print(f"5% terminal reserve quantile: {np.quantile(terminal_reserves, 0.05):.3f}")
```

The same pieces can be visualized with the plotting helpers:

```python
from ruin_theory import plot_path, plot_ruin_curve, plot_ruin_time_histogram, simulate_path

u = np.linspace(0.0, 8.0, 100)
probabilities = ultimate_ruin_exponential(model, u)
estimate = estimate_ruin_probability(model, horizon=10.0, n_simulations=5_000, seed=123)
path = simulate_path(model, horizon=10.0, seed=123)

_, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
plot_path(path, ax=axes[0])
plot_ruin_curve(u, probabilities, ax=axes[1], label="ultimate")
plot_ruin_time_histogram(estimate, ax=axes[2])
plt.show()
```

## Current scope

This first development pass prioritizes a correct, extensible core over a broad
catalog of closed forms. Implemented formula paths currently cover classical
Cramer-Lundberg primary-claim models with linear severity scaling:

- ultimate ruin for exponential and hyperexponential severities;
- ultimate ruin for phase-type severities in the Cramer-Lundberg model;
- finite-time ruin for exponential severities;
- exact finite-time ruin for integer-valued claim sizes and real reserves by stable
  Seal/Takacs formulas, Picard-Lefevre's formula and inventory recursions;
- exact finite-time inventory recursions for deterministic increasing
  boundaries, automatic inverse crossing dates and non-homogeneous Poisson
  arrival means or interval claim-size intensity measures;
- finite-horizon non-homogeneous discrete-time ruin recursions with discounted
  premiums, lower/upper discretization bounds, dependent or exchangeable claim
  scenarios, surplus/deficit-at-ruin distributions and periodwise Lundberg
  bounds;
- Picard-Lefevre generalized-Appell base polynomials, coefficients and
  homogeneous finite-time boundary formulas;
- constant-interest exponential ruin, double-barrier win-first probabilities,
  maximum-before-default hazards and sensitivity plots;
- horizontal dividend-barrier analytics, geometric/renewal dividend
  distributions, path simulation and comparison plots;
- time-in-red and integrated-negative-part diagnostics, Loisel derivative
  identity checks, Poisson-exponential infinite-horizon closed forms and
  optimal reserve allocation by equalizing active-line red times;
- Markov-modulated multirisk finite-time recursions with common shocks,
  arbitrary solvency regions, statewise vector compound-Poisson increments and
  dependence-impact plots;
- multirisk dividend and insolvency-penalty CTMC approximations with branch
  barriers, status-dependent premium interactions, ruin-state distributions and
  convergence diagnostics;
- Lundberg adjustment coefficients, bounds and light-tail asymptotics;
- Pollaczek-Khinchine Monte Carlo via equilibrium claim sampling;
- discrete Pollaczek-Khinchine/Panjer approximations for ultimate ruin;
- limited moments, coverage transformations and severity discretization;
- Panjer aggregate distributions for Poisson, binomial, geometric and
  negative-binomial frequencies;
- constant prevention optimization for ruin probability, adjustment coefficient
  and expected surplus in the Gauchon et al. (2020) model, with custom convex
  decreasing prevention-response functions;
- periodic prevention calendars with projected-log KKT allocation for
  exponential responses, constrained optimization for custom response functions,
  lagged calendars, annual Lundberg/net-profit helpers and heavy-tail
  tail-pressure optimization;
- discrete-time INAR/BINAR by-claim simulation, ruin estimation and
  diagnostics;
- Gerber-Shiu discounted penalty diagnostics with deficit-at-ruin and
  surplus-before-ruin plots;
- De Vylder three-moment approximation for supported severity families.

Sparre-Andersen arrivals, by-claims, INAR/BINAR dependent by-claims, capital
injections and nonlinear prevention are available in simulation and prevention
optimization.
Matrix-valued/closed-form Gerber-Shiu solvers remain planned beyond the current
simulation diagnostics. The two-claim-type prevention model remains planned
beyond the current single-risk and periodic prevention optimizers.
Premium-dependent finite-time dynamics and richer
matrix-valued dependence solvers remain planned beyond the current lattice
finite-time formulas; the detailed finite-time implementation roadmap is in
`docs/features.md`.

## Scientific references

Core references include Lundberg (1903, 1926), Cramer (1930, 1955), Sparre
Andersen (1957), Gerber (1979), Rolski et al. (1999), and Asmussen and
Albrecher (2010). Numerical validation examples and software comparisons cite
Dutang, Goulet and Pigeon (2008) and Goulet's `actuar` ruin-theory notes.
Finite-time discrete formulas follow Picard and Lefevre (1997, 1998),
Rulliere and Loisel (2004), Seal (1969), Takacs (1962), De Vylder (1999),
Ignatov, Kaishev and Krachunov (2001), and the finite-horizon reviews and
extensions by Lefevre and Loisel.
Time-in-red and reserve-allocation diagnostics follow dos Reis (1993), Gerber
(1988), Loisel (2005), and Dickson and dos Reis (1996).
Markov-modulated multirisk common-shock recursions and dependence diagnostics
follow Picard, Lefevre and Coulibaly (2003), Loisel (2004, 2005), and related
work by Cossette, Landriault and Marceau.
Multirisk dividend and insolvency-penalty approximations follow Asmussen and
Kella (2000), Frostig (2004), and Loisel's 2005 ISFA working paper on
Markov-modulated multirisk models with common shocks.
Prevention features are guided by Ehrlich and Becker (1972), Gauchon et al.
(2020, 2021), Schmidli (2008), and the seasonal-prevention manuscript by
Minier, Valla and Lefevre, with heavy-tail periodic prevention following the
2026 manuscript by Valla, Rivoire, Minier, Guibert and Loisel.

Claude Lefevre's lecture notes and comments helped orient the mathematical
reading for this project; formal references cite the original scientific
sources. We thank Claude Lefevre for the notes, discussions and guidance.
