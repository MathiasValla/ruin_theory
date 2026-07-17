# Validation

Validation is organized around small reproducible numerical targets. The goal is
to make the package auditable for a software-paper submission and useful for
teaching.

## Validation Notebooks

The notebooks in `notebooks/validation/` reproduce methods that were previously
validated mainly by internal consistency checks:

- `markov_modulated_multirisk_reproduction.ipynb`
- `multirisk_ctmc_dividends_reproduction.ipynb`
- `loisel_privault_sensitivity_reproduction.ipynb`
- `inar_binar_byclaim_reproduction.ipynb`

Run the notebook code cells without requiring Jupyter:

```bash
python scripts/check_validation_notebooks.py
```

Regenerate the notebooks:

```bash
python scripts/create_validation_notebooks.py
```

## Reproduction Targets

The manuscript and supplementary material should include at least one compact
numerical result for each method family.

| Source family | Small reproducible result | Current artifact |
| --- | --- | --- |
| Dutang, Goulet and Pigeon; `actuar` notes | Adjustment coefficient `0.1667`; exponential ruin table starts at `0.6000` and ends near `1.237e-09`; Beekman-Panjer Pareto bounds match the R notes. | `tests/test_reference_actuar_examples.py`; `examples/r_actuar_package_python.py` |
| Classical Cramer-Lundberg / Asmussen-Albrecher | For `lambda=3`, `c=1`, exponential rate `5`, `psi(u)=0.6 exp(-2u)` and adjustment coefficient `R=2`. | `tests/test_formulas.py`; `examples/quickstart.py` |
| Phase-type and matrix-analytic methods | One-phase PH/ME representations reduce to the exponential formula; two-branch PH matches the hyperexponential formula `(24 exp(-u)+exp(-6u))/35`. | `tests/test_formulas.py`; `tests/test_matrix_analytic.py` |
| Goffard-Loisel-Pommeret orthogonal-polynomial ultimate ruin approximation | Exponential case with `lambda=3`, `c=1`, exponential rate `5`: order-0 Laguerre approximation equals exact values `[0.6, 0.0812011699, 0.0109893833, 0.00002723996]` at reserves `[0,1,2,5]`; compound-geometric moments are `[1.0,0.3,0.3]`. | `tests/test_polynomial_expansion.py`; `examples/orthogonal_polynomial_ruin.py` |
| Goffard-Sarantsev level-dependent Levy-driven convergence rates | Compound-Poisson gamma-claim example with safety loading `eta=0.1`, premium `2.2` and zero diffusion gives `lambda_star=0.0312707` and convergence rate `0.00319329`; gap bound at horizons `[0,1]` with moment `3` is `[4.0,3.98724721]`. | `tests/test_levy_driven.py`; `examples/levy_convergence_rate.py` |
| Goffard ordered exit and Goffard-Lefevre ordered/dual-risk models | Uniform order-statistic rectangle check gives `0.5`; deterministic ordered two-sided survival is `0.9997623443` in the Poisson example; Poisson-exponential dual-risk atom is `exp(-1)=0.3678794412` and the CDF at `t=1` equals the atom. | `tests/test_ordered_risk.py`; `examples/ordered_risk_models.py` |
| Seal/Takacs/Picard-Lefevre/De Vylder finite-time formulas | De Vylder deterministic-claim table: ruin probabilities `0.765864440648`, `0.039901595038`, `0.000692886838`, `0.000004740559`, `0.000000014338` for reserves `0,5,10,15,20`. | `tests/test_finite_discrete.py`; `examples/finite_time_discrete_ruin.py` |
| Rulliere-Loisel / Lefevre-Loisel boundary formulas | Boundary recursion and generalized Appell computation reproduce the linear-boundary inventory formula for the same claim PMF and horizon. | `tests/test_finite_discrete.py` |
| Non-homogeneous finite-horizon discrete-time recursions | Two-period hand check gives ruin probability `0.75`; discount factors `[1.0, 1.1, 1.32]` for interest rates `[0.1, 0.2]`. | `tests/test_finite_discrete_time.py`; `examples/finite_discrete_time_castaner.py` |
| Interest-force and win-first formulas | Positive interest-force exponential formula matches the Segerdahl-style integral to relative tolerance `1e-12`; win-first probability equals `phi(u)/phi(u+v)`. | `tests/test_interest.py` |
| Dividend-barrier formulas | Barrier period-count PMF with hit probability `0.6` and continuation probability `0.25` is `[0.4, 0.45, 0.1125, 0.028125, 0.00703125]`. | `tests/test_dividends.py`; `examples/barrier_dividends.py` |
| Red time and reserve allocation | Piecewise-linear path `[1,-1,-2,2]` over times `[0,1,2,3]` has time-in-red `2.0` and negative area `2.25`; exponential closed forms satisfy the Loisel derivative identity. | `tests/test_red_time.py`; `examples/red_time_allocation.py` |
| Gerber-Shiu discounted penalties | Default Gerber-Shiu penalty recovers ruin probability; exponential closed form matches ultimate ruin; Poisson first-claim probability is reproduced. | `tests/test_gerber_shiu.py`; `examples/gerber_shiu_diagnostics.py` |
| Markov-modulated multirisk common shocks | One-period two-line manual check gives `any_line=0.75`, `total=0.25`, `hybrid=0.25`; positive-dependence impact is `-0.25`. | `notebooks/validation/markov_modulated_multirisk_reproduction.ipynb`; `tests/test_markov_modulated.py` |
| Multirisk CTMC dividends and insolvency penalties | One-line barrier case has state count `2`, expected time to ruin `0.5`, ruin probability `1.0`, expected dividends `[1.5]`, expected deficit `[1.0]`. | `notebooks/validation/multirisk_ctmc_dividends_reproduction.ipynb`; `tests/test_multirisk_dividends.py` |
| Gauchon et al. (2020) constant prevention | Exponential-frequency optimum matches `c - 1/a`; expected-surplus optimum matches `log(a)/a` in the test parameterization. | `tests/test_prevention_optimization.py`; `examples/matrix_analytic_prevention.py` |
| Gauchon et al. (2021) two-claim prevention | Zero-surplus usefulness condition returns positive prevention and the optimized large-claim arrival rate decreases relative to no prevention. | `tests/test_prevention_optimization.py`; `examples/matrix_analytic_prevention.py` |
| Seasonal and dynamic prevention | No-seasonality case returns a flat calendar; lagged prevention shifts spending one period before the pressure peak; projected-log KKT rule is reproduced. | `tests/test_prevention_optimization.py`; `examples/matrix_analytic_prevention.py` |
| KLR worsening-risk and regular-variation asymptotics | KLR table for speeds `[0.01,0.02,0.05,0.1,0.2]`: shape values near `[0.497,0.351,0.222,0.157,0.111]`; scale values near `[0.124,0.147,0.185,0.220,0.262]`. | `tests/test_climate_change.py`; `tests/test_regular_variation.py` |
| Loisel-Privault sensitivity | IBP estimator for `d psi(u,T)/du` agrees with finite differences of the exact exponential finite-time formula at `u=[0.5,1.0,1.5]` within four Monte Carlo standard errors. | `notebooks/validation/loisel_privault_sensitivity_reproduction.ipynb`; `tests/test_simulation.py` |
| Panjer recursion and loss-model discretization | Degenerate severities reproduce exact binomial, Poisson-thinning, geometric and negative-binomial count laws; TVaR/quantile conventions are checked. | `tests/test_aggregate.py`; `tests/test_loss_utilities.py` |
| INAR/BINAR by-claim simulations | INAR expected by-claim counts for four periods are `[11.0, 11.1, 11.11, 11.111]`; BINAR expected counts match the matrix recursion. | `notebooks/validation/inar_binar_byclaim_reproduction.ipynb`; `tests/test_integer_byclaims.py` |
| Package-level public API and plotting | Public API smoke tests verify exported names; plotting tests ensure diagnostics return Matplotlib axes without requiring display backends. | `tests/test_public_api.py`; `tests/test_plotting.py` |
