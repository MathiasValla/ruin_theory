# Prevention Manuscripts: Validation Scope

Sources: Minier, Valla and Lefevre, *Seasonal prevention in ruin theory:
periodic control, Lundberg bounds, and a national storm-loss-day application*,
supplied revision (10); Valla, *How long can premiums compensate infinite-mean
claims?*, supplied revision of 5 October 2026. Both are ongoing manuscripts.
The implementation is subject to scientific review, not a certification of
the source manuscripts' theoretical claims.

## Seasonal Prevention

| Case | Implementation | Numerical check |
| --- | --- | --- |
| Fixed and free budgets, exponential/quadratic/reciprocal responses | `optimize_seasonal_prevention` | Analytic inverse-marginal allocations compared with constrained SLSQP |
| Pointwise caps, annual ceilings and mandatory minimum spending | Same function | Budget feasibility and boundary tests; minimum spending is required in every interval |
| Rate versus interval-amount conventions | `budget_convention` | Explicit cost weights and separate response variables; monthly amounts must be converted before simulation |
| General convex responses | Callable response and SLSQP | Solver convergence checked; no nonconvex global-optimality assertion |
| Continuous seasonal profiles | `integrate_seasonal_pressure` | Quadrature and refinement of step calendars |
| Delayed effects | `lag_steps` | Uniform intervals, benefit shifted separately from cost |
| Finite dynamic calendars | Existing remaining-budget DP and `compare_seasonal_prevention` | Actual durations and matching horizon; not reserve-feedback stochastic control |
| Seasonal light-tail adjustment coefficients | `seasonal_lundberg_coefficient` | Affine, cosine and triangular exponential profiles checked against meshes |
| Root-optimal calendars | `optimize_seasonal_lundberg` | Nested convex allocation checked against a heterogeneous-severity grid and an analytic free-budget case |
| Finite-time ruin, starting season and recorded paths | `compare_seasonal_prevention` | Shared-event coupling, strict negative reserve, continuous negative-drift crossings, individual and paired MC errors |
| Historical storm objectives | `examples/seasonal_storm_replication.py` | Four loss/root pairs match the source table's rounding using private standardized inputs |
| Historical calibration uncertainty | Same script, `--bootstrap 5000` | Whole years resampled; calendar reoptimized; frequency, loss and relative timing gain reported |
| Historical reserve grid | Same script, `--historical-paths 50000` | Capitals 1, 2 and 4; independent estimates compared with combined MC errors |
| Historical/projected climate scenarios | Same script, `--paths N` | All 16 calibrations; fixed-control and scenario-repriced premium designs are distinct |

The empirical adapter verifies standardized monthly inputs rather than
silently replacing calibrated losses with raw series. Its inputs and cached
outputs are private and excluded from distributions. Redistribution rights
and an accessible data archive must be settled before JSS submission.

Mandatory minimum rates cover the source's simple threshold variant.
Optional on/off activation with discontinuous benefits is a different,
nonconvex problem and is not claimed to have a globally solved optimizer.
An averaged adjustment coefficient alone does not justify a uniform-in-phase
Lundberg bound. Source assumptions and phase prefactors must be checked.

## Heavy-Tail Prevention

`examples/prevention_paper_reproductions.py --full` uses the source grids and
sample sizes for the baseline mean curve, stable-limit Laplace transform,
tail-index robustness, frequency/heavy-severity/light-severity interventions,
and finite-mean equal-loss comparison. The six experiment blocks are not
substituted with finite-horizon or one-big-jump approximations.

The annual model uses a Pareto-I/light mixture. Reducing heavy frequency moves
probability mass to the light component. Zero light losses permit exact
geometric event skipping; nonzero light losses require annual crossings.
The limit is a safety condition, not a censoring convention: if even one path
has not ruined, no complete mean estimate is returned. Completed cases are
cached under the ignored output directory; the cache key includes model,
premium, seed, replication count, limit and implementation hash.

The finite-mean comparison uses Lomax/exponential streams, equal removed
annual mean loss and a common prevention cost, with exact stationary-excess
Pollaczek--Khinchine sampling. It is an ultimate probability experiment,
not an annual infinite-mean mixture experiment.

`examples/compare_heavy_tail_paper.py` compares independent runs with the
authors' CSVs. It reports discrepancies in combined Monte Carlo standard
errors. A five-SE screen is a diagnostic, not a simultaneous significance
test, and does not demand equality with first-order asymptotic predictions.

The 7 October 2026 full execution completed all 83 annual cases without
censoring and the 600,000-replication-per-policy PK comparison. All 292
comparisons with the authors' independent CSVs were within five combined
standard errors; the maximum standardized difference was 2.954.

## Review

Charles's reviewer packet includes these implementations and the existing
`integer_byclaims` INAR/BINAR modules, tests and validation notebook.
Review notes should distinguish mathematical correctness, API/usability and
teaching value. Passing tests and independent numerical comparisons do not
replace a specialist's review of conventions and source fidelity.
