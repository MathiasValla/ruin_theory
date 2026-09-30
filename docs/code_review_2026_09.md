# Package Review: September 2026

Review completed on 2026-09-30 against baseline commit
`b4c5846eba1c0103b269c39edf3bbe46bc6b8986`. This is a development review,
not certification of the mathematical content or approval for journal submission.
The package remains a public research preview. AI/Codex assistance was used for
this review, its code changes, tests and documentation; human scientific review
remains necessary.

## Open Findings And Scientific Limits

These items remain relevant after the corrections below, in priority order.

1. **P2: zero-interest win-first probabilities with nonpositive loading are
   unsupported.** `interest.py`, `win_first_probability_exponential_interest_force`,
   uses a ratio of positive ultimate non-ruin probabilities. For premium 1,
   arrival rate 2 and exponential rate 1, both probabilities are zero and the
   function raises. Finite-barrier success is still meaningful; a separate
   scale-function/limiting implementation is needed. This restriction is now
   explicit in the method documentation. Positive-interest cases where subtraction
   rounds non-ruin to zero also require care.
2. **P2: generic moment quadrature can miss probability mass at extreme scales.**
   `losses.py`, `limited_moment` and continuous `coverage_transform` moments,
   still rely on quadrature for unsupported closed-form families. Direct formulas
   now cover the reproduced deterministic, empirical, exponential, Gamma and
   Erlang limited-moment cases; that does not establish accuracy for every
   custom law or transformed continuous distribution. Check integration scales
   and an independent moment oracle before using extreme parameters.
3. **P2: high-order polynomial/root calculations are not uniformly stable.**
   `finite_discrete.py` generalized Appell coefficients and signed-time formulas,
   `polynomial_expansion.py` raw-moment coefficient fitting, and adjustment roots
   near critical loading still need conditioning studies. Stable PMFs and tail
   evaluation do not establish stability of the entire approximation. Compare
   inventory recursions, grid refinements, exact special cases and alternate
   precision where appropriate.
4. **Finite approximations retain explicit modeling limits.** In
   `finite_discrete_time.py`, missing PMF mass is charged to ruin and deficit
   diagnostics omit the unknown tail. In `markov_modulated.py`, truncated
   compound-Poisson laws are conditional on retained counts; inspect the returned
   error bound. In `multirisk_dividends.py`, lower truncation and grid refinement
   remain caller responsibilities, claim vectors use lattice units, and dense
   terminal-rate storage still limits large models. Boundary callbacks are
   assumed continuous and nondecreasing, not certified by the software.
5. **Optimization and asymptotics are not global guarantees.** Prevention
   response convexity is checked only on sampled points; the dynamic calendar
   is a finite-grid optimum. Levy convergence rates use a finite level grid.
   One-big-jump calculations and climate equivalents are asymptotic quantities,
   not exact finite-capital probabilities. Custom regular-variation and
   infinite-mean assumptions need mathematical justification, especially at
   tail index one.
6. **Validation gaps remain.** No formal type checker or minimum-dependency
   environment was run. Local verification used Python 3.11.13, NumPy 2.4.6 and
   SciPy 1.17.1 on macOS arm64. The configured Linux/Python 3.10-3.12 CI matrix
   has not been run remotely for these uncommitted changes. Coverage is not a
   correctness proof; several modules remain below 80% individually.

The four existing notebooks exercise compact mathematical cases and diagnostics.
Their successful execution is **not** a claim that every numerical table in the
cited papers has been independently reproduced. The provenance-table reviewers
should identify the exact equation/table/figure, parameters, convention, expected
value, tolerance, tested commit and unresolved questions for each accepted result.

## Corrected High-Priority Findings

All entries below have regression tests. Existing public exports were preserved.
Some numerical values and invalid-input behavior intentionally changed.

| Priority | Component | Reproduced defect and correction | Regression location |
| --- | --- | --- | --- |
| P1 | `simulation.py`, IBP sensitivity | Missing inverse premium slope. With `c=2`, the old density estimate was about twice the exponential finite-difference reference. The estimator now includes `1/c`, uses stable rare-claim normalization, and documents its supported linear-premium domain. | `tests/test_simulation_regressions.py` |
| P1 | `multirisk_dividends.py` | Interior premium transitions had intensity `c`, not `c/h`. Changing the lattice step changed the physical drift. Use `c/h`, preserve monetary reward rates, reject nonabsorbing reachable classes, and check the state cap before allocating the grid. | `tests/test_multirisk_dividends.py` |
| P1 | `finite_discrete_time.py` | Dependent scenarios could become solvent again after recorded ruin: claims `[2,0]`, premiums `[1,2]` returned final ruin zero. Ruin is now an absorbing event along each scenario. | `tests/test_finite_discrete_time.py` |
| P1 | `finite_discrete.py`, `aggregate.py` | Poisson/Panjer recursion seeds underflowed at ordinary large intensities; binomial Panjer cancellation produced mass around `7.8e10`. Stable log propagation, frequency splitting and nonnegative convolution replace those failing paths. Appell bases avoid cancelling `exp(Lambda)` against tiny probabilities. | `tests/test_finite_discrete.py`, `tests/test_aggregate.py` |
| P1 | `finite_discrete.py` | Constant/flat integer boundaries mixed strict and non-strict ruin conventions. Shared constraint construction now distinguishes first attainment, strict exceedance and terminal equality. Positive short intervals are retained even at high intensity. | `tests/test_finite_discrete.py` |
| P1 | `ordered_risk.py` | Dual Poisson-exponential ruin-time density was twice an independent Poisson/Gamma mixture and overflowed through the Bessel function. Remove the extra factor, use scaled Bessel evaluation and preserve the atom convention. | `tests/test_ordered_risk.py` |
| P1 | `formulas.py`, `interest.py` | Conditional expected ruin time missed the premium time scale. Positive-interest ruin could underflow to zero (instead of about `0.08136` in a small-force case). Correct scaling and use a scaled incomplete-gamma evaluation. | `tests/test_formulas.py`, `tests/test_interest.py` |
| P1 | `prevention.py` | Two-claim optimization evaluated infeasible endpoints; projected-log allocation underspent its equality budget; generic optimization depended on pressure units. Isolate feasible positive drift, bisect the log multiplier and normalize objective weights. | `tests/test_prevention_optimization.py` |
| P1 | `prevention.py` | Heavy-tail within-period integration omitted part of prevention cost and missed deterministic reserve crossings. Integrate the existing linear reserve segment consistently, including integrable zero endpoints. | `tests/test_prevention_optimization.py` |
| P1 | `climate_change.py`, `regular_variation.py` | Unscaled positive tail integrals returned negative or negligible values at large capital; a climate asymptotic was not invariant to monetary units. Rescale quadrature, restore the missing scale, and enforce supported strict power conditions. | `tests/test_climate_change.py`, `tests/test_regular_variation.py` |
| P1 | `integer_byclaims.py` | Gamma claims crashed through eager evaluation of absent `metadata['rate']`. Select stored scale/rate explicitly. Dormant infinite-mean branches now contribute zero, not NaN. | `tests/test_integer_byclaims.py` |
| P1 | `matrix_analytic.py` | Repeated dense PH convolutions lost rare count probabilities through CDF subtraction. A single sparse count/phase generator retains the probabilities and avoids repeated matrix exponentials. | `tests/test_matrix_analytic.py` |

## Additional Corrections

- `results.py`: select the claim at the actual ruin event, not a later event
  whose timestamp happens to be numerically close.
- `models.py`: disabled by-claims, zero severity and zero frequency do not
  multiply an infinite severity mean by zero.
- `gerber_shiu.py`: rationalize the exponential root/prefactor to avoid
  cancellation at large discount rates; respect explicit censoring and the
  shortest available observation horizon for supplied paths.
- `red_time.py`: use the observed horizon, integrate all lines on their common
  observed interval, and keep distinct pre/post-jump endpoints.
- `dividends.py`: use stable small-interest growth/barrier/payment expressions
  and record barrier-hit knots before continuous dividend accrual.
- `distributions.py`: own constructor input arrays, ignore inactive mixture
  poles, reject NaN queries and avoid a unit-dependent PH/ME stability cutoff.
- `losses.py`: stabilize Gamma/Erlang/exponential moments, directly evaluate
  covered discrete losses, and preserve atoms at zero during discretization.
  Limited moments at a Gamma cap of `1e8` now return the correct raw-moment
  limit instead of zero.
- `aggregate.py`, `renewal.py`: sum survival tails directly, preserve tiny
  equilibrium masses, repair TVaR boundary handling and document four-ULP
  snapping for decimal lattice quotients (`0.3/0.1`).
- `ordered_risk.py`: use guarded precision for large rectangle recursions;
  adapt automatic OSPP truncation to its actual count law and requested tolerance.
- `finite_discrete_time.py`: require complete PMFs for MGF roots, handle missing
  roots conservatively, omit zero-mass diagnostic support and validate callbacks.
- `polynomial_expansion.py`: replace cancellation-prone Laguerre tail sums with
  the equivalent linear-time recurrence.
- `levy_driven.py`: stabilize small-argument Gamma/inverse-Gaussian exponents,
  retain the finite inverse-Gaussian endpoint, and reject nonfinite derivatives.
- `prevention.py`: reuse response-grid evaluations and Bellman indices; respect
  tiny caps, inactive severity components and floating-point root tolerances.
- `plotting.py`: render continuous growth/payments continuously, show singleton
  CDF jumps, preserve full solvency regions, aggregate true projected marginal
  probabilities and keep convergence axes stable when overlaid. Tests inspect
  artist coordinates, not only labels.

## Compatibility Notes

- IBP sensitivity now supports **positive linear premium income only**. A custom
  callable must agree with that contract; sampled checks are not a proof of
  linearity. Nonlinear income is rejected because its inverse-derivative weights
  were not implemented correctly. This is an intentional tightening, not a
  claim that Loisel-Privault's paper only treats linear premiums. The reference
  entry now points to the [published 2009 article](https://doi.org/10.1016/j.cam.2008.10.066).
- Infinite-mean integral/asymptotic helpers enforce `alpha * beta > 1` within
  their supported domain; special boundary cases are not silently generalized.
- Invalid fractional grid indices, nonfinite derivatives, deficient MGF PMFs
  and nontransient CTMCs now fail explicitly in the repaired paths.
- No public names were removed from `__all__` (308 unique exports). No test was
  deleted as redundant and no broad formatting or new framework was introduced.
- Stored figures, manuscript values and reviewer notes computed from the old
  implementation may need regeneration, especially for nonunit premiums,
  nonunit CTMC steps and dual-risk densities. The existing manuscript PDF was
  not revised during this code review.

## Measured Performance

These are local microbenchmarks, not universal speed guarantees. Old/new kernels
used the same environment and inputs. Baseline code was read from the commit
above, not restored over the working tree. Correctness changes are separated
from timing comparisons where necessary.

| Workload | Baseline | Revised | Observation |
| --- | ---: | ---: | --- |
| IBP, 5,000 paths, 1,000 capital values, horizon 4, seed 2026 | 254.11 ms | 113.55 ms | 2.24x faster; traced peak allocation 81.91 MB to 2.88 MB (28.45x smaller) |
| Dynamic calendar, four periods, three cycles, 201 budget points | 65.67 ms | 4.33 ms | 15.16x faster; identical allocations |
| PH Erlang-2 renewal counts, horizon 5, cutoff 60 | 31.31 ms | 1.36 ms | 22.96x faster; maximum PMF difference `2.04e-15` |
| Lattice PMF, four-point severity, mean 5, retained index 600 | 1.58 ms | 0.45 ms | 3.52x faster |
| Laguerre tails, order 40, 1,000 points in `[0,10]` | 27.43 ms | 0.23 ms | 119.84x faster; old cancellation also repaired |
| CTMC, 450 states and 193 retained terminal states, unit grid | 16.50 ms | 10.30 ms | 1.60x faster; matching rewards and terminal probabilities |
| Full 60-dimensional rectangle | 0.68 ms | 0.82 ms | Slower, but fixes probability zero to the correct one |

The IBP benchmark is reproducible with
`python scripts/benchmark_sensitivity.py --output /tmp/ibp.json`.
It uses `c=6`, `lambda=5`, Exp(1) claims, capital values evenly spaced in
`[0,10]`, one warmup and the median of three timing runs. Peak Python allocation
is measured separately with `tracemalloc`, not process RSS. The old result lacks
`1/c`: after accounting for that intentional correction, density differences
were below `3.9e-15` and standard-error differences below `1.6e-15`.
The optimization uses interval endpoint searches and online moments instead
of allocating a path-by-capital matrix unless pathwise output is requested.

The dynamic-calendar benchmark uses weights `[1,5,1,3]`, budget `0.8`, cap `1`,
response `exp(-3p)` and seven repeats of five calls. Response evaluations drop
from 131,076 to 912. The CTMC optimization solves once for occupation times
with the transposed generator, then evaluates all reward/terminal vectors by
dot product; it does not remove the finite-state approximation.

## Verification

| Check | Result |
| --- | --- |
| Full tests, warnings treated as errors | 456 passed; baseline 260 passed |
| Combined statement/branch coverage | 80.96%; baseline 79.15% |
| Statement coverage / branch coverage | 86.10% / 66.55% |
| Existing validation notebooks | All four code-cell execution checks passed |
| Example scripts | All 19 ran successfully under the noninteractive Agg backend in a temporary copy |
| Ruff configured lint | Passed |
| Ruff security rules for `src`, excluding test-assert rule S101 | Passed; not a formal security audit |
| Syntax compilation and `git diff --check` | Passed |
| Strict documentation build | Passed |
| Wheel/sdist build, metadata, private-material guard and installed-wheel smoke | Passed; wheel imported outside the checkout and the classical probability reference was 0.6 |

Two examples call `plt.show()` and emit the expected noninteractive-backend
warning under Agg; neither failed. Example execution is not a full visual audit
of every figure. The plotting review inspected a dedicated six-panel regression
figure, in addition to artist-coordinate tests.

CI now enforces 80% combined coverage and has a distribution job: build, metadata
validation, rejection of PDFs/private reference directories, and a wheel import
and numerical smoke test outside the checkout. The release guard itself has
tests for both archive formats and forbidden entries. No tracked PDF/reference
directory was found. Local private references were neither published nor deleted.
No package upload, Git push, issue closure or release creation was performed.

To repeat the main checks after installing the development, docs and release extras:

```sh
ruff check .
pytest -W error --cov=ruin_theory --cov-fail-under=80 --cov-report=term-missing
python scripts/check_validation_notebooks.py
mkdocs build --strict
python -m build
python -m twine check dist/*
python scripts/check_release_artifacts.py dist/*.tar.gz dist/*.whl
```

## Review Method And Scope

The user-requested workflows were read and applied as review procedures:

- [Everything Claude Code](https://github.com/affaan-m/ECC/tree/d3b8a3e908904e242ed2dbe66af62cca71131419):
  Python reviewer, verification loop and refactor-clean guidance.
- [Deep `/cleanse`](https://github.com/ulfaslak/saas_tmplt/blob/4cf92f75a2a7cbc19f733cc5fd3f32e3e6f72fbb/.claude/commands/cleanse.md):
  full source/contract review, numerical duplication review and verification.

No Claude runtime slash command, global hooks or plugin was installed. The
procedures were adapted to an existing scientific Python library, not applied
as an automatic code-rewriting tool. Database/UI-specific cleanup did not apply.
An open-issue search returned no matches; only the main worktree was present.

All 26 package Python modules were read across the review, with disjoint review
groups for foundations, finite-time methods, multirisk/climate, prevention,
plotting/public API, and the shared simulation/result/dividend/red-time layer.
Matching tests, the packaging configuration, CI, relevant method documentation,
provenance/validation entries and examples were inspected. Confirmed defects
were reproduced before fixes; controls, parameterized cases and independent
special-case oracles were retained. Integration tests and examples were run
again after the groups' changes stabilized.

This broad source review does not replace paper-by-paper mathematical review,
cross-platform testing, numerical error analysis or the authors' approval of
the manuscript. The next gate is the scoped scientific review described in the
provenance table, with separate mathematical, API/usability and teaching feedback.
