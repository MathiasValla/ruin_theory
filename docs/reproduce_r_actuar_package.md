# Reproducing `R_actuar_package.pdf`

Two scripts recreate the numerical examples from the ruin-theory sections of
`R_actuar_package.pdf` using this package.

- [examples/reproduce_r_actuar_package.py](../examples/reproduce_r_actuar_package.py)
  prints the numerical outputs.
- [examples/r_actuar_package_python.py](../examples/r_actuar_package_python.py)
  generates a PDF report and standalone PNG figures under `output/`.

Run:

```bash
uv run python examples/reproduce_r_actuar_package.py
uv run python examples/r_actuar_package_python.py
```

The second command writes:

```text
output/pdf/r_actuar_package_python.pdf
output/figures/fig_adjustment_coefficients.png
output/figures/fig_ruin_curves.png
output/figures/fig_lundberg_bound.png
output/figures/fig_beekman_panjer_bounds.png
output/figures/fig_heavy_tail_asymptotic.png
output/figures/fig_simulation_diagnostics.png
```

## Adjustment Coefficient

The PDF example uses exponential claim sizes with rate 1, exponential waiting
times with rate 2, premium rate 2.4, and safety loading 20 percent. In the
Cramer-Lundberg parameterization this is:

```python
from ruin_theory import CramerLundbergProcess, adjustment_coefficient, exponential

model = CramerLundbergProcess(
    premium_rate=2.4,
    claim_arrival_rate=2.0,
    claim_distribution=exponential(rate=1.0),
)
print(adjustment_coefficient(model))
```

Expected output:

```text
0.1667
```

## Proportional Reinsurance Coefficients

The actuar example evaluates the adjustment coefficient under retained
proportions `0.75, 0.8, 0.9, 1.0` and premium function `p(alpha)=2.6 alpha-0.2`.
For exponential claims, retaining `alpha X` changes the exponential rate from
1 to `1 / alpha`.

Expected output:

```text
[0.1905 0.1862 0.1765 0.1667]
```

## Exponential Ruin Function

The PDF call

```r
ruin(claims = "e", par.claims = list(rate = 5),
     wait = "e", par.wait = list(rate = 3))
```

uses premium rate 1. The matching Python model is:

```python
import numpy as np
from ruin_theory import CramerLundbergProcess, exponential, ultimate_ruin_exponential

model = CramerLundbergProcess(
    premium_rate=1.0,
    claim_arrival_rate=3.0,
    claim_distribution=exponential(rate=5.0),
)
print(ultimate_ruin_exponential(model, np.arange(11)))
```

Expected output:

```text
[6.000e-01 8.120e-02 1.099e-02 1.487e-03 2.013e-04 2.724e-05
 3.687e-06 4.989e-07 6.752e-08 9.138e-09 1.237e-09]
```

## Hyperexponential Ruin Function

The Gerber example in the PDF uses equal mixture weights on exponential rates
3 and 7, arrival rate 3, and premium rate 1. The package reproduces the known
closed form:

```text
psi(u) = (24 exp(-u) + exp(-6u)) / 35
```

The script prints `True` for equality with this formula over `u=0,...,10`.

## Beekman/Panjer Pareto Bounds

The PDF approximates infinite-time ruin for a Lomax/Pareto-II claim law

```text
P(x) = 1 - (4 / (4 + x))^5
```

with mean 1 and safety loading 20 percent. The equilibrium distribution is
again Lomax, now with shape 4 and scale 4. The example uses the package-level
discrete Pollaczek-Khinchine/Panjer machinery: lower/upper endpoint
discretization of the equilibrium severity followed by a compound-geometric
recursion for the all-time maximum.

Expected output:

```text
u    lower       upper
 0  0.6719160  0.83333
 5  0.2892792  0.51572
10  0.1361541  0.32938
15  0.0662486  0.21200
20  0.0329848  0.13700
25  0.0167551  0.08877
30  0.0086802  0.05764
35  0.0045911  0.03749
40  0.0024843  0.02443
45  0.0013790  0.01595
50  0.0007877  0.01043
```

## Remaining Gaps

The package now includes the general building blocks needed for the visible
discretization, Panjer, VaR and TVaR examples, as well as phase-type claim-size
laws and Cramer-Lundberg ultimate ruin for phase-type claims. Remaining
matrix-analytic extensions include:

- phase-type interarrival times in the deterministic ruin solvers;
- matrix-based Gerber-Shiu quantities.
