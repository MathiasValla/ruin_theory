# Scientific References And Acknowledgments

This package follows the notation and computational priorities of classical
ruin theory. The references below should be cited when using the corresponding
features in scientific work.

## Core Ruin Theory

- Lundberg, F. (1903). *I Approximerad Framstallning av
  Sannolikhetsfunktionen. II Aterforsakring av Kollektivrisker*. Almqvist &
  Wiksell, Uppsala.
- Lundberg, F. (1926). *Forsakringsteknisk Riskutjamning*. F. Englunds
  Boktryckeri AB, Stockholm.
- Cramer, H. (1930). *On the Mathematical Theory of Risk*. Skandia Jubilee
  Volume, Stockholm.
- Cramer, H. (1955). *Collective Risk Theory*. Jubilee volume of
  Forsakringsbolaget Skandia, Stockholm.
- Sparre Andersen, E. (1957). On the collective theory of risk in the case of
  contagion between the claims. *Transactions of the XVth International
  Congress of Actuaries*, New York, II, 219-229.
- Gerber, H. U. (1979). *An Introduction to Mathematical Risk Theory*. S. S.
  Huebner Foundation, Philadelphia.
- Rolski, T., Schmidli, H., Schmidt, V., and Teugels, J. (1999). *Stochastic
  Processes for Insurance and Finance*. Wiley, Chichester.
- Asmussen, S., and Albrecher, H. (2010). *Ruin Probabilities*, 2nd edition.
  World Scientific, Singapore.

## Closed Forms, Bounds And Approximations

- Asmussen, S., and Rolski, T. (1991). Computational methods in risk theory: a
  matrix-algorithmic approach. *Insurance: Mathematics and Economics*, 10,
  259-274.
- Asmussen, S., and Rolski, T. (1994). Risk theory in a periodic environment:
  Lundberg's inequality and the Cramer-Lundberg approximation. *Mathematics
  of Operations Research*, 19, 410-433.
- Beekman, J. A. (1968). Collective risk results. *Transactions of the Society
  of Actuaries*, 20, 182-199.
- De Vylder, F. (1978). A practical solution to the problem of ultimate ruin
  probability. *Scandinavian Actuarial Journal*, 1978, 114-119.
- Dufresne, F., and Gerber, H. U. (1988). The probability and severity of ruin
  for combinations of exponential claim amount distributions and their
  translations. *Insurance: Mathematics and Economics*, 7, 75-80.
- Gerber, H. U., Goovaerts, M. J., and Kaas, R. (1987). On the probability and
  severity of ruin. *ASTIN Bulletin*, 17, 151-163.
- Gerber, H. U., and Shiu, E. S. W. (1997). The joint distribution of the time
  of ruin, the surplus immediately before ruin, and the deficit at ruin.
  *Insurance: Mathematics and Economics*, 21, 129-137.
- Gerber, H. U., and Shiu, E. S. W. (1998). On the time value of ruin. *North
  American Actuarial Journal*, 2, 48-72.
- Picard, P., and Lefevre, C. (1997). The probability of ruin in finite time
  with discrete claim size distribution. *Scandinavian Actuarial Journal*,
  1997(1), 58-69.
- Picard, P., and Lefevre, C. (1998). The moments of ruin time in the classical
  risk model with discrete claim size distribution. *Insurance: Mathematics
  and Economics*, 23(2), 157-172.
- Rulliere, D., and Loisel, S. (2005). The win-first probability under
  interest force. Working paper, Universite Lyon 1.
- Segerdahl, C.-O. (1942). Uber einige risikotheoretische Fragestellungen.
  *Skandinavisk Aktuarietidskrift*, 25, 43-83.
- Seal, H. L. (1969). *Stochastic Theory of a Risk Business*. Wiley, New York.
- Sundt, B., and Teugels, J. L. (1995). Ruin estimates under interest force.
  *Insurance: Mathematics and Economics*, 16(1), 7-22.
- Sundt, B., and Teugels, J. L. (1997). The adjustment function in ruin
  estimates under interest force. *Insurance: Mathematics and Economics*,
  19(1), 85-94.
- Takacs, L. (1962). A generalization of the ballot problem and its application
  in the theory of queues. *Journal of the American Statistical Association*,
  57(298), 327-337.
- De Vylder, F. E. (1999). Numerical finite-time ruin probabilities by the
  Picard-Lefevre formula. *Scandinavian Actuarial Journal*, 1999(2), 97-105.
- Ignatov, Z. G., Kaishev, V. K., and Krachunov, R. S. (2001). An improved
  finite-time ruin probability formula and its Mathematica implementation.
  *Insurance: Mathematics and Economics*, 29(3), 375-386.
- Lefevre, C., and Loisel, S. (2009). Finite-time ruin probabilities for
  discrete, possibly dependent, claim severities. *Methodology and Computing
  in Applied Probability*, 11(3), 425-441.
- Loisel, S. (2004). *Contribution a l'etude de processus univaries et
  multivaries de la theorie de la ruine*. PhD thesis, Universite Claude
  Bernard Lyon 1.
- Rulliere, D., and Loisel, S. (2004). Another look at the Picard-Lefevre
  formula for finite-time ruin probabilities. Preprint presented at the 7th
  IME Conference, Lyon.
- Castaner, A., Claramunt, M. M., Gathy, M., Lefevre, C., and Marmol, M.
  (2010). Ruin problems for a discrete time risk model with non-homogeneous
  conditions. Manuscript.
- Panjer, H. H. (1981). Recursive evaluation of a family of compound
  distributions. *ASTIN Bulletin*, 12, 22-26.
- Klugman, S. A., Panjer, H. H., and Willmot, G. E. (2012). *Loss Models: From
  Data to Decisions*, 4th edition. Wiley, New York.

## Actuarial Software References

- Dutang, C., Goulet, V., and Pigeon, M. (2008). actuar: An R package for
  actuarial science. *Journal of Statistical Software*, 25(7).
  doi:10.18637/jss.v025.i07.
- Goulet, V. (2026). *Risk and ruin theory features of actuar*. Package note
  for the R package `actuar`.

These sources motivate the package's user-facing API style for distribution
helpers, adjustment coefficients, ruin-function front doors, discretization and
aggregate-distribution methods.

## Prevention In Ruin Models

- Ehrlich, I., and Becker, G. S. (1972). Market insurance, self-insurance, and
  self-protection. *Journal of Political Economy*, 80(4), 623-648.
- Gauchon, R., Loisel, S., Rulliere, J.-L., and Trufin, J. (2020). Optimal
  prevention strategies in the classical risk model. *Insurance: Mathematics
  and Economics*, 91, 202-208.
- Gauchon, R., Loisel, S., Rulliere, J.-L., and Trufin, J. (2021). Optimal
  prevention of large risks with two types of claims. *Scandinavian Actuarial
  Journal*, 2021(4), 323-334.
- Schmidli, H. (2008). *Stochastic Control in Insurance*. Springer, London.
- Minier, C., Valla, M., and Lefevre, C. (manuscript). Seasonal prevention in
  ruin theory: periodic control, Lundberg bounds, and storm-loss-day
  simulations.
- Valla et al. (2026, manuscript). How long can premiums compensate infinite-mean claims?
  Ruin-time asymptotics for heavy-tailed compound events.

## By-Claims And Discrete Dependence

The package supports independent event-level by-claims in continuous-time
simulation and INAR/BINAR dependent by-claim counts in discrete time.

- McKenzie, E. (1985). Some simple models for discrete variate time series.
  *Water Resources Bulletin*, 21(4), 645-650.
- Al-Osh, M. A., and Alzaid, A. A. (1987). First-order integer-valued
  autoregressive (INAR(1)) process. *Journal of Time Series Analysis*, 8(3),
  261-275.
- Du, J. G., and Li, Y. (1991). The integer-valued autoregressive (INAR(p))
  model. *Journal of Time Series Analysis*, 12(2), 129-142.
- Pedeli, X., and Karlis, D. (2011). A bivariate INAR(1) process with
  application. *Statistical Modelling*, 11(4), 325-349.

The concrete ruin-simulation conventions for this package are also checked
against the local reference scripts `ModèleINARen1Dcalculsderuine.py`,
`ModèleINARen1Dmoyennesempiriques.py`, and
`ModèleBINARen2Dcalculsde_psi_sum.py`.

## Acknowledgment

Claude Lefevre's lecture notes and comments helped orient the mathematical
reading of ruin theory for this project. The package documentation cites the
original articles, monographs and software papers whenever a result or API is
implemented. We thank Claude Lefevre for the notes, discussions and guidance
that shaped the prevention and periodic-risk directions.
