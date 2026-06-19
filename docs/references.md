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
- Valla, M., Rivoire, P., Minier, C., Guibert, Q., and Loisel, S. (2026,
  manuscript). How long can premiums compensate infinite-mean claims?
  Ruin-time asymptotics for heavy-tailed compound events.

## By-Claims And Discrete Dependence

The current package supports event-level by-claims in continuous-time
simulation. The planned INAR/BINAR layer is motivated by local project scripts
and should be documented separately when promoted to public API. Relevant
background for dependent count processes and multivariate integer-valued
autoregression should be cited with the future implementation.

## Acknowledgment

Claude Lefevre's lecture notes and comments helped orient the mathematical
reading of ruin theory for this project. The package documentation cites the
original articles, monographs and software papers whenever a result or API is
implemented. We thank Claude Lefevre for the notes, discussions and guidance
that shaped the prevention and periodic-risk directions.
