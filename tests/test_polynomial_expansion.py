import numpy as np
import pytest

from ruin_theory import (
    CramerLundbergProcess,
    LaguerreRuinExpansion,
    compound_geometric_moments_from_severity,
    exponential,
    fit_ultimate_ruin_polynomial_expansion,
    ultimate_ruin_exponential,
    ultimate_ruin_polynomial_expansion,
)


def test_laguerre_expansion_is_exact_for_exponential_claims_with_xi_gamma():
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    surplus = np.array([0.0, 1.0, 2.0, 5.0])

    np.testing.assert_allclose(
        ultimate_ruin_polynomial_expansion(model, surplus, order=0),
        ultimate_ruin_exponential(model, surplus),
        rtol=2e-12,
        atol=1e-15,
    )


def test_laguerre_expansion_reusable_coefficients_match_front_door():
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    surplus = np.linspace(0.0, 4.0, 9)
    expansion = fit_ultimate_ruin_polynomial_expansion(model, order=5)

    np.testing.assert_allclose(
        expansion.evaluate(surplus),
        ultimate_ruin_polynomial_expansion(model, surplus, order=5),
    )


def test_laguerre_expansion_validates_integrability_condition():
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )

    with pytest.raises(ValueError, match="xi < 2"):
        fit_ultimate_ruin_polynomial_expansion(model, order=4, xi=4.1)


def test_compound_geometric_moments_for_exponential_equilibrium():
    moments = compound_geometric_moments_from_severity(exponential(rate=5.0), rho=0.6, order=2)

    # Exponential claims have the same equilibrium law. For geometric N with
    # mean rho/(1-rho), E[M]=E[N]/5 and Var(M)=E[N]/25 + Var(N)/25.
    np.testing.assert_allclose(moments, [1.0, 0.3, 0.3])


def test_review_laguerre_tail_integrals_match_direct_quadrature_at_high_order():
    from scipy import integrate, special

    expansion = LaguerreRuinExpansion(np.ones(41), xi=1.0, rho=0.5, order=40)
    points = np.array([[0.01, 0.5], [2.0, 10.0]])
    expected = np.array([
        integrate.quad(
            lambda x: np.exp(-x) * special.eval_laguerre(40, x),
            z, np.inf, epsabs=1e-12,
        )[0] for z in points.ravel()
    ]).reshape(points.shape)
    np.testing.assert_allclose(expansion.tail_integrals(points)[40], expected, atol=2e-12)
    np.testing.assert_array_equal(expansion.tail_integrals(0.0)[1:], 0.0)
