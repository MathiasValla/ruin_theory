"""Infinite-mean regularly varying ruin tests."""

import numpy as np
import pytest
from scipy import special

from ruin_theory import (
    InfiniteMeanPremiumModel,
    InfiniteMeanRuinModel,
    PolynomialPremiumGrowth,
    RegularlyVaryingTail,
    calibrate_polynomial_premium_coefficient,
    finite_mean_equilibrium_tail,
    finite_mean_equilibrium_tail_asymptotic,
    finite_mean_regular_variation_curve,
    finite_mean_regularly_varying_ruin_asymptotic,
    infinite_mean_constant,
    infinite_mean_one_big_jump_asymptotic,
    infinite_mean_one_big_jump_integral,
    infinite_mean_ruin_asymptotic,
    infinite_mean_ruin_curve,
    pareto_infinite_mean_model,
    premium_power_calibration_grid,
    premium_power_condition,
    regular_variation_tail_diagnostic,
)


def test_pareto_polynomial_model_matches_existing_klr_case():
    old = InfiniteMeanPremiumModel(
        claim_arrival_rate=1.2,
        tail_index=0.8,
        pareto_scale=2.0,
        premium_coefficient=1.5,
        premium_power=1.6,
    )
    new = pareto_infinite_mean_model(
        claim_arrival_rate=1.2,
        tail_index=0.8,
        pareto_scale=2.0,
        premium_coefficient=1.5,
        premium_power=1.6,
    )

    assert infinite_mean_one_big_jump_asymptotic(new, 100.0) == pytest.approx(
        infinite_mean_ruin_asymptotic(old, 100.0),
    )
    assert infinite_mean_one_big_jump_integral(new, 100.0) / infinite_mean_ruin_asymptotic(
        old,
        100.0,
    ) == pytest.approx(1.0, rel=0.12)


def test_premium_power_condition_and_constant():
    condition = premium_power_condition(tail_index=0.8, premium_power=1.5)

    assert condition.holds
    assert condition.threshold == pytest.approx(1.25)
    assert condition.margin == pytest.approx(0.25)
    assert infinite_mean_constant(0.8, 1.5) > 0.0
    with pytest.raises(ValueError, match="premium_power"):
        infinite_mean_constant(0.8, 1.0)


def test_calibrate_polynomial_premium_coefficient_hits_target():
    tail = RegularlyVaryingTail(tail_index=0.8, scale=1.0)
    capitals = np.array([50.0, 100.0, 200.0])
    result = calibrate_polynomial_premium_coefficient(
        tail,
        capitals,
        target_probability=0.02,
        premium_power=1.6,
        claim_arrival_rate=1.0,
    )

    model = InfiniteMeanRuinModel(
        claim_arrival_rate=1.0,
        tail=tail,
        premium=PolynomialPremiumGrowth(result.required_coefficient, 1.6),
    )
    curve = infinite_mean_ruin_curve(model, capitals, method="asymptotic")

    assert result.condition.holds
    assert result.required_coefficient > 0.0
    assert result.achieved_asymptotic == pytest.approx(0.02)
    assert np.max(curve.probabilities) == pytest.approx(0.02)


def test_premium_power_grid_marks_invalid_powers():
    tail = RegularlyVaryingTail(tail_index=0.8, scale=1.0)
    grid = premium_power_calibration_grid(
        tail,
        [100.0],
        [1.0, 1.3, 1.8],
        target_probability=0.05,
    )

    np.testing.assert_array_equal(grid.condition_holds, [False, True, True])
    assert np.isnan(grid.required_coefficients[0])
    assert np.all(grid.required_coefficients[1:] > 0.0)
    assert grid.threshold == pytest.approx(1.25)


def test_regular_variation_tail_diagnostic_converges_to_power_ratio():
    tail = RegularlyVaryingTail(tail_index=0.7, scale=2.0)
    diagnostic = regular_variation_tail_diagnostic(
        tail,
        thresholds=np.logspace(2, 6, 12),
        multipliers=[2.0, 5.0],
    )

    assert diagnostic.ratios.shape == (2, 12)
    np.testing.assert_allclose(diagnostic.targets, [2.0**-0.7, 5.0**-0.7])
    assert np.max(diagnostic.relative_errors[:, -3:]) < 0.02


def test_finite_mean_equilibrium_tail_matches_pareto_ii_formula():
    tail = RegularlyVaryingTail(tail_index=2.5, scale=4.0)
    capital = np.array([0.0, 4.0, 20.0, 100.0])

    exact = finite_mean_equilibrium_tail(tail, capital)
    expected = (1.0 + capital / 4.0) ** (1.0 - 2.5)

    assert tail.mean() == pytest.approx(4.0 / 1.5)
    np.testing.assert_allclose(exact, expected)


def test_finite_mean_karamata_equivalent_converges_to_integrated_tail():
    tail = RegularlyVaryingTail(tail_index=2.5, scale=4.0)
    capital = np.geomspace(10.0, 1_000_000.0, 12)

    exact = finite_mean_equilibrium_tail(tail, capital)
    asymptotic = finite_mean_equilibrium_tail_asymptotic(tail, capital)
    ratio = asymptotic / exact

    assert ratio[-1] == pytest.approx(1.0, rel=1e-5)
    assert np.all(ratio < 1.0)
    assert finite_mean_equilibrium_tail_asymptotic(tail, 100.0).shape == ()


def test_finite_mean_regular_variation_curve_and_ruin_asymptotic():
    tail = RegularlyVaryingTail(tail_index=3.0, scale=2.0)
    capital = np.array([20.0, 100.0, 500.0])
    curve = finite_mean_regular_variation_curve(tail, capital, rho=0.4)
    ruin = finite_mean_regularly_varying_ruin_asymptotic(tail, capital, rho=0.4)

    np.testing.assert_allclose(curve.ruin_asymptotic, ruin)
    np.testing.assert_allclose(curve.ruin_probabilities, 0.4 / 0.6 * curve.equilibrium_tail)
    assert curve.mean == pytest.approx(1.0)
    assert curve.rho == pytest.approx(0.4)
    assert curve.tail_index == pytest.approx(3.0)


def test_finite_mean_custom_tail_uses_supplied_mean_and_quadrature():
    tail = RegularlyVaryingTail(
        tail_index=2.0,
        survival_function=lambda x: (1.0 + np.asarray(x, dtype=float)) ** -2.0,
        mean_value=1.0,
    )

    exact = finite_mean_equilibrium_tail(tail, [0.0, 1.0, 9.0])

    np.testing.assert_allclose(exact, [1.0, 0.5, 0.1], rtol=1e-10)


def test_regular_variation_argument_validation_and_warning():
    with pytest.raises(ValueError, match="tail_index"):
        RegularlyVaryingTail(tail_index=0.0)
    with pytest.raises(ValueError, match="tail.tail_index"):
        InfiniteMeanRuinModel(
            claim_arrival_rate=1.0,
            tail=RegularlyVaryingTail(tail_index=1.2),
            premium=PolynomialPremiumGrowth(coefficient=1.0, power=1.0),
        )
    with pytest.warns(UserWarning, match="premium.power"):
        InfiniteMeanRuinModel(
            claim_arrival_rate=1.0,
            tail=RegularlyVaryingTail(tail_index=0.8),
            premium=PolynomialPremiumGrowth(coefficient=1.0, power=1.0),
        )
    with pytest.raises(ValueError, match="target_probability"):
        calibrate_polynomial_premium_coefficient(
            RegularlyVaryingTail(tail_index=0.8),
            [100.0],
            target_probability=1.0,
            premium_power=1.6,
        )
    with pytest.raises(ValueError, match="greater than one"):
        finite_mean_equilibrium_tail_asymptotic(RegularlyVaryingTail(tail_index=1.0), [10.0])
    with pytest.raises(ValueError, match="rho"):
        finite_mean_regularly_varying_ruin_asymptotic(
            RegularlyVaryingTail(tail_index=2.0),
            [10.0],
            rho=1.0,
        )


@pytest.mark.parametrize("power", [1.0, 1.25])
def test_integral_rejects_powers_outside_supported_infinite_horizon_domain(power):
    with pytest.warns(UserWarning, match="premium.power"):
        model = pareto_infinite_mean_model(
            claim_arrival_rate=1.0, tail_index=0.8,
            premium_coefficient=1.0, premium_power=power,
        )
    with pytest.raises(ValueError, match="premium.power|premium_power"):
        infinite_mean_one_big_jump_integral(model, 10.0)


def test_infinite_mean_model_rejects_explicit_finite_mean_at_alpha_one():
    # This survival has index -1 but integral one, by y = 1 + log(1 + x).
    tail = RegularlyVaryingTail(
        1.0, survival_function=lambda x: 1.0 / ((1.0 + x) * (1.0 + np.log1p(x))**2),
        mean_value=1.0,
    )
    with pytest.raises(ValueError, match="infinite mean"):
        InfiniteMeanRuinModel(1.0, tail, PolynomialPremiumGrowth(1.0, 2.0))


@pytest.mark.parametrize("capital", [1e-6, 10.0, 1e12])
def test_one_big_jump_quadrature_matches_exact_pareto_integral_across_scales(capital):
    model = pareto_infinite_mean_model(
        claim_arrival_rate=1.2, tail_index=0.8, pareto_scale=2.0,
        premium_coefficient=1.5, premium_power=1.6,
    )
    expected = (
        1.2 * 2.0**0.8 * (capital + 2.0)**(1.0 / 1.6 - 0.8)
        * 1.5**(-1.0 / 1.6) * special.beta(1.0 / 1.6, 0.8 - 1.0 / 1.6) / 1.6
    )
    assert infinite_mean_one_big_jump_integral(model, capital) == pytest.approx(
        expected, rel=1e-7,
    )


def test_custom_finite_mean_equilibrium_quadrature_at_large_capital():
    tail = RegularlyVaryingTail(
        2.0, survival_function=lambda x: (1.0 + x)**-2.0, mean_value=1.0,
    )
    result = finite_mean_equilibrium_tail(tail, [1e12])
    np.testing.assert_allclose(result, [1.0 / (1.0 + 1e12)], rtol=1e-8, atol=0.0)
