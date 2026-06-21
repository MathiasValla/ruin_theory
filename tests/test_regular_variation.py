"""Infinite-mean regularly varying ruin tests."""

import numpy as np
import pytest

from ruin_theory import (
    InfiniteMeanPremiumModel,
    InfiniteMeanRuinModel,
    PolynomialPremiumGrowth,
    RegularlyVaryingTail,
    calibrate_polynomial_premium_coefficient,
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


def test_regular_variation_argument_validation_and_warning():
    with pytest.raises(ValueError, match="tail_index"):
        RegularlyVaryingTail(tail_index=1.2)
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
