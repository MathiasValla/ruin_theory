"""Worsening-risk climate-change ruin model tests."""

import numpy as np
import pytest

from ruin_theory.climate_change import (
    InfiniteMeanPremiumModel,
    WorseningParetoModel,
    climate_change_ruin_table,
    estimate_worsening_pareto_ruin_probability,
    infinite_mean_ruin_asymptotic,
    infinite_mean_ruin_integral,
    klr_scale_asymptotic,
    klr_shape_asymptotic,
    simulate_worsening_pareto_path,
)


def test_worsening_pareto_model_matches_klr_parameterization():
    shape = WorseningParetoModel(
        initial_capital=500.0,
        claim_arrival_rate=1.0,
        pareto_scale=1.0,
        initial_shape=1.5,
        worsening_speed=0.1,
        safety_loading=1.0,
        mode="shape",
    )
    scale = WorseningParetoModel(
        initial_capital=500.0,
        claim_arrival_rate=1.0,
        pareto_scale=1.0,
        initial_shape=1.5,
        worsening_speed=0.1,
        safety_loading=1.0,
        mode="scale",
    )

    assert shape.shape_at(0.0) == pytest.approx(1.5)
    assert shape.shape_at(20.0) == pytest.approx(1.0 + 0.5 / 3.0)
    assert scale.scale_at(20.0) == pytest.approx(3.0)
    assert shape.mean_claim_at(20.0) == pytest.approx(6.0)
    assert scale.mean_claim_at(20.0) == pytest.approx(6.0)
    assert shape.premium_rate_at(20.0) == pytest.approx(12.0)
    assert shape.uninsurability_time(12.0) == pytest.approx(20.0)
    assert shape.cumulative_premium(20.0) == pytest.approx(160.0)


def test_klr_asymptotics_reproduce_table_values():
    speeds = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    shape_values = [
        klr_shape_asymptotic(
            initial_capital=500.0,
            claim_arrival_rate=1.0,
            pareto_scale=1.0,
            initial_shape=1.5,
            worsening_speed=float(speed),
            safety_loading=1.0,
        )
        for speed in speeds
    ]
    scale_values = [
        klr_scale_asymptotic(
            initial_capital=500.0,
            claim_arrival_rate=1.0,
            pareto_scale=1.0,
            initial_shape=1.5,
            worsening_speed=float(speed),
            safety_loading=1.0,
        )
        for speed in speeds
    ]

    np.testing.assert_allclose(shape_values, [0.497, 0.351, 0.222, 0.157, 0.111], rtol=5e-3)
    np.testing.assert_allclose(scale_values, [0.124, 0.147, 0.185, 0.220, 0.262], rtol=7e-3)


def test_infinite_mean_asymptotic_matches_tail_integral():
    model = InfiniteMeanPremiumModel(
        claim_arrival_rate=1.2,
        tail_index=0.8,
        pareto_scale=2.0,
        premium_coefficient=1.5,
        premium_power=1.6,
    )

    integral = infinite_mean_ruin_integral(model, 100.0)
    asymptotic = infinite_mean_ruin_asymptotic(model, 100.0)

    assert integral > 0.0
    assert asymptotic > 0.0
    assert integral / asymptotic == pytest.approx(1.0, rel=0.12)


def test_worsening_pareto_simulation_and_table_are_reproducible():
    model = WorseningParetoModel(
        initial_capital=8.0,
        claim_arrival_rate=0.8,
        pareto_scale=1.0,
        initial_shape=1.7,
        worsening_speed=0.05,
        safety_loading=0.5,
        mode="scale",
    )
    path = simulate_worsening_pareto_path(model, horizon=5.0, seed=123, stop_at_ruin=False)
    estimate = estimate_worsening_pareto_ruin_probability(
        model,
        horizon=5.0,
        n_simulations=50,
        seed=123,
    )
    table = climate_change_ruin_table(
        [0.05, 0.1],
        initial_capital=30.0,
        initial_shape=1.5,
        safety_loading=1.0,
        n_simulations=20,
        seed=123,
    )

    assert path.times[0] == 0.0
    assert path.times[-1] == pytest.approx(5.0)
    assert 0.0 <= estimate.probability <= 1.0
    np.testing.assert_allclose(table.horizons, [40.0, 20.0])
    assert np.all(table.shape_asymptotic > 0.0)
    assert np.all(table.scale_asymptotic > 0.0)


def test_climate_change_argument_validation():
    with pytest.raises(ValueError, match="initial_shape"):
        WorseningParetoModel(1.0, 1.0, 1.0, 1.0, 0.1, 1.0)
    with pytest.raises(ValueError, match="mode"):
        WorseningParetoModel(1.0, 1.0, 1.0, 1.5, 0.1, 1.0, mode="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="premium_power"):
        InfiniteMeanPremiumModel(1.0, 0.8, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="ci_method"):
        estimate_worsening_pareto_ruin_probability(
            WorseningParetoModel(1.0, 1.0, 1.0, 1.5, 0.1, 1.0),
            1.0,
            n_simulations=1,
            ci_method="bad",
        )
    with pytest.raises(ValueError, match="max_events"):
        estimate_worsening_pareto_ruin_probability(
            WorseningParetoModel(1.0, 1.0, 1.0, 1.5, 0.1, 1.0),
            1.0,
            n_simulations=1,
            max_events=0,
        )
