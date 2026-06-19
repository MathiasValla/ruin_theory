"""INAR/BINAR by-claim simulation tests."""

import numpy as np
import pytest

from ruin_theory import (
    BINARByClaimModel,
    INARByClaimModel,
    RuinEstimate,
    deterministic,
    estimate_binar_byclaim_ruin_probability,
    estimate_inar_byclaim_ruin_probability,
    exponential,
    simulate_binar_byclaim_path,
    simulate_binar_byclaim_terminal_reserves,
    simulate_inar_byclaim_path,
    simulate_inar_byclaim_terminal_reserves,
)


def test_inar_expected_counts_match_closed_form_recursion():
    model = INARByClaimModel(
        initial_capital=0.0,
        premium_per_period=37.0,
        primary_count_mean=10.0,
        initial_byclaim_mean=10.0,
        reproduction=0.1,
        primary_distribution=deterministic(2.0),
        byclaim_distribution=deterministic(3.0),
    )

    np.testing.assert_allclose(model.expected_byclaim_counts(4), [11.0, 11.1, 11.11, 11.111])
    expected_terminal = 4 * 37.0 - 4 * 10.0 * 2.0 - np.sum([11.0, 11.1, 11.11, 11.111]) * 3.0
    assert model.expected_terminal_reserve(4) == pytest.approx(expected_terminal)


def test_inar_path_reserve_is_premium_minus_primary_and_byclaim_losses():
    model = INARByClaimModel(
        initial_capital=100.0,
        premium_per_period=20.0,
        primary_count_mean=3.0,
        initial_byclaim_mean=2.0,
        reproduction=0.4,
        primary_distribution=deterministic(2.0),
        byclaim_distribution=deterministic(1.5),
    )

    path = simulate_inar_byclaim_path(model, periods=6, seed=123, stop_at_ruin=False)

    assert path.primary_counts.shape == (6, 1)
    assert path.byclaim_counts.shape == (6, 1)
    increments = 20.0 - path.primary_losses.sum(axis=1) - path.byclaim_losses.sum(axis=1)
    np.testing.assert_allclose(path.reserves, 100.0 + np.r_[0.0, np.cumsum(increments)])
    np.testing.assert_allclose(path.primary_losses[:, 0], 2.0 * path.primary_counts[:, 0])
    np.testing.assert_allclose(path.byclaim_losses[:, 0], 1.5 * path.byclaim_counts[:, 0])


def test_inar_terminal_reserve_fast_simulation_matches_theoretical_mean():
    model = INARByClaimModel(
        initial_capital=0.0,
        premium_per_period=37.0,
        primary_count_mean=10.0,
        initial_byclaim_mean=10.0,
        reproduction=0.1,
        primary_distribution=deterministic(2.0),
        byclaim_distribution=deterministic(3.0),
    )

    reserves = simulate_inar_byclaim_terminal_reserves(
        model,
        periods=8,
        n_simulations=60_000,
        seed=123,
    )

    assert reserves.mean() == pytest.approx(model.expected_terminal_reserve(8), abs=0.45)


def test_binar_expected_counts_match_matrix_recursion():
    model = BINARByClaimModel(
        initial_capital=1000.0,
        premium_per_period=15000.0,
        primary_count_means=(5.0, 7.0),
        initial_byclaim_means=(1.0, 1.0),
        reproduction_matrix=((0.41, 0.1), (0.05, 0.3)),
        primary_distributions=(deterministic(10.0), deterministic(1.0)),
        byclaim_distributions=(deterministic(0.5), deterministic(0.5)),
    )

    expected = np.empty((3, 2))
    matrix = np.array([[0.41, 0.1], [0.05, 0.3]])
    previous = matrix @ np.ones(2) + np.array([5.0, 7.0])
    expected[0] = previous
    for index in range(1, 3):
        previous = matrix @ previous + np.array([5.0, 7.0])
        expected[index] = previous

    np.testing.assert_allclose(model.expected_byclaim_counts(3), expected)


def test_binar_path_and_terminal_reserves_are_consistent():
    model = BINARByClaimModel(
        initial_capital=100.0,
        premium_per_period=20.0,
        primary_count_means=(2.0, 1.0),
        initial_byclaim_means=(1.0, 1.0),
        reproduction_matrix=((0.2, 0.1), (0.05, 0.3)),
        primary_distributions=(deterministic(2.0), deterministic(3.0)),
        byclaim_distributions=(deterministic(1.0), deterministic(4.0)),
    )

    path = simulate_binar_byclaim_path(model, periods=5, seed=123, stop_at_ruin=False)
    reserves = simulate_binar_byclaim_terminal_reserves(
        model,
        periods=5,
        n_simulations=50_000,
        seed=123,
    )

    assert path.primary_counts.shape == (5, 2)
    assert path.byclaim_counts.shape == (5, 2)
    increments = 20.0 - path.primary_losses.sum(axis=1) - path.byclaim_losses.sum(axis=1)
    np.testing.assert_allclose(path.reserves, 100.0 + np.r_[0.0, np.cumsum(increments)])
    assert reserves.mean() == pytest.approx(model.expected_terminal_reserve(5), abs=0.35)


def test_integer_byclaim_ruin_estimators_return_ruin_estimates():
    inar = INARByClaimModel(
        initial_capital=0.0,
        premium_per_period=0.0,
        primary_count_mean=10.0,
        initial_byclaim_mean=10.0,
        reproduction=0.9,
        primary_distribution=deterministic(1.0),
        byclaim_distribution=deterministic(1.0),
    )
    binar = BINARByClaimModel(
        initial_capital=0.0,
        premium_per_period=0.0,
        primary_count_means=(5.0, 7.0),
        initial_byclaim_means=(1.0, 1.0),
        reproduction_matrix=((0.41, 0.1), (0.05, 0.3)),
        primary_distributions=(deterministic(1.0), deterministic(1.0)),
        byclaim_distributions=(deterministic(1.0), deterministic(1.0)),
    )

    inar_estimate = estimate_inar_byclaim_ruin_probability(
        inar,
        periods=5,
        n_simulations=2000,
        seed=123,
    )
    binar_estimate = estimate_binar_byclaim_ruin_probability(
        binar,
        periods=5,
        n_simulations=2000,
        seed=123,
    )

    assert isinstance(inar_estimate, RuinEstimate)
    assert isinstance(binar_estimate, RuinEstimate)
    assert inar_estimate.probability > 0.99
    assert binar_estimate.probability > 0.99
    assert np.all(np.isfinite(inar_estimate.ruin_times[inar_estimate.ruin_times < np.inf]))


def test_integer_byclaim_validation_rejects_bad_parameters():
    with pytest.raises(ValueError, match="reproduction"):
        INARByClaimModel(
            initial_capital=0.0,
            premium_per_period=1.0,
            primary_count_mean=1.0,
            initial_byclaim_mean=1.0,
            reproduction=1.2,
            primary_distribution=exponential(1.0),
            byclaim_distribution=exponential(1.0),
        )
    with pytest.raises(ValueError, match="shape"):
        BINARByClaimModel(
            initial_capital=0.0,
            premium_per_period=1.0,
            primary_count_means=(1.0, 1.0),
            initial_byclaim_means=(1.0, 1.0),
            reproduction_matrix=((0.2, 0.1, 0.0), (0.1, 0.2, 0.0)),
            primary_distributions=(exponential(1.0), exponential(1.0)),
            byclaim_distributions=(exponential(1.0), exponential(1.0)),
        )
