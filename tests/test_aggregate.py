"""Aggregate-distribution and Panjer recursion checks."""

from math import exp, factorial

import numpy as np
import pytest

from ruin_theory import (
    AggregateDistribution,
    compound_poisson_distribution,
    panjer_recursion,
)


def test_binomial_with_degenerate_severity_is_exact_on_lattice():
    aggregate = panjer_recursion(
        [0.0, 0.0, 1.0],
        "binomial",
        frequency_params={"n": 3, "p": 0.4},
    )

    expected = np.zeros(7)
    expected[[0, 2, 4, 6]] = [
        0.6**3,
        3 * 0.4 * 0.6**2,
        3 * 0.4**2 * 0.6,
        0.4**3,
    ]
    np.testing.assert_allclose(aggregate.grid, np.arange(7))
    np.testing.assert_allclose(aggregate.pmf, expected, rtol=1e-14, atol=1e-15)
    assert aggregate.mean() == pytest.approx(2.0 * 3.0 * 0.4)
    assert aggregate.variance() == pytest.approx(4.0 * 3.0 * 0.4 * 0.6)


def test_compound_poisson_bernoulli_thins_to_poisson():
    lam = 2.5
    claim_probability = 0.3
    aggregate = compound_poisson_distribution(
        [1.0 - claim_probability, claim_probability],
        rate=lam,
        max_aggregate=9,
    )

    thinned_mean = lam * claim_probability
    expected = np.array(
        [exp(-thinned_mean) * thinned_mean**k / factorial(k) for k in range(10)]
    )
    np.testing.assert_allclose(aggregate.pmf, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(aggregate.cdf_values(), np.cumsum(expected), rtol=1e-13)


def test_geometric_frequency_with_unit_severity_is_geometric_count_law():
    probability = 0.25
    aggregate = panjer_recursion(
        [0.0, 1.0],
        {"model": "geometric", "probability": probability},
        max_aggregate=8,
    )

    expected = probability * (1.0 - probability) ** np.arange(9)
    np.testing.assert_allclose(aggregate.pmf, expected, rtol=1e-14, atol=1e-15)
    assert aggregate.cdf(2) == pytest.approx(expected[:3].sum())
    assert aggregate.survival(2) == pytest.approx(1.0 - expected[:3].sum())


def test_compound_geometric_with_zero_severity_mass_matches_closed_recursion():
    severity = np.array([0.2, 0.5, 0.3])
    probability = 1.0 / 6.0
    q = 1.0 - probability
    aggregate = panjer_recursion(
        severity,
        "geometric",
        frequency_params={"p": probability},
        max_aggregate=7,
    )

    expected = np.zeros(8)
    expected[0] = probability / (1.0 - q * severity[0])
    scale = q / (1.0 - q * severity[0])
    for k in range(1, expected.size):
        upper = min(k, severity.size - 1)
        expected[k] = scale * sum(
            severity[j] * expected[k - j] for j in range(1, upper + 1)
        )

    np.testing.assert_allclose(aggregate.pmf, expected, rtol=1e-14, atol=1e-15)


def test_negative_binomial_frequency_with_unit_severity_is_count_law():
    r = 2.5
    p = 0.4
    aggregate = panjer_recursion(
        [0.0, 1.0],
        "negative_binomial",
        frequency_params={"r": r, "p": p},
        max_aggregate=6,
    )

    expected = np.empty(7)
    expected[0] = p**r
    for k in range(1, expected.size):
        expected[k] = ((k + r - 1.0) / k) * (1.0 - p) * expected[k - 1]
    np.testing.assert_allclose(aggregate.pmf, expected, rtol=1e-14, atol=1e-15)


def test_quantile_value_at_risk_and_tvar_follow_discrete_tail_convention():
    aggregate = AggregateDistribution(grid=[0, 1, 2, 3], pmf=[0.1, 0.2, 0.4, 0.3])

    np.testing.assert_allclose(aggregate.cdf([0, 1.5, 3]), [0.1, 0.3, 1.0])
    assert aggregate.ppf(0.6) == 2.0
    assert aggregate.quantile(0.6) == 2.0
    assert aggregate.value_at_risk(0.6) == 2.0
    assert aggregate.tail_value_at_risk(0.5) == pytest.approx(2.6)
    assert aggregate.tail_value_at_risk(0.9) == pytest.approx(3.0)
    assert aggregate.ppf(1.0) == 3.0


def test_invalid_aggregate_and_panjer_arguments_are_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        panjer_recursion([0.2, 0.2], "poisson", frequency_params={"lambda": 1.0})
    with pytest.raises(ValueError, match="support must start at zero"):
        panjer_recursion(
            [0.5, 0.5],
            "poisson",
            frequency_params={"lambda": 1.0},
            support=[1, 2],
        )
    with pytest.raises(ValueError, match="frequency model"):
        panjer_recursion([0.5, 0.5], "logarithmic", frequency_params={"p": 0.5})
    with pytest.raises(ValueError, match="p must be in"):
        panjer_recursion([0.0, 1.0], "binomial", frequency_params={"n": 2, "p": 1.0})
    with pytest.raises(ValueError, match="q exceeds"):
        AggregateDistribution(grid=[0, 1], pmf=[0.25, 0.25]).ppf(0.75)
    with pytest.raises(ValueError, match="NaN"):
        AggregateDistribution(grid=[0, 1], pmf=[0.5, 0.5]).cdf(np.nan)
    with pytest.raises(ValueError, match="NaN"):
        AggregateDistribution(grid=[0, 1], pmf=[0.5, 0.5]).survival([0.0, np.nan])
    with pytest.raises(ValueError, match="TVaR requires"):
        AggregateDistribution(grid=[0, 1], pmf=[0.25, 0.25]).tail_value_at_risk(0.25)
