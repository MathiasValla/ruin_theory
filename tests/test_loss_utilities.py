import math

import numpy as np
import pytest

from ruin_theory import (
    coverage_transform,
    deterministic,
    discretize,
    empirical,
    empirical_limited_moment,
    empirical_moment,
    exponential,
    limited_moment,
    raw_moment,
)


def test_raw_moment_matches_exponential_deterministic_and_empirical_laws():
    assert raw_moment(exponential(rate=2.0), 1) == pytest.approx(0.5)
    assert raw_moment(exponential(rate=2.0), 2) == pytest.approx(0.5)
    assert raw_moment(deterministic(3.0), 3) == pytest.approx(27.0)

    sample = np.array([0.0, 2.0, 4.0])
    assert raw_moment(empirical(sample), 2) == pytest.approx(20.0 / 3.0)
    assert empirical_moment(sample, order=2) == pytest.approx(20.0 / 3.0)


def test_limited_moment_matches_exponential_deterministic_and_empirical_laws():
    distribution = exponential(rate=2.0)

    assert limited_moment(distribution, 3.0) == pytest.approx((1.0 - math.exp(-6.0)) / 2.0)
    assert limited_moment(distribution, 3.0, order=2) == pytest.approx(
        0.5 * (1.0 - math.exp(-6.0) * (1.0 + 6.0)),
    )
    np.testing.assert_allclose(
        limited_moment(deterministic(4.0), np.array([2.0, 5.0]), order=1),
        [2.0, 4.0],
    )

    sample = np.array([0.0, 2.0, 4.0])
    assert empirical_limited_moment(sample, 3.0, order=2) == pytest.approx(13.0 / 3.0)
    assert limited_moment(empirical(sample), 3.0, order=2) == pytest.approx(13.0 / 3.0)


def test_coverage_transform_supports_ordinary_and_franchise_deductibles():
    losses = np.array([50.0, 125.0, 300.0])
    np.testing.assert_allclose(
        coverage_transform(
            losses,
            deductible=100.0,
            limit=150.0,
            coinsurance=0.8,
            inflation=1.1,
        ),
        [0.0, 30.0, 120.0],
    )

    np.testing.assert_allclose(
        coverage_transform(losses, franchise_deductible=100.0, limit=150.0, coinsurance=0.5),
        [0.0, 62.5, 75.0],
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        coverage_transform(losses, deductible=1.0, franchise_deductible=1.0)


def test_coverage_transform_distribution_path_tracks_atoms_moments_and_samples():
    base = exponential(rate=1.0)
    covered = coverage_transform(base, deductible=1.0, limit=2.0, coinsurance=0.5)

    np.testing.assert_allclose(
        covered.survival(np.array([-1.0, 0.0, 0.25, 1.0])),
        [1.0, math.exp(-1.0), math.exp(-1.5), 0.0],
    )
    np.testing.assert_allclose(
        covered.cdf(np.array([-1.0, 0.0, 0.25, 1.0])),
        [0.0, 1.0 - math.exp(-1.0), 1.0 - math.exp(-1.5), 1.0],
    )

    expected_mean = 0.5 * math.exp(-1.0) * (1.0 - math.exp(-2.0))
    expected_second = 2.0 * 0.5**2 * math.exp(-1.0) * (
        1.0 - math.exp(-2.0) * (1.0 + 2.0)
    )
    assert covered.mean() == pytest.approx(expected_mean)
    assert covered.variance() == pytest.approx(expected_second - expected_mean**2)

    rng = np.random.default_rng(123)
    sample = covered.sample(200, rng=rng)
    assert sample.shape == (200,)
    assert np.all((0.0 <= sample) & (sample <= 1.0))
    assert discretize(covered, from_=0.0, to=1.0, step=0.25, method="upper").total_mass <= 1.0


def test_coverage_transform_distribution_path_supports_franchise_coverage():
    covered = coverage_transform(
        exponential(rate=1.0),
        franchise_deductible=1.0,
        limit=2.0,
    )

    np.testing.assert_allclose(
        covered.survival(np.array([0.0, 0.5, 1.5, 2.0])),
        [math.exp(-1.0), math.exp(-1.0), math.exp(-1.5), 0.0],
    )
    assert covered.mean() == pytest.approx(2.0 * math.exp(-1.0) - math.exp(-2.0))


def test_discretize_upper_lower_and_rounding_match_endpoint_formulas():
    distribution = exponential(rate=1.0)
    f = distribution.cdf

    upper = discretize(distribution, from_=0.0, to=2.0, step=1.0, method="upper")
    np.testing.assert_allclose(upper.support, [0.0, 1.0])
    np.testing.assert_allclose(
        upper.pmf,
        [f(1.0) - f(0.0), f(2.0) - f(1.0)],
    )
    lower = discretize(distribution, from_=0.0, to=2.0, step=1.0, method="lower")
    np.testing.assert_allclose(lower.support, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        lower.pmf,
        [f(0.0), f(1.0) - f(0.0), f(2.0) - f(1.0)],
    )
    rounding = discretize(distribution, from_=0.0, to=2.0, step=1.0, method="rounding")
    np.testing.assert_allclose(
        rounding.pmf,
        [f(0.5), f(1.5) - f(0.5)],
    )


def test_discretize_unbiased_preserves_probability_and_truncated_first_moment():
    distribution = exponential(rate=1.0)

    grid = discretize(distribution, from_=0.0, to=2.0, step=1.0, method="unbiased")
    masses = grid.pmf
    expected = np.array(
        [
            math.exp(-1.0),
            (1.0 - math.exp(-1.0)) ** 2,
            math.exp(-1.0) - 2.0 * math.exp(-2.0),
        ],
    )
    np.testing.assert_allclose(masses, expected, rtol=1e-14, atol=1e-14)

    assert grid.total_mass == pytest.approx(distribution.cdf(2.0))
    assert grid.mean == pytest.approx(
        limited_moment(distribution, 2.0) - 2.0 * distribution.survival(2.0),
    )


def test_discretize_deterministic_claim_is_exact_on_lower_grid():
    masses = discretize(deterministic(2.0), from_=0.0, to=4.0, step=1.0, method="lower").pmf

    np.testing.assert_allclose(masses, [0.0, 0.0, 1.0, 0.0, 0.0])


def test_loss_helpers_validate_arguments():
    with pytest.raises(ValueError, match="non-negative"):
        raw_moment(exponential(1.0), -1)
    with pytest.raises(TypeError, match="ClaimDistribution"):
        raw_moment(np.array([1.0]))
    with pytest.raises(TypeError, match="ClaimDistribution"):
        limited_moment(np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="epsabs"):
        limited_moment(exponential(1.0), 1.0, epsabs=0.0)
    with pytest.raises(ValueError, match="integer multiple"):
        discretize(exponential(1.0), from_=0.0, to=1.0, step=0.3)
    with pytest.raises(ValueError, match="method"):
        discretize(exponential(1.0), from_=0.0, to=1.0, step=0.5, method="sideways")
    with pytest.raises(TypeError, match="ClaimDistribution"):
        discretize(np.array([0.5, 0.5]), from_=0.0, to=1.0, step=0.5)
