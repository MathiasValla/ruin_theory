import numpy as np
import pytest

from ruin_theory import (
    ByClaimModel,
    ClaimDistribution,
    CramerLundbergProcess,
    FrequencyModel,
    PreventionProgram,
    deterministic,
    empirical,
    exponential,
    mixture_exponential,
)


def test_distribution_sample_validates_shape_and_values():
    bad_shape = ClaimDistribution(
        name="bad_shape",
        mean_value=1.0,
        variance_value=0.0,
        sampler=lambda rng, n: np.ones(n + 1),
    )
    with pytest.raises(ValueError, match="returned"):
        bad_shape.sample(2, rng=np.random.default_rng(1))

    negative = ClaimDistribution(
        name="negative",
        mean_value=1.0,
        variance_value=0.0,
        sampler=lambda rng, n: -np.ones(n),
    )
    with pytest.raises(ValueError, match="negative"):
        negative.sample(1, rng=np.random.default_rng(1))


def test_factories_reject_non_finite_or_misshaped_inputs():
    with pytest.raises(ValueError, match="rate"):
        exponential(np.inf)
    with pytest.raises(ValueError, match="data"):
        empirical([1.0, np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        mixture_exponential([[1.0], [2.0]])


def test_mixture_exponential_has_zero_density_below_support():
    distribution = mixture_exponential([2.0, 5.0], weights=[0.25, 0.75])

    np.testing.assert_allclose(distribution.pdf(np.array([-1.0, 0.0])), [0.0, 4.25])
    np.testing.assert_allclose(distribution.cdf(np.array([-1.0, 0.0])), [0.0, 0.0])
    np.testing.assert_allclose(distribution.survival(np.array([-1.0, 0.0])), [1.0, 1.0])


def test_frequency_model_validates_direct_construction():
    with pytest.raises(ValueError, match="kind"):
        FrequencyModel(kind="compound")
    with pytest.raises(ValueError, match="cannot also define"):
        FrequencyModel(kind="poisson", rate=1.0, interarrival_distribution=deterministic(1.0))


def test_prevention_custom_transform_is_simulation_only_for_mean_intensity():
    prevention = PreventionProgram(severity_transform=lambda claims: np.minimum(claims, 1.0))
    model = CramerLundbergProcess(
        premium_rate=2.0,
        claim_arrival_rate=1.0,
        claim_distribution=deterministic(3.0),
        prevention=prevention,
    )

    np.testing.assert_allclose(prevention.apply_severity(np.array([0.5, 3.0])), [0.5, 1.0])
    with pytest.raises(NotImplementedError, match="severity_transform"):
        _ = model.expected_claim_amount


def test_prevention_frequency_windows_validate_and_apply():
    prevention = PreventionProgram(
        frequency_multiplier=0.25,
        frequency_windows=((2.0, 5.0, 0.0), (8.0, 9.0, 2.0)),
    )

    assert prevention.apply_frequency(4.0) == 1.0
    assert prevention.apply_frequency(4.0, time=3.0) == 0.0
    assert prevention.apply_frequency(4.0, time=5.0) == 1.0
    assert prevention.apply_frequency(4.0, time=8.5) == 8.0
    assert prevention.next_frequency_change_after(0.0) == 2.0

    with pytest.raises(ValueError, match="end"):
        PreventionProgram(frequency_windows=((2.0, 2.0, 1.0),))
    with pytest.raises(ValueError, match="overlap"):
        PreventionProgram(frequency_windows=((1.0, 3.0, 0.5), (2.0, 4.0, 1.0)))


def test_by_claim_sampling_handles_zero_secondary_counts():
    class FakeRng:
        def binomial(self, n, p, size):
            return np.ones(size, dtype=int)

        def poisson(self, lam, size):
            return np.array([0, 2, 1])

    by_claim = ByClaimModel(probability=1.0, distribution=deterministic(5.0), count_mean=1.0)

    np.testing.assert_allclose(by_claim.sample_total(3, rng=FakeRng()), [0.0, 10.0, 5.0])


def test_by_claim_sampling_supports_geometric_secondary_counts():
    class FakeRng:
        def binomial(self, n, p, size):
            return np.ones(size, dtype=int)

        def geometric(self, p, size):
            assert p == 0.25
            return np.array([1, 3, 2])

    by_claim = ByClaimModel(
        probability=1.0,
        distribution=deterministic(5.0),
        count_mean=3.0,
        count_distribution="geometric",
    )

    np.testing.assert_allclose(by_claim.sample_total(3, rng=FakeRng()), [0.0, 10.0, 5.0])


def test_by_claim_count_pgf_matches_count_distribution():
    poisson = ByClaimModel(probability=1.0, distribution=deterministic(1.0), count_mean=2.0)
    geometric = ByClaimModel(
        probability=1.0,
        distribution=deterministic(1.0),
        count_mean=3.0,
        count_distribution="geometric",
    )

    assert poisson.count_pgf(1.0) == pytest.approx(1.0)
    assert poisson.count_pgf(0.5) == pytest.approx(np.exp(-1.0))
    assert geometric.count_pgf(1.0) == pytest.approx(1.0)
    assert geometric.count_pgf(0.25) == pytest.approx(0.25 / (1.0 - 0.75 * 0.25))
    assert np.isinf(geometric.count_pgf(2.0))


def test_by_claim_rejects_unknown_count_distribution():
    with pytest.raises(ValueError, match="count_distribution"):
        ByClaimModel(
            probability=1.0,
            distribution=deterministic(5.0),
            count_distribution="binomial",
        )
