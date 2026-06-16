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


def test_by_claim_sampling_handles_zero_secondary_counts():
    class FakeRng:
        def binomial(self, n, p, size):
            return np.ones(size, dtype=int)

        def poisson(self, lam, size):
            return np.array([0, 2, 1])

    by_claim = ByClaimModel(probability=1.0, distribution=deterministic(5.0), count_mean=1.0)

    np.testing.assert_allclose(by_claim.sample_total(3, rng=FakeRng()), [0.0, 10.0, 5.0])
