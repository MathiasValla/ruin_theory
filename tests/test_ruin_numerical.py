import numpy as np
import pytest

from ruin_theory import (
    ByClaimModel,
    CapitalInjectionModel,
    CramerLundbergProcess,
    PreventionProgram,
    deterministic,
    discrete_pollaczek_khinchine_ultimate_ruin,
    equilibrium_severity_pmf,
    exponential,
    ultimate_ruin_exponential,
    ultimate_ruin_panjer,
)


def test_discrete_pk_uses_floor_surplus_grid_convention():
    surplus = np.array([0.0, 0.99, 1.0, 1.01, 2.0])

    ruin = discrete_pollaczek_khinchine_ultimate_ruin(
        [0.0, 1.0],
        surplus,
        step=1.0,
        rho=0.5,
    )

    np.testing.assert_allclose(ruin, [0.5, 0.5, 0.25, 0.25, 0.125])


def test_discrete_pk_accepts_safety_loading_instead_of_rho():
    ruin = discrete_pollaczek_khinchine_ultimate_ruin(
        [0.0, 1.0],
        [0.0, 1.0],
        step=1.0,
        safety_loading=1.0,
    )

    np.testing.assert_allclose(ruin, [0.5, 0.25])


def test_equilibrium_severity_pmf_discretizes_exponential_integrated_tail():
    pmf = equilibrium_severity_pmf(
        exponential(rate=2.0),
        step=0.5,
        max_value=2.0,
        method="lower",
    )

    expected_cdf = 1.0 - np.exp(-2.0 * np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    expected = np.r_[expected_cdf[0], np.diff(expected_cdf)]
    np.testing.assert_allclose(pmf, expected)


def test_equilibrium_severity_pmf_validates_grid_scale_and_method():
    distribution = exponential(rate=2.0)

    with pytest.raises(ValueError, match="integer multiple"):
        equilibrium_severity_pmf(distribution, step=1.0, max_value=0.5)
    with pytest.raises(ValueError, match="integer multiple"):
        equilibrium_severity_pmf(distribution, step=1.0, max_value=2.5)
    with pytest.raises(ValueError, match="method"):
        equilibrium_severity_pmf(distribution, step=1.0, max_value=2.0, method="middle")
    with pytest.raises(ValueError, match="scale"):
        equilibrium_severity_pmf(distribution, step=1.0, max_value=2.0, scale=0.0)


def test_ultimate_ruin_panjer_exponential_brackets_closed_form():
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=0.5,
        claim_distribution=exponential(rate=1.0),
    )
    surplus = np.array([0.0, 1.0, 2.0, 4.0])
    exact = ultimate_ruin_exponential(model, surplus)

    lower = ultimate_ruin_panjer(
        model,
        surplus,
        step=0.02,
        max_value=40.0,
        discretization="upper",
    )
    upper = ultimate_ruin_panjer(
        model,
        surplus,
        step=0.02,
        max_value=40.0,
        discretization="lower",
    )

    assert np.all(lower <= exact + 2e-12)
    assert np.all(exact <= upper + 2e-12)
    np.testing.assert_allclose(0.5 * (lower + upper), exact, atol=0.006)


def test_ultimate_ruin_panjer_can_return_details():
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=0.5,
        claim_distribution=exponential(rate=1.0),
    )

    result = ultimate_ruin_panjer(
        model,
        [0.0, 1.0],
        step=0.1,
        max_value=10.0,
        return_result=True,
    )

    assert result.step == pytest.approx(0.1)
    assert result.rho == pytest.approx(0.5)
    assert result.ruin_probabilities.shape == (2,)
    assert np.all(np.diff(result.aggregate_cdf) >= -1e-14)
    assert "floor(u/step)" in result.convention


def test_ultimate_ruin_panjer_rejects_unsupported_model_features_and_bad_inputs():
    claim_distribution = exponential(rate=1.0)

    with pytest.raises(ValueError, match="frequency prevention"):
        ultimate_ruin_panjer(
            CramerLundbergProcess(
                premium_rate=2.0,
                claim_arrival_rate=0.5,
                claim_distribution=claim_distribution,
                prevention=PreventionProgram(frequency_windows=((0.0, 1.0, 0.5),)),
            ),
            step=0.1,
            max_value=2.0,
        )
    with pytest.raises(ValueError, match="by-claims"):
        ultimate_ruin_panjer(
            CramerLundbergProcess(
                premium_rate=2.0,
                claim_arrival_rate=0.5,
                claim_distribution=claim_distribution,
                by_claims=(
                    ByClaimModel(probability=0.2, distribution=deterministic(1.0)),
                ),
            ),
            step=0.1,
            max_value=2.0,
        )
    with pytest.raises(ValueError, match="capital injections"):
        ultimate_ruin_panjer(
            CramerLundbergProcess(
                premium_rate=2.0,
                claim_arrival_rate=0.5,
                claim_distribution=claim_distribution,
                capital_injections=(
                    CapitalInjectionModel(rate=0.1, distribution=deterministic(1.0)),
                ),
            ),
            step=0.1,
            max_value=2.0,
        )
    with pytest.raises(ValueError, match="net profit"):
        ultimate_ruin_panjer(
            CramerLundbergProcess(
                premium_rate=0.5,
                claim_arrival_rate=1.0,
                claim_distribution=claim_distribution,
            ),
            step=0.1,
            max_value=2.0,
        )
    with pytest.raises(ValueError, match="step"):
        ultimate_ruin_panjer(
            CramerLundbergProcess(
                premium_rate=2.0,
                claim_arrival_rate=0.5,
                claim_distribution=claim_distribution,
            ),
            step=0.0,
            max_value=2.0,
        )
    with pytest.raises(ValueError, match="max_value"):
        ultimate_ruin_panjer(
            CramerLundbergProcess(
                premium_rate=2.0,
                claim_arrival_rate=0.5,
                claim_distribution=claim_distribution,
            ),
            step=0.1,
            max_value=0.0,
        )


def test_beekman_panjer_lomax_bounds_match_actuar_reference():
    def equilibrium_lomax_cdf(x):
        values = np.asarray(x, dtype=float)
        return np.where(values < 0.0, 0.0, 1.0 - (4.0 / (4.0 + values)) ** 4)

    max_value = 200
    points = np.arange(max_value + 1, dtype=float)
    lower_endpoint_pmf = np.zeros(max_value + 1, dtype=float)
    lower_endpoint_pmf[:-1] = equilibrium_lomax_cdf(points[1:]) - equilibrium_lomax_cdf(
        points[:-1]
    )
    upper_endpoint_pmf = np.empty(max_value + 1, dtype=float)
    upper_endpoint_pmf[0] = equilibrium_lomax_cdf(0.0)
    upper_endpoint_pmf[1:] = equilibrium_lomax_cdf(points[1:]) - equilibrium_lomax_cdf(
        points[:-1]
    )
    surplus = np.arange(0, 55, 5)

    lower = discrete_pollaczek_khinchine_ultimate_ruin(
        lower_endpoint_pmf,
        surplus,
        step=1.0,
        rho=5.0 / 6.0,
    )
    upper = discrete_pollaczek_khinchine_ultimate_ruin(
        upper_endpoint_pmf,
        surplus,
        step=1.0,
        rho=5.0 / 6.0,
    )

    np.testing.assert_allclose(
        lower,
        [
            0.6719160,
            0.2892792,
            0.1361541,
            0.0662486,
            0.0329848,
            0.0167551,
            0.0086802,
            0.0045911,
            0.0024843,
            0.0013790,
            0.0007877,
        ],
        atol=5e-8,
    )
    np.testing.assert_allclose(
        upper,
        [
            0.83333,
            0.51572,
            0.32938,
            0.21200,
            0.13700,
            0.08877,
            0.05764,
            0.03749,
            0.02443,
            0.01595,
            0.01043,
        ],
        atol=5e-6,
    )
