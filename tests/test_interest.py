import math

import numpy as np
import pytest
from scipy import integrate

from ruin_theory import (
    maximum_before_default_hazard,
    maximum_before_default_survival,
    non_ruin_exponential_interest_force,
    ultimate_ruin_exponential_interest_force,
    win_first_probability_exponential_interest_force,
    win_first_probability_from_non_ruin,
    win_first_time_bound,
)


def test_interest_force_formula_reduces_to_classical_exponential_ruin():
    u = np.array([0.0, 1.0, 2.0])
    ruin = ultimate_ruin_exponential_interest_force(
        u,
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_rate=5.0,
        interest_force=0.0,
    )

    np.testing.assert_allclose(ruin, 0.6 * np.exp(-2.0 * u))


def test_interest_force_exponential_formula_matches_segerdahl_integral():
    premium = 1.2
    arrival = 0.7
    claim_rate = 1.4
    force = 0.08
    u = 2.0

    def integrand(z: float) -> float:
        return math.exp(-claim_rate * z) * (1.0 + force * z / premium) ** (
            arrival / force - 1.0
        )

    integral = integrate.quad(integrand, u, math.inf, epsabs=1e-12)[0]
    integral_zero = integrate.quad(integrand, 0.0, math.inf, epsabs=1e-12)[0]
    expected = arrival * integral / (premium + arrival * integral_zero)

    ruin = ultimate_ruin_exponential_interest_force(
        u,
        premium_rate=premium,
        claim_arrival_rate=arrival,
        claim_rate=claim_rate,
        interest_force=force,
    )

    assert ruin == pytest.approx(expected, rel=1e-12)


def test_win_first_is_quotient_of_non_ruin_probabilities():
    kwargs = dict(
        premium_rate=1.2,
        claim_arrival_rate=0.7,
        claim_rate=1.4,
        interest_force=0.08,
    )
    u = np.array([0.0, 1.0, 2.0])
    v = 1.5
    phi_u = non_ruin_exponential_interest_force(u, **kwargs)
    phi_uv = non_ruin_exponential_interest_force(u + v, **kwargs)

    win_first = win_first_probability_exponential_interest_force(u, v, **kwargs)

    np.testing.assert_allclose(win_first, phi_u / phi_uv)
    assert win_first_probability_exponential_interest_force(1.0, 0.0, **kwargs) == pytest.approx(
        1.0
    )


def test_generic_win_first_and_maximum_before_default_survival():
    non_ruin = lambda x: 1.0 - 0.5 * np.exp(-x)

    assert win_first_probability_from_non_ruin(1.0, 2.0, non_ruin) == pytest.approx(
        non_ruin(1.0) / non_ruin(3.0)
    )
    assert maximum_before_default_survival(2.0, non_ruin) == pytest.approx(
        non_ruin(0.0) / non_ruin(2.0)
    )


def test_maximum_before_default_hazard_matches_log_derivative():
    non_ruin = lambda x: 1.0 - 0.5 * np.exp(-x)
    x = np.array([0.0, 1.0, 2.0])
    expected = 0.5 * np.exp(-x) / (1.0 - 0.5 * np.exp(-x))

    hazard = maximum_before_default_hazard(x, non_ruin, step=1e-6)

    np.testing.assert_allclose(hazard, expected, rtol=2e-5, atol=1e-8)


def test_win_first_time_bound_matches_no_claim_deterministic_growth():
    assert win_first_time_bound(2.0, 3.0, premium_rate=1.5) == pytest.approx(2.0)
    assert win_first_time_bound(
        2.0,
        3.0,
        premium_rate=1.5,
        interest_force=0.1,
    ) == pytest.approx(math.log1p(3.0 / (2.0 + 1.5 / 0.1)) / 0.1)


def test_interest_force_validates_arguments():
    with pytest.raises(ValueError, match="initial_capital"):
        ultimate_ruin_exponential_interest_force(
            -1.0,
            premium_rate=1.0,
            claim_arrival_rate=1.0,
            claim_rate=1.0,
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        win_first_probability_from_non_ruin(1.0, 1.0, lambda x: 1.0 / (1.0 + x))
    with pytest.raises(TypeError, match="callable"):
        maximum_before_default_survival(1.0, 3.0)
