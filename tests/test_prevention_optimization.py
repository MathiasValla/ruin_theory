import math

import numpy as np
import pytest

from ruin_theory import (
    ConstantPreventionResult,
    ExpectedSurplusPreventionResult,
    adjustment_coefficient,
    exponential,
    optimize_constant_prevention,
    optimize_expected_surplus_prevention,
)


def test_constant_prevention_optimizer_matches_exponential_frequency_closed_form():
    premium_rate = 10.0
    decay = 0.2

    result = optimize_constant_prevention(
        exponential(rate=1.0),
        premium_rate=premium_rate,
        frequency_function=lambda p: math.exp(-decay * p),
    )

    assert isinstance(result, ConstantPreventionResult)
    assert result.amount == pytest.approx(premium_rate - 1.0 / decay, abs=2e-5)
    assert result.boundary == "interior"
    assert result.claim_arrival_rate == pytest.approx(math.exp(-1.0))
    assert result.net_premium_rate == pytest.approx(5.0)
    assert result.loss_ratio == pytest.approx(math.exp(-1.0) / 5.0)
    assert result.safety_loading == pytest.approx(5.0 * math.e - 1.0)
    assert result.non_ruin_probability_at_zero == pytest.approx(1.0 - math.exp(-1.0) / 5.0)
    assert result.prevention_program.frequency_multiplier == pytest.approx(result.claim_arrival_rate)
    assert result.model.premium_rate == pytest.approx(result.net_premium_rate)
    assert result.model.claim_arrival_rate == pytest.approx(result.claim_arrival_rate)
    assert result.adjustment_coefficient == pytest.approx(adjustment_coefficient(result.model))


def test_constant_prevention_optimizer_returns_zero_when_prevention_is_not_efficient():
    result = optimize_constant_prevention(
        exponential(rate=1.0),
        premium_rate=10.0,
        frequency_function=lambda p: math.exp(-0.05 * p),
    )

    assert result.amount == pytest.approx(0.0)
    assert result.boundary == "zero"
    assert result.loss_ratio == pytest.approx(0.1)


def test_constant_prevention_optimizer_handles_activation_threshold():
    premium_rate = 10.0
    threshold = 3.0
    decay = 0.5

    def frequency(amount: float) -> float:
        if amount <= threshold:
            return 1.0
        return math.exp(-decay * (amount - threshold))

    result = optimize_constant_prevention(
        exponential(rate=1.0),
        premium_rate=premium_rate,
        frequency_function=frequency,
        activation_threshold=threshold,
    )

    assert result.amount == pytest.approx(premium_rate - 1.0 / decay, abs=2e-5)
    assert result.boundary == "interior"
    assert result.loss_ratio < 0.1


def test_constant_prevention_optimizer_compares_threshold_candidate_to_zero():
    premium_rate = 10.0
    threshold = 7.0
    decay = 0.5

    def frequency(amount: float) -> float:
        if amount <= threshold:
            return 1.0
        return math.exp(-decay * (amount - threshold))

    result = optimize_constant_prevention(
        exponential(rate=1.0),
        premium_rate=premium_rate,
        frequency_function=frequency,
        activation_threshold=threshold,
    )

    assert result.amount == pytest.approx(0.0)
    assert result.boundary == "zero"
    assert result.loss_ratio == pytest.approx(0.1)


def test_constant_prevention_optimizer_respects_max_prevention_boundary():
    result = optimize_constant_prevention(
        exponential(rate=1.0),
        premium_rate=10.0,
        frequency_function=lambda p: math.exp(-0.2 * p),
        max_prevention=4.0,
    )

    assert result.amount == pytest.approx(4.0, abs=2e-5)
    assert result.boundary == "upper"


def test_expected_surplus_optimizer_matches_gauchon_closed_form_condition():
    decay = 2.0
    result = optimize_expected_surplus_prevention(
        exponential(rate=1.0),
        premium_rate=10.0,
        frequency_function=lambda p: math.exp(-decay * p),
        horizon=2.0,
        initial_capital=3.0,
    )

    assert isinstance(result, ExpectedSurplusPreventionResult)
    assert result.amount == pytest.approx(math.log(decay) / decay, abs=2e-5)
    assert result.boundary == "interior"
    assert result.net_drift == pytest.approx(10.0 - result.amount - 1.0 / decay)
    assert result.expected_surplus == pytest.approx(3.0 + 2.0 * result.net_drift)


def test_expected_surplus_optimizer_can_differ_from_ruin_optimizer():
    claims = exponential(rate=1.0)
    frequency = lambda p: math.exp(-0.2 * p)

    ruin_optimum = optimize_constant_prevention(
        claims,
        premium_rate=10.0,
        frequency_function=frequency,
    )
    surplus_optimum = optimize_expected_surplus_prevention(
        claims,
        premium_rate=10.0,
        frequency_function=frequency,
        horizon=1.0,
    )

    assert ruin_optimum.amount == pytest.approx(5.0, abs=2e-5)
    assert surplus_optimum.amount == pytest.approx(0.0)
    assert surplus_optimum.boundary == "zero"


def test_expected_surplus_optimizer_respects_threshold_and_upper_bound():
    threshold = 0.1
    decay = 3.0

    def frequency(amount: float) -> float:
        if amount <= threshold:
            return 1.0
        return math.exp(-decay * (amount - threshold))

    result = optimize_expected_surplus_prevention(
        exponential(rate=1.0),
        premium_rate=8.0,
        frequency_function=frequency,
        horizon=1.0,
        activation_threshold=threshold,
        max_prevention=5.0,
    )

    expected_amount = threshold + math.log(decay) / decay
    assert result.amount == pytest.approx(expected_amount, abs=2e-5)
    assert threshold <= result.amount <= 5.0
    assert result.net_drift > 8.0 - 1.0


def test_constant_prevention_optimizer_validates_arguments():
    with pytest.raises(TypeError, match="ClaimDistribution"):
        optimize_constant_prevention(
            object(),
            premium_rate=1.0,
            frequency_function=lambda p: 1.0,
        )
    with pytest.raises(ValueError, match="finite positive values"):
        optimize_constant_prevention(
            exponential(rate=1.0),
            premium_rate=1.0,
            frequency_function=lambda p: 0.0,
        )
    with pytest.raises(ValueError, match="premium_rate"):
        optimize_constant_prevention(
            exponential(rate=1.0),
            premium_rate=0.0,
            frequency_function=lambda p: 1.0,
        )
    with pytest.raises(TypeError, match="callable"):
        optimize_constant_prevention(
            exponential(rate=1.0),
            premium_rate=1.0,
            frequency_function=1.0,
        )
    with pytest.raises(ValueError, match="finite positive values"):
        optimize_constant_prevention(
            exponential(rate=1.0),
            premium_rate=1.0,
            frequency_function=lambda p: np.nan,
        )
    with pytest.raises(ValueError, match="horizon"):
        optimize_expected_surplus_prevention(
            exponential(rate=1.0),
            premium_rate=1.0,
            frequency_function=lambda p: 1.0,
            horizon=0.0,
        )
