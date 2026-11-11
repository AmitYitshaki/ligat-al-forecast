"""Ordinal likelihood, weights, result order, reference implementation and failure gates."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from statsmodels.miscmodels.ordinal_model import OrderedModel

from ligat.models.ordered_logit import (
    PRECISION_LOSS_GRADIENT_TOL,
    OrderedEloPredictor,
    _log_probabilities,
    fit_ordered_logit,
    ordered_probabilities,
    time_weights,
)


def sample() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(304)
    x = rng.normal(size=1800)
    cumulative = np.column_stack((expit(-0.9 - 1.3 * x), expit(0.5 - 1.3 * x)))
    u = rng.random(len(x))
    # Reverse OrderedModel's away/draw/home to the judge's H/D/A.
    y = 2 - (u[:, None] > cumulative).sum(axis=1)
    return x, y


def test_reference_statsmodels_parameters_and_likelihood() -> None:
    x, y = sample()
    actual = fit_ordered_logit(x, y)
    reference = OrderedModel(2 - y, x[:, None], distr="logit").fit(
        method="bfgs", disp=False, gtol=1e-10, maxiter=1000
    )
    np.testing.assert_allclose(
        [actual.beta, actual.theta1, actual.delta], reference.params, rtol=0, atol=1e-4
    )
    assert abs(actual.log_likelihood - reference.llf) < 1e-6
    thresholds = reference.model.transform_threshold_params(reference.params)
    assert actual.theta2 == pytest.approx(thresholds[2], abs=1e-4)
    np.testing.assert_allclose(
        ordered_probabilities(actual, x),
        reference.model.predict(reference.params)[:, ::-1],
        atol=1e-6,
    )
    assert actual.converged and actual.beta > 0


def test_analytic_scores_match_numeric_derivative_including_draws() -> None:
    x = np.array([-3, -0.4, 0, 0.7, 4])
    params = np.array([1.3, -0.9, np.log(1.4)])
    _, scores = _log_probabilities(params, x)
    numeric = approx_derivative(lambda p: _log_probabilities(p, x)[0].ravel(), params)
    np.testing.assert_allclose(scores.reshape(-1, 3), numeric, rtol=1e-7, atol=1e-8)


def test_weights_equal_repeated_games_and_global_rescaling_and_zero_weight_removal() -> None:
    x, y = sample()
    weights = np.arange(len(x)) % 3 + 1
    weighted = fit_ordered_logit(x, y, weights)
    repeated = fit_ordered_logit(np.repeat(x, weights), np.repeat(y, weights))
    rescaled = fit_ordered_logit(x, y, weights * 7)
    for other in (repeated, rescaled):
        np.testing.assert_allclose(
            [weighted.beta, weighted.theta1, weighted.delta],
            [other.beta, other.theta1, other.delta],
            atol=1e-8,
        )
    dropped = fit_ordered_logit(x[1:], y[1:])
    weights = np.ones(len(x))
    weights[0] = 0
    ignored = fit_ordered_logit(x, y, weights)
    np.testing.assert_allclose(
        [dropped.beta, dropped.theta1, dropped.delta],
        [ignored.beta, ignored.theta1, ignored.delta],
        atol=1e-8,
    )


def test_hda_order_monotonicity_extremes_and_no_home_advantage_offset() -> None:
    x, y = sample()
    fit = fit_ordered_logit(x, y)
    p = ordered_probabilities(fit, [-1000, -1, 0, 1, 1000])
    assert p[0, 2] == 1 and p[-1, 0] == 1
    assert (np.diff(p[:, 0]) >= 0).all()
    assert (np.diff(p[:, 2]) <= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-15)
    model = OrderedEloPredictor()
    train = pd.DataFrame({"elo_diff": x * 400, "outcome": y, "season_id": 8})
    model.fit(train)
    games = pd.DataFrame({"elo_diff": [0, 400, -400], "home_advantage": [999] * 3})
    np.testing.assert_allclose(model.predict(games), ordered_probabilities(fit, [0, 1, -1]))
    assert model.training_seasons == (8,)
    with pytest.raises(RuntimeError, match="before prediction"):
        OrderedEloPredictor().predict(games)


def test_time_weights_previous_season_one_and_validation_excluded() -> None:
    np.testing.assert_array_equal(time_weights([8, 9, 10], 11, None), [1, 1, 1])
    np.testing.assert_allclose(time_weights([8, 9, 10], 11, 2), [0.5, np.sqrt(0.5), 1])
    with pytest.raises(ValueError, match="earlier"):
        time_weights([8, 11], 11, 2)
    x, y = sample()
    train = pd.DataFrame(
        {"elo_diff": x * 400, "outcome": y, "season_id": np.arange(len(x)) % 3 + 8}
    )
    weighted = OrderedEloPredictor(2)
    with patch("ligat.models.ordered_logit.fit_ordered_logit", wraps=fit_ordered_logit) as mocked:
        weighted.fit(train)
        np.testing.assert_array_equal(mocked.call_args.args[0], train.elo_diff.to_numpy() / 400)
        np.testing.assert_array_equal(
            mocked.call_args.args[2], time_weights(train.season_id, 11, 2)
        )
    with pytest.raises(ValueError, match="not LOSO"):
        weighted.predict(pd.DataFrame({"elo_diff": [0], "season_id": [9]}))


@pytest.mark.parametrize("weights", [[-1, 1, 1], [0, 0, 0], [1, 2], [np.nan, 1, 1]])
def test_invalid_weights_stop(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="Weights"):
        fit_ordered_logit([-1, 0, 1], [2, 1, 0], weights)


def test_optimizer_failure_and_negative_beta_stop() -> None:
    x, y = sample()
    with patch(
        "ligat.models.ordered_logit.minimize",
        return_value=SimpleNamespace(success=False, status=1, message="failed", x=np.zeros(3)),
    ):
        with pytest.raises(RuntimeError, match="did not converge"):
            fit_ordered_logit(x, y)
    with pytest.raises(RuntimeError, match="Negative"):
        fit_ordered_logit(x, 2 - y)
    with pytest.raises(ValueError, match="three outcomes"):
        fit_ordered_logit([0, 1, 2], [0, 0, 1])


def test_normal_success_records_gradient_without_fallback() -> None:
    assert PRECISION_LOSS_GRADIENT_TOL == 1e-6
    x, y = sample()
    fit = fit_ordered_logit(x, y)
    assert fit.converged
    assert fit.accepted_by_gradient_check is False
    assert np.isfinite(fit.max_abs_gradient)
    assert fit.max_abs_gradient <= PRECISION_LOSS_GRADIENT_TOL


@pytest.mark.parametrize("weight_scale", [1, 1e9])
def test_precision_loss_at_minimum_uses_recomputed_normalized_gradient(
    monkeypatch: pytest.MonkeyPatch, weight_scale: float
) -> None:
    x, y = sample()
    ordinary = fit_ordered_logit(x, y)
    optimum = np.array([ordinary.beta, ordinary.theta1, ordinary.delta])
    monkeypatch.setattr(
        "ligat.models.ordered_logit.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=2,
            message="precision loss",
            x=optimum,
            jac=np.ones(3),
            nit=ordinary.iterations,
        ),
    )
    fit = fit_ordered_logit(x, y, np.full(len(x), weight_scale))
    assert fit.converged and fit.accepted_by_gradient_check
    assert fit.max_abs_gradient <= PRECISION_LOSS_GRADIENT_TOL
    assert fit.max_abs_gradient == pytest.approx(ordinary.max_abs_gradient, abs=1e-15)
    np.testing.assert_array_equal([fit.beta, fit.theta1, fit.delta], optimum)


def test_precision_loss_far_from_minimum_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = sample()
    fit = fit_ordered_logit(x, y)
    monkeypatch.setattr(
        "ligat.models.ordered_logit.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=2,
            message="precision loss",
            jac=np.zeros(3),
            x=np.array([fit.beta + 5, fit.theta1 + 2, fit.delta + 1]),
        ),
    )
    with pytest.raises(RuntimeError, match="status=2; message=precision loss"):
        fit_ordered_logit(x, y)


@pytest.mark.parametrize("status", [0, 1, 3])
def test_other_failure_status_at_minimum_still_stops(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    x, y = sample()
    fit = fit_ordered_logit(x, y)
    monkeypatch.setattr(
        "ligat.models.ordered_logit.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=status,
            message="not converged",
            x=np.array([fit.beta, fit.theta1, fit.delta]),
        ),
    )
    with pytest.raises(RuntimeError, match=f"status={status}; message=not converged"):
        fit_ordered_logit(x, y)


@pytest.mark.parametrize("success,status,value", [(True, 0, np.nan), (False, 2, np.inf)])
def test_nonfinite_parameters_stop_even_with_success_or_precision_loss(
    monkeypatch: pytest.MonkeyPatch, success: bool, status: int, value: float
) -> None:
    x, y = sample()
    monkeypatch.setattr(
        "ligat.models.ordered_logit.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=success,
            status=status,
            message="nonfinite",
            x=np.array([value, 0, 0]),
        ),
    )
    with pytest.raises(RuntimeError, match=f"status={status}; message=nonfinite"):
        fit_ordered_logit(x, y)


def test_gradient_acceptance_does_not_bypass_negative_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = sample()
    positive = fit_ordered_logit(x, y)
    mirrored = np.array([-positive.beta, -positive.theta2, positive.delta])
    monkeypatch.setattr(
        "ligat.models.ordered_logit.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, status=2, message="precision loss", x=mirrored, nit=1
        ),
    )
    with pytest.raises(RuntimeError, match="Negative"):
        fit_ordered_logit(x, 2 - y)
