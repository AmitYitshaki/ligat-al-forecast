"""Season-weight transport and weighted Poisson likelihood, with legacy identity checks."""

import numpy as np
import pytest
import statsmodels.api as sm
from pandas.testing import assert_frame_equal

from ligat.evaluate import team_goal_backtest as judge
from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_metrics import poisson_nll
from ligat.models import form_models
from ligat.models.form_models import fit_poisson
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_team_goal_backtest import MeanGoalSpy, factory_with_history


@pytest.mark.parametrize("half_life", [None, 1, 2, 4])
def test_judge_passes_manual_raw_weights_with_aligned_paired_rows(goal_sources, half_life):
    """For validation 15: seasons 14, 12, 8 receive 1, 0.5, 0.125 when half_life is 2."""
    data = prepare(goal_sources)
    factory, models = factory_with_history("per_fold")
    result = judge.evaluate_team_goals(
        factory,
        data,
        feature_columns=("elo_diff",),
        half_life=half_life,
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
    )
    model = models[0]
    X, y, context = model.fit_calls[0]
    weights = model.weight_calls[0]
    expected = np.array(
        [
            1.0 if half_life is None else 0.5 ** ((14 - int(s)) / half_life)
            for s in context.season_id
        ]
    )
    np.testing.assert_array_equal(weights, expected)
    assert weights.index.equals(X.index) and weights.index.equals(y.index)
    assert weights.groupby(level="game_id").nunique().eq(1).all()
    assert weights.loc[context.season_id.eq(14)].eq(1).all()
    if half_life == 2:
        assert weights.loc[context.season_id.eq(12)].eq(0.5).all()
        assert weights.loc[context.season_id.eq(8)].eq(0.125).all()
    assert result.per_team["lambda"].eq(np.average(y, weights=expected)).all()
    assert (
        result.per_team.half_life.isna().all()
        if half_life is None
        else (result.per_team.half_life.eq(half_life).all())
    )
    # Evaluation remains an unweighted mean over observed validation rows.
    assert result.summary.poisson_nll == pytest.approx(result.per_team.poisson_nll.mean())


def test_default_and_explicit_none_use_unit_weights_and_identical_outputs(goal_sources):
    data = prepare(goal_sources)
    a = judge.evaluate_team_goals(MeanGoalSpy, data, feature_columns=("elo_diff",))
    factory, models = factory_with_history("per_fold")
    b = judge.evaluate_team_goals(factory, data, feature_columns=("elo_diff",), half_life=None)
    assert_frame_equal(a.per_team, b.per_team)
    assert_frame_equal(a.per_fold, b.per_fold)
    for model in models:
        assert model.weight_calls[0].eq(1).all()


@pytest.mark.parametrize("half_life", [1, 2, 4])
def test_daily_models_reject_season_decay_before_any_fit(goal_sources, half_life):
    factory, models = factory_with_history("per_matchday")
    with pytest.raises(ValueError, match="per_matchday.*season half_life"):
        judge.evaluate_team_goals(
            factory, prepare(goal_sources), feature_columns=("elo_diff",), half_life=half_life
        )
    assert not any(model.fit_calls for model in models)


def test_daily_none_never_requests_season_weights(goal_sources, monkeypatch):
    monkeypatch.setattr(judge, "time_weights", lambda *a: pytest.fail("Daily season weighting"))
    factory, models = factory_with_history("per_matchday")
    result = judge.evaluate_team_goals(
        factory, prepare(goal_sources), feature_columns=("elo_diff",)
    )
    assert result.per_team.half_life.isna().all()
    assert all(model.weight_calls == [None] for model in models)


@pytest.mark.parametrize("half_life", [-1, 0, 0.5, 3, np.nan, np.inf, True, "2"])
def test_undeclared_half_lives_fail_before_model_creation(goal_sources, half_life):
    def factory():
        pytest.fail("Premature model creation")

    with pytest.raises(ValueError, match="None, 1, 2 or 4"):
        judge.evaluate_team_goals(
            factory, prepare(goal_sources), feature_columns=("elo_diff",), half_life=half_life
        )


def test_every_fold_weights_only_strictly_earlier_training_seasons(goal_sources, monkeypatch):
    calls = []
    original = judge.time_weights

    def guarded(seasons, validation, half_life):
        assert (seasons < validation).all()
        calls.append(validation)
        return original(seasons, validation, half_life)

    monkeypatch.setattr(judge, "time_weights", guarded)
    judge.evaluate_team_goals(
        MeanGoalSpy, prepare(goal_sources), feature_columns=("elo_diff",), half_life=2
    )
    assert calls == list(range(11, 26))


@pytest.mark.parametrize("declared", [False, None])
def test_models_without_explicit_weight_support_fail_before_fit(goal_sources, declared):
    class Unsupported:
        refit_policy = "per_fold"
        if declared is not None:
            supports_sample_weight = declared

        def fit(self, *args, **kwargs):
            pytest.fail("Unsupported weights reached fit")

    with pytest.raises(ValueError, match="support for sample_weight"):
        judge.evaluate_team_goals(
            Unsupported, prepare(goal_sources), feature_columns=("elo_diff",), half_life=2
        )


@pytest.fixture
def poisson_sample():
    rng = np.random.default_rng(14052)
    x = np.column_stack((np.ones(80), rng.normal(size=(80, 2))))
    y = rng.poisson(np.exp(x @ np.array([0.2, 0.3, -0.2])))
    return x, y


def test_weighted_objective_and_gradient_match_independent_nll_and_finite_differences(
    poisson_sample, monkeypatch
):
    """Check the optimizer's actual objective and gradient at several non-optimal points."""
    x, y = poisson_sample
    weights = np.linspace(0.05, 3, len(y))
    minimize = form_models.minimize
    checked = []

    def audited(objective, initial, **kwargs):
        for beta in (initial, np.array([0.2, -0.1, 0.4]), np.array([-0.3, 0.4, -0.2])):
            value, analytical = objective(beta)
            expected = np.average(poisson_nll(y, np.exp(x @ beta)), weights=weights)
            assert value == pytest.approx(expected, abs=1e-13)
            step = 1e-6
            numerical = []
            for dimension in np.eye(len(beta)):
                plus = objective(beta + step * dimension)[0]
                minus = objective(beta - step * dimension)[0]
                numerical.append((plus - minus) / (2 * step))
            np.testing.assert_allclose(analytical, numerical, rtol=0, atol=1e-8)
            checked.append(beta)
        return minimize(objective, initial, **kwargs)

    monkeypatch.setattr(form_models, "minimize", audited)
    fit_poisson(x, y, weights=weights)
    assert len(checked) == 3


def test_integer_weights_equal_duplicating_observations(poisson_sample):
    x, y = poisson_sample
    weights = np.ones(len(y), dtype=int)
    weights[0] = 2
    actual = fit_poisson(x, y, weights=weights)
    expected = fit_poisson(np.repeat(x, weights, axis=0), np.repeat(y, weights))
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-7, rtol=0)


def test_weighted_fit_matches_statsmodels_and_weight_rescaling(poisson_sample):
    x, y = poisson_sample
    weights = np.linspace(0.05, 3, len(y))
    actual = fit_poisson(x, y, weights=weights)
    expected = sm.GLM(y, x, family=sm.families.Poisson(), freq_weights=weights).fit(tol=1e-12)
    np.testing.assert_allclose(actual.coefficients, expected.params, atol=1e-7, rtol=0)
    scaled = fit_poisson(x, y, weights=weights * 7)
    np.testing.assert_allclose(actual.coefficients, scaled.coefficients, atol=1e-7, rtol=0)


def test_none_weights_preserve_legacy_fit_exactly(poisson_sample):
    x, y = poisson_sample
    assert fit_poisson(x, y) == fit_poisson(x, y, weights=None)


@pytest.mark.parametrize(
    "weights",
    [
        [-1, 1, 1, 1],
        [np.nan, 1, 1, 1],
        [np.inf, 1, 1, 1],
        [1, 1, 1],
        [0, 0, 0, 0],
        [[1, 1, 1, 1]],
        1,
        [1j, 1, 1, 1],
        [1e308, 1e308, 1, 1],
    ],
)
def test_invalid_poisson_weights_are_rejected(weights):
    x = np.column_stack((np.ones(4), [0, 1, 0, 1]))
    with pytest.raises(ValueError, match="weights"):
        fit_poisson(x, [1, 2, 1, 3], weights=weights)


def test_zero_weight_rows_do_not_affect_fit_or_numerical_stability(poisson_sample):
    x, y = poisson_sample
    # This row would overflow exp(x @ beta), but has no likelihood contribution.
    actual = fit_poisson(
        np.vstack((x, [1, 1e200, -1e200])), np.append(y, 99), weights=np.append(np.ones(len(y)), 0)
    )
    expected = fit_poisson(x, y)
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-7, rtol=0)


def test_zero_weights_cannot_hide_empty_goal_mass_or_effective_rank():
    x = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
    with pytest.raises(ValueError, match="weighted goal mass"):
        fit_poisson(x, [0, 0, 1], weights=[1, 1, 0])
    with pytest.raises(ValueError, match="full-rank"):
        fit_poisson(x, [1, 2, 3], weights=[1, 0, 0])
