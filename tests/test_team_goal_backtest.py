"""Spy-model checks for both chronological M2 refit policies, coverage and row NLL."""

from dataclasses import replace
from math import factorial, log

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare


class MeanGoalSpy:
    """Test-only constant model: training mean, recording every fit and prediction input."""

    supports_sample_weight = True

    def __init__(self, refit_policy="per_fold"):
        self.refit_policy = refit_policy
        self.fit_calls, self.predict_calls = [], []
        self.weight_calls = []

    def fit(self, X, y, sample_weight=None, *, context):
        assert X.index.equals(y.index) and y.index.equals(context.index)
        self.fit_calls.append((X.copy(deep=True), y.copy(deep=True), context.copy(deep=True)))
        self.weight_calls.append(None if sample_weight is None else sample_weight.copy(deep=True))
        if sample_weight is not None:
            assert X.index.equals(sample_weight.index)
        self.rate = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X, *, context):
        self.predict_calls.append((X.copy(deep=True), context.copy(deep=True)))
        return pd.Series(self.rate, index=X.index, name="lambda")


def factory_with_history(policy):
    created = []

    def factory():
        model = MeanGoalSpy(policy)
        created.append(model)
        return model

    return factory, created


def test_blind_judge_never_calls_scores_or_attaches_validation_goals(goal_sources, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Blind judge called a scoring function")

    monkeypatch.setattr("ligat.evaluate.team_goal_backtest.poisson_nll", forbidden)
    data = prepare(goal_sources)
    result = evaluate_team_goals(MeanGoalSpy, data, feature_columns=("elo_diff",), blind=True)
    assert result.per_team["lambda"].gt(0).all()
    assert "goals_scored" not in result.per_team and "poisson_nll" not in result.per_team
    assert "poisson_nll" not in result.per_fold and "poisson_nll" not in result.summary
    assert result.summary["rows"] == len(result.per_team)


@pytest.mark.parametrize("refit_policy", ["per_fold", "per_matchday"])
def test_spy_proves_target_separation_chronology_pairs_and_provenance(goal_sources, refit_policy):
    """Every fresh fit sees exactly allowed history, and predict receives no current targets."""
    data = prepare(goal_sources, ("elo_diff", "goal_difference"))
    original = [f.copy(deep=True) for f in (data.X, data.y, data.context)]
    factory, models = factory_with_history(refit_policy)
    result = evaluate_team_goals(factory, data, feature_columns=("elo_diff",))
    assert result.refit_policy == refit_policy
    assert len(models) == (15 if refit_policy == "per_fold" else result.per_fold.fits.sum())
    for model in models:
        assert len(model.fit_calls) == len(model.predict_calls) == 1
        X, y, history = model.fit_calls[0]
        predicted, context = model.predict_calls[0]
        assert tuple(X.columns) == tuple(predicted.columns) == ("elo_diff",)
        assert not {"y", "target", "goals_scored", "home_goals", "away_goals"} & set(context)
        assert history.date.max() < context.as_of.min()
        assert not X.index.isin(predicted.index).any()
        season = int(context.season_id.iloc[0])
        if refit_policy == "per_fold":
            allowed = data.context.season_id.lt(season)
            assert set(history.season_id) == set(range(8, season))
        else:
            assert context.as_of.nunique() == 1
            allowed = data.context.date.lt(context.as_of.iloc[0])
        assert X.index.equals(data.X.index[allowed])
        assert_series_equal(y, data.y.loc[allowed])
        assert history.groupby(level="game_id").size().eq(2).all()
        assert context.groupby(level="game_id").size().eq(2).all()
        scored = result.per_team.loc[predicted.index]
        assert scored.train_max_date.eq(history.date.max()).all()
        assert scored.train_games.eq(len(history) // 2).all()
        assert set(scored.train_seasons) == {tuple(sorted(history.season_id.unique()))}
        assert scored["lambda"].eq(y.mean()).all()
    assert_frame_equal(data.X, original[0])
    assert_series_equal(data.y, original[1])
    assert_frame_equal(data.context, original[2])


@pytest.mark.parametrize("refit_policy", ["per_fold", "per_matchday"])
def test_coverage_fold_depth_and_manual_nll_are_exact(goal_sources, refit_policy):
    """Warmup is unpredicted; 15 folds and 14 deep flags retain two equal-weight rows per game."""
    data = prepare(goal_sources)
    result = evaluate_team_goals(
        lambda: MeanGoalSpy(refit_policy), data, feature_columns=("elo_diff",)
    )
    expected = data.y.index[data.context.season_id.ge(11)]
    assert result.per_team.index.equals(expected)
    assert result.per_team.index.is_unique
    assert result.per_team.groupby(level="game_id").size().eq(2).all()
    assert result.per_fold.season_id.tolist() == list(range(11, 26))
    assert result.per_fold.deep_fold.tolist() == [False] + [True] * 14
    assert not result.per_team.loc[result.per_team.season_id.eq(11), "deep_fold"].any()
    for row in result.per_team.itertuples():
        # lambda is a Python keyword, so use indexed access below instead.
        key = row.Index
        rate, y = result.per_team.loc[key, ["lambda", "goals_scored"]]
        assert result.per_team.loc[key, "poisson_nll"] == pytest.approx(
            rate - y * log(rate) + log(factorial(int(y))), abs=1e-14
        )
    for fold in result.per_fold.itertuples():
        scores = result.per_team.loc[result.per_team.season_id.eq(fold.season_id)]
        expected_nll = scores.groupby(level="game_id").poisson_nll.mean().mean()
        assert fold.poisson_nll == pytest.approx(expected_nll)
        assert fold.rows == 2 * fold.games
    assert result.summary.poisson_nll == pytest.approx(result.per_team.poisson_nll.mean())
    assert result.summary.poisson_nll == pytest.approx(
        np.average(result.per_fold.poisson_nll, weights=result.per_fold.games)
    )
    assert len(set(result.per_fold.games)) > 1


def test_same_day_results_are_invisible_but_earlier_validation_results_can_train(goal_sources):
    """Perturb one day's targets: neither same-day match changes; the next day does change."""
    data = prepare(goal_sources)
    changed_y = data.y.copy()
    changed_y.loc[(1100, 1)] += 30
    changed = replace(data, y=changed_y)
    for refit_policy in ("per_fold", "per_matchday"):
        a, b = [
            evaluate_team_goals(
                lambda p=refit_policy: MeanGoalSpy(p), source, feature_columns=("elo_diff",)
            )
            for source in (data, changed)
        ]
        first_day = a.per_team.loc[1100, "as_of"].iloc[0]
        same_day = a.per_team.season_id.eq(11) & a.per_team.as_of.eq(first_day)
        assert a.per_team.loc[same_day].index.get_level_values("game_id").nunique() == 2
        assert_series_equal(a.per_team.loc[same_day, "lambda"], b.per_team.loc[same_day, "lambda"])
        later = a.per_team.season_id.eq(11) & a.per_team.as_of.gt(first_day)
        assert later.any()
        if refit_policy == "per_fold":
            assert_series_equal(a.per_team.loc[later, "lambda"], b.per_team.loc[later, "lambda"])
        else:
            assert not a.per_team.loc[later, "lambda"].equals(b.per_team.loc[later, "lambda"])


def test_shuffled_sources_produce_identical_evaluation(goal_sources):
    a = prepare(goal_sources)
    b = prepare(tuple(f.sample(frac=1, random_state=12) for f in goal_sources))
    results = [
        evaluate_team_goals(MeanGoalSpy, source, feature_columns=("elo_diff",)) for source in (a, b)
    ]
    assert_frame_equal(results[0].per_team, results[1].per_team)
    assert_frame_equal(results[0].per_fold, results[1].per_fold)
    assert_series_equal(results[0].summary, results[1].summary)


def test_explicit_glm_training_start_records_actual_depth_without_dropping_validation(goal_sources):
    data = prepare(goal_sources)
    result = evaluate_team_goals(
        MeanGoalSpy, data, feature_columns=("elo_diff",), policy=DevelopmentPolicy(training_start=9)
    )
    assert result.per_fold.deep_fold.tolist() == [False, False] + [True] * 13
    assert result.per_team.season_id.unique().tolist() == list(range(11, 26))
    assert result.per_fold.train_seasons.iloc[1] == (9, 10, 11)


@pytest.mark.parametrize(
    "damage", ["zero", "negative", "nan", "infinite", "bool", "complex", "order", "keys", "array"]
)
def test_bad_prediction_values_or_index_fail_closed(goal_sources, damage):
    class Bad(MeanGoalSpy):
        def predict(self, X, *, context):
            rates = super().predict(X, context=context)
            if damage in ("zero", "negative", "nan", "infinite"):
                rates.iloc[0] = {"zero": 0, "negative": -1, "nan": np.nan, "infinite": np.inf}[
                    damage
                ]
            elif damage == "bool":
                rates = rates.astype(bool)
            elif damage == "complex":
                rates = rates.astype(complex) + 1j
            elif damage == "order":
                rates = rates.iloc[::-1]
            elif damage == "keys":
                rates = rates.reset_index(drop=True)
            else:
                return rates.to_numpy()
            return rates

    with pytest.raises(ValueError, match="positive finite rates|exact requested key order"):
        evaluate_team_goals(Bad, prepare(goal_sources), feature_columns=("elo_diff",))


def test_external_weights_and_excluded_seasons_are_rejected_before_fitting(goal_sources):
    data = prepare(goal_sources)

    def factory():
        pytest.fail("Premature model creation")

    with pytest.raises(ValueError, match="computed by the judge"):
        evaluate_team_goals(factory, data, feature_columns=("elo_diff",), sample_weight=data.y)
    with pytest.raises(ValueError, match="all preceding seasons"):
        evaluate_team_goals(
            factory,
            data,
            feature_columns=("elo_diff",),
            policy=DevelopmentPolicy(
                excluded_seasons=(11,), validation_seasons=tuple(range(12, 26))
            ),
        )


def test_missing_seasons_bad_policy_and_changed_refit_declaration_fail(goal_sources):
    data = prepare(goal_sources)
    included = data.context.season_id.ne(8)
    partial = GoalData(data.X.loc[included], data.y.loc[included], data.context.loc[included])
    with pytest.raises(ValueError, match="all development seasons"):
        evaluate_team_goals(MeanGoalSpy, partial, feature_columns=("elo_diff",))
    with pytest.raises(ValueError, match="must declare"):
        evaluate_team_goals(lambda: MeanGoalSpy("unknown"), data, feature_columns=("elo_diff",))
    models = iter([MeanGoalSpy("per_fold"), MeanGoalSpy("per_matchday")])
    with pytest.raises(ValueError, match="changed"):
        evaluate_team_goals(lambda: next(models), data, feature_columns=("elo_diff",))


def test_late_previous_season_fixture_is_rejected_per_fold_and_filtered_per_day(goal_sources):
    prematch, matches = [f.copy(deep=True) for f in goal_sources]
    same_day = matches.loc[matches.game_id.eq(1100), "date"].iloc[0].normalize()
    matches.loc[matches.game_id.eq(1000), "date"] = same_day
    prematch.loc[1000, "date"] = same_day
    prematch.loc[1000, "as_of"] = same_day
    with pytest.raises(ValueError, match="at or after"):
        evaluate_team_goals(
            MeanGoalSpy,
            prepare((prematch, matches)),
            feature_columns=("elo_diff",),
        )
    data = prepare((prematch, matches))
    factory, models = factory_with_history("per_matchday")
    evaluate_team_goals(factory, data, feature_columns=("elo_diff",))
    inspected = 0
    for model in models:
        _, _, history = model.fit_calls[0]
        _, context = model.predict_calls[0]
        if context.season_id.iloc[0] != 11:
            continue
        assert history.date.lt(context.as_of.iloc[0]).all()
        assert (1000 in history.game_id.to_numpy()) == (context.as_of.iloc[0] > same_day)
        inspected += 1
    assert inspected > 1


def test_judge_snapshots_are_isolated_from_model_mutation(goal_sources):
    """A model's changes to supplied X, y and context cannot affect scores or subsequent fits."""

    class Mutating(MeanGoalSpy):
        def fit(self, X, y, sample_weight=None, *, context):
            super().fit(X, y, sample_weight, context=context)
            X.iloc[:, :] = -999
            y.iloc[:] = 999
            context["date"] = context.date + pd.Timedelta(days=9999)
            return self

        def predict(self, X, *, context):
            rates = super().predict(X, context=context)
            X.iloc[:, :] = 999
            context["as_of"] = context.as_of - pd.Timedelta(days=9999)
            return rates

    data = prepare(goal_sources)
    expected = evaluate_team_goals(MeanGoalSpy, data, feature_columns=("elo_diff",))
    actual = evaluate_team_goals(Mutating, data, feature_columns=("elo_diff",))
    assert_frame_equal(actual.per_team, expected.per_team)
    assert_frame_equal(actual.per_fold, expected.per_fold)


def test_explicit_validation_subset_keeps_original_fold_ids_and_complete_training_history(
    goal_sources,
):
    data = prepare(goal_sources)
    result = evaluate_team_goals(
        MeanGoalSpy,
        data,
        feature_columns=("elo_diff",),
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(12, 18, 25)),
    )
    assert result.per_fold.season_id.tolist() == [12, 18, 25]
    assert result.per_fold.fold.tolist() == [2, 8, 15]
    assert result.per_fold.train_seasons.iloc[1] == tuple(range(8, 18))
    assert result.per_team.index.equals(data.X.index[data.context.season_id.isin([12, 18, 25])])
