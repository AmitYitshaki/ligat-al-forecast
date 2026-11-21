"""Hand-counted row metrics, scope alignment and calibration quantile groups."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ligat.evaluate.goal_diagnostics import goal_outcome_metrics
from ligat.evaluate.goal_metrics import poisson_nll
from ligat.evaluate.goal_reporting import calibration_deciles, readable_goal_metrics
from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation, evaluate_team_goals
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_team_goal_backtest import MeanGoalSpy


def evaluation(rates, *, goals=None, seasons=None):
    n = len(rates)
    assert n and n % 2 == 0
    seasons = [21] * (n // 2) if seasons is None else list(seasons)
    rows = pd.DataFrame(
        {
            "game_id": np.repeat(np.arange(n // 2) + 100, 2),
            "team_id": np.tile([20, 10], n // 2),
            "is_home": np.tile([True, False], n // 2),
            "season_id": np.repeat(seasons, 2),
            "lambda": rates,
            "goals_scored": np.zeros(n, dtype=int) if goals is None else goals,
        }
    ).set_index(["game_id", "team_id"])
    rows["fold"] = rows.season_id - 10
    rows["deep_fold"] = rows.season_id.ge(12)
    rows["poisson_nll"] = poisson_nll(rows.goals_scored, rows["lambda"])
    folds = (
        rows.reset_index()
        .groupby(["fold", "season_id"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            deep_fold=("deep_fold", "first"),
            poisson_nll=("poisson_nll", "mean"),
        )
    )
    return TeamGoalEvaluation(rows, folds, pd.Series(dtype=float), "per_fold")


def test_probability_translation_and_exact_row_rates_in_all_three_scopes():
    seasons = [11, 12, 21, 25]
    a = evaluation([1, 2, 3, 4, 5, 6, 7, 8], seasons=seasons)
    b = evaluation([2, 1, 3, 5, 4, 6, 8, 7], seasons=seasons)
    result = readable_goal_metrics(
        a, b, r0=evaluation([10] * 8, seasons=seasons), r1=evaluation([5] * 8, seasons=seasons)
    )
    for scope, count, loss in [("all", 8, 4.5), ("deep", 6, 5.5), ("last5", 4, 6.5)]:
        row = result.summary.loc[scope]
        values = np.arange(1, 9)[{"all": 0, "deep": 2, "last5": 4}[scope] :]
        assert row.rows == count
        assert row.poisson_nll == loss  # y=0 => NLL=lambda, without rounding or fitting.
        assert row.observed_probability_percent == pytest.approx(100 * np.exp(-loss), abs=1e-12)
        assert row.elo_share_percent == pytest.approx(100 * (10 - loss) / 5)
        assert row.elo_share_defined
        assert row.mae == values.mean()
        assert row.rmse == pytest.approx(np.sqrt(np.mean(values**2)))
    row = result.summary.loc["all"]
    assert (row.better_rows, row.worse_rows, row.tie_rows) == (3, 3, 2)
    assert (row.better_percent, row.worse_percent, row.tie_percent) == (37.5, 37.5, 25)
    assert row.better_non_tie_percent == 50
    assert result.summary.loc["deep", ["better_rows", "worse_rows", "tie_rows"]].tolist() == [
        2,
        2,
        2,
    ]
    assert result.summary.loc["last5", ["better_rows", "worse_rows", "tie_rows"]].tolist() == [
        1,
        2,
        1,
    ]
    seasonal = result.by_season.set_index("season_id")
    assert seasonal.better_rows.to_dict() == {11: 1, 12: 1, 21: 0, 25: 1}
    assert seasonal.worse_rows.to_dict() == {11: 1, 12: 0, 21: 1, 25: 1}
    assert seasonal.tie_rows.to_dict() == {11: 0, 12: 1, 21: 1, 25: 0}


@pytest.mark.parametrize(("rate", "expected"), [(4, 0), (2, 100), (3, 50)])
def test_elo_contribution_has_zero_hundred_and_intermediate_values(rate, expected):
    a = evaluation([rate] * 4)
    result = readable_goal_metrics(a, a, r0=evaluation([4] * 4), r1=evaluation([2] * 4))
    assert (result.summary.elo_share_percent == expected).all()
    assert result.summary.elo_share_defined.all()


@pytest.mark.parametrize("r1_rate", [4, 5])
def test_elo_nonpositive_denominator_is_explicitly_undefined(r1_rate):
    a = evaluation([3] * 4)
    result = readable_goal_metrics(a, a, r0=evaluation([4] * 4), r1=evaluation([r1_rate] * 4))
    assert result.summary.elo_share_percent.isna().all()
    assert not result.summary.elo_share_defined.any()


def test_identical_models_report_all_ties_and_no_non_tie_percentage():
    a = evaluation([1, 2, 3, 4])
    result = readable_goal_metrics(a, a, r0=a, r1=a)
    assert result.summary.better_percent.eq(0).all()
    assert result.summary.worse_percent.eq(0).all()
    assert result.summary.tie_percent.eq(100).all()
    assert result.summary.better_non_tie_percent.isna().all()


def test_row_ties_are_exact_and_not_an_isclose_tolerance():
    a = evaluation([1, 1])
    b = evaluation([1 + 1e-13, 1 - 1e-13])
    row = readable_goal_metrics(a, b, r0=a, r1=b).summary.loc["all"]
    assert row.better_rows == row.worse_rows == 1
    assert row.tie_rows == 0


def test_rmse_pools_rows_instead_of_averaging_fold_rmse():
    a = evaluation([1, 1] + [3] * 6, seasons=[21, 22, 22, 22])
    row = readable_goal_metrics(a, a, r0=a, r1=a).summary.loc["all"]
    assert row.mae == 2.5
    assert row.rmse == pytest.approx(np.sqrt(7))
    assert row.rmse != pytest.approx((1 + 3) / 2)


def test_scopes_apply_the_candidate_population_to_baselines_with_different_depths():
    a = evaluation([1, 1, 3, 3], seasons=[11, 12])
    zero, one = evaluation([4] * 4, seasons=[11, 12]), evaluation([2] * 4, seasons=[11, 12])
    zero = replace(
        zero,
        per_team=zero.per_team.assign(deep_fold=True),
        per_fold=zero.per_fold.assign(deep_fold=True),
    )
    result = readable_goal_metrics(a, a, r0=zero, r1=one)
    row = result.summary.loc["deep"]
    assert row.rows == 2 and row.elo_share_percent == 50
    empty = result.summary.loc["last5"]
    assert empty.rows == 0 and pd.isna(empty.rmse) and not empty.elo_share_defined


def test_readable_metrics_align_by_keys_and_never_modify_evaluations():
    a = evaluation([1, 2, 3, 4])
    original = a.per_team.copy(deep=True)
    shuffled = replace(a, per_team=a.per_team.sample(frac=1, random_state=4))
    expected = readable_goal_metrics(a, a, r0=a, r1=a)
    actual = readable_goal_metrics(shuffled, a, r0=shuffled, r1=a)
    pd.testing.assert_frame_equal(actual.summary, expected.summary)
    pd.testing.assert_frame_equal(actual.by_season, expected.by_season)
    pd.testing.assert_frame_equal(a.per_team, original)


@pytest.mark.parametrize("damage", ["keys", "goals", "side", "fold"])
def test_readable_metrics_reject_misaligned_reference_and_baseline(damage):
    a = evaluation([1, 2, 3, 4])
    b = evaluation([1, 2, 3, 4], goals=[1] * 4) if damage == "goals" else a
    if damage == "keys":
        b = replace(b, per_team=b.per_team.rename(index={20: 30}, level="team_id"))
    elif damage == "side":
        b = replace(b, per_team=b.per_team.assign(is_home=~b.per_team.is_home))
    elif damage == "fold":
        b = replace(b, per_team=b.per_team.assign(fold=99), per_fold=b.per_fold.assign(fold=99))
    for supplied in ((a, b, a, a), (a, a, b, a), (a, a, a, b)):
        with pytest.raises(ValueError, match="identical"):
            readable_goal_metrics(supplied[0], supplied[1], r0=supplied[2], r1=supplied[3])


def test_calibration_quantile_groups_never_split_ties_and_conserve_all_rows():
    rates = [0.5] * 8 + [1.0] * 14 + [2.0] * 10 + [3.0] * 8
    goals = [0, 1] * 20
    a = evaluation(rates, goals=goals, seasons=[12] * 10 + [21] * 10)
    result = calibration_deciles(a)
    for scope, count in [("all", 40), ("last5", 20)]:
        bins = result.loc[result.scope.eq(scope)]
        assert 1 <= len(bins) <= 10
        assert bins.n.sum() == count
        source = a.per_team if scope == "all" else a.per_team.loc[a.per_team.season_id.eq(21)]
        for value in source["lambda"].unique():
            matches = bins.loc[bins.lambda_min.le(value) & bins.lambda_max.ge(value)]
            assert len(matches) == 1
        for row in bins.itertuples():
            group = source.loc[source["lambda"].between(row.lambda_min, row.lambda_max)]
            assert len(group) == row.n
            assert row.lambda_mean == pytest.approx(group["lambda"].mean())
            assert row.goals_mean == pytest.approx(group.goals_scored.mean())


def test_calibration_constant_lambda_forms_one_group_per_nonempty_scope():
    a = evaluation([1.5] * 6, goals=[0, 1, 2, 3, 4, 5])
    result = calibration_deciles(a)
    assert result.scope.tolist() == ["all", "last5"]
    assert result.n.tolist() == [6, 6]
    assert result.lambda_mean.tolist() == [1.5, 1.5]
    assert result.goals_mean.tolist() == [2.5, 2.5]
    early = evaluation([1.5] * 2, seasons=[11])
    assert calibration_deciles(early).scope.tolist() == ["all"]


def test_calibration_uses_quantile_boundaries_on_distinct_lambdas():
    result = calibration_deciles(evaluation(np.arange(1, 21)))
    for scope in ("all", "last5"):
        groups = result.loc[result.scope.eq(scope)]
        assert groups.bin.tolist() == list(range(1, 11))
        assert groups.n.tolist() == [2] * 10
        assert groups.lambda_mean.tolist() == list(np.arange(1.5, 20, 2))


def test_reports_accept_actual_judge_output_without_training_baselines(goal_sources):
    a = evaluate_team_goals(MeanGoalSpy, prepare(goal_sources), feature_columns=("elo_diff",))
    result = readable_goal_metrics(a, a, r0=a, r1=a)
    assert result.summary.loc["all", "rows"] == len(a.per_team)
    assert result.summary.loc["all", "poisson_nll"] == pytest.approx(a.summary.poisson_nll)
    assert result.summary.loc["all", "tie_percent"] == 100
    assert calibration_deciles(a).query('scope == "all"').n.sum() == len(a.per_team)
    diagnostics = goal_outcome_metrics(a)
    assert len(diagnostics.per_game) * 2 == len(a.per_team)
    assert diagnostics.per_fold.games.tolist() == a.per_fold.games.tolist()
