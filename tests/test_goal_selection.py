"""Team-key and observed-goal checks before M2 fold comparisons."""

from dataclasses import replace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.evaluate.goal_selection import paired_goal_statistics
from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation, evaluate_team_goals
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_team_goal_backtest import MeanGoalSpy


def evaluation(improvement=0):
    rows = []
    for season in range(11, 26):
        for game in range(1 + season % 3):
            for team in (1, 2):
                rows.append(
                    {
                        "game_id": 100 * season + game,
                        "team_id": team,
                        "season_id": season,
                        "fold": season - 10,
                        "deep_fold": season >= 12,
                        "goals_scored": 3 if team == 1 else 1,
                        "poisson_nll": 2 - improvement,
                    }
                )
    team = pd.DataFrame(rows).set_index(["game_id", "team_id"])
    folds = (
        team.reset_index()
        .groupby(["fold", "season_id"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            deep_fold=("deep_fold", "first"),
            poisson_nll=("poisson_nll", "mean"),
        )
    )
    return TeamGoalEvaluation(team, folds, pd.Series(dtype=float), "per_fold")


def test_m2_pairing_checks_targets_and_is_independent_of_row_order():
    a, b = evaluation(0.1), evaluation()
    expected = paired_goal_statistics(a, b)
    shuffled = replace(
        a, per_team=a.per_team.sample(frac=1, random_state=14), per_fold=a.per_fold.iloc[::-1]
    )
    actual = paired_goal_statistics(shuffled, b)
    assert actual.mean_improvement == pytest.approx(0.1)
    assert_frame_equal(expected.differences, actual.differences)


def test_wrapper_accepts_actual_team_judge_output(goal_sources):
    result = evaluate_team_goals(MeanGoalSpy, prepare(goal_sources), feature_columns=("elo_diff",))
    stats = paired_goal_statistics(result, result)
    assert stats.n == 14 and stats.mean_improvement == 0 and stats.se == 0
    assert stats.last5_folds == 5 and stats.last5_folds_better == 0


@pytest.mark.parametrize(
    "damage",
    [
        "key",
        "goal",
        "duplicate",
        "missing",
        "fold",
        "depth",
        "fold_games",
        "fold_score",
        "negative_goal",
        "fractional_goal",
    ],
)
def test_m2_mismatched_keys_targets_and_inconsistent_fold_summaries_fail(damage):
    a, b = evaluation(0.1), evaluation()
    rows = a.per_team.copy()
    folds = a.per_fold.copy()
    if damage == "key":
        rows = rows.rename(index={1: 999}, level="team_id")
    elif damage == "goal":
        rows.iloc[0, rows.columns.get_loc("goals_scored")] += 1
    elif damage == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif damage == "missing":
        rows = rows.iloc[1:]
    elif damage == "fold":
        rows.iloc[0, rows.columns.get_loc("fold")] = 99
    elif damage == "depth":
        rows.iloc[0, rows.columns.get_loc("deep_fold")] = True
    elif damage == "fold_games":
        folds.loc[0, "games"] += 1
    elif damage == "fold_score":
        folds.loc[0, "poisson_nll"] += 0.1
    elif damage == "negative_goal":
        rows.iloc[0, rows.columns.get_loc("goals_scored")] = -1
    else:
        rows = rows.astype({"goals_scored": float})
        rows.iloc[0, rows.columns.get_loc("goals_scored")] = 0.5
    with pytest.raises(ValueError):
        paired_goal_statistics(replace(a, per_team=rows, per_fold=folds), b)
