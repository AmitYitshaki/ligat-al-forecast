"""M2 pairing guard before invoking the shared, metric-independent selection core."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.selection import PairedStatistics, _numeric, _prepare_folds, paired_statistics
from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation


def _team_rows(result: TeamGoalEvaluation, score: str) -> pd.DataFrame:
    rows = result.per_team
    if (
        not isinstance(rows.index, pd.MultiIndex)
        or rows.index.names != ["game_id", "team_id"]
        or not rows.index.is_unique
        or rows.index.to_frame(index=False).isna().any().any()
        or not rows.columns.is_unique
        or rows.empty
        or not {"goals_scored", "fold", "season_id", "deep_fold", score} <= set(rows)
    ):
        raise ValueError(
            "M2 pairing requires unique complete team-game keys, targets and row scores"
        )
    rows = rows.sort_index().copy(deep=True)
    _numeric(rows.goals_scored, "goals_scored")
    y = rows.goals_scored.to_numpy(float)
    if (y < 0).any() or (y % 1 != 0).any():
        raise ValueError("M2 goals_scored must be nonnegative integers")
    if not rows.groupby(level="game_id").size().eq(2).all():
        raise ValueError("M2 pairing requires two team rows per game")
    return rows


def _check_fold_scores(result: TeamGoalEvaluation, rows: pd.DataFrame, score: str) -> None:
    _numeric(rows[score], score)
    records = []
    for (fold, season), group in rows.groupby(["fold", "season_id"], sort=False):
        if group.deep_fold.isna().any() or group.deep_fold.nunique() != 1:
            raise ValueError("M2 rows disagree on their fold depth")
        by_game = group.groupby(level="game_id")[score].mean()
        records.append(
            {
                "fold": fold,
                "season_id": season,
                "games": len(by_game),
                "deep_fold": group.deep_fold.iloc[0],
                score: float(by_game.mean()),
            }
        )
    # The same fixture cannot be split between different folds/seasons.
    if not rows.groupby(level="game_id")[["fold", "season_id"]].nunique().eq(1).all().all():
        raise ValueError("M2 paired rows must share their fold and validation season")
    expected = _prepare_folds(pd.DataFrame(records), score)
    actual = _prepare_folds(result.per_fold, score)
    if len(actual) != len(expected) or any(
        not np.array_equal(actual[c].to_numpy(), expected[c].to_numpy())
        for c in ("fold", "season_id", "games", "deep_fold")
    ):
        raise ValueError("M2 fold population differs from its team rows")
    if not np.allclose(actual[score], expected[score], rtol=0, atol=1e-12):
        raise ValueError("M2 fold scores differ from their team-row scores")


def paired_goal_statistics(
    candidate: TeamGoalEvaluation,
    reference: TeamGoalEvaluation,
    *,
    score: str = "poisson_nll",
) -> PairedStatistics:
    """Require identical team keys, observed goals and fold assignment before scoring.

    Also check that fold populations and scores summarize their supplied rows.
    The 1e-12 aggregation check tolerates arithmetic roundoff only; the shared
    calculation uses the original fold scores without changing their values.
    """
    a, b = (_team_rows(result, score) for result in (candidate, reference))
    if not a.index.equals(b.index) or any(
        not np.array_equal(a[c].to_numpy(), b[c].to_numpy())
        for c in ("goals_scored", "fold", "season_id", "deep_fold")
    ):
        raise ValueError(
            "M2 comparisons require identical team-game keys, goals and fold assignments"
        )
    for result, rows in ((candidate, a), (reference, b)):
        _check_fold_scores(result, rows, score)
    return paired_statistics(candidate.per_fold, reference.per_fold, score=score)
