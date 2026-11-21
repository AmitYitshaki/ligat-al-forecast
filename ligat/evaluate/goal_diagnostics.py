"""Independent 6x6 Poisson diagnostics for M2; legacy conversion stays unchanged."""

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from scipy.stats import poisson  # type: ignore[import-untyped]

from ligat.evaluate.goal_reporting import _goal_rows, _scopes
from ligat.evaluate.metrics import log_loss, rps, validate_probabilities
from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation


@dataclass(frozen=True)
class GoalOutcomeMetrics:
    """Paired game probabilities, rule-compatible fold scores and three scope means."""

    per_game: pd.DataFrame
    per_fold: pd.DataFrame
    summary: pd.DataFrame


def poisson_score_matrix(lambda_home: ArrayLike, lambda_away: ArrayLike) -> NDArray[np.float64]:
    """Return n x 6 x 6, home on rows and away on columns, with complete 5+ tails.

    Inputs are aligned, nonempty one-dimensional nonnegative finite rate arrays.
    Each marginal has Poisson masses 0..4 and sf(4), numerically stable 1-F(4).
    No goal truncation, clipping or probability renormalization is performed.
    """
    home, away = (np.asarray(values, dtype=float) for values in (lambda_home, lambda_away))
    if (
        home.ndim != 1
        or away.shape != home.shape
        or not len(home)
        or not np.isfinite(home).all()
        or not np.isfinite(away).all()
        or (home < 0).any()
        or (away < 0).any()
    ):
        raise ValueError("Expected aligned nonempty finite nonnegative home/away rate arrays")
    marginals = []
    for rates in (home, away):
        masses = poisson.pmf(np.arange(5)[None, :], rates[:, None])
        marginals.append(np.column_stack((masses, poisson.sf(4, rates))))
    return np.asarray(marginals[0][:, :, None] * marginals[1][:, None, :], dtype=float)


def outcome_probabilities(matrix: ArrayLike) -> NDArray[np.float64]:
    """H/D/A probabilities; the full diagonal, including 5+:5+, counts as a draw."""
    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 3
        or values.shape[1:] != (6, 6)
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=(1, 2)), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Expected nonnegative finite n x 6 x 6 matrices with unit mass")
    return validate_probabilities(
        np.column_stack(
            (
                np.tril(values, -1).sum(axis=(1, 2)),
                values.trace(axis1=1, axis2=2),
                np.triu(values, 1).sum(axis=(1, 2)),
            )
        )
    )


def goal_outcome_metrics(evaluation: TeamGoalEvaluation) -> GoalOutcomeMetrics:
    """Join home/away by is_home and game_id, then score actual H/D/A per game.

    Fold log loss and RPS include fold, season_id, games and deep_fold so B2a/B2b
    consume them directly. Scope means pool games, including shallow folds in
    all/last5. This pure reporting function returns diagnostics for subsequent
    leakage/rule-10 review; it neither trains a model nor makes a selection.
    """
    rows = _goal_rows(evaluation).reset_index()
    home, away = (rows.loc[rows.is_home.eq(side)].set_index("game_id") for side in (True, False))
    away = away.reindex(home.index)
    games = home.loc[:, ["fold", "season_id", "deep_fold"]].copy()
    for side, frame in (("home", home), ("away", away)):
        games[f"{side}_team_id"] = frame.team_id
        games[f"lambda_{side}"] = frame["lambda"]
        games[f"{side}_goals"] = frame.goals_scored
    p = outcome_probabilities(poisson_score_matrix(games.lambda_home, games.lambda_away))
    games[["p_home", "p_draw", "p_away"]] = p
    games["outcome"] = np.where(
        games.home_goals.gt(games.away_goals),
        0,
        np.where(games.home_goals.eq(games.away_goals), 1, 2),
    )
    games["log_loss"] = log_loss(games.outcome, p)
    games["rps"] = rps(games.outcome, p)
    per_fold = (
        games.reset_index()
        .groupby(["fold", "season_id"], as_index=False, sort=True)
        .agg(
            games=("game_id", "nunique"),
            deep_fold=("deep_fold", "first"),
            log_loss=("log_loss", "mean"),
            rps=("rps", "mean"),
        )
    )
    records = []
    for scope, mask in _scopes(games).items():
        group = games.loc[mask]
        records.append(
            {
                "scope": scope,
                "games": len(group),
                "log_loss": float(group.log_loss.mean()) if len(group) else None,
                "rps": float(group.rps.mean()) if len(group) else None,
            }
        )
    return GoalOutcomeMetrics(
        games.sort_index(), per_fold, pd.DataFrame(records).set_index("scope")
    )
