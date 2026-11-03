"""Development-only goal judge: targets are separate from every prediction input."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ligat.evaluate.backtest import (
    IDENTITY_COLUMNS,
    TARGET_COLUMNS,
    _prepare_games,
    _validate_folds,
    development_folds,
    summarize,
)
from ligat.evaluate.goal_metrics import poisson_nll, poisson_outcomes
from ligat.evaluate.metrics import log_loss, rps


class GoalPredictor(Protocol):
    """Receive explicit features and separate observed targets; predict n×2 rates."""

    def fit(self, features: pd.DataFrame, goals: pd.DataFrame) -> None: ...

    def predict_goals(self, features: pd.DataFrame) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class GoalEvaluationResult:
    """Paired per-game scores for comparison, plus actual team-row Poisson scores."""

    per_game: pd.DataFrame
    per_fold: pd.DataFrame
    summary: pd.Series
    per_team: pd.DataFrame
    mode: str = "walk_forward"
    label: str = "Chronological goal judge: 15 development folds"


def summarize_goals(per_game: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Weight NLL by team rows (two per game); retain H/D/A diagnostics unchanged."""
    folds, summary = summarize(per_game)
    if not np.isfinite(per_game.poisson_nll.to_numpy(float)).all():
        raise ValueError("Poisson NLL must be finite")
    nll = per_game.groupby("season_id").poisson_nll.mean()
    folds["rows"] = 2 * folds.games
    folds["poisson_nll"] = folds.season_id.map(nll)
    summary["rows"] = 2 * int(summary.games)
    for suffix, value in {
        "": np.average(folds.poisson_nll, weights=folds.rows),
        "_std": folds.poisson_nll.std(ddof=1),
        "_min": folds.poisson_nll.min(),
        "_max": folds.poisson_nll.max(),
    }.items():
        summary[f"poisson_nll{suffix}"] = float(value)
    last = folds.tail(5)
    summary["last5_poisson_nll"] = float(np.average(last.poisson_nll, weights=last.rows))
    return folds, summary


def evaluate_goals(
    predictor_factory: Callable[[], GoalPredictor],
    games: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    folds: pd.DataFrame | None = None,
    max_goals: int = 10,
) -> GoalEvaluationResult:
    """Fit on past seasons; send targets only to fitting and subsequent scoring."""
    frame = _prepare_games(games)
    if not {"home_goals", "away_goals"} <= set(frame):
        raise ValueError("Goal evaluation requires both goal targets")
    expected = development_folds(frame)
    plan = expected if folds is None else folds
    _validate_folds(plan, expected)
    features = list(feature_columns)
    if set(features) & TARGET_COLUMNS or len(set(features)) != len(features):
        raise ValueError("Predict features must not include target columns")
    if not set(features) <= set(frame):
        raise ValueError("Requested goal features are missing")
    projection = list(dict.fromkeys([c for c in IDENTITY_COLUMNS if c in frame] + features))
    parts, team_parts = [], []
    for fold in plan.sort_values("validation_season").itertuples(index=False):
        train = frame.loc[frame.season_id.isin(fold.train_seasons)]
        val = frame.loc[frame.season_id.eq(fold.validation_season)]
        if train.empty or val.empty:
            raise ValueError("Empty goal fold")
        predictor = predictor_factory()
        predictor.fit(train[projection].copy(), train[["home_goals", "away_goals"]].copy())
        rates = np.asarray(predictor.predict_goals(val[projection].copy(deep=True)), float)
        if rates.shape != (len(val), 2):
            raise ValueError("Expected one home/away rate pair per game")
        probabilities, omitted = poisson_outcomes(rates, max_goals=max_goals)
        y = val[["home_goals", "away_goals"]].to_numpy(float)
        nll = poisson_nll(y.ravel(), rates.ravel()).reshape(-1, 2)
        scored = val[["game_id", "season_id", "home_goals", "away_goals", "outcome"]].copy()
        scored["fold"] = fold.fold
        scored[["lambda_home", "lambda_away"]] = rates
        scored[["p_home", "p_draw", "p_away"]] = probabilities
        scored["poisson_nll"] = nll.mean(axis=1)
        scored["log_loss"] = log_loss(val.outcome, probabilities)
        scored["rps"] = rps(val.outcome, probabilities)
        scored["omitted_mass"] = omitted
        parts.append(scored)
        team = pd.DataFrame(
            {
                "game_id": np.repeat(val.game_id.to_numpy(), 2),
                "season_id": np.repeat(val.season_id.to_numpy(), 2),
                "fold": fold.fold,
                "is_home": np.tile([True, False], len(val)),
                "goals_scored": y.ravel(),
                "lambda": rates.ravel(),
                "poisson_nll": nll.ravel(),
            }
        )
        if {"home_team_id", "away_team_id"} <= set(val):
            team["team_id"] = val[["home_team_id", "away_team_id"]].to_numpy().ravel()
        team_parts.append(team)
    per_game = pd.concat(parts, ignore_index=True)
    per_fold, summary = summarize_goals(per_game)
    return GoalEvaluationResult(
        per_game, per_fold, summary, pd.concat(team_parts, ignore_index=True)
    )
