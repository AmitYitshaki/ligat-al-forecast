"""A small, development-only judge for fitted H/D/A predictors."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS
from ligat.data.repository import Repository
from ligat.evaluate.metrics import log_loss, rps, validate_outcomes, validate_probabilities
from ligat.evaluate.splits import expanding_window_folds
from ligat.models.base import Predictor

Mode = Literal["walk_forward", "loso"]
RETROSPECTIVE_LABEL = "רטרוספקטיבי, לא לבחירה"
IDENTITY_COLUMNS = ("game_id", "season_id", "date", "home_team_id", "away_team_id")
# Predict gets an explicit projection, not every apparently harmless EDA column.
# Match-point targets are excluded too; future prematch point features need names
# distinguishing them from these target aliases.
TARGET_COLUMNS = frozenset(
    {
        "outcome",
        "result",
        "target",
        "y",
        "home_goals",
        "away_goals",
        "total_goals",
        "goal_difference",
        "goals_scored",
        "goals_conceded",
        "points",
        "home_points",
        "away_points",
        "ht_home_goals",
        "ht_away_goals",
        "ht_result",
    }
)


@dataclass(frozen=True)
class EvaluationResult:
    """One configuration's per-game scores, fold scores and aggregate diagnostics."""

    per_game: pd.DataFrame
    per_fold: pd.DataFrame
    summary: pd.Series
    mode: Mode
    label: str


class _CountsRepository(Repository):
    """Reuse the approved fold builder with already-loaded development data only."""

    def __init__(self, games: pd.DataFrame) -> None:
        self.games = games

    def read_filtered(self, name: str, filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
        if name != "matches" or filters != [("season_id", "in", list(DEVELOPMENT_SEASONS))]:
            raise ValueError("The judge's fold builder may request development matches only")
        return self.games[["season_id"]].copy()


def _prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.copy(deep=True)
    if "game_id" not in frame and frame.index.name == "game_id":
        frame = frame.reset_index()
    if not {"game_id", "season_id"} <= set(frame):
        raise ValueError("Games require game_id and season_id")
    if frame.empty or frame.game_id.isna().any() or frame.game_id.duplicated().any():
        raise ValueError("Expected one nonempty, unique row per game_id")
    if not frame.season_id.isin(DEVELOPMENT_SEASONS).all():
        raise ValueError("The judge only accepts development seasons 8–25")
    frame["season_id"] = frame.season_id.astype(int)
    has_goals = {"home_goals", "away_goals"} <= set(frame)
    if has_goals:
        goals = frame[["home_goals", "away_goals"]].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(goals).all() or (goals < 0).any() or (goals % 1 != 0).any():
            raise ValueError("Observed goals must be complete nonnegative integers")
        derived = np.select([goals[:, 0] > goals[:, 1], goals[:, 0] == goals[:, 1]], [0, 1], 2)
    if "outcome" in frame:
        targets = validate_outcomes(frame.outcome.to_numpy(), len(frame))
        if has_goals and not np.array_equal(targets, derived):
            raise ValueError("Outcome codes contradict the observed goals")
    elif has_goals:
        targets = derived
    else:
        raise ValueError("Games require outcome codes or both observed goal columns")
    frame["outcome"] = targets
    return frame.reset_index(drop=True)


def development_folds(games: pd.DataFrame) -> pd.DataFrame:
    """Build exactly the approved 15 folds, without reading holdout counts or data."""
    frame = _prepare_games(games)
    if set(frame.season_id.unique()) != set(DEVELOPMENT_SEASONS):
        raise ValueError("Walk-forward evaluation requires all 18 development seasons")
    return expanding_window_folds(repository=_CountsRepository(frame), include_holdout=False)


def loso_folds(games: pd.DataFrame) -> pd.DataFrame:
    """Retrospective stability plan: all other supplied development seasons train."""
    frame = _prepare_games(games)
    seasons = sorted(int(s) for s in frame.season_id.unique())
    if len(seasons) < 2:
        raise ValueError("LOSO requires at least two development seasons")
    return pd.DataFrame(
        [
            {
                "fold": f"loso_{season}",
                "train_seasons": tuple(s for s in seasons if s != season),
                "validation_season": season,
            }
            for season in seasons
        ]
    )


def _validate_folds(plan: pd.DataFrame, expected: pd.DataFrame) -> None:
    required = {"fold", "train_seasons", "validation_season"}
    if not required <= set(plan) or plan.fold.duplicated().any():
        raise ValueError("Invalid fold plan")
    for row in plan.itertuples(index=False):
        training = tuple(row.train_seasons)
        if (
            row.fold in ("final_validation", "test")
            or row.validation_season not in DEVELOPMENT_SEASONS
            or not set(training) <= set(DEVELOPMENT_SEASONS)
            or row.validation_season in training
        ):
            raise ValueError(
                "Final-validation/test folds and training-validation overlap are forbidden"
            )
    actual_rows = {
        (row.fold, tuple(row.train_seasons), row.validation_season) for row in plan.itertuples()
    }
    expected_rows = {
        (row.fold, tuple(row.train_seasons), row.validation_season) for row in expected.itertuples()
    }
    if actual_rows != expected_rows or len(plan) != len(expected):
        raise ValueError("Fold plan differs from the approved evaluation mode")


def summarize(per_game: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Weight overall/last-five means by game count; fold dispersion is ddof=1."""
    if per_game.empty or not per_game.season_id.isin(DEVELOPMENT_SEASONS).all():
        raise ValueError("Scores require nonempty development games")
    if per_game.duplicated(["fold", "game_id"]).any():
        raise ValueError("A game may occur only once in a fold")
    scores = per_game[["log_loss", "rps"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    folds = (
        per_game.groupby(["fold", "season_id"], sort=False)
        .agg(games=("game_id", "size"), log_loss=("log_loss", "mean"), rps=("rps", "mean"))
        .reset_index()
        .sort_values("season_id")
        .reset_index(drop=True)
    )
    if folds.fold.duplicated().any() or folds.season_id.duplicated().any():
        raise ValueError("Each validation season must correspond to one fold")
    last = folds.tail(5)
    summary: dict[str, float | int] = {
        "games": len(per_game),
        "folds": len(folds),
        "last5_games": int(last.games.sum()),
    }
    for metric in ("log_loss", "rps"):
        summary[metric] = float(np.average(folds[metric], weights=folds.games))
        summary[f"{metric}_std"] = float(folds[metric].std(ddof=1))
        summary[f"{metric}_min"] = float(folds[metric].min())
        summary[f"{metric}_max"] = float(folds[metric].max())
        summary[f"last5_{metric}"] = float(np.average(last[metric], weights=last.games))
    return folds, pd.Series(summary)


def evaluate(
    predictor_factory: Callable[[], Predictor],
    games: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = (),
    folds: pd.DataFrame | None = None,
    mode: Mode = "walk_forward",
) -> EvaluationResult:
    """Fit fresh predictors per fold and score only after target-free prediction.

    Validation inputs contain identity columns and explicitly requested prematch
    features only. Training additionally receives outcome codes and, if present,
    observed goals. Inputs are copied and no repository I/O occurs here.
    """
    if mode not in ("walk_forward", "loso"):
        raise ValueError("Unknown evaluation mode")
    frame = _prepare_games(games)
    expected = development_folds(frame) if mode == "walk_forward" else loso_folds(frame)
    plan = expected if folds is None else folds
    _validate_folds(plan, expected)
    features = list(feature_columns)
    if set(features) & TARGET_COLUMNS or len(set(features)) != len(features):
        raise ValueError("Predict features must not include result/goal/point target columns")
    if not set(features) <= set(frame):
        raise ValueError("Requested prematch feature columns are missing")
    prediction_columns = list(dict.fromkeys([c for c in IDENTITY_COLUMNS if c in frame] + features))
    training_columns = [*prediction_columns, "outcome"]
    training_columns += [c for c in ("home_goals", "away_goals") if c in frame]
    parts = []
    for row in plan.sort_values("validation_season").itertuples(index=False):
        training = frame.loc[frame.season_id.isin(row.train_seasons), training_columns].copy()
        validation = frame.loc[frame.season_id.eq(row.validation_season)].copy()
        if training.empty or validation.empty:
            raise ValueError("Training and validation folds must contain games")
        predictor = predictor_factory()
        predictor.fit(training)
        p = validate_probabilities(
            predictor.predict(validation[prediction_columns].copy(deep=True)), len(validation)
        )
        y = validation.outcome.to_numpy()
        scored = validation[["game_id", "season_id"]].copy()
        scored["fold"] = row.fold
        scored[["p_home", "p_draw", "p_away"]] = p
        scored["outcome"] = y
        scored["log_loss"] = log_loss(y, p)
        scored["rps"] = rps(y, p)
        parts.append(scored)
    per_game = pd.concat(parts, ignore_index=True)
    per_fold, summary = summarize(per_game)
    return EvaluationResult(
        per_game,
        per_fold,
        summary,
        mode,
        "כרונולוגי: 15 קיפולי פיתוח" if mode == "walk_forward" else RETROSPECTIVE_LABEL,
    )
