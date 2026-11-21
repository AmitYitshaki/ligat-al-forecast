"""M2 chronological team-row judge; legacy experiment judges remain independent."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.backtest import DevelopmentPolicy, policy_folds
from ligat.evaluate.goal_inputs import GoalData, _feature_columns
from ligat.evaluate.goal_metrics import poisson_nll
from ligat.features.pipeline import DEVELOPMENT_SEASONS
from ligat.models.goal_model import RefitPolicy, TeamGoalModel
from ligat.models.ordered_logit import time_weights

MIN_TRAIN_SEASONS = 4


@dataclass(frozen=True)
class TeamGoalEvaluation:
    """OOF rates, row/fold Poisson NLL and the actual history behind every prediction."""

    per_team: pd.DataFrame
    per_fold: pd.DataFrame
    summary: pd.Series
    refit_policy: RefitPolicy


def _fold_plan(data: GoalData, policy: DevelopmentPolicy) -> pd.DataFrame:
    if policy.excluded_seasons:
        raise ValueError("M2 trains on all preceding seasons from its declared training start")
    if set(data.context.season_id) != set(DEVELOPMENT_SEASONS):
        raise ValueError("M2 fold evaluation requires all development seasons 8-25")
    home = data.context.loc[data.context.is_home, ["game_id", "season_id"]].copy()
    home["home_goals"] = data.y.loc[home.index].to_numpy()
    away = data.context.loc[~data.context.is_home, ["game_id"]].copy()
    away["away_goals"] = data.y.loc[away.index].to_numpy()
    away = away.set_index("game_id")
    home["away_goals"] = away.away_goals.reindex(home.game_id).to_numpy()
    # Reuse the approved fold builder with in-memory games: no repository read.
    return policy_folds(home.reset_index(drop=True), policy)


def _predict_batch(
    model: TeamGoalModel,
    data: GoalData,
    training: pd.Index,
    validation: pd.Index,
    *,
    fold: int,
    deep_fold: bool,
    refit_policy: RefitPolicy,
    validation_season: int,
    half_life: float | None,
) -> pd.DataFrame:
    train_context = data.context.loc[training]
    val_context = data.context.loc[validation]
    if train_context.empty or val_context.empty:
        raise ValueError("Goal training and prediction batches must be nonempty")
    maximum = train_context.date.max()
    if not maximum < val_context.as_of.min():
        raise ValueError("Goal training dates must be strictly earlier than prediction as_of")
    if training.isin(validation).any():
        raise ValueError("Goal training and prediction keys must be disjoint")
    weights = None
    if refit_policy == "per_fold":
        weights = pd.Series(
            time_weights(train_context.season_id, validation_season, half_life),
            index=training,
            name="sample_weight",
        )
        if getattr(model, "supports_sample_weight", False) is not True:
            raise ValueError("Goal model does not declare support for sample_weight")
    fitted = model.fit(
        data.X.loc[training].copy(deep=True),
        data.y.loc[training].copy(deep=True),
        sample_weight=weights,
        context=train_context.copy(deep=True),
    )
    if fitted is not model:
        raise ValueError("TeamGoalModel.fit must return the fitted model itself")
    rates = fitted.predict(
        data.X.loc[validation].copy(deep=True), context=val_context.copy(deep=True)
    )
    if (
        not isinstance(rates, pd.Series)
        or not rates.index.equals(validation)
        or not pd.api.types.is_numeric_dtype(rates.dtype)
        or pd.api.types.is_bool_dtype(rates.dtype)
        or pd.api.types.is_complex_dtype(rates.dtype)
    ):
        raise ValueError(
            "Goal predict must return a numeric Series in the exact requested key order"
        )
    values = rates.to_numpy(dtype=float, na_value=np.nan)
    scores = poisson_nll(data.y.loc[validation].to_numpy(float), values)
    result = val_context.drop(columns=["game_id", "team_id"]).copy()
    result["fold"] = fold
    result["goals_scored"] = data.y.loc[validation]
    result["lambda"] = values
    result["poisson_nll"] = scores
    seasons = tuple(sorted(int(s) for s in train_context.season_id.unique()))
    result["train_seasons"] = [seasons] * len(result)
    result["train_max_date"] = maximum
    result["train_games"] = len(training) // 2
    result["deep_fold"] = deep_fold
    result["refit_policy"] = refit_policy
    result["half_life"] = half_life
    return result


def evaluate_team_goals(
    model_factory: Callable[[], TeamGoalModel],
    data: GoalData,
    *,
    feature_columns: Sequence[str],
    policy: DevelopmentPolicy | None = None,
    half_life: float | None = None,
    sample_weight: pd.Series | None = None,
) -> TeamGoalEvaluation:
    """Predict seasons 11-25 from allowed history, fitting a fresh model per batch.

    Default training starts at 8; an explicit policy may start at 9 for the GLM
    or restrict the reported validation seasons. per_matchday additionally uses
    already played validation fixtures, strictly before the current local day.
    Fold depth uses the initial training seasons, never the partial validation
    season. For per_fold, the judge passes raw season weights (ones when
    half_life=None), without changing their scale. Daily models receive None
    weights and must implement their own declared day-decay in their adapter.
    External observation weights are forbidden: select half_life instead.
    """
    if sample_weight is not None:
        raise ValueError("sample_weight is computed by the judge; select half_life instead")
    if half_life is not None and (
        isinstance(half_life, (bool, np.bool_))
        or not isinstance(half_life, (int, float, np.integer, np.floating))
        or half_life not in (1, 2, 4)
    ):
        raise ValueError("Season half_life must be None, 1, 2 or 4")
    columns = _feature_columns(feature_columns, data.X.columns)
    # Validate independent snapshots even if a caller changed a mutable frame.
    data = GoalData(
        data.X.loc[:, columns].copy(deep=True), data.y.copy(deep=True), data.context.copy(deep=True)
    )
    plan = _fold_plan(data, policy or DevelopmentPolicy(training_start=8))
    parts, fold_records = [], []
    declared: RefitPolicy | None = None
    for row in plan.itertuples(index=False):
        model = model_factory()
        if model.refit_policy not in ("per_fold", "per_matchday"):
            raise ValueError("Goal model must declare per_fold or per_matchday refit policy")
        if declared is not None and model.refit_policy != declared:
            raise ValueError("Goal model factory changed its declared refit policy")
        declared = model.refit_policy
        if declared == "per_matchday" and half_life is not None:
            raise ValueError("per_matchday does not accept season half_life; use day decay in C4")
        train = data.context.season_id.isin(row.train_seasons)
        val = data.context.season_id.eq(row.validation_season)
        deep = len(row.train_seasons) >= MIN_TRAIN_SEASONS
        validation = data.context.loc[val]
        # A per-fold fit cannot silently drop a preceding season's late fixture.
        # The daily policy instead filters every fixture at its actual as_of.
        if declared == "per_fold" and (
            data.context.loc[train, "date"].max() >= validation.as_of.min()
        ):
            raise ValueError("Goal fold contains training dates at or after validation as_of")
        batches = (
            [(None, validation)]
            if declared == "per_fold"
            else list(validation.groupby("as_of", sort=True))
        )
        fold_parts = []
        for number, (day, batch) in enumerate(batches):
            if number:
                model = model_factory()
                if model.refit_policy != declared:
                    raise ValueError("Goal model factory changed its declared refit policy")
            allowed = (
                train
                if day is None
                else (
                    data.context.season_id.isin((*row.train_seasons, row.validation_season))
                    & data.context.date.lt(day)
                )
            )
            fold_parts.append(
                _predict_batch(
                    model,
                    data,
                    data.X.index[allowed],
                    batch.index,
                    fold=int(row.fold),
                    deep_fold=deep,
                    refit_policy=declared,
                    validation_season=int(row.validation_season),
                    half_life=half_life,
                )
            )
        scored = pd.concat(fold_parts)
        # Each game has two equally weighted team rows; averaging game means
        # explicitly gives the required game-weighted fold and overall NLL.
        game_scores = scored.groupby(level="game_id").poisson_nll.mean()
        fold_records.append(
            {
                "fold": int(row.fold),
                "season_id": int(row.validation_season),
                "train_seasons": tuple(row.train_seasons),
                "deep_fold": deep,
                "games": len(game_scores),
                "rows": len(scored),
                "poisson_nll": float(game_scores.mean()),
                "fits": len(batches),
                "half_life": half_life,
            }
        )
        parts.append(scored)
    per_team = pd.concat(parts).sort_index()
    expected = data.X.index[data.context.season_id.isin(plan.validation_season)]
    if not per_team.index.is_unique or not per_team.index.equals(expected.sort_values()):
        raise ValueError("Goal OOF coverage must contain each requested team-game key exactly once")
    per_fold = pd.DataFrame(fold_records)
    summary = pd.Series(
        {
            "games": int(per_fold.games.sum()),
            "rows": len(per_team),
            "folds": len(per_fold),
            "poisson_nll": float(np.average(per_fold.poisson_nll, weights=per_fold.games)),
        }
    )
    assert declared is not None
    return TeamGoalEvaluation(per_team, per_fold, summary, declared)
