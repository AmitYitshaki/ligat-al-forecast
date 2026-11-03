"""EXP-001 orchestration, reusing the B2 judge without selecting a final model."""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.evaluate.backtest import EvaluationResult, Mode, evaluate
from ligat.evaluate.comparison import Comparison, compare
from ligat.features.elo import (
    EloParams,
    EloResult,
    PreparedElo,
    compute_prepared_elo,
    prepare_elo,
)
from ligat.models.ordered_logit import OrderedEloPredictor


@dataclass(frozen=True)
class CalibrationConfig:
    """One rating formula and one conversion time-weight rule."""

    elo: EloParams
    half_life: float | None = None

    def __post_init__(self) -> None:
        if self.half_life is not None:
            value = float(self.half_life)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Half-life must be positive or None")
            object.__setattr__(self, "half_life", value)

    @property
    def config_id(self) -> str:
        """Keep unweighted Elo IDs; distinguish every weighted conversion stably."""
        if self.half_life is None:
            return self.elo.config_id
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "elo-time-" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    """The grid and predeclared comparison rules, loaded before any scoring."""

    experiment_id: str
    grid: tuple[CalibrationConfig, ...]
    pilot: tuple[CalibrationConfig, ...]
    half_lives: tuple[float | None, ...]
    stage2_top_n: int
    leakage_threshold: float
    raw: dict[str, Any]
    sha256: str
    source_text: str


def _grid(settings: dict[str, Any]) -> tuple[CalibrationConfig, ...]:
    keys = list(EloParams.__dataclass_fields__)
    return tuple(
        CalibrationConfig(EloParams(**dict(zip(keys, values, strict=True))))
        for values in product(*(settings[key] for key in keys))
    )


def load_experiment(path: Path = Path("config/experiment.yaml")) -> ExperimentSpec:
    """Read the frozen grid, pilot corners and rules; never run anything on import."""
    content = path.read_bytes()
    raw = yaml.safe_load(content)
    grid = _grid(raw["elo"])
    pilot_settings = {
        key: value if isinstance(value, list) else [value] for key, value in raw["pilot"].items()
    }
    pilot = _grid(pilot_settings)
    if len({c.config_id for c in grid}) != len(grid) or not set(pilot) <= set(grid):
        raise ValueError("Duplicate grid configurations or pilot outside the declared grid")
    half_lives = tuple(
        None if value is None else float(value) for value in raw["stage2"]["half_lives"]
    )
    for half_life in half_lives:
        CalibrationConfig(EloParams(), half_life)
    threshold = float(raw["leakage_log_loss_threshold"])
    top_n = int(raw["stage2"]["top_n"])
    if not math.isfinite(threshold) or threshold <= 0 or top_n < 1:
        raise ValueError("Invalid leakage threshold or stage-two size")
    if raw["simplicity"] != {
        "additions": [
            "dead_game_discount",
            "use_goal_multiplier",
            "promoted_initialization",
            "time_weight",
        ],
        "tie_break": "log_loss",
        "approval_required": True,
    }:
        raise ValueError("Unsupported simplicity rule; review before changing methodology")
    refinement = raw["refinement"]
    if (
        refinement["step"] != 0.1
        or refinement["paired_se_multiplier"] != 1
        or refinement["approval_required"] is not True
        or refinement["reuse_references"] != ["coarse_winner", "undiscounted_control"]
        or set(refinement["discounts"]) != set(raw["elo"]["dead_game_discount"])
    ):
        raise ValueError("Unsupported refinement rule; review before changing methodology")
    return ExperimentSpec(
        str(raw["experiment_id"]),
        grid,
        pilot,
        half_lives,
        top_n,
        threshold,
        raw,
        hashlib.sha256(content).hexdigest(),
        content.decode("utf-8"),
    )


@dataclass(frozen=True)
class CalibrationInput:
    """Reusable immutable rating batches and the judge's one-row-per-game targets."""

    elo: PreparedElo
    games: pd.DataFrame


def prepare_calibration(tables: AnalysisTables, nothing_to_play: pd.Series) -> CalibrationInput:
    """Prepare all configurations once, excluding every unrequested EDA column."""
    prepared = prepare_elo(tables, nothing_to_play)
    games = pd.DataFrame(
        [game for season in prepared.seasons for day in season.days for game in day.games],
        columns=[
            "game_id",
            "season_id",
            "date",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "dead",
        ],
    ).drop(columns="dead")
    if set(games.season_id) != set(DEVELOPMENT_SEASONS):
        raise ValueError("Calibration requires all 18 development seasons, and no holdouts")
    return CalibrationInput(prepared, games)


@dataclass(frozen=True)
class CalibrationRun:
    """A configuration's evaluated outputs, fitted converters and replay endpoints."""

    config: CalibrationConfig
    evaluation: EvaluationResult
    fits: pd.DataFrame
    snapshots: pd.DataFrame
    seconds: float


@dataclass(frozen=True)
class CalibrationBatch:
    """Explicitly requested runs plus the judge's review flags; no approved selection."""

    runs: tuple[CalibrationRun, ...]
    comparison: Comparison


def _evaluate_ratings(
    prepared: CalibrationInput, ratings: EloResult, config: CalibrationConfig, mode: Mode
) -> tuple[EvaluationResult, pd.DataFrame]:
    if ratings.config_id != config.elo.config_id:
        raise ValueError("Ratings belong to another Elo configuration")
    home_index = pd.MultiIndex.from_frame(prepared.games[["game_id", "home_team_id"]])
    games = prepared.games.copy()
    games["elo_diff"] = ratings.prematch.elo_diff.reindex(home_index).to_numpy()
    models: list[OrderedEloPredictor] = []

    def factory() -> OrderedEloPredictor:
        model = OrderedEloPredictor(config.half_life)
        models.append(model)
        return model

    result = evaluate(factory, games, feature_columns=("elo_diff",), mode=mode)
    records = []
    for model, fold in zip(models, result.per_fold.itertuples(), strict=True):
        fit = model.fit_result
        assert fit is not None  # evaluate cannot predict with an unfitted converter.
        records.append(
            {
                "fold": fold.fold,
                "season_id": fold.season_id,
                "training_seasons": model.training_seasons,
                **asdict(fit),
                "theta2": fit.theta2,
            }
        )
    return result, pd.DataFrame(records)


def run_configuration(
    prepared: CalibrationInput,
    config: CalibrationConfig,
    *,
    leakage_threshold: float,
    mode: Mode = "walk_forward",
    ratings: EloResult | None = None,
) -> CalibrationRun:
    """Run one explicitly supplied configuration; fail before returning suspicious results."""
    if not math.isfinite(leakage_threshold) or leakage_threshold <= 0:
        raise ValueError("A positive finite leakage threshold is required")
    if config.half_life is not None and mode != "walk_forward":
        raise ValueError("Time-weighted fits require chronological folds")
    started = perf_counter()
    ratings = ratings or compute_prepared_elo(prepared.elo, config.elo)
    result, fits = _evaluate_ratings(prepared, ratings, config, mode)
    if mode == "walk_forward" and result.summary.log_loss < leakage_threshold:
        # Do not leak a score or configuration ranking in pilot exception messages.
        raise RuntimeError("Leakage threshold breached: stop the experiment and investigate")
    return CalibrationRun(config, result, fits, ratings.season_snapshots, perf_counter() - started)


def run_grid(
    prepared: CalibrationInput,
    configs: Sequence[CalibrationConfig],
    *,
    leakage_threshold: float,
    mode: Mode = "walk_forward",
) -> CalibrationBatch:
    """Evaluate only the supplied grid, serially, reusing Elo across time-weight variants."""
    if not configs or len({c.config_id for c in configs}) != len(configs):
        raise ValueError("Supply nonempty, distinct configurations explicitly")
    runs = []
    cached: EloResult | None = None
    for config in configs:
        if cached is not None and cached.config_id != config.elo.config_id:
            cached = None
        started = perf_counter()
        if cached is None:
            cached = compute_prepared_elo(prepared.elo, config.elo)
        run = run_configuration(
            prepared, config, leakage_threshold=leakage_threshold, mode=mode, ratings=cached
        )
        runs.append(replace(run, seconds=perf_counter() - started))
    comparison = compare(
        {r.config.config_id: r.evaluation for r in runs}, for_selection=mode == "walk_forward"
    )
    return CalibrationBatch(tuple(runs), comparison)


def stage2_configs(batch: CalibrationBatch, spec: ExperimentSpec) -> tuple[CalibrationConfig, ...]:
    """Build time-weight variants of the ten leaders; an exact cutoff tie needs a decision."""
    ranked = batch.comparison.rankings
    n = spec.stage2_top_n
    if len(ranked) < n or any(r.config.half_life is not None for r in batch.runs):
        raise ValueError("Stage two requires enough unweighted first-stage configurations")
    if len(ranked) > n and ranked.iloc[n - 1].log_loss == ranked.iloc[n].log_loss:
        raise ValueError("Exact tie at stage-two cutoff: Amit's decision is required")
    configs = {r.config.config_id: r.config for r in batch.runs}
    return tuple(
        replace(configs[name], half_life=h) for name in ranked.index[:n] for h in spec.half_lives
    )


def combine_stages(*batches: CalibrationBatch) -> CalibrationBatch:
    """Compare the full coarse grid and stage two together, retaining shared controls once."""
    runs: dict[str, CalibrationRun] = {}
    for batch in batches:
        for run in batch.runs:
            name = run.config.config_id
            if name in runs:
                if not runs[name].evaluation.per_game.equals(run.evaluation.per_game):
                    raise ValueError("Repeated configuration has different scores across stages")
            else:
                runs[name] = run
    comparison = compare({name: run.evaluation for name, run in runs.items()})
    return CalibrationBatch(tuple(runs.values()), comparison)


def hyperparameter_summary(batch: CalibrationBatch) -> pd.DataFrame:
    """Describe best and mean loss per value; these are not causal effects."""
    rows = []
    for run in batch.runs:
        rows.append(
            {
                **asdict(run.config.elo),
                "half_life": run.config.half_life,
                "log_loss": float(run.evaluation.summary.log_loss),
            }
        )
    frame = pd.DataFrame(rows)
    parts = []
    for parameter in (*EloParams.__dataclass_fields__, "half_life"):
        part = frame.groupby(parameter, dropna=False).log_loss.agg(["min", "mean", "count"])
        part = part.reset_index().rename(
            columns={
                parameter: "value",
                "min": "best_log_loss",
                "mean": "mean_log_loss",
                "count": "configs",
            }
        )
        part.insert(0, "parameter", parameter)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def structural_additions(config: CalibrationConfig) -> int:
    """Count only the four predeclared structural additions, never numeric K/HFA/regression."""
    return sum(
        (
            config.elo.dead_game_discount != 1,
            config.elo.use_goal_multiplier,
            config.elo.promoted_initialization == "previous_promoted_mean",
            config.half_life is not None,
        )
    )


def simplicity_proposals(batch: CalibrationBatch) -> tuple[str, ...]:
    """Propose the simplest close configurations, breaking ties by loss; never approve a choice."""
    ranked = batch.comparison.rankings
    if "within_one_se" not in ranked or ranked.within_one_se.isna().any():
        raise ValueError("An undetermined one-SE reference requires Amit's decision")
    configs = {r.config.config_id: r.config for r in batch.runs}
    close = ranked.loc[ranked.within_one_se].copy()
    close["additions"] = [structural_additions(configs[name]) for name in close.index]
    close = close.loc[close.additions.eq(close.additions.min())]
    return tuple(close.index[close.log_loss.eq(close.log_loss.min())])


def refinement_applicable(proposed: CalibrationRun, control: CalibrationRun) -> bool:
    """Require discount != 1 and a paired improvement > 1 SE over its identical d=1 control."""
    if proposed.config.elo.dead_game_discount == 1:
        return False
    expected = replace(proposed.config, elo=replace(proposed.config.elo, dead_game_discount=1))
    if control.config != expected:
        raise ValueError("Refinement control must differ only in discount=1")
    comparison = compare(
        {
            proposed.config.config_id: proposed.evaluation,
            control.config.config_id: control.evaluation,
        }
    )
    if comparison.all_winners != (proposed.config.config_id,):
        return False
    row = comparison.rankings.loc[control.config.config_id]
    return bool(row.paired_mean_difference > row.paired_se)


@dataclass(frozen=True)
class RefinementPlan:
    """A grid proposal only; existing coarse/reference runs must not be refitted."""

    candidates: tuple[CalibrationConfig, ...]
    to_run: tuple[CalibrationConfig, ...]
    references: tuple[CalibrationConfig, ...]


def build_refinement_grid(proposed: CalibrationConfig, spec: ExperimentSpec) -> RefinementPlan:
    """Build the approved tenths between coarse neighbors; never execute EXP-001b.

    The coarse winner and its identical discount=1 control always accompany an
    applicable comparison, from already-computed results, including time weights.
    """
    discounts = spec.raw["refinement"]["discounts"]
    current = proposed.elo.dead_game_discount
    if current not in discounts:
        raise ValueError("Refine only a winner from the coarse discount grid")
    if current == 1:
        return RefinementPlan((), (), ())
    references = (proposed, replace(proposed, elo=replace(proposed.elo, dead_game_discount=1)))
    candidates = tuple(
        replace(proposed, elo=replace(proposed.elo, dead_game_discount=float(d)))
        for d in discounts[current]
    )
    return RefinementPlan(
        candidates, tuple(c for c in candidates if c not in references), references
    )
