"""EXP-002 configuration, frozen inputs and explicitly requested fold runs."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.evaluate.backtest import EvaluationResult, evaluate
from ligat.evaluate.comparison import Comparison, compare
from ligat.evaluate.goal_backtest import GoalEvaluationResult, evaluate_goals
from ligat.features.form_artifacts import E0050_PARAMS
from ligat.features.registry import FORM_CANDIDATES, FORM_COLUMNS, validate_feature_columns
from ligat.features.table_state import game_rows
from ligat.models.form_models import FormOrderedPredictor, FormPoissonPredictor

Route = Literal["P", "O"]


@dataclass(frozen=True)
class FormCandidate:
    """One window rule and home/away aggregation, shared by attack and defense."""

    candidate: str
    variant: str

    def __post_init__(self) -> None:
        if self.candidate not in FORM_CANDIDATES or self.variant not in ("all", "venue"):
            raise ValueError("Unknown form candidate or variant")

    @property
    def config_id(self) -> str:
        """Readable stable ID, unrelated to scores."""
        return f"{self.candidate}_{self.variant}"

    @property
    def complexity(self) -> int:
        """Only venue adds structural complexity; window types have no ordering."""
        return int(self.variant == "venue")


@dataclass(frozen=True)
class FormExperimentSpec:
    """Exact commented configuration loaded before executing any models."""

    grid: tuple[FormCandidate, ...]
    pilot: tuple[FormCandidate, ...]
    log_offset: float
    max_goals: int
    leakage_threshold: float
    raw: dict[str, Any]
    sha256: str
    source_text: str


def load_form_experiment(path: Path = Path("config/exp002.yaml")) -> FormExperimentSpec:
    """Validate the predeclared grid and methodology; never run on import."""
    content = path.read_bytes()
    raw = yaml.safe_load(content)
    grid = tuple(FormCandidate(c, v) for c, v in product(raw["candidates"], raw["variants"]))
    pilot = tuple(FormCandidate(**r) for r in raw["pilot"])
    if (
        raw["experiment_id"] != "EXP-002"
        or len(grid) != 22
        or len(set(grid)) != 22
        or set(raw["candidates"]) != set(FORM_CANDIDATES)
        or set(pilot) != {FormCandidate("w10", "all"), FormCandidate("ewma10", "venue")}
        or len(pilot) != 2
        or not set(pilot) <= set(grid)
        or raw["poisson"]
        != {
            "log_offset": 0.1,
            "with_elo": [True, False],
            "max_goals": 10,
            "selection_metric": "poisson_nll",
        }
        or raw["ordered"] != {"selection_metric": "log_loss", "elo_first": True}
        or raw["evaluation"]
        != {
            "seasons": list(DEVELOPMENT_SEASONS),
            "folds": 15,
            "leakage_log_loss_threshold": 0.98,
            "paired_se_multiplier": 1,
            "half_life": None,
            "loso": False,
        }
        or raw["simplicity"]
        != {"all": 0, "venue": 1, "tie_break": "selection_metric", "approval_required": True}
    ):
        raise ValueError("EXP-002 differs from the approved specification")
    return FormExperimentSpec(
        grid,
        pilot,
        0.1,
        10,
        0.98,
        raw,
        hashlib.sha256(content).hexdigest(),
        content.decode("utf-8"),
    )


def load_form_inputs(spec: FormExperimentSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load checksummed development-only derived artifacts; never recompute form or Elo."""
    expected_exp001 = spec.raw["inputs"]["exp001_sha256"]
    if hashlib.sha256(Path("config/experiment.yaml").read_bytes()).hexdigest() != expected_exp001:
        raise ValueError("Frozen EXP-001 config changed")
    frames = {}
    for key in ("form", "elo"):
        entry = spec.raw["inputs"][key]
        path = Path(entry["path"])
        if not path.resolve().is_relative_to(Path("data/derived").resolve()):
            raise ValueError("EXP-002 inputs must come from derived storage")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen EXP-002 input checksum mismatch")
        meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if meta["artifact_sha256"] != entry["sha256"] or meta["seasons"] != list(
            DEVELOPMENT_SEASONS
        ):
            raise ValueError("Input metadata does not describe development-only artifact")
        if key == "elo" and meta["config_id"] != E0050_PARAMS.config_id:
            raise ValueError("EXP-002 requires the selected E0050 Elo")
        frames[key] = pd.read_parquet(path)
    return frames["form"], frames["elo"]


@dataclass(frozen=True)
class FormExperimentInput:
    """One row per fixture plus immutable-index paired prematch form candidates."""

    games: pd.DataFrame
    form: pd.DataFrame


def prepare_form_experiment(
    tables: AnalysisTables, form: pd.DataFrame, elo: pd.DataFrame
) -> FormExperimentInput:
    """Use the common game_rows projection and join exact frozen team-game keys."""
    rows = game_rows(tables)
    if set(rows.season_id) != set(DEVELOPMENT_SEASONS):
        raise ValueError("EXP-002 requires all development seasons and no holdouts")
    # The clean projection uses nullable Int64; derived artifacts use NumPy
    # int64. Compare canonical numeric keys, rather than pandas storage dtypes.
    index = pd.MultiIndex.from_frame(rows[["game_id", "team_id"]].astype("int64")).sort_values()
    if (
        not form.index.is_unique
        or not elo.index.is_unique
        or not form.sort_index().index.equals(index)
        or not elo.sort_index().index.equals(index)
        or tuple(form.columns) != FORM_COLUMNS
        or not np.isfinite(form.to_numpy(float)).all()
    ):
        raise ValueError("Frozen form/Elo rows do not match projected game keys")
    validate_feature_columns(form, "form")
    validate_feature_columns(elo, "elo")
    home = rows.loc[rows.is_home].copy()
    games = (
        home[
            [
                "game_id",
                "season_id",
                "date",
                "team_id",
                "opponent_id",
                "goals_scored",
                "goals_conceded",
            ]
        ]
        .rename(
            columns={
                "team_id": "home_team_id",
                "opponent_id": "away_team_id",
                "goals_scored": "home_goals",
                "goals_conceded": "away_goals",
            }
        )
        .reset_index(drop=True)
    )
    hi = pd.MultiIndex.from_frame(games[["game_id", "home_team_id"]])
    games["elo_diff"] = elo.elo_diff.reindex(hi).to_numpy(float)
    if not np.isfinite(games.elo_diff.to_numpy()).all():
        raise ValueError("Nonfinite prematch Elo difference")
    return FormExperimentInput(games, form.copy(deep=True))


def candidate_games(prepared: FormExperimentInput, candidate: FormCandidate | None) -> pd.DataFrame:
    """Cross-match paired keys; venue columns already encode each team's current role."""
    games = prepared.games.copy(deep=True)
    if candidate is None:
        return games
    indices = {
        side: pd.MultiIndex.from_frame(games[["game_id", f"{side}_team_id"]])
        for side in ("home", "away")
    }
    for side, index in indices.items():
        for metric in ("gf", "ga", "pts"):
            column = f"{metric}_{candidate.variant}_{candidate.candidate}"
            games[f"{side}_{metric}"] = prepared.form[column].reindex(index).to_numpy(float)
    games["points_difference"] = games.home_pts - games.away_pts
    return games


@dataclass(frozen=True)
class FormRun:
    """A requested route/configuration with diagnostics; never an approved selection."""

    route: Route
    candidate: FormCandidate | None
    with_elo: bool
    evaluation: EvaluationResult | GoalEvaluationResult
    fits: pd.DataFrame
    seconds: float

    @property
    def config_id(self) -> str:
        """Name candidates and references without using any observed score."""
        suffix = self.candidate.config_id if self.candidate else "reference"
        return f"{self.route}_{suffix}_{'elo' if self.with_elo else 'no_elo'}"


def run_form_configuration(
    prepared: FormExperimentInput,
    spec: FormExperimentSpec,
    *,
    route: Route,
    candidate: FormCandidate | None,
    with_elo: bool = True,
) -> FormRun:
    """Evaluate only the requested configuration, serially on the approved 15 folds."""
    if route not in ("P", "O") or (route == "O" and not with_elo):
        raise ValueError("Unknown or unsupported EXP-002 route")
    if candidate is not None and candidate not in spec.grid:
        raise ValueError("Candidate outside declared grid")
    start = perf_counter()
    games = candidate_games(prepared, candidate)
    records = []
    if route == "P":
        models_p: list[FormPoissonPredictor] = []

        def factory_p() -> FormPoissonPredictor:
            model = FormPoissonPredictor(
                with_elo=with_elo, with_form=candidate is not None, log_offset=spec.log_offset
            )
            models_p.append(model)
            return model

        features = (["home_gf", "home_ga", "away_gf", "away_ga"] if candidate else []) + (
            ["elo_diff"] if with_elo else []
        )
        evaluation: EvaluationResult | GoalEvaluationResult = evaluate_goals(
            factory_p, games, feature_columns=features, max_goals=spec.max_goals
        )
        models: Sequence[FormPoissonPredictor | FormOrderedPredictor] = models_p
    else:
        models_o: list[FormOrderedPredictor] = []

        def factory_o() -> FormOrderedPredictor:
            model = FormOrderedPredictor(with_form=candidate is not None)
            models_o.append(model)
            return model

        features = ["elo_diff"] + (["points_difference"] if candidate else [])
        evaluation = evaluate(factory_o, games, feature_columns=features)
        models = models_o
    if float(evaluation.summary.log_loss) < spec.leakage_threshold:
        raise RuntimeError("Leakage threshold breached: stop EXP-002 and investigate")
    for model, fold in zip(models, evaluation.per_fold.itertuples(), strict=True):
        fit = model.fit_result
        assert fit is not None
        record = {
            "fold": fold.fold,
            "season_id": fold.season_id,
            "training_seasons": model.training_seasons,
            **asdict(fit),
        }
        records.append(record)
    return FormRun(
        route, candidate, with_elo, evaluation, pd.DataFrame(records), perf_counter() - start
    )


def compare_form_candidates(runs: Sequence[FormRun], route: Route) -> Comparison:
    """Compare route-specific selection loss, excluding descriptive no-Elo controls."""
    selected = [r for r in runs if r.route == route and r.candidate is not None and r.with_elo]
    if len({r.config_id for r in selected}) != len(selected):
        raise ValueError("Duplicate EXP-002 candidate results")
    result = compare(
        {r.config_id: r.evaluation for r in selected},
        metric="poisson_nll" if route == "P" else "log_loss",
    )
    result.rankings["complexity"] = {
        r.config_id: r.candidate.complexity for r in selected if r.candidate is not None
    }
    return result


def zero_counts(prepared: FormExperimentInput, spec: FormExperimentSpec) -> pd.DataFrame:
    """Descriptive zero counts per team-row candidate, before the uniform log offset."""
    return pd.DataFrame(
        [
            {
                "candidate": c.candidate,
                "variant": c.variant,
                "rows": len(prepared.form),
                **{
                    f"{m}_zeros": int(prepared.form[f"{m}_{c.variant}_{c.candidate}"].eq(0).sum())
                    for m in ("gf", "ga")
                },
            }
            for c in spec.grid
        ]
    )
