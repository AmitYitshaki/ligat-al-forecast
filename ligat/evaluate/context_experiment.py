"""Declared EXP-003 contexts, reused judges/models, and explicitly requested runs."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from numpy.typing import NDArray

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.evaluate.backtest import DevelopmentPolicy, EvaluationResult, evaluate
from ligat.evaluate.comparison import compare
from ligat.evaluate.form_experiment import FormCandidate, candidate_games, prepare_form_experiment
from ligat.evaluate.goal_backtest import GoalEvaluationResult, evaluate_goals
from ligat.features.registry import SEASON_CONTEXT_COLUMNS, validate_feature_columns
from ligat.features.table_state import game_rows
from ligat.models.baselines import B0
from ligat.models.form_models import FormOrderedPredictor, FormPoissonPredictor
from ligat.models.ordered_logit import fit_ordered_logit

CANDIDATES = (
    "value_avg",
    "value_total",
    "position",
    "position_mean",
    "ppm",
    "gd",
    "value_avg+position",
)
Route = Literal["A", "O", "P"]
REFERENCE = {"A": "elo", "O": "elo", "P": "poisson"}


@dataclass(frozen=True)
class ContextConfiguration:
    """A declared candidate/reference and the explicit season-11 sensitivity flag."""

    route: Route
    candidate: str | None = None
    reference: str | None = None
    sensitivity: bool = False

    def __post_init__(self) -> None:
        if self.route not in REFERENCE or (self.candidate is None) == (self.reference is None):
            raise ValueError("One declared route and exactly one candidate/reference are required")
        if self.candidate is not None and self.candidate not in CANDIDATES:
            raise ValueError("Unknown EXP-003 candidate")
        allowed = ("b0", "elo") if self.route == "A" else (REFERENCE[self.route],)
        if self.reference is not None and self.reference not in allowed:
            raise ValueError("Reference does not match its route")
        if self.sensitivity and (
            self.route == "A"
            or self.candidate not in (None, "value_avg", "value_total", "value_avg+position")
        ):
            raise ValueError("Unknown sensitivity configuration")

    @property
    def config_id(self) -> str:
        """Stable identity without any observed score."""
        return f"{self.route}_{self.candidate or self.reference}" + (
            "_no_s11" if self.sensitivity else ""
        )

    @property
    def columns(self) -> tuple[str, ...]:
        """The combined candidate has two coefficients, never a product interaction."""
        return tuple(f"z_{name}" for name in self.candidate.split("+")) if self.candidate else ()


def context_configurations() -> tuple[ContextConfiguration, ...]:
    """Exactly 25 main and eight sensitivity configurations; does not fit models."""
    main = [ContextConfiguration("A", reference="b0")]
    routes: tuple[Route, ...] = ("A", "O", "P")
    for route in routes:
        main.append(ContextConfiguration(route, reference=REFERENCE[route]))
        main.extend(ContextConfiguration(route, candidate=name) for name in CANDIDATES)
    sensitivity_routes: tuple[Route, ...] = ("O", "P")
    for route in sensitivity_routes:
        main.append(ContextConfiguration(route, reference=REFERENCE[route], sensitivity=True))
        main.extend(
            ContextConfiguration(route, candidate=name, sensitivity=True)
            for name in ("value_avg", "value_total", "value_avg+position")
        )
    return tuple(main)


@dataclass(frozen=True)
class ContextExperimentSpec:
    """The exact commented declaration loaded before any pilot fitting."""

    raw: dict[str, Any]
    source_text: str
    sha256: str


def load_context_experiment(path: Path = Path("config/exp003.yaml")) -> ContextExperimentSpec:
    """Validate the predeclared candidates, folds, routes, and blind pilot identities."""
    content = path.read_bytes()
    raw = yaml.safe_load(content)
    if (
        raw["experiment_id"] != "EXP-003"
        or raw["candidates"] != list(CANDIDATES)
        or raw["routes"] != ["A", "O", "P"]
        or raw["poisson"]
        != {"log_offset": 0.1, "max_goals": 10, "candidate": "ewma5", "variant": "venue"}
        or raw["evaluation"]
        != {
            "training_start": 9,
            "validation_seasons": list(range(11, 26)),
            "half_life": None,
            "loso": False,
            "leakage_log_loss_threshold": 0.98,
        }
        or raw["sensitivity"]
        != {
            "excluded_season": 11,
            "routes": ["O", "P"],
            "candidates": ["value_avg", "value_total", "value_avg+position"],
        }
        or raw["pilot"]
        != {
            "validation_seasons": [11, 18, 25],
            "configurations": ["O_elo", "O_gd", "P_poisson", "P_value_avg"],
        }
        or raw["seed"] is not None
        or set(raw["inputs"]) != {"context", "elo", "form"}
    ):
        raise ValueError("EXP-003 differs from its predeclared specification")
    return ContextExperimentSpec(raw, content.decode("utf-8"), hashlib.sha256(content).hexdigest())


@dataclass(frozen=True)
class ContextInputs:
    """Frozen derived frames and the checksum-pinned corrected context audit."""

    context: pd.DataFrame
    elo: pd.DataFrame
    form: pd.DataFrame
    audit: pd.DataFrame


def load_context_inputs(spec: ContextExperimentSpec) -> ContextInputs:
    """Verify all three artifacts and sidecars before the first DataFrame read."""
    metadata = {}
    paths = {}
    for name in ("context", "elo", "form"):
        item = spec.raw["inputs"][name]
        path = Path(item["path"])
        if not path.resolve().is_relative_to(Path("data/derived").resolve()):
            raise ValueError("EXP-003 inputs must be inside derived storage")
        sidecar = path.with_suffix(".metadata.json")
        for target, expected in ((path, item["sha256"]), (sidecar, item["metadata_sha256"])):
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen EXP-003 input checksum mismatch")
        metadata[name] = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata[name]["artifact_sha256"] != item["sha256"] or metadata[name]["seasons"] != list(
            DEVELOPMENT_SEASONS
        ):
            raise ValueError("Input metadata is not the declared development artifact")
        paths[name] = path
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    return ContextInputs(
        frames["context"], frames["elo"], frames["form"], pd.DataFrame(metadata["context"]["audit"])
    )


@dataclass(frozen=True)
class ContextExperimentInput:
    """Projected fixture rows with paired derived contexts and no recomputed feature block."""

    games: pd.DataFrame
    team_context: pd.DataFrame
    input_sha256: str


def prepare_context_experiment(
    tables: AnalysisTables, inputs: ContextInputs
) -> ContextExperimentInput:
    """Build only declared log ratios/differences, preserving season-8 missing values."""
    rows = game_rows(tables)
    if set(rows.season_id) != set(DEVELOPMENT_SEASONS):
        raise ValueError("EXP-003 requires development seasons 8–25 only")
    base = prepare_form_experiment(tables, inputs.form, inputs.elo)
    context = inputs.context.copy(deep=True)
    if (
        not context.index.is_unique
        or not context.sort_index().index.equals(inputs.form.sort_index().index)
        or tuple(context.columns) != SEASON_CONTEXT_COLUMNS
    ):
        raise ValueError("Frozen context keys or columns do not match form/Elo")
    validate_feature_columns(context, "season_context")
    identity = rows.set_index(["game_id", "team_id"])
    context["season_id"] = identity.season_id.reindex(context.index).to_numpy(int)
    audit = inputs.audit
    if audit.season_id.duplicated().any() or set(audit.season_id) != set(DEVELOPMENT_SEASONS):
        raise ValueError("Context audit requires one row per development season")
    context["position_mean"] = context.prev_position_pct.copy()
    promoted = context.promoted_in
    context.loc[promoted, "position_mean"] = context.loc[promoted, "season_id"].map(
        audit.set_index("season_id").promoted_position_mean
    )
    games = candidate_games(base, FormCandidate("ewma5", "venue"))
    paired = {}
    for side in ("home", "away"):
        index = pd.MultiIndex.from_frame(games[["game_id", f"{side}_team_id"]])
        paired[side] = context.reindex(index).reset_index(drop=True)
    known = games.season_id.gt(8).to_numpy()
    for candidate, column in (
        ("value_avg", "prev_value_share_avg"),
        ("value_total", "prev_value_share_total"),
        ("position", "prev_position_pct"),
        ("position_mean", "position_mean"),
    ):
        home = paired["home"][column].to_numpy(float)
        away = paired["away"][column].to_numpy(float)
        if (
            not np.isfinite(home[known]).all()
            or not np.isfinite(away[known]).all()
            or (home[known] <= 0).any()
            or (away[known] <= 0).any()
        ):
            raise ValueError("Known previous-season ratios require positive finite values")
        transformed = np.full(len(games), np.nan)
        transformed[known] = np.log(home[known] / away[known])
        games[f"z_{candidate}"] = transformed
    for candidate, column in (("ppm", "prev_ppm"), ("gd", "prev_gd_per_game")):
        games[f"z_{candidate}"] = paired["home"][column].to_numpy(float) - paired["away"][
            column
        ].to_numpy(float)
    if not np.isfinite(
        games.loc[known, [f"z_{name}" for name in CANDIDATES[:-1]]].to_numpy(float)
    ).all():
        raise ValueError("Nonfinite known context contrast")
    digest = hashlib.sha256(
        games.to_json(orient="split", double_precision=15, date_format="iso").encode()
    )
    digest.update(context.to_json(orient="split", double_precision=15).encode())
    return ContextExperimentInput(games, context, digest.hexdigest())


class ContextOrderedPredictor(FormOrderedPredictor):
    """Use the existing ordinal likelihood; only Elo is subject to the sign guard."""

    def __init__(self, configuration: ContextConfiguration) -> None:
        super().__init__(with_form=False)
        self.configuration = configuration
        self.with_elo = configuration.route == "O" or configuration.reference == "elo"

    def design(self, features: pd.DataFrame) -> NDArray[np.float64]:
        columns = [features.elo_diff.to_numpy(float) / 400] if self.with_elo else []
        columns.extend(features[column].to_numpy(float) for column in self.configuration.columns)
        return np.column_stack(columns)

    def fit(self, train: pd.DataFrame) -> None:
        self.fit_result = None
        self.training_seasons = tuple(sorted(int(s) for s in train.season_id.unique()))
        self.fit_result = fit_ordered_logit(
            self.design(train), train.outcome.to_numpy(), nonnegative_first=self.with_elo
        )


class ContextPoissonPredictor(FormPoissonPredictor):
    """Append antisymmetric context columns to the unchanged EXP-002 goal design."""

    def __init__(self, columns: tuple[str, ...]) -> None:
        super().__init__(with_elo=True, with_form=True, log_offset=0.1)
        self.context_columns = columns

    def design(self, features: pd.DataFrame) -> NDArray[np.float64]:
        base = super().design(features)
        if not self.context_columns:
            return base
        z = features.loc[:, self.context_columns].to_numpy(float)
        paired = np.stack((z, -z), axis=1).reshape(2 * len(features), len(self.context_columns))
        return np.column_stack((base, paired))

    def fit(self, features: pd.DataFrame, goals: pd.DataFrame) -> None:
        super().fit(features, goals)
        assert self.fit_result is not None
        if self.fit_result.coefficients[4] < 0:
            raise RuntimeError("Negative Poisson Elo coefficient: stop EXP-003")


@dataclass(frozen=True)
class ContextRun:
    """One explicitly requested evaluation, with fit diagnostics and no model selection."""

    configuration: ContextConfiguration
    evaluation: EvaluationResult | GoalEvaluationResult
    fits: pd.DataFrame
    seconds: float

    @property
    def config_id(self) -> str:
        """The predeclared identity used in checksummed result files."""
        return self.configuration.config_id


def run_context_configuration(
    prepared: ContextExperimentInput,
    spec: ContextExperimentSpec,
    configuration: ContextConfiguration,
    *,
    validation_seasons: tuple[int, ...] | None = None,
    regression_training_start: int | None = None,
) -> ContextRun:
    """Fit a single requested identity; callers never implicitly run the 33-item grid."""
    if configuration not in context_configurations():
        raise ValueError("Configuration outside the declared EXP-003 grid")
    if regression_training_start is not None and (
        regression_training_start != 8
        or configuration.candidate is not None
        or configuration.sensitivity
        or configuration.reference == "b0"
    ):
        raise ValueError("Training from 8 is reserved for reference regression tests")
    seasons = (
        tuple(s for s in range(11, 26) if not configuration.sensitivity or s != 11)
        if validation_seasons is None
        else validation_seasons
    )
    if validation_seasons is not None and (
        configuration.config_id not in spec.raw["pilot"]["configurations"]
        or validation_seasons != tuple(spec.raw["pilot"]["validation_seasons"])
    ):
        raise ValueError("Explicit partial runs are restricted to the declared blind pilot")
    policy = DevelopmentPolicy(
        regression_training_start or 9, (11,) if configuration.sensitivity else (), seasons
    )
    start = perf_counter()
    models: list[Any] = []

    def factory() -> Any:
        if configuration.route == "P":
            model: Any = ContextPoissonPredictor(configuration.columns)
        elif configuration.reference == "b0":
            model = B0()
        else:
            model = ContextOrderedPredictor(configuration)
        models.append(model)
        return model

    games = prepared.games
    if configuration.route == "P":
        features = ["home_gf", "home_ga", "away_gf", "away_ga", "elo_diff", *configuration.columns]
        evaluation: EvaluationResult | GoalEvaluationResult = evaluate_goals(
            factory, games, feature_columns=features, policy=policy
        )
    else:
        features = (
            ["elo_diff"] if configuration.route == "O" or configuration.reference == "elo" else []
        ) + list(configuration.columns)
        evaluation = evaluate(factory, games, feature_columns=features, policy=policy)
    if float(evaluation.summary.log_loss) < spec.raw["evaluation"]["leakage_log_loss_threshold"]:
        raise RuntimeError("Leakage threshold breached: stop EXP-003 without displaying the score")
    records = []
    for model, fold in zip(models, evaluation.per_fold.itertuples(), strict=True):
        fit = getattr(model, "fit_result", None)
        if fit is None:
            diagnostics = {
                "converged": True,
                "iterations": 0,
                "accepted_by_gradient_check": False,
                "max_abs_gradient": 0.0,
            }
            elo_coefficient = None
        else:
            diagnostics = asdict(fit)
            elo_coefficient = (
                (fit.coefficients[4] if configuration.route == "P" else fit.beta)
                if configuration.route in ("O", "P") or configuration.reference == "elo"
                else None
            )
        training = tuple(range(policy.training_start, int(fold.season_id)))
        training = tuple(s for s in training if s not in policy.excluded_seasons)
        records.append(
            {
                "fold": fold.fold,
                "season_id": fold.season_id,
                "training_seasons": getattr(model, "training_seasons", training),
                "elo_coefficient": elo_coefficient,
                **diagnostics,
            }
        )
    return ContextRun(configuration, evaluation, pd.DataFrame(records), perf_counter() - start)


def _paired_statistics(left: ContextRun, right: ContextRun) -> dict[str, Any]:
    metric: Literal["poisson_nll", "log_loss"] = (
        "poisson_nll" if left.configuration.route == "P" else "log_loss"
    )
    # Reuse the judge's pairing check before calculating reference-relative fold differences.
    seasons = tuple(int(s) for s in left.evaluation.per_fold.season_id)
    compare(
        {left.config_id: left.evaluation, right.config_id: right.evaluation},
        metric=metric,
        validation_seasons=seasons,
    )
    a = left.evaluation.per_fold.set_index("season_id")
    b = right.evaluation.per_fold.set_index("season_id").reindex(a.index)
    differences = a[metric] - b[metric]
    counts = a["rows" if metric == "poisson_nll" else "games"]
    mean = float(np.average(differences, weights=counts))
    se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    recent = a.index >= 21
    return {
        "mean_improvement": mean,
        "se": se,
        "z": mean / se if se else np.nan,
        "folds_better": int(differences.gt(0).sum()),
        "folds": len(differences),
        "last5_improvement": float(np.average(differences[recent], weights=counts[recent])),
        "last5_folds": int(recent.sum()),
    }


def context_comparisons(runs: Sequence[ContextRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank paired route panels and report reference/direct contrasts; never select a model."""
    rows, direct = [], []
    for route in ("A", "O", "P"):
        for sensitivity in (False, True):
            panel = [
                r
                for r in runs
                if r.configuration.route == route and r.configuration.sensitivity == sensitivity
            ]
            if not panel:
                continue
            by_id = {r.config_id: r for r in panel}
            reference = next(
                (r for r in panel if r.configuration.reference == REFERENCE[route]), None
            )
            if reference is None:
                raise ValueError("Every comparison panel needs its paired route reference")
            seasons = tuple(int(s) for s in reference.evaluation.per_fold.season_id)
            ranking = compare(
                {r.config_id: r.evaluation for r in panel},
                metric="poisson_nll" if route == "P" else "log_loss",
                validation_seasons=seasons,
            ).rankings
            references = [reference]
            if route == "A":
                b0 = next((r for r in panel if r.configuration.reference == "b0"), None)
                if b0 is None:
                    raise ValueError("Route A also requires its B0 reference")
                references.append(b0)
            for paired_reference in references:
                for run in panel:
                    statistics = (
                        {}
                        if run.config_id == paired_reference.config_id
                        else _paired_statistics(paired_reference, run)
                    )
                    rows.append(
                        {
                            "config_id": run.config_id,
                            "route": route,
                            "sensitivity": sensitivity,
                            "reference": paired_reference.config_id,
                            **ranking.loc[run.config_id].to_dict(),
                            **statistics,
                        }
                    )
            if route in ("O", "P"):
                suffix = "_no_s11" if sensitivity else ""
                for default, alternative in (
                    ("value_avg", "value_total"),
                    ("position", "position_mean"),
                ):
                    if (
                        f"{route}_{default}{suffix}" in by_id
                        and f"{route}_{alternative}{suffix}" in by_id
                    ):
                        direct.append(
                            {
                                "route": route,
                                "sensitivity": sensitivity,
                                "default": default,
                                "alternative": alternative,
                                **_paired_statistics(
                                    by_id[f"{route}_{default}{suffix}"],
                                    by_id[f"{route}_{alternative}{suffix}"],
                                ),
                            }
                        )
    direct_frame = pd.DataFrame(direct)
    if direct_frame.empty:
        direct_frame = pd.DataFrame(
            columns=[
                "route",
                "sensitivity",
                "default",
                "alternative",
                "mean_improvement",
                "se",
                "z",
                "folds_better",
                "folds",
                "last5_improvement",
                "last5_folds",
            ]
        )
    return pd.DataFrame(rows), direct_frame
