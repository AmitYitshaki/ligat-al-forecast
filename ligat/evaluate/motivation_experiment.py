"""Preregistered EXP-004: frozen motivation contrasts added to the Elo converter."""

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from numpy.typing import NDArray

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.evaluate.backtest import DevelopmentPolicy, EvaluationResult, evaluate
from ligat.features.table_state import game_rows
from ligat.models.ordered_logit import (
    OrderedEloPredictor,
    fit_ordered_logit,
    ordered_probabilities,
    time_weights,
)
from ligat.motivation import representations

BASE_IDS = (
    "C0",
    "L2-diff",
    "L2-sides",
    "L3-diff",
    "L3-sides",
    "L5-diff",
    "L5-sides",
    "Lc-diff",
    "Lc-sides",
    "Comp",
)
PILOT_IDS = ("C0", "C0-hl4", "L2-diff", "Lc-sides", "Comp-hl4")
PILOT_SEASONS = (12, 18, 25)
VALIDATION_SEASONS = tuple(range(11, 26))
DEEP_SEASONS = tuple(range(12, 26))
INPUTS = {
    "motivation": {
        "path": "data/derived/motivation/motivation_v1.parquet",
        "sha256": "890e58d33b2f50ded50a254e4b88d05cf226cd5bd56c94e16601dcaf941f17f4",
    },
    "motivation_metadata": {
        "path": "data/derived/motivation/motivation_v1.metadata.json",
        "sha256": "da15d684277739b573265d1cbfa83a2b02ea6b0dc88a906496629cd8b8bd2e3d",
    },
    "elo": {
        "path": "data/derived/elo/elo_E0050_prematch.parquet",
        "sha256": "759b9b830b617b9dec9c519242544c733ffa068e4cd503f9603851cb25857d9f",
    },
    "elo_metadata": {
        "path": "data/derived/elo/elo_E0050_prematch.metadata.json",
        "sha256": "83e926b1381fcf575412d6264413993b24589b367d2e58cd66aa19a7bdcd30bf",
    },
}
Metric = Literal["log_loss", "rps"]
Stage = Literal["baseline", "pilot", "full"]
AUDIT_SETTINGS = {
    "improvement_above": 0.005,
    "placebo_permutations": 200,
    "placebo_seed": 14112026,
    "zero_column_tolerance": 1e-8,
}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotivationConfiguration:
    """One declared representation/shape and its paired conversion time weighting."""

    name: str
    half_life: int | None = None

    def __post_init__(self) -> None:
        if self.name not in BASE_IDS or self.half_life not in (None, 4):
            raise ValueError("Configuration outside the declared EXP-004 grid")

    @property
    def config_id(self) -> str:
        """Identity fixed before observing any score."""
        return self.name + ("-hl4" if self.half_life else "")

    @property
    def reference_id(self) -> str:
        """All representations compare against the C0 with identical decay."""
        return "C0-hl4" if self.half_life and self.name != "C0" else "C0"

    @property
    def columns(self) -> tuple[str, ...]:
        """Design columns produced only by the shared representation functions."""
        if self.name == "C0":
            return ()
        level, form = ("Comp", "diff") if self.name == "Comp" else self.name.split("-")
        suffixes = (
            ("title_diff", "cut_diff", "survival_diff")
            if level == "Comp"
            else ("m_diff",)
            if form == "diff"
            else ("m_home", "m_away")
        )
        return tuple(f"{level}_{suffix}" for suffix in suffixes)


def motivation_configurations() -> tuple[MotivationConfiguration, ...]:
    """Exactly twenty configurations in preregistered simplicity order; no fitting."""
    return tuple(MotivationConfiguration(name, decay) for name in BASE_IDS for decay in (None, 4))


def _declaration() -> dict[str, Any]:
    return {
        "experiment_id": "EXP-004",
        "inputs": INPUTS,
        "configurations": [
            {"config_id": c.config_id, **asdict(c)} for c in motivation_configurations()
        ],
        "evaluation": {
            "training_start": 8,
            "validation_seasons": list(VALIDATION_SEASONS),
            "deep_seasons": list(DEEP_SEASONS),
            "last5_seasons": list(range(21, 26)),
            "minimum_training_seasons": 4,
            "loso": False,
        },
        "comparisons": {
            c.config_id: c.reference_id for c in motivation_configurations() if c.config_id != "C0"
        },
        "leakage": {"log_loss_below": 0.98},
        "audit": AUDIT_SETTINGS,
        "pilot": {"validation_seasons": list(PILOT_SEASONS), "configurations": list(PILOT_IDS)},
        "seed": None,
    }


@dataclass(frozen=True)
class MotivationExperimentSpec:
    """Exact commented declaration retained for the experiment manifest."""

    raw: dict[str, Any]
    source_text: str
    sha256: str


def load_motivation_experiment(path: Path = Path("config/exp004.yaml")) -> MotivationExperimentSpec:
    """Reject any deviation, including changed hashes, folds, decay or pilot identities."""
    content = path.read_bytes()
    raw = yaml.safe_load(content)
    if json.dumps(raw, sort_keys=True) != json.dumps(_declaration(), sort_keys=True):
        raise ValueError("EXP-004 differs from its predeclared specification")
    return MotivationExperimentSpec(
        raw, content.decode("utf-8"), hashlib.sha256(content).hexdigest()
    )


@dataclass(frozen=True)
class MotivationInputs:
    """Checksum-verified, development-only frozen team-game inputs."""

    motivation: pd.DataFrame
    elo: pd.DataFrame


def load_motivation_inputs(spec: MotivationExperimentSpec) -> MotivationInputs:
    """Check all four files before reading; Parquet scans explicitly exclude holdouts."""
    if spec.raw != _declaration():
        raise ValueError("Modified EXP-004 specification")
    for entry in spec.raw["inputs"].values():
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen EXP-004 input checksum mismatch")
    for name in ("motivation", "elo"):
        meta = json.loads(Path(INPUTS[name + "_metadata"]["path"]).read_text(encoding="utf-8"))
        if (
            meta["seasons"] != list(DEVELOPMENT_SEASONS)
            or meta["artifact_sha256"] != INPUTS[name]["sha256"]
        ):
            raise ValueError("Input metadata must describe the pinned development artifact")
    frames = {
        name: pd.read_parquet(
            INPUTS[name]["path"], filters=[("season_id", "in", list(DEVELOPMENT_SEASONS))]
        )
        for name in ("motivation", "elo")
    }
    return MotivationInputs(frames["motivation"], frames["elo"])


@dataclass(frozen=True)
class MotivationExperimentInput:
    """One row per fixture, prematch contrasts and explicitly separate observed targets."""

    games: pd.DataFrame
    input_sha256: str
    motivation: pd.DataFrame | None = None
    participants: pd.DataFrame | None = None
    elo: pd.DataFrame | None = None


def prepare_motivation_experiment(
    tables: AnalysisTables, inputs: MotivationInputs
) -> MotivationExperimentInput:
    """Validate exact venue/identity/date pairing before calling shared representations."""
    seasons = tables.team_matches.season_id
    if seasons.isna().any() or not set(seasons) <= set(DEVELOPMENT_SEASONS):
        raise ValueError("EXP-004 rejects seasons outside development 8-25 before preparation")
    rows = game_rows(tables)
    if set(rows.season_id) != set(DEVELOPMENT_SEASONS):
        raise ValueError("EXP-004 requires all development seasons 8-25 only")
    if rows[["game_id", "team_id"]].isna().any().any():
        raise ValueError("NA in team-game identity keys is forbidden")
    # Frozen artifacts use NumPy int64; the clean projection can use nullable Int64.
    # Normalize storage only after rejecting missing keys; identity checks stay exact.
    rows = rows.astype({"game_id": "int64", "team_id": "int64"})
    identity = rows.set_index(["game_id", "team_id"]).sort_index()
    motivation = inputs.motivation.sort_index()
    elo = inputs.elo.sort_index()
    if any(not f.index.is_unique or not f.index.equals(identity.index) for f in (motivation, elo)):
        raise ValueError("Missing or duplicate frozen team-game rows")
    for column in ("opponent_id", "is_home", "season_id", "as_of"):
        if motivation[column].isna().any() or not motivation[column].eq(identity[column]).all():
            raise ValueError("Frozen motivation identity, is_home or as_of mismatch")
    if (
        not motivation.is_home.isin([True, False]).all()
        or not np.isfinite(elo.elo_diff.to_numpy(float)).all()
    ):
        raise ValueError("Unknown venue or nonfinite prematch Elo")
    home = rows.loc[rows.is_home]
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
    ai = pd.MultiIndex.from_frame(games[["game_id", "away_team_id"]])
    games["elo_diff"] = elo.elo_diff.reindex(hi).to_numpy(float)
    if not np.array_equal(games.elo_diff.to_numpy(), -elo.elo_diff.reindex(ai).to_numpy(float)):
        raise ValueError("Elo sides do not match the same fixture")
    for config in motivation_configurations()[::2]:
        if config.name == "C0":
            continue
        level, form = ("Comp", "diff") if config.name == "Comp" else config.name.split("-")
        values = representations.game_contrast(
            motivation, level, cast(Literal["diff", "sides"], form)
        ).reindex(games.game_id)
        games[list(config.columns)] = values.to_numpy(float)
    l3 = representations.game_contrast(motivation, "L3", "sides").reindex(games.game_id)
    games["l3_involved"] = l3.lt(1).any(axis=1).to_numpy()
    if not np.isfinite(
        games[
            [
                "elo_diff",
                *dict.fromkeys(c for config in motivation_configurations() for c in config.columns),
            ]
        ].to_numpy(float)
    ).all():
        raise ValueError("Nonfinite motivation design")
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(games, index=True).to_numpy().tobytes()
    ).hexdigest()
    return MotivationExperimentInput(
        games, digest, motivation.copy(deep=True), rows.copy(deep=True), elo.copy(deep=True)
    )


class MotivationOrderedPredictor(OrderedEloPredictor):
    """Unchanged ordinal likelihood, with unconstrained motivation slopes beside Elo."""

    def __init__(self, configuration: MotivationConfiguration) -> None:
        super().__init__(configuration.half_life)
        self.configuration = configuration

    def design(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Elo/400 first, then the already prepared shared representation columns."""
        return np.column_stack(
            [
                features.elo_diff.to_numpy(float) / 400,
                *(features[c].to_numpy(float) for c in self.configuration.columns),
            ]
        )

    def fit(self, train: pd.DataFrame) -> None:
        """Use exactly OrderedEloPredictor for C0, including its one-column path."""
        if self.configuration.name == "C0":
            super().fit(train)
            return
        self.fit_result = None
        self.training_seasons = tuple(sorted(int(s) for s in train.season_id.unique()))
        weights = (
            None
            if self.half_life is None
            else time_weights(train.season_id, max(self.training_seasons) + 1, self.half_life)
        )
        self.fit_result = fit_ordered_logit(
            self.design(train), train.outcome.to_numpy(), weights, nonnegative_first=True
        )

    def predict(self, games: pd.DataFrame) -> NDArray[np.float64]:
        """The judge supplies an explicit target-free projection only."""
        if self.configuration.name == "C0":
            return super().predict(games)
        if self.fit_result is None:
            raise RuntimeError("Fit the converter before prediction")
        if not games.season_id.eq(max(self.training_seasons) + 1).all():
            raise ValueError("EXP-004 supports chronological prediction only")
        return ordered_probabilities(self.fit_result, self.design(games))


@dataclass(frozen=True)
class MotivationRun:
    """One explicitly requested run; scores remain machine data during blind QA."""

    configuration: MotivationConfiguration
    evaluation: EvaluationResult
    fits: pd.DataFrame
    seconds: float
    stage: Stage
    audit_required: bool = False
    audit: dict[str, Any] | None = None

    @property
    def config_id(self) -> str:
        """Preregistered identity, independent of scores."""
        return self.configuration.config_id


def check_leakage(evaluation: EvaluationResult, reference: EvaluationResult | None = None) -> bool:
    """H-K22: stop on absolute loss; flag paired improvement without disclosing scores."""
    if float(evaluation.summary.log_loss) < 0.98:
        raise RuntimeError("Leakage threshold breached: stop EXP-004 without displaying the score")
    if reference is not None:
        a, b = _paired_folds(evaluation, reference)
        improvement = float(np.average(b.log_loss - a.log_loss, weights=a.games))
        if improvement > AUDIT_SETTINGS["improvement_above"]:
            LOGGER.warning("EXP-004 paired improvement flagged for automatic audit")
            return True
    return False


def run_motivation_configuration(
    prepared: MotivationExperimentInput,
    spec: MotivationExperimentSpec,
    configuration: MotivationConfiguration,
    *,
    stage: Stage,
    reference: MotivationRun | None = None,
    audit_cache: dict[str, Any] | None = None,
) -> MotivationRun:
    """Run one configuration only; pilot and reference regression have restricted scopes."""
    if spec.raw != _declaration() or configuration not in motivation_configurations():
        raise ValueError("Configuration/specification outside EXP-004")
    if (
        stage not in ("baseline", "pilot", "full")
        or (stage == "baseline" and configuration.name != "C0")
        or (stage == "pilot" and configuration.config_id not in PILOT_IDS)
    ):
        raise ValueError("Undeclared run scope")
    if configuration.config_id != "C0" and (
        reference is None
        or reference.config_id != configuration.reference_id
        or reference.stage != stage
    ):
        raise ValueError("An evaluated C0 with matching decay, stage and games is required")
    seasons = PILOT_SEASONS if stage == "pilot" else VALIDATION_SEASONS
    policy = DevelopmentPolicy(training_start=8, validation_seasons=seasons)
    models: list[MotivationOrderedPredictor] = []

    def factory() -> MotivationOrderedPredictor:
        model = MotivationOrderedPredictor(configuration)
        models.append(model)
        return model

    start = perf_counter()
    evaluation = evaluate(
        factory, prepared.games, feature_columns=("elo_diff", *configuration.columns), policy=policy
    )
    flagged = check_leakage(evaluation, reference.evaluation if reference else None)
    records = []
    for model, fold in zip(models, evaluation.per_fold.itertuples(), strict=True):
        assert model.fit_result is not None
        if model.training_seasons != tuple(range(8, int(fold.season_id))):
            raise ValueError("Fitted training seasons differ from the declared policy")
        records.append(
            {
                "fold": fold.fold,
                "season_id": fold.season_id,
                "training_seasons": model.training_seasons,
                "training_games": int(prepared.games.season_id.isin(model.training_seasons).sum()),
                "validation_games": fold.games,
                **asdict(model.fit_result),
                "theta2": model.fit_result.theta2,
            }
        )
    run = MotivationRun(
        configuration, evaluation, pd.DataFrame(records), perf_counter() - start, stage, flagged
    )
    if flagged:
        assert reference is not None
        audit = audit_motivation_configuration(prepared, run, reference, cache=audit_cache)
        run = MotivationRun(
            configuration, evaluation, run.fits, perf_counter() - start, stage, True, audit
        )
    return run


def _paired_folds(
    left: EvaluationResult, right: EvaluationResult
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["fold", "season_id", "game_id"]
    a, b = (r.per_game.set_index(keys).sort_index() for r in (left, right))
    if (
        not a.index.is_unique
        or not b.index.is_unique
        or not a.index.equals(b.index)
        or not np.array_equal(a.outcome, b.outcome)
    ):
        raise ValueError("Comparisons require exactly the same games and outcomes")
    from ligat.evaluate.backtest import summarize

    af, _ = summarize(a.reset_index())
    bf, _ = summarize(b.reset_index())
    return af.set_index("season_id"), bf.set_index("season_id")


def paired_statistics(
    candidate: EvaluationResult, reference: EvaluationResult, metric: Metric, seasons: Sequence[int]
) -> dict[str, Any]:
    """Positive is improvement; weighted mean, unweighted paired ddof=1 SE."""
    if (
        metric not in ("log_loss", "rps")
        or len(set(seasons)) != len(seasons)
        or len(seasons) < 2
        or not set(seasons) <= set(VALIDATION_SEASONS)
    ):
        raise ValueError("Invalid paired statistics scope")
    a, b = _paired_folds(candidate, reference)
    if not set(seasons) <= set(a.index):
        raise ValueError("Missing requested paired folds")
    a, b = a.loc[list(seasons)], b.loc[list(seasons)]
    d = b[metric] - a[metric]
    recent = a.index >= 21
    return {
        "mean_improvement": float(np.average(d, weights=a.games)),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "folds_better": int(d.gt(0).sum()),
        "folds": len(d),
        "last5_folds_better": int(d[recent].gt(0).sum()),
        "last5_folds": int(recent.sum()),
        "last5_improvement": float(np.average(d[recent], weights=a.games[recent]))
        if recent.any()
        else None,
    }


def entry_disagreement(log_stats: Mapping[str, Any], rps_stats: Mapping[str, Any]) -> bool:
    """Rule 10: opposite signs and at least one non-negligible paired difference."""
    a, b = float(log_stats["mean_improvement"]), float(rps_stats["mean_improvement"])
    return a * b < 0 and (abs(a) > log_stats["se"] or abs(b) > rps_stats["se"])


def selection_disagreement(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Rule 10: a metric winner lies outside the other metric's one-SE close set."""
    return not set(left["winners"]) <= set(right["close"]) or not set(right["winners"]) <= set(
        left["close"]
    )


def _ranking(
    runs: Sequence[MotivationRun], metric: Metric, seasons: Sequence[int], *, close: bool
) -> dict[str, Any]:
    scores = {
        r.config_id: float(
            np.average(
                r.evaluation.per_fold.set_index("season_id").loc[list(seasons), metric],
                weights=r.evaluation.per_fold.set_index("season_id").loc[list(seasons), "games"],
            )
        )
        for r in runs
    }
    winners = tuple(name for name, score in scores.items() if score == min(scores.values()))
    near = []
    if close:
        # Exact tied leaders are all references: require closeness to every leader.
        by_id = {r.config_id: r for r in runs}
        for r in runs:
            if all(
                (s := paired_statistics(by_id[w].evaluation, r.evaluation, metric, seasons))[
                    "mean_improvement"
                ]
                <= s["se"]
                for w in winners
            ):
                near.append(r.config_id)
    return {"winners": winners, "close": tuple(near), "scores": scores}


def motivation_comparisons(runs: Sequence[MotivationRun]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full-run paired statistics and review proposals only; never approve a model."""
    by_id = {r.config_id: r for r in runs}
    if (
        len(by_id) != len(runs)
        or set(by_id) != {c.config_id for c in motivation_configurations()}
        or any(
            r.stage != "full" or tuple(r.evaluation.per_fold.season_id) != VALIDATION_SEASONS
            for r in runs
        )
    ):
        raise ValueError("Decision statistics require all twenty full chronological runs")
    rows = []
    entry_flags = {}
    for run in runs:
        if run.config_id == "C0":
            continue
        stats = {}
        for scope, seasons in (("deep", DEEP_SEASONS), ("all15", VALIDATION_SEASONS)):
            for metric in ("log_loss", "rps"):
                s = paired_statistics(
                    run.evaluation,
                    by_id[run.configuration.reference_id].evaluation,
                    metric,
                    seasons,
                )
                passes = s["mean_improvement"] > s["se"] and s["last5_folds_better"] >= 4
                rows.append(
                    {
                        "config_id": run.config_id,
                        "reference": run.configuration.reference_id,
                        "scope": scope,
                        "metric": metric,
                        **s,
                        "passes_rule8": passes,
                    }
                )
                if scope == "deep":
                    stats[metric] = s
        entry_flags[run.config_id] = entry_disagreement(stats["log_loss"], stats["rps"])
    comparison = pd.DataFrame(rows)
    metrics: tuple[Metric, ...] = ("log_loss", "rps")
    ranks = {metric: _ranking(runs, metric, DEEP_SEASONS, close=True) for metric in metrics}
    recent = _ranking(runs, "log_loss", tuple(range(21, 26)), close=False)
    proposals = {}
    eligible_rankings = {}
    for metric in metrics:
        eligible_ids = set(
            comparison.loc[
                comparison.scope.eq("deep")
                & comparison.metric.eq(metric)
                & comparison.passes_rule8,
                "config_id",
            ]
        )
        eligible = [r for r in runs if r.configuration.name != "C0" and r.config_id in eligible_ids]
        ranking = (
            _ranking(eligible, metric, DEEP_SEASONS, close=True)
            if eligible
            else {"winners": (), "close": ()}
        )
        proposals[metric] = next(
            (c.config_id for c in motivation_configurations() if c.config_id in ranking["close"]),
            None,
        )
        eligible_rankings[metric] = ranking
    rule6 = set(ranks["log_loss"]["winners"]) != set(recent["winners"])
    rule10_selection = selection_disagreement(ranks["log_loss"], ranks["rps"])
    needs_review = rule6 or rule10_selection or any(entry_flags.values())
    review = {
        "rankings": ranks,
        "last5_ranking": recent,
        "rule4_proposals": proposals,
        "rule4_rankings": eligible_rankings,
        "rule6": rule6,
        "rule10_entry": entry_flags,
        "rule10_selection": rule10_selection,
        "status": "מחכה לעמית" if needs_review else "הצעה בלבד; נדרש אישור עמית",
        "selected_configuration": None,
    }
    return comparison, review


def verify_motivation_baselines(
    unweighted: MotivationRun, weighted: MotivationRun
) -> dict[str, Any]:
    """Hard reproduction gate using EXP-001 identities/provenance, never score ranking."""
    from ligat.evaluate.calibration_artifacts import verify_calibration

    if (
        unweighted.config_id != "C0"
        or weighted.config_id != "C0-hl4"
        or any(
            r.stage != "baseline" or tuple(r.evaluation.per_fold.season_id) != VALIDATION_SEASONS
            for r in (unweighted, weighted)
        )
    ):
        raise ValueError("Baseline reproduction requires both controls on all fifteen folds")
    directory = Path("reports/experiments/EXP-001/combined")
    verify_calibration(directory / "metadata.json")
    meta = json.loads(Path(INPUTS["elo_metadata"]["path"]).read_text(encoding="utf-8"))
    configurations = json.loads((directory / "configurations.json").read_text(encoding="utf-8"))
    controls = [
        c for c in configurations if c["elo"] == meta["params"] and c["half_life"] in (None, 4)
    ]
    if len(controls) != 2 or {c["half_life"] for c in controls} != {None, 4}:
        raise ValueError("EXP-001 metadata does not identify the two controls uniquely")
    stored_fits = pd.read_csv(directory / "fold_scores.csv", float_precision="round_trip")
    result = {}
    for run, target in ((unweighted, 1.020424), (weighted, 1.020373)):
        identifier = next(
            c["config_id"] for c in controls if c["half_life"] == run.configuration.half_life
        )
        if run.config_id == "C0" and identifier != meta["config_id"]:
            raise ValueError("E0050 identity does not agree with its metadata")
        prediction = pd.read_parquet(
            "data/derived/elo/exp001_predictions_combined.parquet",
            filters=[
                ("season_id", "in", list(VALIDATION_SEASONS)),
                ("config_id", "==", identifier),
            ],
        )
        value = float(run.evaluation.summary.log_loss)
        if abs(value - target) > 1e-6:
            raise RuntimeError("EXP-004 baseline log-loss reproduction failed")
        actual = run.evaluation.per_game.set_index(["season_id", "game_id"]).sort_index()
        probability_error = None
        if not prediction.empty:
            expected = prediction.set_index(["season_id", "game_id"]).sort_index()
            if (
                not expected.index.is_unique
                or not actual.index.equals(expected.index)
                or not np.array_equal(actual.outcome, expected.outcome)
            ):
                raise RuntimeError("EXP-001 reference prediction keys do not match")
            probability_error = float(
                np.max(
                    np.abs(
                        actual[["p_home", "p_draw", "p_away"]].to_numpy()
                        - expected[["p_home", "p_draw", "p_away"]].to_numpy()
                    )
                )
            )
            if probability_error > 1e-9:
                raise RuntimeError("EXP-004 baseline per-game probability reproduction failed")
        elif run.config_id == "C0":
            raise RuntimeError("Required E0050 saved predictions missing")
        fits = (
            stored_fits.loc[stored_fits.config_id.eq(identifier)]
            .set_index("season_id")
            .sort_index()
        )
        computed = run.fits.set_index("season_id").sort_index()
        if not fits.index.equals(computed.index):
            raise RuntimeError("Saved baseline fit folds do not match")
        columns = ["beta", "theta1", "delta", "theta2"]
        fit_error = float(np.max(np.abs(fits[columns].to_numpy() - computed[columns].to_numpy())))
        if fit_error > 1e-9:
            raise RuntimeError("EXP-004 baseline coefficient/threshold reproduction failed")
        result[run.config_id] = {
            "log_loss": value,
            "exp001_config_id": identifier,
            "probabilities_available": not prediction.empty,
            "max_probability_error": probability_error,
            "max_fit_error": fit_error,
            "games": len(actual),
            "folds": len(fits),
        }
    return {"passed": True, "controls": result}


def motivation_diagnostics(
    runs: Sequence[MotivationRun], prepared: MotivationExperimentInput
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B1b only: paired L3-involved losses and explicit one-side expected-points changes."""
    if any(r.stage != "full" for r in runs) or {r.config_id for r in runs} != {
        c.config_id for c in motivation_configurations()
    }:
        raise ValueError("Additional diagnostics require the complete full run")
    from ligat.models.ordered_logit import OrderedFit

    involved = set(prepared.games.loc[prepared.games.l3_involved, "game_id"])
    by_id = {r.config_id: r for r in runs}
    subset, effects = [], []
    median = float(
        prepared.games.loc[prepared.games.season_id.isin(VALIDATION_SEASONS), "elo_diff"].median()
    )
    for run in runs:
        scores = run.evaluation.per_game.loc[run.evaluation.per_game.game_id.isin(involved)]
        ref = by_id[run.configuration.reference_id].evaluation.per_game
        ref = ref.loc[ref.game_id.isin(involved)]
        for season in VALIDATION_SEASONS:
            a, b = (f.loc[f.season_id.eq(season)] for f in (scores, ref))
            subset.append(
                {
                    "config_id": run.config_id,
                    "reference": run.configuration.reference_id,
                    "season_id": season,
                    "games": len(a),
                    "log_loss": float(a.log_loss.mean()) if len(a) else None,
                    "reference_log_loss": float(b.log_loss.mean()) if len(b) else None,
                }
            )
        if run.configuration.name == "C0":
            continue
        for fit_scope, frame in (
            ("last_fold", run.fits.tail(1)),
            ("deep_mean", run.fits.loc[run.fits.season_id.isin(DEEP_SEASONS)]),
        ):
            betas = np.array([[r.beta, *r.additional_betas] for r in frame.itertuples()]).mean(
                axis=0
            )
            fit = OrderedFit(
                float(betas[0]),
                float(frame.theta1.mean()),
                float(frame.delta.mean()),
                0.0,
                True,
                0,
                False,
                0.0,
                tuple(betas[1:]),
            )
            count = 3 if run.configuration.name == "Comp" else 1
            for component in range(count):
                for side in ("home", "away"):
                    for gap in (0.0, median):
                        before = np.zeros((1, len(betas)))
                        before[0, 0] = gap / 400
                        after = before.copy()
                        if run.configuration.name.endswith("sides"):
                            before[0, 1:] = 1
                            after[0, 1:] = 1
                            after[0, 1 if side == "home" else 2] = 0
                        else:
                            after[0, component + 1] = -1 if side == "home" else 1
                        p0, p1 = (ordered_probabilities(fit, x)[0] for x in (before, after))
                        i = 0 if side == "home" else 2
                        effects.append(
                            {
                                "config_id": run.config_id,
                                "fit_scope": fit_scope,
                                "side": side,
                                "component": ("title", "cut", "survival")[component]
                                if count == 3
                                else "m",
                                "elo_diff": gap,
                                "expected_points_change": float(
                                    3 * (p1[i] - p0[i]) + p1[1] - p0[1]
                                ),
                            }
                        )
    return pd.DataFrame(subset), pd.DataFrame(effects)


def shuffled_motivation(
    motivation: pd.DataFrame, participants: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Move flags and all importance values jointly within season/stage, preserving identity."""
    identity = participants.set_index(["game_id", "team_id"]).sort_index()
    source = motivation.sort_index()
    if not source.index.is_unique or not source.index.equals(identity.index):
        raise ValueError("Shuffle needs exactly aligned team-game identities")
    groups = identity[["season_id", "stage"]].reset_index(drop=True)
    if not set(groups.season_id) <= set(DEVELOPMENT_SEASONS):
        raise ValueError("Shuffle rejects nondevelopment seasons")
    columns = [
        c
        for c in source
        if c in ("nothing_to_play", "leverage_max")
        or c.startswith(("title_", "cut_", "survival_", "survival_direct_"))
    ]
    result = source.copy(deep=True)
    permutation = np.arange(len(source))
    for indices in groups.groupby(
        ["season_id", "stage"], sort=True, observed=True
    ).indices.values():
        permutation[indices] = rng.permutation(indices)
    result[columns] = source.iloc[permutation][columns].set_axis(source.index)
    return result


def audit_decomposition(
    prepared: MotivationExperimentInput, run: MotivationRun, reference: MotivationRun
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribute paired improvement to all three dead-team groups; each scope sums exactly."""
    _paired_folds(run.evaluation, reference.evaluation)
    keys = ["season_id", "game_id"]
    a, b = (r.evaluation.per_game.set_index(keys).sort_index() for r in (run, reference))
    data = prepared.games.set_index(keys).loc[a.index].copy()
    if prepared.motivation is None:
        raise ValueError("Audit requires the frozen motivation rows")
    levels = representations.game_contrast(prepared.motivation, "L2", "sides").reindex(
        data.index.get_level_values("game_id")
    )
    home, away = levels.m_home.eq(0).to_numpy(), levels.m_away.eq(0).to_numpy()
    data["group"] = np.where(
        home ^ away, "one_dead", np.where(home & away, "both_dead", "neither_dead")
    )
    data["dead_side"] = np.select(
        [home & away, home, away], ["both", "home", "away"], default="neither"
    )
    for metric in ("log_loss", "rps"):
        data["improvement_" + metric] = b[metric] - a[metric]
    for side in ("home", "draw", "away"):
        data["reference_p_" + side] = b["p_" + side]
        data["candidate_p_" + side] = a["p_" + side]
    data["outcome"] = a.outcome
    records = []
    for scope in [*a.index.get_level_values("season_id").unique(), "all"]:
        frame = data if scope == "all" else data.xs(scope, level="season_id", drop_level=False)
        for group in ("one_dead", "both_dead", "neither_dead"):
            part = frame.loc[frame.group.eq(group)]
            records.append(
                {
                    "scope": str(scope),
                    "group": group,
                    "games": len(part),
                    "all_games": len(frame),
                    **{
                        metric + "_sum": float(part["improvement_" + metric].sum())
                        for metric in ("log_loss", "rps")
                    },
                    **{
                        metric + "_contribution": float(
                            part["improvement_" + metric].sum() / len(frame)
                        )
                        for metric in ("log_loss", "rps")
                    },
                }
            )
    return pd.DataFrame(records), data.sort_values("improvement_log_loss", ascending=False).head(
        10
    ).reset_index()


def zero_column_check(
    evaluation: EvaluationResult, reference: EvaluationResult, tolerance: float
) -> float:
    """Hard stop on a matrix-path discrepancy, without displaying losses or coefficients."""
    _paired_folds(evaluation, reference)
    keys = ["season_id", "game_id"]
    a, b = (r.per_game.set_index(keys).sort_index() for r in (evaluation, reference))
    columns = ["p_home", "p_draw", "p_away"]
    error = float(np.max(np.abs(a[columns].to_numpy() - b[columns].to_numpy())))
    if error > tolerance:
        raise RuntimeError("EXP-004 zero-column audit failed: stop without displaying scores")
    return error


def _audit_evaluation(
    games: pd.DataFrame, configuration: MotivationConfiguration, seasons: tuple[int, ...]
) -> EvaluationResult:
    result = evaluate(
        lambda: MotivationOrderedPredictor(configuration),
        games,
        feature_columns=("elo_diff", *configuration.columns),
        policy=DevelopmentPolicy(training_start=8, validation_seasons=seasons),
    )
    check_leakage(result)
    return result


def audit_motivation_timing(
    prepared: MotivationExperimentInput, seasons: tuple[int, ...]
) -> dict[str, Any]:
    """Recheck original audit E once per run, with strictly-prior tables and objective flags."""
    from ligat.simulate.rules import load_season_rules, objective_flags
    from ligat.simulate.snapshot import snapshot_as_of

    if prepared.motivation is None or prepared.participants is None or prepared.elo is None:
        raise ValueError("Timing audit requires frozen inputs and participant rows")
    motivation, participants = prepared.motivation, prepared.participants
    identity = participants.set_index(["game_id", "team_id"]).sort_index()
    prematch_path = Path("data/derived/prematch/prematch_v1.parquet")
    table_path = Path("data/derived/prematch/table_state_v2.parquet")
    filters = [("season_id", "in", list(DEVELOPMENT_SEASONS))]
    prematch = pd.read_parquet(
        prematch_path, columns=["season_id", "nothing_to_play"], filters=filters
    ).sort_index()
    frozen = pd.read_parquet(table_path, filters=filters).sort_index()
    valid = (
        motivation.index.equals(prematch.index)
        and motivation.index.equals(identity.index)
        and frozen.index.equals(identity.index)
        and motivation.nothing_to_play.eq(prematch.nothing_to_play).all()
        and motivation.as_of.eq(identity.as_of).all()
    )
    dead_rows = 0
    for (game_id, team_id), row in motivation.loc[motivation.nothing_to_play].iterrows():
        past = participants.loc[
            participants.season_id.eq(row.season_id)
            & participants.team_id.eq(team_id)
            & participants.as_of.lt(row.as_of)
        ]
        valid = (
            valid
            and int(frozen.loc[(game_id, team_id), "games_played"]) == len(past)
            and game_id not in set(past.game_id)
        )
        dead_rows += 1
    rules = load_season_rules()
    levels = representations.game_contrast(motivation, "L2", "sides")
    one = levels.m_home.ne(levels.m_away)
    relevant = prepared.games.loc[
        prepared.games.season_id.isin(seasons) & prepared.games.game_id.isin(levels.index[one])
    ]
    snapshots: dict[tuple[int, pd.Timestamp], Any] = {}
    objective_rows = 0
    for row in relevant.itertuples():
        team = int(row.home_team_id if levels.loc[row.game_id, "m_home"] == 0 else row.away_team_id)
        key = int(row.season_id), pd.Timestamp(row.date).normalize()
        if key not in snapshots:
            snapshots[key] = snapshot_as_of(participants, prepared.elo, rules, *key)
        snapshot = snapshots[key]
        stage = str(snapshot.table.loc[team, "table_group"])
        part = snapshot.table.loc[snapshot.table.table_group.eq(stage)]
        played = (
            snapshot.played.loc[snapshot.played.stage.eq(stage)]
            .groupby("team_id")
            .size()
            .reindex(part.index, fill_value=0)
        )
        flags = objective_flags(
            part, snapshot.rules, stage, snapshot.rules.stage_rounds(stage) - played
        )
        valid = valid and bool(flags.loc[team].all())
        objective_rows += 1
    if not valid:
        raise RuntimeError("EXP-004 timing audit failed: stop without displaying scores")
    return {
        "passed": True,
        "paired_rows": len(motivation),
        "flag_mismatches": 0,
        "importance_as_of_mismatches": 0,
        "dead_rows_checked": dead_rows,
        "timing_mismatches": 0,
        "one_dead_games": objective_rows,
        "snapshot_dates": len(snapshots),
        "objectives_all_decided": True,
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (prematch_path, table_path)
        },
    }


def audit_motivation_configuration(
    prepared: MotivationExperimentInput,
    run: MotivationRun,
    reference: MotivationRun,
    *,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """H-K22 automatic decomposition, zero columns, joint-row placebos, and timing."""
    if prepared.motivation is None or prepared.participants is None:
        raise ValueError("Automatic audit requires frozen team-game inputs")
    start = perf_counter()
    seasons = tuple(int(s) for s in run.evaluation.per_fold.season_id)
    decomposition, top10 = audit_decomposition(prepared, run, reference)
    zero_games = prepared.games.copy(deep=True)
    zero_games[list(run.configuration.columns)] = 0.0
    zero_result = _audit_evaluation(zero_games, run.configuration, seasons)
    error = zero_column_check(
        zero_result, reference.evaluation, float(AUDIT_SETTINGS["zero_column_tolerance"])
    )
    context = {} if cache is None else cache
    key = (prepared.input_sha256, seasons)
    if context.get("timing_key") != key:
        context["timing"] = audit_motivation_timing(prepared, seasons)
        context["timing_key"] = key
    if not context["timing"].get("passed"):
        raise RuntimeError("EXP-004 timing audit failed: stop without displaying scores")
    rng = np.random.default_rng(int(AUDIT_SETTINGS["placebo_seed"]))
    records = []
    base = reference.evaluation.per_game.set_index(["season_id", "game_id"]).sort_index()
    for iteration in range(int(AUDIT_SETTINGS["placebo_permutations"])):
        shuffled = shuffled_motivation(prepared.motivation, prepared.participants, rng)
        level, form = (
            ("Comp", "diff")
            if run.configuration.name == "Comp"
            else run.configuration.name.split("-")
        )
        contrast = representations.game_contrast(
            shuffled, level, cast(Literal["diff", "sides"], form)
        ).reindex(prepared.games.game_id)
        games = prepared.games.copy(deep=True)
        games[list(run.configuration.columns)] = contrast.to_numpy(float)
        result = _audit_evaluation(games, run.configuration, seasons)
        scored = result.per_game.set_index(["season_id", "game_id"]).sort_index()
        records.append(
            {
                "iteration": iteration + 1,
                **{
                    metric + "_improvement": float((base[metric] - scored[metric]).mean())
                    for metric in ("log_loss", "rps")
                },
            }
        )
    placebos = pd.DataFrame(records)
    actual = float(reference.evaluation.summary.log_loss - run.evaluation.summary.log_loss)
    values = placebos.log_loss_improvement.to_numpy()
    summary = {
        "iterations": len(values),
        "seed": int(AUDIT_SETTINGS["placebo_seed"]),
        "mean_improvement": float(values.mean()),
        "percentile95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "above_threshold": int((values > AUDIT_SETTINGS["improvement_above"]).sum()),
        "observed": actual,
        "below_observed": int((values < actual).sum()),
        "at_least_observed": int((values >= actual).sum()),
        "empirical_percentile": float(100 * (values < actual).mean()),
    }
    return {
        "passed": True,
        "decomposition": decomposition,
        "top10": top10,
        "zero_column_max_error": error,
        "timing": context["timing"],
        "placebos": placebos,
        "placebo_summary": summary,
        "seconds": perf_counter() - start,
    }
