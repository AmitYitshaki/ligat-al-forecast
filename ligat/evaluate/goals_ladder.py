"""EXP-005 feature screening and full-only ladder orchestration."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.goal_diagnostics import goal_outcome_metrics
from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goal_metrics import poisson_nll
from ligat.evaluate.goal_reporting import readable_goal_metrics
from ligat.evaluate.goal_selection import paired_goal_statistics
from ligat.evaluate.goals_experiment import (
    GoalBackend,
    GoalExperimentSpec,
    GoalUnit,
    catboost_grid,
    family_features,
    xgb_stage_a,
    xgb_stage_b,
)
from ligat.evaluate.goals_runner import (
    GoalProvenance,
    GoalUnitResult,
    _atomic_json,
    _payload_digest,
    goal_provenance,
    goal_units_status,
    load_goal_inputs,
    run_goal_units,
)
from ligat.evaluate.selection import rule8
from ligat.evaluate.selection_rules import (
    close_configurations,
    leakage_check,
    metric_disagreement_entry,
    metric_disagreement_selection,
    recent_leader_flag,
)
from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation
from ligat.features import registry
from ligat.models.m2_glm import GLMGoalConfig
from ligat.models.m2_maher import MAHER_COLUMNS, MaherGoalConfig
from ligat.models.m2_maher_prior import load_promoted_prior
from ligat.models.m2_xgb import _digest


def candidate_correlations(
    features: pd.DataFrame, *, seasons: pd.Series
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Pearson on development features only; nominal categories are not ordered.

    The returned pairs exceed |r|=0.95. Nothing is removed automatically: the
    caller presents the evidence to Amit and keeps correlation_drops pending.
    """
    allowed = set(registry.M2_BASE_COLUMNS)
    allowed.update(c for group in registry.M2_FAMILY_COLUMNS.values() for c in group)
    if (
        not features.index.equals(seasons.index)
        or not features.columns.is_unique
        or not set(features) <= allowed
        or not seasons.isin(range(8, 26)).all()
    ):
        raise ValueError("Correlation screening requires aligned development-only M2 features")
    nominal = tuple(c for c in ("stage", "team_id", "opponent_id") if c in features)
    numeric = features.drop(columns=list(nominal)).select_dtypes(include=["number", "bool"])
    if set(numeric) != set(features) - set(nominal):
        raise ValueError("Non-nominal candidates must be numeric or boolean")
    correlations = numeric.corr(method="pearson")
    records = []
    for i, first in enumerate(numeric):
        for second in numeric.columns[i + 1 :]:
            value = correlations.loc[first, second]
            if np.isfinite(value) and abs(value) > 0.95:
                records.append(
                    {
                        "first": first,
                        "second": second,
                        "r": float(value),
                        "rows": int(numeric[[first, second]].notna().all(axis=1).sum()),
                    }
                )
    return pd.DataFrame(records, columns=["first", "second", "r", "rows"]), nominal


@dataclass(frozen=True)
class LadderConfiguration:
    """Backend parameters plus judge weights; weights participate in the ranking ID."""

    backend: str
    parameters: dict[str, Any]
    half_life: float | None = None

    @property
    def key(self) -> str:
        return _digest(
            {"backend": self.backend, "parameters": self.parameters, "half_life": self.half_life}
        )

    @property
    def label(self) -> str:
        label = GoalUnit(
            cast(GoalBackend, self.backend), self.parameters, ("is_home",), 11, self.half_life
        ).label
        if self.backend != "maher":
            label += (
                " · בלי דעיכה" if self.half_life is None else f" · דעיכה {self.half_life:g} עונות"
            )
        return label


ScoreProvider = Callable[[LadderConfiguration, tuple[str, ...], str], TeamGoalEvaluation]


class LadderReviewRequired(RuntimeError):
    """Rule 6 requires Amit's review before any dependent stage can continue."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is pd.NA or isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


class GoalLadder:
    """Pure full-ladder logic with an injected scored provider; pilot never invokes it."""

    def __init__(
        self,
        spec: GoalExperimentSpec,
        provider: ScoreProvider,
        root: Path,
        *,
        provenance: dict[str, Any],
    ) -> None:
        self.spec, self.provider, self.root = spec, provider, root
        self.provenance = provenance
        drops = spec.raw["correlation_drops"]
        if not isinstance(drops, list):
            raise ValueError("Ladder requires Amit's approved correlation_drops list")
        if spec.raw["fast_path"] not in ("full", "short"):
            raise ValueError("Ladder requires an explicit full/short path")
        self.drops = set(drops)
        self.cache: dict[tuple[str, tuple[str, ...]], TeamGoalEvaluation] = {}
        self.decisions: list[dict[str, Any]] = []
        self.active_dependencies: set[str] = set()
        self.choice_dependencies: dict[str, set[str]] = {}
        self.open_reviews: dict[str, dict[str, Any]] = {}
        self.used_reviews: set[str] = set()
        self.reviews = dict(spec.raw.get("review_decisions", {}))

    def _columns(self, columns: tuple[str, ...]) -> tuple[str, ...]:
        result = tuple(c for c in columns if c not in self.drops)
        if not result:
            raise ValueError("Approved drops remove every feature")
        return result

    def _evaluate(
        self, cfg: LadderConfiguration, columns: tuple[str, ...], role: str
    ) -> TeamGoalEvaluation:
        key = (cfg.key, columns)
        if key not in self.cache:
            result = self.provider(cfg, columns, role)
            check = leakage_check(result.per_fold, score="log_loss")
            self._record(
                "leakage", role + cfg.key + _digest(columns), flagged_folds=check.flagged_folds
            )
            self.cache[key] = result
        return self.cache[key]

    def _record(self, kind: str, name: str, **evidence: Any) -> None:
        evidence.setdefault("provisional_on", sorted(self.active_dependencies))
        payload = _jsonable({"kind": kind, "name": name, "provenance": self.provenance, **evidence})
        path = self.root / "decisions" / (kind + "-" + _digest({"name": name}) + ".json")
        history = ([path] if path.exists() else []) + sorted(
            path.parent.glob(path.stem + ".r*.json")
        )
        if history:
            previous = json.loads(history[-1].read_text(encoding="utf-8"))
            if previous["sha256"] != _payload_digest(previous["payload"]):
                raise ValueError("Decision signature mismatch; refusing to overwrite")
            comparable = {k: v for k, v in previous["payload"].items() if k != "previous_decision"}
            if comparable == payload:
                self.decisions.append(previous["payload"])
                return
            payload["previous_decision"] = {"path": history[-1].name, "sha256": previous["sha256"]}
            path = path.with_name(path.stem + f".r{len(history):04d}.json")
        if path.exists():
            raise ValueError("Decision revision already exists; refusing to overwrite")
        _atomic_json(path, {"payload": payload, "sha256": _payload_digest(payload)})
        self.decisions.append(payload)

    def _select(
        self, candidates: list[LadderConfiguration], columns: tuple[str, ...], name: str
    ) -> tuple[LadderConfiguration, list[LadderConfiguration]]:
        unique = {c.key: c for c in candidates}
        choice = self.reviews.get(name)
        if choice is not None:
            if choice not in unique:
                raise ValueError(f"Review choice for {name} is not a candidate")
            self.used_reviews.add(name)
        prepare = getattr(self.provider, "prepare", None)
        if prepare is not None:
            prepare(list(unique.values()), columns, name)
        evaluated = {k: self._evaluate(c, columns, name) for k, c in unique.items()}
        frames = {k: e.per_fold for k, e in evaluated.items()}
        declared_order = {k: i for i, k in enumerate(unique)}
        simplicity = sorted(
            unique,
            key=lambda k: (
                unique[k].parameters.get("max_depth", unique[k].parameters.get("depth", 0))
                if unique[k].parameters.get("depth", 0) is not None
                else evaluated[k].summary.get("actual_depth", float("inf")),
                unique[k].parameters.get("n_estimators", unique[k].parameters.get("iterations", 0))
                if unique[k].parameters.get("iterations", 0) is not None
                else evaluated[k].summary.get("actual_iterations", float("inf")),
                declared_order[k],
            ),
        )
        nll = close_configurations(frames, score="poisson_nll", simplicity_order=simplicity)
        rps = close_configurations(frames, score="rps", simplicity_order=simplicity)
        recent = recent_leader_flag(frames, score="poisson_nll")
        dependencies = set(self.active_dependencies)
        mode = self.spec.raw.get("rule6_mode", "stop")
        if recent.flag and choice is None and mode == "provisional":
            dependencies.add(name)
            self.open_reviews[name] = {
                "name": name,
                "deep_leaders": [
                    {"key": k, "label": unique[k].label} for k in recent.deep_ranking.winners
                ],
                "recent_leaders": [
                    {"key": k, "label": unique[k].label} for k in recent.last5_ranking.winners
                ],
                "proposal": nll.proposal,
            }
        self.choice_dependencies[name] = dependencies
        self._record(
            "selection",
            name,
            features=columns,
            proposal=nll.proposal,
            selected=choice or nll.proposal,
            amit_choice=choice,
            rule6_mode=mode,
            provisional_on=sorted(dependencies),
            rule4=nll,
            rule6=recent.flag,
            rule10=metric_disagreement_selection(nll, rps),
            configurations={k: asdict(v) for k, v in unique.items()},
            labels={k: v.label for k, v in unique.items()},
        )
        if recent.flag and choice is None and mode != "provisional":
            raise LadderReviewRequired("Rule 6: recent and deep leaders differ; review required")
        return unique[choice or nll.proposal], [unique[k] for k in nll.ranking.order]

    def _tune(
        self, columns: tuple[str, ...], name: str
    ) -> tuple[LadderConfiguration, list[LadderConfiguration]]:
        a = [LadderConfiguration("xgb", asdict(c)) for c in xgb_stage_a(self.spec)]
        inherited = set(self.active_dependencies)
        winner, _ = self._select(a, columns, name + "-A")
        self.active_dependencies = self.choice_dependencies[name + "-A"]
        from ligat.models.m2_xgb import XGBGoalConfig

        b = [
            LadderConfiguration("xgb", asdict(c), hl)
            for c, hl in xgb_stage_b(self.spec, XGBGoalConfig(**winner.parameters))
        ]
        result = self._select([*a, *b], columns, name + "-AB")
        self.choice_dependencies[name] = set(self.choice_dependencies[name + "-AB"])
        self.active_dependencies = inherited
        return result

    def _entry(
        self,
        cfg: LadderConfiguration,
        candidate: tuple[str, ...],
        reference: tuple[str, ...],
        name: str,
        *,
        can_enter: bool = True,
    ) -> Any:
        a, b = self._evaluate(cfg, candidate, name), self._evaluate(cfg, reference, name)
        stats = paired_goal_statistics(a, b)
        decision = rule8(stats, version=2)
        rps = paired_goal_statistics(a, b, score="rps")
        self._record(
            "entry",
            name,
            candidate=candidate,
            reference=reference,
            config=asdict(cfg),
            rule8=decision,
            entered=bool(decision.passes) and can_enter,
            mean_exceeds_se=decision.mean_exceeds_se,
            rule10=metric_disagreement_entry(stats, rps),
        )
        return decision

    def run(self) -> dict[str, Any]:
        """Tune S1/S2, audit the base, enter families, rank stages and run comparators."""
        base = self._columns(registry.M2_BASE_COLUMNS)
        s1, _ = self._tune(self._columns(registry.M2_S1_COLUMNS), "S1")
        s2, s2_order = self._tune(base, "S2")
        self.active_dependencies = set(self.choice_dependencies["S2"])
        for group, fields in registry.M2_BASE_GROUPS.items():
            removed = tuple(c for c in base if c not in fields)
            self._entry(s2, removed, base, "base-without-" + group, can_enter=False)
        current, entered = base, []
        stages = {"S1": self._columns(registry.M2_S1_COLUMNS), "S2": base}
        alternatives = []
        for family in ("F1-R", "F1-A"):
            candidate = self._columns(family_features(base, family))
            decision = self._entry(s2, candidate, base, family, can_enter=False)
            if decision.passes:
                alternatives.append((family, candidate, decision.stats.mean_improvement))
        automatic_family = (
            max(alternatives, key=lambda item: (item[2], item[0] == "F1-R"))[0]
            if alternatives
            else "none"
        )
        family_choice = self.reviews.get("F1", automatic_family)
        if "F1" in self.reviews:
            self.used_reviews.add("F1")
        if family_choice not in {"none", *(a[0] for a in alternatives)}:
            raise ValueError("F1 review must choose none or a family that passed rule 8")
        if family_choice != "none":
            family, current, _ = next(a for a in alternatives if a[0] == family_choice)
            entered.append(family)
            self._internal(s2, current, family)
        self._record(
            "family",
            "F1",
            entered=bool(entered),
            selected_family=entered[0] if entered else None,
            proposal=automatic_family,
            amit_choice=self.reviews.get("F1"),
            reference=base,
            candidate=current,
        )
        stages["F1"] = current
        for family in ("F2", "F3", "F4", "F5", "F6"):
            candidate = self._columns(family_features(current, family))
            if self._entry(s2, candidate, current, family).passes:
                current = candidate
                entered.append(family)
                self._internal(s2, current, family)
            stages[family] = current
        winners = {"S1": s1, "S2": s2}
        for name, columns in stages.items():
            if name in winners:
                continue
            self.active_dependencies = set(self.choice_dependencies["S2"])
            if self.spec.raw["fast_path"] == "short":
                winners[name], _ = self._select(s2_order[:5], columns, name + "-short")
                self.choice_dependencies[name] = set(self.choice_dependencies[name + "-short"])
            else:
                winners[name], _ = self._tune(columns, name + "-retune")
        comparators = {}
        for name, columns in (("base", base), ("selected", current)):
            self.active_dependencies = (
                set() if name == "base" else set(self.choice_dependencies["S2"])
            )
            configs = [LadderConfiguration("catboost", asdict(c)) for c in catboost_grid(self.spec)]
            winner, _ = self._select(configs, columns, "catboost-" + name)
            comparators["catboost-" + name] = {"config": asdict(winner), "features": columns}
        if "F6" in entered or self.spec.raw.get("catboost_f6_requested", False):
            self.active_dependencies = set(self.choice_dependencies["S2"])
            configs = [LadderConfiguration("catboost", asdict(c)) for c in catboost_grid(self.spec)]
            columns = self._columns(family_features(base, "F6"))
            winner, _ = self._select(configs, columns, "catboost-F6")
            comparators["catboost-F6"] = {"config": asdict(winner), "features": columns}
        self.active_dependencies = set()
        glm = [
            LadderConfiguration("glm", asdict(GLMGoalConfig("base", 9)), hl)
            for hl in self.spec.raw["glm"]["half_life"]
        ]
        winner, _ = self._select(glm, registry.M2_BASE_COLUMNS, "glm-base")
        comparators["glm"] = {"config": asdict(winner), "features": registry.M2_BASE_COLUMNS}
        frozen = load_promoted_prior(Path(self.spec.raw["inputs"]["maher_prior"]))
        maher = [
            LadderConfiguration("maher", asdict(MaherGoalConfig(hl, frozen)))
            for hl in self.spec.raw["maher"]["half_life"]
        ]
        winner, _ = self._select(maher, MAHER_COLUMNS, "maher")
        comparators["maher"] = {"config": asdict(winner), "features": MAHER_COLUMNS}
        unused = set(self.reviews) - self.used_reviews
        if unused:
            raise ValueError("Unknown review selections: " + ", ".join(sorted(unused)))
        for name, review in self.open_reviews.items():
            review["dependents"] = sorted(
                {d["name"] for d in self.decisions if name in d["provisional_on"]}
                | {choice for choice, deps in self.choice_dependencies.items() if name in deps}
            )
        return {
            "features": current,
            "entered": entered,
            "stages": stages,
            "winners": {k: asdict(v) for k, v in winners.items()},
            "top5_s2": [asdict(c) for c in s2_order[:5]],
            "comparators": comparators,
            "decisions": self.decisions,
            "open_reviews": list(self.open_reviews.values()),
            "provisional_dependencies": {k: sorted(v) for k, v in self.choice_dependencies.items()},
        }

    def _internal(self, cfg: LadderConfiguration, columns: tuple[str, ...], family: str) -> None:
        for column in registry.M2_FAMILY_COLUMNS[family]:
            if column in columns:
                self._entry(
                    cfg,
                    tuple(c for c in columns if c != column),
                    columns,
                    family + "-without-" + column,
                    can_enter=False,
                )


def _score_unit_results(results: list[GoalUnitResult], data: GoalData) -> TeamGoalEvaluation:
    """Full-only scoring of already signed blind units, using the existing metrics."""
    teams, folds = [], []
    for result in results:
        team = result.evaluation.per_team.copy(deep=True)
        fold = result.evaluation.per_fold.copy(deep=True)
        team["fold"] = result.unit.season - 10
        fold["fold"] = result.unit.season - 10
        team["goals_scored"] = data.y.reindex(team.index)
        team["poisson_nll"] = poisson_nll(team.goals_scored.to_numpy(), team["lambda"].to_numpy())
        fold["poisson_nll"] = float(team.groupby(level="game_id").poisson_nll.mean().mean())
        teams.append(team)
        folds.append(fold)
    rows, per_fold = pd.concat(teams).sort_index(), pd.concat(folds, ignore_index=True)
    summary = pd.Series(
        {
            "games": int(per_fold.games.sum()),
            "rows": len(rows),
            "folds": len(per_fold),
            "poisson_nll": float(np.average(per_fold.poisson_nll, weights=per_fold.games)),
        }
    )
    depths = [r.metadata.get("actual_params", {}).get("depth") for r in results]
    if all(d is not None for d in depths):
        summary["actual_depth"] = max(depths)
    summary["actual_iterations"] = max(log["iterations"] for r in results for log in r.training_log)
    evaluation = TeamGoalEvaluation(rows, per_fold, summary, results[0].evaluation.refit_policy)
    outcomes = goal_outcome_metrics(evaluation)
    for score in ("rps", "log_loss"):
        rows[score] = rows.index.get_level_values("game_id").map(outcomes.per_game[score])
    merged = per_fold.merge(
        outcomes.per_fold[["fold", "season_id", "rps", "log_loss"]],
        on=["fold", "season_id"],
        validate="one_to_one",
    )
    return TeamGoalEvaluation(rows, merged, summary, evaluation.refit_policy)


class SavedGoalScores:
    """Guarded full-only provider using C5 status, units, journals and index throughout."""

    def __init__(
        self, spec: GoalExperimentSpec, data: GoalData, provenance: GoalProvenance
    ) -> None:
        spec.guard_full()
        self.spec, self.data, self.provenance = spec, data, provenance
        self.cache: dict[tuple[str, tuple[str, ...]], TeamGoalEvaluation] = {}

    def prepare(
        self, candidates: list[LadderConfiguration], columns: tuple[str, ...], role: str
    ) -> None:
        """Dispatch a whole grid as one parallel batch, including all folds."""
        pending = [c for c in candidates if (c.key, columns) not in self.cache]
        if not pending:
            return
        units_by_key = {
            c.key: [
                GoalUnit(
                    cast(GoalBackend, c.backend),
                    c.parameters,
                    columns,
                    season,
                    half_life=c.half_life,
                    role=role,
                )
                for season in self.spec.seasons
            ]
            for c in pending
        }
        units = [u for group in units_by_key.values() for u in group]
        register_full_plan(self.spec.output, units, self.provenance)
        status = goal_units_status(
            units, self.data, self.provenance, self.spec.output, stage="full"
        )
        outdated = [r for r in status if r["status"] == "outdated"]
        if outdated:
            raise ValueError("Full ladder cache is outdated: " + outdated[0]["reason"])
        results = run_goal_units(
            units,
            self.data,
            self.provenance,
            self.spec.output,
            n_jobs=self.spec.n_jobs,
            stage="full",
        )
        by_id = {r.unit.unit_id: r for r in results}
        for candidate in pending:
            group = [by_id[u.unit_id] for u in units_by_key[candidate.key]]
            scored = _score_unit_results(group, self.data)
            leakage_check(scored.per_fold, score="log_loss")
            self.cache[(candidate.key, columns)] = scored

    def __call__(
        self, cfg: LadderConfiguration, columns: tuple[str, ...], role: str
    ) -> TeamGoalEvaluation:
        self.prepare([cfg], columns, role)
        return self.cache[(cfg.key, columns)]


def run_goal_full(spec: GoalExperimentSpec) -> dict[str, Any]:
    """Full entry point; approval, path and correlation gates precede data or scoring."""
    spec.guard_full()
    data = load_goal_inputs(spec)
    provenance = goal_provenance(data, prematch_version=spec.raw["inputs"]["prematch_version"])
    provider = SavedGoalScores(spec, data, provenance)
    engine = GoalLadder(spec, provider, spec.output / "full", provenance=provenance.identity())
    result = engine.run()
    result["provenance"] = asdict(provenance)
    rulers = {}
    ruler_models = {}
    for name, config in (
        ("R0", GLMGoalConfig("r0")),
        ("R1", GLMGoalConfig("r1")),
        ("EXP-002", GLMGoalConfig("ruler", 8)),
        ("EXP-003", GLMGoalConfig("ruler", 9)),
    ):
        ruler_config = LadderConfiguration("glm", asdict(config))
        rulers[name] = provider(ruler_config, config.feature_columns, name)
        ruler_models[name] = {"config": asdict(ruler_config), "features": config.feature_columns}
    reference = provider(
        LadderConfiguration(**result["winners"]["S2"]), result["stages"]["S2"], "S2"
    )
    reports = {}
    models = {
        name: {"config": config, "features": result["stages"][name]}
        for name, config in result["winners"].items()
    }
    models.update(result["comparators"])
    models.update(ruler_models)
    for name, model in models.items():
        ranked_config = LadderConfiguration(**model["config"])
        candidate = provider(ranked_config, model["features"], name)
        report = readable_goal_metrics(candidate, reference, r0=rulers["R0"], r1=rulers["R1"])
        # Comparisons require both models to have their declared minimum history.
        # Temporary masks affect reporting only; saved unit provenance is untouched.
        shared = candidate.per_team.deep_fold & reference.per_team.deep_fold
        paired = []
        for evaluation in (candidate, reference):
            rows, folds = evaluation.per_team.copy(), evaluation.per_fold.copy()
            rows["deep_fold"] = shared
            by_season = rows.groupby("season_id").deep_fold.first()
            folds["deep_fold"] = folds.season_id.map(by_season)
            paired.append(
                TeamGoalEvaluation(rows, folds, evaluation.summary, evaluation.refit_policy)
            )
        dependency_name = "glm-base" if name == "glm" else name
        dependencies = sorted(
            set(result["provisional_dependencies"].get(dependency_name, []))
            | set(result["provisional_dependencies"].get("S2", []))
        )
        reports[name] = {
            "label": ranked_config.label,
            "summary": report.summary.reset_index().to_dict(orient="records"),
            "by_season": report.by_season.reset_index().to_dict(orient="records"),
            "paired": _jsonable(paired_goal_statistics(*paired)),
            "provisional_on": dependencies,
        }
        for review in result["open_reviews"]:
            if review["name"] in dependencies:
                review["dependents"] = sorted(set(review["dependents"]) | {"metrics:" + name})
    result["readable_metrics"] = reports
    payload = cast(dict[str, Any], _jsonable(result))
    _atomic_json(
        spec.output / "full" / "ladder_report.json",
        {"payload": payload, "sha256": _payload_digest(payload)},
    )
    return payload


def register_full_plan(root: Path, units: list[GoalUnit], provenance: GoalProvenance) -> None:
    """Persist the accumulating, target-free unit plan before a full batch starts."""
    path = root / "full" / "unit_plan.json"
    previous = _read_full_plan(path, provenance) if path.exists() else []
    planned = {u.unit_id: u for u in previous}
    for unit in units:
        planned.setdefault(unit.unit_id, unit)
    payload = {
        "provenance": provenance.identity(),
        "units": [asdict(planned[k]) for k in sorted(planned)],
    }
    _atomic_json(path, {"payload": payload, "sha256": _payload_digest(payload)})


def _read_full_plan(path: Path, provenance: GoalProvenance) -> list[GoalUnit]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    payload = saved["payload"]
    if (
        saved["sha256"] != _payload_digest(payload)
        or payload["provenance"] != provenance.identity()
    ):
        raise ValueError("Full unit plan is outdated or damaged")
    return [GoalUnit(**(u | {"features": tuple(u["features"])})) for u in payload["units"]]


def exp005_full_status(spec: GoalExperimentSpec) -> dict[str, Any]:
    """Inspect saved full units without scores or training, including interrupted batches."""
    from collections import Counter

    spec.guard_full()
    data = load_goal_inputs(spec)
    provenance = goal_provenance(data, prematch_version=spec.raw["inputs"]["prematch_version"])
    path = spec.output / "full" / "unit_plan.json"
    if path.exists():
        units = _read_full_plan(path, provenance)
    else:
        units = [
            GoalUnit("xgb", asdict(c), columns, season, role=role)
            for role, columns in (
                ("S1-A", registry.M2_S1_COLUMNS),
                ("S2-A", registry.M2_BASE_COLUMNS),
            )
            for c in xgb_stage_a(spec)
            for season in spec.seasons
        ]
    statuses = goal_units_status(units, data, provenance, spec.output, stage="full")
    return {
        "stage": "full",
        "units": len(units),
        "counts": dict(Counter(r["status"] for r in statuses)),
        "unit_statuses": statuses,
        "deferred": "Data-dependent stages are registered before their batches start"
        if not path.exists()
        else None,
    }
