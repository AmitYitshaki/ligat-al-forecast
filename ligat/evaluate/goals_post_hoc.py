"""Post-hoc EXP-005 diagnostics; all native models and D3 choices remain frozen."""

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any, cast

from ligat.evaluate.goal_reporting import readable_goal_metrics
from ligat.evaluate.goals_experiment import GoalExperimentSpec, GoalUnit
from ligat.evaluate.goals_ladder import _jsonable, _score_unit_results
from ligat.evaluate.goals_runner import (
    _payload_digest,
    load_goal_unit,
    rebuild_goal_index,
    run_goal_units,
)
from ligat.evaluate.selection import paired_statistics
from ligat.evaluate.selection_rules import leakage_check
from ligat.features import registry
from ligat.models.m2_catboost import CatBoostGoalConfig
from ligat.models.m2_xgb import XGBGoalConfig
from ligat.models.training_log import atomic_bytes
from ligat.viz.goal_training_curves import _sources, frozen_full_report


def diagnostic_plan(spec: GoalExperimentSpec, report: dict[str, Any]) -> list[tuple[str, GoalUnit]]:
    selected = report["winners"]["S2"]
    cfg = XGBGoalConfig(**selected["parameters"])
    base = tuple(c for c in registry.M2_BASE_COLUMNS if c not in spec.raw["correlation_drops"])
    all_columns = tuple(
        dict.fromkeys(
            base
            + tuple(
                c
                for name in ["F1-A", "F2", "F3", "F4", "F5"]
                for c in registry.M2_FAMILY_COLUMNS[name]
                if c not in spec.raw["correlation_drops"]
            )
        )
    )
    configurations: list[tuple[str, Any, tuple[str, ...]]] = [
        ("P1", cfg, tuple(c for c in all_columns if c not in ("elo_pre", "elo_diff"))),
        ("P2", cfg, all_columns),
    ]
    for trees, rate in product(
        spec.raw["xgboost"]["stage_a"]["n_estimators"],
        spec.raw["xgboost"]["stage_a"]["learning_rate"],
    ):
        configurations.append(
            (
                "P3",
                replace(
                    cfg,
                    max_depth=1,
                    min_child_weight=30,
                    n_estimators=trees,
                    learning_rate=rate,
                    preset="grid",
                ),
                base,
            )
        )
    for caution in [50, 100]:
        configurations.append(
            (
                "P3",
                replace(
                    cfg,
                    max_depth=2,
                    n_estimators=150,
                    learning_rate=0.05,
                    min_child_weight=caution,
                    preset="grid",
                ),
                base,
            )
        )
    cat = spec.raw["catboost"]
    for trees, penalty in product([150, 400], [3, 10]):
        configurations.append(
            (
                "P4",
                CatBoostGoalConfig(
                    depth=2,
                    iterations=trees,
                    learning_rate=cat["learning_rate"],
                    l2_leaf_reg=penalty,
                    one_hot_max_size=cat["one_hot_max_size"],
                ),
                base,
            )
        )
    return [
        (
            name,
            GoalUnit(
                "catboost" if name == "P4" else "xgb",
                asdict(config),
                columns,
                season,
                half_life=selected["half_life"],
                role=name,
            ),
        )
        for name, config, columns in configurations
        for season in spec.seasons
    ]


def _signature() -> str:
    return _payload_digest(
        {
            f.__name__: inspect.getsource(cast(Callable[..., Any], f))
            for f in [
                diagnostic_plan,
                diagnostic_units,
                _signature,
                run_post_hoc,
                _score_unit_results,
                readable_goal_metrics,
                paired_statistics,
                leakage_check,
            ]
        }
    )


def diagnostic_units(
    units: list[GoalUnit],
    data: Any,
    provenance: Any,
    target: Path,
    binding: dict[str, Any],
    *,
    n_jobs: int = 1,
    model_factory: Any = None,
) -> list[Any]:
    for u in units:
        path = target / "synthetic" / (u.unit_id + ".json")
        if path.exists():
            p = json.loads(path.read_bytes())["payload"]
            if "diagnostic_binding" in p and p["diagnostic_binding"] != binding:
                raise ValueError("Post-hoc cache is outdated")
    results = run_goal_units(
        units,
        data,
        provenance,
        target,
        n_jobs=n_jobs,
        stage="synthetic",
        model_factory=model_factory,
    )
    for u in units:
        path = target / "synthetic" / (u.unit_id + ".json")
        saved = json.loads(path.read_bytes())
        p = saved["payload"]
        if not p.get("diagnostic"):
            p.update(diagnostic=True, diagnostic_binding=binding)
            atomic_bytes(path, json.dumps(dict(payload=p, sha256=_payload_digest(p))).encode())

    rebuild_goal_index(target, stage="synthetic")
    return results


def run_post_hoc(spec: GoalExperimentSpec, *, n_jobs: int = 10) -> dict[str, Any]:
    report = frozen_full_report(spec)
    data, provenance, native, _ = _sources(spec, "full")
    roots = [spec.output / "full", spec.output / "diagnostics" / "learning_curves"]
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for r in roots
        for p in r.rglob("*")
        if p.is_file()
    }
    plan = diagnostic_plan(spec, report)
    target = spec.output / "diagnostics" / "post_hoc"
    binding = dict(
        diagnostic=True,
        code_sha256=_signature(),
        report_sha256=hashlib.sha256(
            (spec.output / "full" / "ladder_report.json").read_bytes()
        ).hexdigest(),
        provenance=provenance.identity(),
    )
    results = diagnostic_units(
        [u for _, u in plan], data, provenance, target, binding, n_jobs=n_jobs
    )
    by_id = {r.unit.unit_id: r for r in results}

    def baseline(name: str) -> Any:
        if name == "S2":
            config = report["winners"]["S2"]
            columns = report["stages"]["S2"]
            group = [
                u
                for u in native
                if u.backend == config["backend"]
                and u.parameters == config["parameters"]
                and u.half_life == config["half_life"]
                and u.features == tuple(columns)
            ]
        else:
            group = [u for u in native if u.role == name]
        assert len(group) == len(spec.seasons), (name, len(group))
        return _score_unit_results(
            [
                load_goal_unit(spec.output / "full" / (u.unit_id + ".json"), u, provenance, data)
                for u in group
            ],
            data,
        )

    baselines = {name: baseline(name) for name in ["S2", "EXP-002", "R0", "R1"]}
    groups: dict[tuple[str, str], list[GoalUnit]] = {}
    for name, u in plan:
        groups.setdefault((name, u.configuration().config_id), []).append(u)
    records = []
    for (name, key), units in groups.items():
        evaluation = _score_unit_results([by_id[u.unit_id] for u in units], data)
        leakage_check(evaluation.per_fold, score="log_loss")
        metrics = readable_goal_metrics(
            evaluation, baselines["S2"], r0=baselines["R0"], r1=baselines["R1"]
        )
        paired = {}
        for ref in ["S2", "EXP-002"]:
            s = paired_statistics(evaluation.per_fold, baselines[ref].per_fold, score="poisson_nll")
            paired[ref] = asdict(s)
            paired[ref]["differences"] = s.differences.to_dict("records")
        folds = evaluation.per_fold
        summary_rows = metrics.summary.reset_index().to_dict("records")
        for row in summary_rows:
            selected = (
                folds
                if row["scope"] == "all"
                else folds.loc[folds.deep_fold]
                if row["scope"] == "deep"
                else folds.loc[folds.season_id.ge(21)]
            )
            row["log_loss"] = float(
                __import__("numpy").average(selected.log_loss, weights=selected.games)
            )
        records.append(
            dict(
                check=name,
                key=key,
                label=units[0].label,
                parameters=units[0].parameters,
                features=units[0].features,
                half_life=units[0].half_life,
                summary=summary_rows,
                paired=paired,
                folds=folds.to_dict("records"),
            )
        )
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items())
    assert set(before) == {p for r in roots for p in r.rglob("*") if p.is_file()}
    hda = [float(f["log_loss"]) for record in records for f in record["folds"]]
    summary = _jsonable(
        dict(
            binding=binding,
            records=records,
            units=len(plan),
            frozen_files=len(before),
            frozen_files_unchanged=True,
            minimum_fold_hda_log_loss=min(hda),
            folds_below_one=sum(v < 1 for v in hda),
        )
    )
    summary["unit_files"] = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (target / "synthetic").iterdir()
        if p.is_file()
    }
    atomic_bytes(
        target / "report.json",
        json.dumps(dict(payload=summary, sha256=_payload_digest(summary)), default=str).encode(),
    )
    return dict(summary)


def load_post_hoc(spec: GoalExperimentSpec) -> dict[str, Any]:
    saved = json.loads((spec.output / "diagnostics" / "post_hoc" / "report.json").read_bytes())
    if saved["sha256"] != _payload_digest(saved["payload"]):
        raise ValueError("Invalid diagnostic report")
    if saved["payload"]["binding"]["code_sha256"] != _signature():
        raise ValueError("Diagnostic code changed")
    frozen_full_report(spec)
    target = spec.output / "diagnostics" / "post_hoc"
    if (
        saved["payload"]["binding"]["report_sha256"]
        != hashlib.sha256((spec.output / "full" / "ladder_report.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Frozen report changed")
    if any(
        hashlib.sha256((target / name).read_bytes()).hexdigest() != sha
        for name, sha in saved["payload"]["unit_files"].items()
    ):
        raise ValueError("Diagnostic source changed")
    return dict(saved["payload"])
