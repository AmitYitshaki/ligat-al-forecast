"""Reconstruct the signed F5 folds, with no writes to the evaluation store."""

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goals_experiment import GoalUnit, load_goal_experiment
from ligat.evaluate.goals_runner import (
    _atomic_json,
    _payload_digest,
    goal_provenance,
    load_goal_inputs,
    load_goal_unit,
)
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.models.m2_xgb import XGBGoalConfig, XGBGoalModel
from ligat.viz.goals_m3_dossier import source_bundle
from ligat.viz.goals_walk_review import LABELS, folds

SEASONS = tuple(range(11, 26))


def inputs(tables: Any = None) -> tuple[Any, Any, Any, dict[int, GoalUnit]]:
    spec = load_goal_experiment()
    report = source_bundle()
    winner = report["report"]["winners"]["F5"]
    columns = tuple(report["report"]["stages"]["F5"])
    if winner["backend"] != "xgb" or winner["half_life"] != 4:
        raise ValueError("Unexpected signed F5 specification")
    units = {s: GoalUnit("xgb", winner["parameters"], columns, s, 4) for s in SEASONS}
    sources = json.loads(Path("data/reports/T-014/E3g_sources.json").read_bytes())
    ids = {
        int(row["season"]): Path(row["path"]).stem
        for row in sources
        if row["candidate"] == LABELS["F5"]
    }
    expected = {s: u.unit_id for s, u in units.items()}
    signed = dict(zip(folds(report, "F5").season_id, folds(report, "F5").unit_id, strict=True))
    if expected != ids or expected != signed:
        raise ValueError("F5 folds do not agree across the signed report and E3g")
    data = load_goal_inputs(spec, tables=tables)
    if set(data.context.season_id) != set(range(8, 26)):
        raise ValueError("Unexpected F5 input seasons")
    provenance = goal_provenance(data, prematch_version=spec.raw["inputs"]["prematch_version"])
    return spec, data, provenance, units


def reconstruct_fold(
    root: Path, unit: GoalUnit, data: Any, provenance: Any
) -> tuple[XGBGoalModel, float]:
    source = load_goal_unit(root / "full" / (unit.unit_id + ".json"), unit, provenance, data)
    model = XGBGoalModel(XGBGoalConfig(**unit.parameters))
    started = perf_counter()
    with threadpool_limits(limits=1):
        evaluation = evaluate_team_goals(
            lambda: model,
            data,
            feature_columns=unit.features,
            policy=DevelopmentPolicy(training_start=8, validation_seasons=(unit.season,)),
            half_life=unit.half_life,
            blind=True,
        )
    elapsed = perf_counter() - started
    native = source.evaluation.per_team["lambda"]
    predicted = evaluation.per_team["lambda"]
    if not predicted.index.equals(native.index):
        raise ValueError(f"Season {unit.season}: validation keys differ")
    np.testing.assert_array_equal(
        predicted.to_numpy().view(np.uint64),
        native.to_numpy().view(np.uint64),
        err_msg=f"Season {unit.season}: lambda differs bit-for-bit",
    )
    np.testing.assert_array_equal(
        np.asarray(model.training_log["train_loss"], dtype=np.float64).view(np.uint64),
        np.asarray(source.training_log[0]["train_loss"], dtype=np.float64).view(np.uint64),
        err_msg=f"Season {unit.season}: training journal differs bit-for-bit",
    )
    return model, elapsed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct() -> dict[str, Any]:
    started = perf_counter()
    spec, data, provenance, units = inputs()
    target = spec.output / "diagnostics" / "f5_explain"
    if (target / "manifest.json").exists():
        raise ValueError("F5 explanation models already published")
    verified = []
    for season, unit in units.items():
        model, elapsed = reconstruct_fold(spec.output, unit, data, provenance)
        if perf_counter() - started >= 600:
            raise TimeoutError("F5 reconstruction exceeded 10 minutes")
        verified.append((season, unit, model, elapsed))
        print(f"Season {season}: lambda and training log exact ({elapsed:.2f}s)", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for season, unit, model, elapsed in verified:
        path = target / f"season-{season}.json"
        model.save(path)
        records.append(
            dict(
                season_id=season,
                unit_id=unit.unit_id,
                model_sha256=sha256(path),
                metadata_sha256=sha256(path.with_suffix(".metadata.json")),
                bit_identical=True,
                elapsed_seconds=elapsed,
                fit_seconds=model.training_log["fit_seconds"],
            )
        )
    payload = dict(
        folds=records,
        elapsed_seconds=perf_counter() - started,
        ladder_sha256=sha256(spec.output / "full" / "ladder_report.json"),
    )
    _atomic_json(target / "manifest.json", dict(payload=payload, sha256=_payload_digest(payload)))
    return payload
