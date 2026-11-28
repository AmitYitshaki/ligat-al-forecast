"""Publish and verify the three approved, already evaluated M2 lambda layers."""

import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.goal_diagnostics import outcome_probabilities, poisson_score_matrix
from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goal_metrics import poisson_nll
from ligat.evaluate.goals_experiment import GoalBackend, GoalExperimentSpec
from ligat.evaluate.goals_runner import (
    GoalProvenance,
    _atomic_json,
    _payload_digest,
    goal_provenance,
    goal_units_status,
    load_goal_unit,
)
from ligat.evaluate.lambda_layer import (
    LAMBDA_ROOT,
    LambdaLayerConfig,
    join_lambda_games,
    lambda_layer_status,
    load_lambda_layer,
    write_lambda_layer,
)
from ligat.evaluate.metrics import log_loss
from ligat.models.training_log import atomic_bytes

MODELS = {"f5": "F5", "glm_base": "glm", "maher": "maher"}
DISPLAY = {
    "f5": ("F5", "M2 שנבחר"),
    "glm_base": ("GLM בבסיס", "מועמד ל-M3"),
    "maher": ("Maher", "מועמד ל-M3"),
}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _signed(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_bytes())
    payload = cast(dict[str, Any], envelope["payload"])
    if set(envelope) != {"payload", "sha256"} or envelope["sha256"] != _payload_digest(payload):
        raise ValueError(f"Invalid manifest signature: {path}")
    return payload


def signed_configurations(report: dict[str, Any]) -> dict[str, LambdaLayerConfig]:
    """Use exact native parameters from the frozen report, including Maher's prior."""
    configs = {}
    for key, name in MODELS.items():
        if name == "F5":
            raw, features = report["winners"][name], report["stages"][name]
        else:
            comparator = report["comparators"][name]
            raw, features = comparator["config"], comparator["features"]
        configs[key] = LambdaLayerConfig(
            cast(GoalBackend, raw["backend"]),
            raw["parameters"],
            tuple(features),
            "as_evaluated",
            half_life=raw["half_life"],
        )
    return configs


def layer_data(data: GoalData) -> GoalData:
    """Give the legacy layer validator its exact physical season-id dtype."""
    return GoalData(data.X, data.y, data.context.astype({"season_id": "int64"}))


def _sources(spec: GoalExperimentSpec, data: GoalData) -> tuple[Any, ...]:
    root = spec.output / "full"
    report_path = root / "ladder_report.json"
    report = _signed(report_path)
    report_sha = _digest(report_path.read_bytes())
    validation_path = root / "validation_views.parquet"
    validation_manifest = _signed(root / "validation_views.manifest.json")
    validation = validation_path.read_bytes()
    if (
        validation_manifest["output_sha256"] != _digest(validation)
        or validation_manifest["binding"]["report_sha256"] != report_sha
    ):
        raise ValueError("Validation scores are not bound to the signed ladder")
    provenance = goal_provenance(data, prematch_version=spec.raw["inputs"]["prematch_version"])
    return report, report_sha, pd.read_parquet(io.BytesIO(validation)), provenance


def _native_folds(
    evaluation: Path, config: LambdaLayerConfig, provenance: GoalProvenance, data: GoalData
) -> tuple[Any, Any]:
    results = [
        load_goal_unit(evaluation / "full" / (u.unit_id + ".json"), u, provenance, data)
        for u in config.units()
    ]
    rows = pd.concat([r.evaluation.per_team for r in results]).sort_index()
    folds = pd.concat([r.evaluation.per_fold for r in results], ignore_index=True)
    return rows, folds[["season_id", "train_seasons", "deep_fold"]]


def _verified_scores(
    layer: Any, native: Any, folds: Any, validation: Any, data: GoalData, summary: list[Any]
) -> dict[str, float]:
    rates = layer.set_index(["game_id", "team_id"]).sort_index()
    if not rates.index.equals(native.index):
        raise ValueError("Exported keys differ from signed native keys")
    np.testing.assert_array_equal(rates["lambda"], native["lambda"])
    goals = data.y.reindex(rates.index)
    scores = rates.assign(nll=poisson_nll(goals, rates["lambda"]))
    seasonal = [
        float(part.groupby(level="game_id").nll.mean().mean())
        for _, part in scores.groupby("season_id", sort=True)
    ]
    np.testing.assert_array_equal(seasonal, validation.poisson_nll.to_numpy())
    deep = set(folds.loc[folds.deep_fold, "season_id"])
    scopes = {
        "all": scores,
        "deep": scores.loc[scores.season_id.isin(deep)],
        "last5": scores.loc[scores.season_id.between(21, 25)],
    }
    nll = {key: float(value.nll.mean()) for key, value in scopes.items()}
    for record in summary:
        if nll[record["scope"]] != record["poisson_nll"]:
            raise ValueError("Exported summary NLL differs from signed ladder")
    paired = join_lambda_games(layer, data)
    if len(layer) != 7442 or len(paired) != 3721 or set(paired.season_id) != set(range(11, 26)):
        raise ValueError("Unexpected lambda layer coverage")
    home = data.context.is_home
    actual_h = goals.reindex(home.loc[home].index).droplevel("team_id").reindex(paired.index)
    actual_a = goals.reindex(home.loc[~home].index).droplevel("team_id").reindex(paired.index)
    actual = np.where(actual_h > actual_a, 0, np.where(actual_h == actual_a, 1, 2))
    probabilities = outcome_probabilities(
        poisson_score_matrix(paired.lambda_home, paired.lambda_away)
    )
    if float(log_loss(actual, probabilities).mean()) < 0.98:
        raise ValueError("H/D/A leakage threshold crossed in published layer")
    return nll


def publish(spec: GoalExperimentSpec, data: GoalData, root: Path = LAMBDA_ROOT) -> dict[str, Any]:
    """Preflight all 45 units before writing any lambda bundle files."""
    report, report_sha, validation, provenance = _sources(spec, data)
    data = layer_data(data)
    configs = signed_configurations(report)
    for key, config in configs.items():
        status = lambda_layer_status(
            config, data, root / key, provenance=provenance, evaluation_root=spec.output
        )
        unit_statuses = goal_units_status(
            config.units(), data, provenance, spec.output, stage="full"
        )
        if status["status"] == "outdated" or any(u["status"] != "complete" for u in unit_statuses):
            raise ValueError(f"{key}: all 15 native units must be complete before export")
    records = {}
    for key, config in configs.items():
        units = config.units()
        unit_ids = [u.unit_id for u in units]
        signed_scores = validation.loc[validation.unit_id.isin(unit_ids)].sort_values("fold")
        if signed_scores.unit_id.tolist() != unit_ids:
            raise ValueError(f"Missing signed seasonal NLL for {key}")
        native, folds = _native_folds(spec.output, config, provenance, data)
        thin = (11, 12) if key == "glm_base" else (11,)
        if tuple(folds.loc[~folds.deep_fold, "season_id"]) != thin:
            raise ValueError(f"Unexpected thin folds for {key}")
        path = root / key
        layer = write_lambda_layer(
            config, data, path, provenance=provenance, evaluation_root=spec.output
        )
        loaded = load_lambda_layer(
            config, data, path, provenance=provenance, evaluation_root=spec.output
        )
        pd.testing.assert_frame_equal(layer, loaded)
        scores = _verified_scores(
            loaded,
            native,
            folds,
            signed_scores,
            data,
            report["readable_metrics"][MODELS[key]]["summary"],
        )
        stream = io.BytesIO()
        folds.to_parquet(stream, index=False)
        raw = stream.getvalue()
        fold_path = path / "folds.parquet"
        if fold_path.exists() and fold_path.read_bytes() != raw:
            raise ValueError(f"Refusing to replace a conflicting folds file for {key}")
        if not fold_path.exists():
            atomic_bytes(fold_path, raw)
        name, role = DISPLAY[key]
        records[key] = dict(
            name=name,
            role=role,
            native_config_id=units[0].configuration().config_id,
            unit_ids=unit_ids,
            config_id=config.config_id,
            lambda_sha256=_digest((path / "lambda.parquet").read_bytes()),
            folds_sha256=_digest(raw),
            rows=len(layer),
            games=layer.game_id.nunique(),
            seasons=len(folds),
            deep_seasons=int(folds.deep_fold.sum()),
            nll=scores,
        )
    payload = dict(format_version=1, ladder_sha256=report_sha, models=records)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if _signed(manifest_path) != payload:
            raise ValueError("A conflicting lambda bundle is already published")
    else:
        _atomic_json(manifest_path, dict(payload=payload, sha256=_payload_digest(payload)))
    return payload


def load_bundle_summary(
    spec: GoalExperimentSpec, data: GoalData, root: Path = LAMBDA_ROOT
) -> dict[str, Any]:
    """Read and recheck the finished bundle and the hashes of both files per model."""
    manifest = _signed(root / "manifest.json")
    report, report_sha, validation, provenance = _sources(spec, data)
    data = layer_data(data)
    if manifest["ladder_sha256"] != report_sha or set(manifest["models"]) != set(MODELS):
        raise ValueError("Unexpected lambda bundle sources or model list")
    for key, config in signed_configurations(report).items():
        record = manifest["models"][key]
        path = root / key
        raw = (path / "folds.parquet").read_bytes()
        if (
            _digest(raw) != record["folds_sha256"]
            or _digest((path / "lambda.parquet").read_bytes()) != record["lambda_sha256"]
            or record["config_id"] != config.config_id
            or record["unit_ids"] != [u.unit_id for u in config.units()]
        ):
            raise ValueError(f"Bundle signature mismatch: {key}")
        layer = load_lambda_layer(
            config, data, path, provenance=provenance, evaluation_root=spec.output
        )
        native, folds = _native_folds(spec.output, config, provenance, data)
        saved_folds = pd.read_parquet(io.BytesIO(raw))
        saved_folds["train_seasons"] = saved_folds.train_seasons.map(tuple)
        pd.testing.assert_frame_equal(folds, saved_folds)
        if (
            record["rows"] != len(layer)
            or record["games"] != layer.game_id.nunique()
            or record["seasons"] != len(folds)
            or record["deep_seasons"] != int(folds.deep_fold.sum())
        ):
            raise ValueError(f"Bundle coverage mismatch: {key}")
        signed = validation.loc[validation.unit_id.isin(record["unit_ids"])].sort_values("fold")
        scores = _verified_scores(
            layer, native, folds, signed, data, report["readable_metrics"][MODELS[key]]["summary"]
        )
        if scores != record["nll"]:
            raise ValueError(f"Bundle summary score mismatch: {key}")
    return manifest
