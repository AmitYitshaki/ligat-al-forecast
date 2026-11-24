"""Signed chronological M2 rates for M3 under an explicit training-window policy."""

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa

from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goals_experiment import GoalBackend, GoalUnit
from ligat.evaluate.goals_runner import (
    GoalProvenance,
    ModelFactory,
    _canonical,
    _payload_digest,
    goal_provenance,
    goal_units_status,
    load_goal_unit,
    run_goal_units,
)
from ligat.models.m2_xgb import _digest
from ligat.models.training_log import atomic_bytes

LAMBDA_COLUMNS = (
    "game_id",
    "team_id",
    "season_id",
    "as_of",
    "lambda",
    "train_end_season",
    "config_id",
    "prematch_version",
    "train_end_date",
    "trained_on_current_season",
)
_LAYER_FORMAT_VERSION = 2
LAMBDA_ROOT = Path("data/derived/m2_lambda_v1")
_SEASONS = tuple(range(11, 26))
_LOCAL_TIME = pd.DatetimeTZDtype(unit="ns", tz="Asia/Jerusalem")
LAMBDA_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(int, pa.Check.gt(0)),
        "team_id": pa.Column(int, pa.Check.gt(0)),
        "season_id": pa.Column(int, pa.Check.isin(_SEASONS)),
        "as_of": pa.Column(_LOCAL_TIME),
        "lambda": pa.Column(float, [pa.Check.gt(0), pa.Check(np.isfinite)]),
        "train_end_season": pa.Column(int, pa.Check.in_range(8, 24)),
        "config_id": pa.Column(str),
        "prematch_version": pa.Column(str),
        "train_end_date": pa.Column(_LOCAL_TIME),
        "trained_on_current_season": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda f: f.train_end_season.lt(f.season_id).all()),
        pa.Check(lambda f: f.train_end_date.lt(f.as_of).all()),
    ],
    strict=True,
    ordered=True,
    unique=["game_id", "team_id"],
)


@dataclass(frozen=True)
class LambdaLayerConfig:
    """The selected backend, native parameters, features and judge decay for M3."""

    backend: GoalBackend
    parameters: dict[str, Any]
    features: tuple[str, ...]
    refit_policy: Literal["as_evaluated", "prior_seasons_only"]
    half_life: float | None = None

    def __post_init__(self) -> None:
        if self.refit_policy not in ("as_evaluated", "prior_seasons_only"):
            raise ValueError("Explicit refit_policy must be as_evaluated or prior_seasons_only")
        self.units()

    def units(self) -> tuple[GoalUnit, ...]:
        """Declare fifteen blind units; as_evaluated IDs match the evaluation exactly."""
        return tuple(
            GoalUnit(
                self.backend,
                dict(self.parameters),
                self.features,
                season,
                self.half_life,
                "lambda-layer",
                prior_seasons_only=self.refit_policy == "prior_seasons_only",
            )
            for season in _SEASONS
        )

    @property
    def config_id(self) -> str:
        """Bind native settings, feature order, weights and the export history policy."""
        return "m2-lambda-" + _digest(
            {"format_version": _LAYER_FORMAT_VERSION, "configuration": asdict(self)}
        )


def validate_lambda_layer(frame: pd.DataFrame, data: GoalData) -> None:
    """Check positive rates, exact development coverage and each row's training boundary."""
    try:
        LAMBDA_SCHEMA.validate(frame)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        raise ValueError(f"Invalid lambda layer schema: {exc}") from exc
    indexed = frame.set_index(["game_id", "team_id"]).sort_index()
    expected = data.context.loc[data.context.season_id.isin(_SEASONS)].sort_index()
    if (
        not indexed.index.equals(expected.index)
        or set(indexed.season_id) != set(_SEASONS)
        or not indexed.season_id.equals(expected.season_id)
        or not indexed.as_of.eq(expected.as_of).all()
        or indexed.config_id.nunique() != 1
        or indexed.prematch_version.nunique() != 1
    ):
        raise ValueError("Lambda layer coverage, timing or configuration is inconsistent")


def _signature(config: LambdaLayerConfig, provenance: GoalProvenance) -> str:
    return _digest(
        {
            "format_version": _LAYER_FORMAT_VERSION,
            "config_id": config.config_id,
            "provenance": provenance.identity(),
            "layer_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


def _load_verified(
    config: LambdaLayerConfig, data: GoalData, provenance: GoalProvenance, root: Path
) -> pd.DataFrame:
    saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or set(saved) != {"payload", "sha256"}:
        raise ValueError("Invalid lambda manifest envelope")
    payload = saved["payload"]
    if (
        not isinstance(payload, dict)
        or saved["sha256"] != _payload_digest(payload)
        or payload.get("format_version") != _LAYER_FORMAT_VERSION
        or payload.get("signature") != _signature(config, provenance)
        or payload.get("config_id") != config.config_id
        or payload.get("config_sha256") != config.config_id.removeprefix("m2-lambda-")
        or payload.get("code_sha256") != provenance.code_sha256
        or payload.get("input_sha256") != provenance.input_sha256
        or payload.get("unit_ids") != [u.unit_id for u in config.units()]
    ):
        raise ValueError("Lambda layer signature mismatch")
    raw = (root / "lambda.parquet").read_bytes()
    if hashlib.sha256(raw).hexdigest() != payload["output_sha256"]:
        raise ValueError("Lambda parquet signature mismatch")
    frame = pd.read_parquet(io.BytesIO(raw))
    validate_lambda_layer(frame, data)
    if (
        not frame.config_id.eq(config.config_id).all()
        or not frame.prematch_version.eq(provenance.prematch_version).all()
        or len(frame) != payload["rows"]
    ):
        raise ValueError("Lambda manifest disagrees with its output")
    return frame


def lambda_layer_status(
    config: LambdaLayerConfig,
    data: GoalData,
    root: Path = LAMBDA_ROOT,
    *,
    provenance: GoalProvenance | None = None,
    evaluation_root: Path | None = None,
) -> dict[str, str]:
    """Read-only C5 status: missing, complete, or outdated; never train or overwrite."""
    provenance = provenance or goal_provenance(data)
    store, stage = _unit_store(config, root, evaluation_root)
    statuses = goal_units_status(config.units(), data, provenance, store, stage=stage)
    outdated = next((s for s in statuses if s["status"] == "outdated"), None)
    if outdated is not None:
        return {"status": "outdated", "reason": outdated["reason"]}
    table, manifest = root / "lambda.parquet", root / "manifest.json"
    if not table.exists() and not manifest.exists():
        return {"status": "missing", "reason": "Layer not published; valid units can resume"}
    try:
        _load_verified(config, data, provenance, root)
        if any(s["status"] != "complete" for s in statuses):
            raise ValueError("Lambda layer has missing training units")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "outdated", "reason": str(exc)}
    return {"status": "complete", "reason": "Verified current layer, units and journals"}


def load_lambda_layer(
    config: LambdaLayerConfig,
    data: GoalData,
    root: Path = LAMBDA_ROOT,
    *,
    provenance: GoalProvenance | None = None,
    evaluation_root: Path | None = None,
) -> pd.DataFrame:
    """Load only a verified current M3 input; reject obsolete inputs, code or settings."""
    provenance = provenance or goal_provenance(data)
    status = lambda_layer_status(
        config, data, root, provenance=provenance, evaluation_root=evaluation_root
    )
    if status["status"] != "complete":
        raise ValueError(f"Lambda layer {status['status']}: {status['reason']}")
    return _load_verified(config, data, provenance, root)


def write_lambda_layer(
    config: LambdaLayerConfig,
    data: GoalData,
    root: Path = LAMBDA_ROOT,
    *,
    n_jobs: int = 15,
    provenance: GoalProvenance | None = None,
    model_factory: ModelFactory | None = None,
    evaluation_root: Path | None = None,
) -> pd.DataFrame:
    """Export OOF goals for M3, resuming C5 units and publishing the signed manifest last.

    No selection or scoring takes place. The caller supplies the selected
    configuration. A custom model factory is a synthetic testing seam and
    requires its own explicit provenance, preventing an accidental native cache.
    Production as_evaluated exports require evaluation_root (the EXP-005 output
    root), read its completed full units, and never fit or mutate that store.
    """
    if model_factory is not None and provenance is None:
        raise ValueError("A synthetic model factory requires explicit provenance")
    if config.refit_policy == "as_evaluated" and evaluation_root is None and model_factory is None:
        raise ValueError(
            "as_evaluated requires evaluation_root to reuse completed evaluation units"
        )
    provenance = provenance or goal_provenance(data)
    store, stage = _unit_store(config, root, evaluation_root)
    status = lambda_layer_status(
        config, data, root, provenance=provenance, evaluation_root=evaluation_root
    )
    if status["status"] == "outdated":
        raise ValueError(f"Lambda layer outdated: {status['reason']}; refusing to overwrite")
    if status["status"] == "complete":
        return load_lambda_layer(
            config, data, root, provenance=provenance, evaluation_root=evaluation_root
        )
    if evaluation_root is not None:
        statuses = goal_units_status(config.units(), data, provenance, store, stage=stage)
        if any(s["status"] != "complete" for s in statuses):
            raise ValueError("Export requires completed evaluation units; it never trains into D3")
        results = [
            load_goal_unit(store / stage / (u.unit_id + ".json"), u, provenance, data)
            for u in config.units()
        ]
    else:
        results = run_goal_units(
            config.units(),
            data,
            provenance,
            store,
            n_jobs=n_jobs,
            stage=stage,
            model_factory=model_factory,
        )
    rows = pd.concat([r.evaluation.per_team for r in results]).reset_index()
    season_ends = data.context.groupby("season_id").date.max().to_dict()
    rows["train_end_season"] = [
        max(s for s in history if s < season and season_ends[s] < as_of)
        for history, season, as_of in zip(
            rows.train_seasons, rows.season_id, rows.as_of, strict=True
        )
    ]
    rows["trained_on_current_season"] = [
        season in history
        for season, history in zip(rows.season_id, rows.train_seasons, strict=True)
    ]
    rows["train_end_season"] = rows.train_end_season.astype("int64")
    rows["train_end_date"] = rows.train_max_date
    rows["config_id"] = config.config_id
    rows["prematch_version"] = provenance.prematch_version
    frame = rows.loc[:, LAMBDA_COLUMNS].sort_values(["game_id", "team_id"]).reset_index(drop=True)
    for key in ("game_id", "team_id", "season_id"):
        frame[key] = frame[key].astype("int64")
    for key in ("as_of", "train_end_date"):
        frame[key] = frame[key].astype(_LOCAL_TIME)
    validate_lambda_layer(frame, data)
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    raw = buffer.getvalue()
    # Validate the actual bytes that will be published, including timezone/dtypes.
    validate_lambda_layer(pd.read_parquet(io.BytesIO(raw)), data)
    payload = {
        "format_version": _LAYER_FORMAT_VERSION,
        "signature": _signature(config, provenance),
        "configuration": asdict(config),
        "config_id": config.config_id,
        "config_sha256": config.config_id.removeprefix("m2-lambda-"),
        "code_sha256": provenance.code_sha256,
        "native_config_id": config.units()[0].configuration().config_id,
        "provenance": asdict(provenance),
        "input_sha256": provenance.input_sha256,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "unit_ids": [u.unit_id for u in config.units()],
        "rows": len(frame),
    }
    atomic_bytes(root / "lambda.parquet", raw)
    atomic_bytes(
        root / "manifest.json",
        (_canonical({"payload": payload, "sha256": _payload_digest(payload)}) + "\n").encode(),
    )
    return frame


def _unit_store(
    config: LambdaLayerConfig, root: Path, evaluation_root: Path | None
) -> tuple[Path, str]:
    if evaluation_root is not None:
        if config.refit_policy != "as_evaluated":
            raise ValueError("Evaluation-unit reuse requires as_evaluated")
        return evaluation_root, "full"
    return root / "units", "lambda"


def join_lambda_games(frame: pd.DataFrame, data: GoalData) -> pd.DataFrame:
    """Pair each fixture's two validated team rates into lambda_home and lambda_away."""
    validate_lambda_layer(frame, data)
    indexed = frame.set_index(["game_id", "team_id"])
    indexed["is_home"] = data.context.is_home.reindex(indexed.index)
    home = indexed.loc[indexed.is_home].reset_index().set_index("game_id")
    away = indexed.loc[~indexed.is_home].reset_index().set_index("game_id")
    result = home.loc[:, ["season_id", "as_of", "config_id", "prematch_version"]].copy()
    result["lambda_home"] = home["lambda"]
    result["lambda_away"] = away["lambda"].reindex(result.index)
    if result.isna().any().any() or not result.index.is_unique:
        raise ValueError("Lambda game pairing must be complete and unique")
    return result.sort_index()
