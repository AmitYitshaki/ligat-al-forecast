"""Blind EXP-005 units: deterministic workers, atomic signed files and exact resume."""

import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import psutil  # type: ignore[import-untyped]
from joblib import Parallel, delayed, parallel_config  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from ligat.analysis.prepare import AnalysisTables, load_analysis_tables
from ligat.data.repository import Repository
from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import GoalData, prepare_goal_data
from ligat.evaluate.goals_experiment import GoalExperimentSpec, GoalUnit, pilot_units
from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation, evaluate_team_goals
from ligat.features import registry
from ligat.features.prematch_v2 import validate_prematch_v2_frame
from ligat.models.goal_model import TeamGoalModel
from ligat.models.m2_catboost import CatBoostGoalModel
from ligat.models.m2_glm import GLMGoalModel
from ligat.models.m2_identity import IDENTITY_COLUMNS
from ligat.models.m2_maher import MaherGoalModel
from ligat.models.m2_maher_prior import season_roster
from ligat.models.m2_xgb import XGBGoalModel, _digest
from ligat.models.training_log import TrainingJournal, atomic_bytes, journal_bytes

ModelFactory = Callable[[GoalUnit], TeamGoalModel]
_FORBIDDEN = {"goals_scored", "poisson_nll", "rps", "log_loss", "brier", "accuracy"}


@dataclass(frozen=True)
class GoalProvenance:
    input_sha256: str
    code_sha256: str
    git_commit: str
    prematch_version: str = "prematch_v2"

    def identity(self) -> dict[str, str]:
        return {
            "input_sha256": self.input_sha256,
            "code_sha256": self.code_sha256,
            "prematch_version": self.prematch_version,
        }


@dataclass(frozen=True)
class GoalUnitResult:
    unit: GoalUnit
    evaluation: TeamGoalEvaluation
    elapsed_seconds: float
    fit_seconds: float
    peak_rss_mb: float
    metadata: dict[str, Any]
    cached: bool
    training_log: list[dict[str, Any]]


class PilotInterrupted(RuntimeError):
    """Intentional, recoverable stop; already completed signed units remain reusable."""


def goal_data_fingerprint(data: GoalData) -> str:
    parts = {}
    for name, frame in (("X", data.X), ("y", data.y), ("context", data.context)):
        values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8").tobytes()
        parts[name] = hashlib.sha256(values).hexdigest()
    parts["columns"] = _digest({"X": list(data.X.columns), "context": list(data.context.columns)})
    return _digest(parts)


def goal_provenance(data: GoalData, *, prematch_version: str = "prematch_v2") -> GoalProvenance:
    root = Path(__file__).resolve().parents[2]
    sources = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "ligat").rglob("*.py"))
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root).strip()
    libraries = {
        name: version(name)
        for name in ("numpy", "pandas", "scipy", "xgboost", "catboost", "joblib", "threadpoolctl")
    }
    return GoalProvenance(
        goal_data_fingerprint(data),
        _digest({"sources": sources, "libraries": libraries}),
        commit,
        prematch_version,
    )


def load_goal_inputs(spec: GoalExperimentSpec, *, tables: AnalysisTables | None = None) -> GoalData:
    """Verify frozen M0 bytes; explicitly load development 8-25 only."""
    path = Path(spec.raw["inputs"]["prematch"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec.raw["inputs"]["prematch_sha256"]:
        raise ValueError("EXP-005 frozen prematch input signature mismatch")
    seasons = tuple(range(8, 26))
    prematch = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    validate_prematch_v2_frame(prematch)
    if tables is None:
        tables = load_analysis_tables(Repository(), seasons=list(seasons))
    columns = tuple(
        dict.fromkeys(
            (
                *registry.M2_BASE_COLUMNS,
                *(c for family in registry.M2_FAMILY_COLUMNS.values() for c in family),
            )
        )
    )
    return prepare_goal_data(prematch, tables.match_table, feature_columns=columns, seasons=seasons)


def _model(unit: GoalUnit, data: GoalData) -> TeamGoalModel:
    from ligat.models.m2_catboost import CatBoostGoalConfig
    from ligat.models.m2_glm import GLMGoalConfig
    from ligat.models.m2_maher import MaherGoalConfig
    from ligat.models.m2_xgb import XGBGoalConfig

    configuration = unit.configuration()
    ids = (
        tuple(sorted(int(v) for v in data.context.team_id.unique()))
        if set(IDENTITY_COLUMNS) & set(unit.features)
        else None
    )
    if isinstance(configuration, XGBGoalConfig):
        return XGBGoalModel(
            configuration,
            team_ids=ids,
            identity_columns=tuple(c for c in IDENTITY_COLUMNS if c in unit.features)
            if ids is not None
            else IDENTITY_COLUMNS,
        )
    if isinstance(configuration, CatBoostGoalConfig):
        return CatBoostGoalModel(
            configuration,
            team_ids=ids,
            identity_columns=tuple(c for c in IDENTITY_COLUMNS if c in unit.features)
            if ids is not None
            else IDENTITY_COLUMNS,
        )
    if isinstance(configuration, GLMGoalConfig):
        return GLMGoalModel(configuration)
    if isinstance(configuration, MaherGoalConfig):
        return MaherGoalModel(configuration, roster=season_roster(data))
    raise ValueError("Unknown goal configuration")


def _signature(unit: GoalUnit, provenance: GoalProvenance) -> str:
    return _digest(
        {"format_version": 2, "unit": unit.identity(), "provenance": provenance.identity()}
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported goal artifact value: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(
        value, default=_json_default, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish only complete JSON; orphan temporary files are never completed units."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _path(root: Path, stage: str, unit: GoalUnit) -> Path:
    if stage not in ("pilot", "full", "synthetic", "lambda"):
        raise ValueError("Invalid goal artifact stage")
    return root / stage / (unit.unit_id + ".json")


def load_goal_unit(
    path: Path, unit: GoalUnit, provenance: GoalProvenance, data: GoalData
) -> GoalUnitResult:
    """Reject mismatched content or provenance; never replace a conflicting cache entry."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or set(saved) != {"payload", "sha256"}:
        raise ValueError("Invalid goal unit envelope")
    payload = saved["payload"]
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("Unsupported goal unit format")
    if saved["sha256"] != _payload_digest(payload) or payload["signature"] != _signature(
        unit, provenance
    ):
        raise ValueError("Goal unit signature mismatch; refusing to overwrite existing result")
    logs = _read_journal(path, payload)
    rows = pd.DataFrame(payload["per_team"]).loc[:, payload["per_team_columns"]]
    if _FORBIDDEN & set(rows):
        raise ValueError("Blind goal unit contains forbidden targets or scores")
    for c in ("date", "as_of", "train_max_date"):
        rows[c] = pd.to_datetime(rows[c], utc=True).dt.tz_convert("Asia/Jerusalem")
    rows["train_seasons"] = rows.train_seasons.map(tuple)
    rows = rows.set_index(["game_id", "team_id"]).sort_index()
    expected = data.X.index[data.context.season_id.eq(unit.season)].sort_values()
    if (
        not rows.index.equals(expected)
        or not rows.index.is_unique
        or not np.isfinite(rows["lambda"]).all()
        or rows["lambda"].le(0).any()
    ):
        raise ValueError("Goal cached unit has invalid coverage or lambdas")
    folds = pd.DataFrame(payload["per_fold"]).loc[:, payload["per_fold_columns"]]
    if _FORBIDDEN & set(folds):
        raise ValueError("Blind folds contain forbidden scores")
    folds["train_seasons"] = folds.train_seasons.map(tuple)
    summary = pd.Series(payload["summary"]).reindex(payload["summary_index"])
    if _FORBIDDEN & set(summary.index):
        raise ValueError("Blind summary contains forbidden scores")
    evaluation = TeamGoalEvaluation(rows, folds, summary, payload["refit_policy"])
    return GoalUnitResult(
        unit,
        evaluation,
        payload["elapsed_seconds"],
        payload["fit_seconds"],
        payload["peak_rss_mb"],
        payload["metadata"],
        True,
        logs,
    )


def run_goal_unit(
    unit: GoalUnit,
    data: GoalData,
    provenance: GoalProvenance,
    root: Path,
    *,
    stage: str = "pilot",
    model_factory: ModelFactory | None = None,
) -> GoalUnitResult:
    """Evaluate one fold through the blind judge and commit one atomic signed JSON file."""
    path = _path(root, stage, unit)
    if path.exists():
        return load_goal_unit(path, unit, provenance, data)
    if path.with_suffix(".trainlog.json.gz").exists():
        raise ValueError("Orphan training journal; refusing to overwrite")
    logs: list[dict[str, Any]] = []
    fit_times = []
    metadata: dict[str, Any] = {}

    class Timed:
        def __init__(self, model: TeamGoalModel) -> None:
            self.model = model
            self.refit_policy = model.refit_policy
            self.supports_sample_weight = model.supports_sample_weight
            self.requires_as_of = getattr(model, "requires_as_of", False)

        def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: pd.Series | None = None,
            *,
            context: pd.DataFrame,
            as_of: pd.Timestamp | None = None,
            prediction_features: pd.DataFrame | None = None,
        ) -> "Timed":
            started = time.perf_counter()
            if self.requires_as_of:
                from ligat.models.goal_model import MatchdayGoalModel

                cast(MatchdayGoalModel, self.model).fit(
                    X, y, context=context, as_of=as_of, prediction_features=prediction_features
                )
            else:
                self.model.fit(X, y, sample_weight, context=context)
            fit_times.append(time.perf_counter() - started)
            logs.append(cast(TrainingJournal, self.model).training_log)
            if isinstance(self.model, CatBoostGoalModel):
                metadata["actual_params"] = self.model.actual_params
            return self

        def predict(self, X: pd.DataFrame, *, context: pd.DataFrame) -> pd.Series:
            return self.model.predict(X, context=context)

    def factory() -> TeamGoalModel:
        return Timed(_model(unit, data) if model_factory is None else model_factory(unit))

    started = time.perf_counter()
    start = 9 if unit.backend == "glm" and unit.parameters["training_start"] == 9 else 8
    with threadpool_limits(limits=1):
        evaluation = evaluate_team_goals(
            factory,
            data,
            feature_columns=unit.features,
            policy=DevelopmentPolicy(training_start=start, validation_seasons=(unit.season,)),
            half_life=unit.half_life,
            blind=True,
            prior_seasons_only=unit.prior_seasons_only,
        )
    elapsed = time.perf_counter() - started
    memory = psutil.Process().memory_info()
    peak = float(getattr(memory, "peak_wset", memory.rss)) / (1024**2)
    journal = journal_bytes(logs, compressed=True)
    payload = {
        "completed_at": datetime.now(UTC).isoformat(),
        "training_log_sha256": hashlib.sha256(journal).hexdigest(),
        "format_version": 2,
        "signature": _signature(unit, provenance),
        "unit": asdict(unit),
        "provenance": asdict(provenance),
        "label": unit.label,
        "config_id": unit.configuration().config_id,
        "refit_policy": evaluation.refit_policy,
        "per_team": evaluation.per_team.reset_index().to_dict(orient="records"),
        "per_team_columns": ["game_id", "team_id", *evaluation.per_team.columns],
        "per_fold": evaluation.per_fold.to_dict(orient="records"),
        "per_fold_columns": list(evaluation.per_fold.columns),
        "summary": evaluation.summary.to_dict(),
        "summary_index": list(evaluation.summary.index),
        "elapsed_seconds": elapsed,
        "fit_seconds": sum(fit_times),
        "peak_rss_mb": peak,
        "metadata": metadata,
    }
    saved = {"payload": payload, "sha256": _payload_digest(payload)}
    atomic_bytes(path.with_suffix(".trainlog.json.gz"), journal)
    _atomic_json(path, saved)
    return GoalUnitResult(unit, evaluation, elapsed, sum(fit_times), peak, metadata, False, logs)


def run_goal_units(
    units: Sequence[GoalUnit],
    data: GoalData,
    provenance: GoalProvenance,
    root: Path,
    *,
    n_jobs: int,
    stage: str = "pilot",
    model_factory: ModelFactory | None = None,
    stop_after_units: int | None = None,
) -> list[GoalUnitResult]:
    """Resume verified units first; execute only pending units in single-thread loky workers."""
    if isinstance(n_jobs, bool) or not 1 <= n_jobs <= 15:
        raise ValueError("One to fifteen workers required")
    if stop_after_units is not None and (
        isinstance(stop_after_units, bool) or stop_after_units < 1
    ):
        raise ValueError("stop_after_units must be a positive count")
    unique = {u.unit_id: u for u in units}
    completed = []
    pending = []
    for unit in unique.values():
        path = _path(root, stage, unit)
        if path.exists():
            completed.append(load_goal_unit(path, unit, provenance, data))
        else:
            pending.append(unit)
    if not pending:
        rebuild_goal_index(root, stage=stage)
        return sorted(completed, key=lambda r: r.unit.unit_id)
    count = 0
    # An intentional interruption waits for its finite batch to publish both
    # files, rather than terminating another worker between journal and unit.
    scheduled = pending if stop_after_units is None else pending[:stop_after_units]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        iterator = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(run_goal_unit)(
                unit, data, provenance, root, stage=stage, model_factory=model_factory
            )
            for unit in scheduled
        )
        try:
            for result in iterator:
                completed.append(result)
                count += 1
                if stop_after_units is not None and count >= stop_after_units:
                    raise PilotInterrupted(
                        "Pilot intentionally interrupted; resume the same signed unit directory"
                    )
        finally:
            iterator.close()
            # The supplied sequence is one batch. Its index is a derived view;
            # unit/journal pairs remain durable throughout the batch. Rebuilding
            # after every unit would repeatedly parse every prior prediction.
            rebuild_goal_index(root, stage=stage)
    return sorted(completed, key=lambda r: r.unit.unit_id)


def run_goal_pilot(
    spec: GoalExperimentSpec, *, stop_after_units: int | None = None
) -> dict[str, Any]:
    """Only engineering evidence is returned or saved; full approval is never changed."""
    data = load_goal_inputs(spec)
    provenance = goal_provenance(data, prematch_version=spec.raw["inputs"]["prematch_version"])
    stop = threading.Event()
    peak = [0.0]
    peak_main = [0.0]
    peak_workers = [0.0]
    per_process_peak: dict[int, float] = {}

    def sample_memory() -> None:
        process = psutil.Process()
        while not stop.is_set():
            total, workers = 0.0, 0.0
            for member in [process, *process.children(recursive=True)]:
                try:
                    rss = member.memory_info().rss / (1024**2)
                    per_process_peak[member.pid] = max(per_process_peak.get(member.pid, 0), rss)
                    total += rss
                    if member.pid == process.pid:
                        peak_main[0] = max(peak_main[0], rss)
                    else:
                        workers += rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak[0] = max(peak[0], total)
            peak_workers[0] = max(peak_workers[0], workers)
            stop.wait(0.1)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = time.perf_counter()
    interrupted = False
    try:
        results = run_goal_units(
            pilot_units(spec),
            data,
            provenance,
            spec.output,
            n_jobs=spec.n_jobs,
            stop_after_units=stop_after_units,
        )
    except PilotInterrupted:
        interrupted = True
        results = []
    finally:
        stop.set()
        monitor.join()
    if interrupted:
        partial = {
            "stage": "pilot",
            "interrupted": True,
            "wall_seconds": time.perf_counter() - started,
            "peak_total_rss_mb": peak[0],
            "peak_main_rss_mb": peak_main[0],
            "peak_workers_rss_mb": peak_workers[0],
            "process_peak_rss_mb": {str(pid): value for pid, value in per_process_peak.items()},
            "completed_files": len(list((spec.output / "pilot").glob("unit-*.json"))),
        }
        _atomic_json(spec.output / "pilot" / "interruption_report.json", partial)
        raise PilotInterrupted(
            "Pilot intentionally interrupted; resume the same signed unit directory"
        )
    report = {
        "stage": "pilot",
        "units": len(results),
        "cached_units": sum(r.cached for r in results),
        "wall_seconds": time.perf_counter() - started,
        "peak_total_rss_mb": peak[0],
        "peak_main_rss_mb": peak_main[0],
        "peak_workers_rss_mb": peak_workers[0],
        "process_peak_rss_mb": {str(pid): value for pid, value in per_process_peak.items()},
        "rows": sum(len(r.evaluation.per_team) for r in results),
        "n_jobs": spec.n_jobs,
        "timings": [
            {
                "unit_id": r.unit.unit_id,
                "backend": r.unit.backend,
                "role": r.unit.role,
                "parameters": r.unit.parameters,
                "season": r.unit.season,
                "half_life": r.unit.half_life,
                "fit_seconds": r.fit_seconds,
                "elapsed_seconds": r.elapsed_seconds,
                "fits": int(r.evaluation.per_fold.fits.sum()),
                "peak_rss_mb": r.peak_rss_mb,
            }
            for r in results
        ],
    }
    target = spec.output / "pilot" / "engineering_report.json"
    _atomic_json(target, report)
    return report


def _read_journal(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    journal = path.with_suffix(".trainlog.json.gz")
    try:
        raw = journal.read_bytes()
        if hashlib.sha256(raw).hexdigest() != payload["training_log_sha256"]:
            raise ValueError("Training journal signature mismatch")
        logs = json.loads(gzip.decompress(raw))
        if not isinstance(logs, list) or not logs:
            raise ValueError("Missing training fit records")
        for log in logs:
            losses = log.get("train_loss")
            zero_iterations = (
                losses == []
                and log.get("iterations") == 0
                and log.get("converged") is True
                and log.get("recorded") is True
            )
            if log.get("dataset") != "training" or (not losses and not zero_iterations):
                raise ValueError("Invalid training journal record")
        return logs
    except (OSError, KeyError, ValueError, EOFError) as exc:
        raise ValueError(f"Invalid training journal for {path.name}: {exc}") from exc


def rebuild_goal_index(root: Path, *, stage: str = "pilot") -> Path:
    """Rebuild the display index from signed unit/journal pairs, without fitting."""
    columns = [
        "unit_id",
        "backend",
        "config_id",
        "label",
        "features",
        "fold",
        "fit_seconds",
        "peak_rss_mb",
        "iterations",
        "first_train_loss",
        "last_train_loss",
        "converged",
        "signature",
        "completed_at",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for path in sorted((root / stage).glob("unit-*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        payload = saved["payload"]
        if saved["sha256"] != _payload_digest(payload):
            raise ValueError(f"Goal unit signature mismatch: {path.name}")
        logs = _read_journal(path, payload)
        unit = payload["unit"]
        writer.writerow(
            dict(
                zip(
                    columns,
                    [
                        path.stem,
                        unit["backend"],
                        payload["config_id"],
                        payload["label"],
                        json.dumps(unit["features"]),
                        unit["season"],
                        payload["fit_seconds"],
                        payload["peak_rss_mb"],
                        sum(log["iterations"] for log in logs),
                        next((log["train_loss"][0] for log in logs if log["train_loss"]), None),
                        next(
                            (log["train_loss"][-1] for log in reversed(logs) if log["train_loss"]),
                            None,
                        ),
                        all(log["converged"] for log in logs),
                        saved["sha256"],
                        payload["completed_at"],
                    ],
                    strict=True,
                )
            )
        )
    target = root / stage / "index.csv"
    atomic_bytes(target, stream.getvalue().encode("utf-8"))
    return target


def exp005_status(config: GoalExperimentSpec) -> list[dict[str, str]]:
    """Classify planned pilot units against current inputs, source and library versions."""
    data = load_goal_inputs(config)
    provenance = goal_provenance(data, prematch_version=config.raw["inputs"]["prematch_version"])
    return goal_units_status(pilot_units(config), data, provenance, config.output)


def goal_units_status(
    units: Sequence[GoalUnit],
    data: GoalData,
    provenance: GoalProvenance,
    root: Path,
    *,
    stage: str = "pilot",
) -> list[dict[str, str]]:
    """Shared read-only cache classification for pilot, full batches and derived layers."""
    result = []
    for unit in units:
        path = _path(root, stage, unit)
        status, reason = "missing", "No saved unit"
        if path.exists() or path.with_suffix(".trainlog.json.gz").exists():
            try:
                load_goal_unit(path, unit, provenance, data)
                status, reason = "complete", "Verified current unit and training journal"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                status, reason = "outdated", str(exc)
        result.append({"unit_id": unit.unit_id, "status": status, "reason": reason})
    return result
