"""Checksummed EXP-001 results, with explicit provenance for an uncommitted code review."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.elo_calibration import CalibrationBatch, ExperimentSpec


@dataclass(frozen=True)
class CalibrationProvenance:
    """Actual input, flag-artifact and working-source hashes, not just the base commit."""

    commit: str
    input_sha256: str
    table_state_sha256: str
    source_sha256: dict[str, str]


def _allowed(path: Path) -> None:
    for protected in ("data/raw", "data/clean", "docs", "config", "notebooks"):
        if path.resolve().is_relative_to(Path(protected).resolve()):
            raise ValueError("Calibration artifacts cannot replace source data or documents")


def save_calibration(
    batch: CalibrationBatch,
    spec: ExperimentSpec,
    provenance: CalibrationProvenance,
    *,
    directory: Path,
    snapshots_path: Path | None = None,
    prediction_ids: tuple[str, ...] = (),
    predictions_path: Path | None = None,
    stage: str,
) -> Path:
    """Save a new run directory and a final completion manifest; never overwrite a run.

    Production callers supply data/derived paths for endpoints and per-game
    predictions. A disposable pilot can keep all files under its temporary root.
    Only explicitly requested configurations get persisted per-game predictions.
    """
    ids = {r.config.config_id for r in batch.runs}
    if len(prediction_ids) > 10 or not set(prediction_ids) <= ids:
        raise ValueError("Persist per-game predictions for at most ten evaluated configurations")
    if stage not in ("pilot", "stage1", "stage2", "loso"):
        raise ValueError("Unknown experiment stage")
    hashes = [
        provenance.input_sha256,
        provenance.table_state_sha256,
        *provenance.source_sha256.values(),
    ]
    if (
        len(provenance.commit) != 40
        or not hashes
        or not provenance.source_sha256
        or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)
    ):
        raise ValueError("Full commit, input, table-state and working-source hashes are required")
    snapshots_path = snapshots_path or directory / "snapshots.parquet"
    predictions_path = predictions_path or directory / "predictions.parquet"
    for path in (directory, snapshots_path, predictions_path):
        _allowed(path)
    if snapshots_path.exists() or (prediction_ids and predictions_path.exists()):
        raise FileExistsError("Derived results already exist; use a separate run destination")
    directory.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}

    def record(path: Path, rows: int | None = None) -> None:
        files[Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": rows,
        }

    settings = directory / "experiment.yaml"
    # Keep the exact commented configuration loaded before the run, not a later edit.
    settings.write_text(spec.source_text, encoding="utf-8", newline="\n")
    record(settings)
    configs = directory / "configurations.json"
    configs.write_text(
        json.dumps(
            [{"config_id": r.config.config_id, **asdict(r.config)} for r in batch.runs], indent=2
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    record(configs)
    scores = []
    for run in batch.runs:
        joined = run.evaluation.per_fold.merge(
            run.fits, on=["fold", "season_id"], validate="one_to_one"
        )
        joined.insert(0, "config_id", run.config.config_id)
        scores.append(joined)
    folds = pd.concat(scores, ignore_index=True)
    target = directory / "fold_scores.csv"
    folds.to_csv(target, index=False, lineterminator="\n", float_format="%.17g")
    record(target, len(folds))
    target = directory / "comparison.csv"
    batch.comparison.rankings.to_csv(target, lineterminator="\n", float_format="%.17g")
    record(target, len(batch.comparison.rankings))
    snapshot_parts = []
    for run in batch.runs:
        part = run.snapshots.reset_index()
        part.insert(0, "config_id", run.config.config_id)
        snapshot_parts.append(part)
    snapshots = pd.concat(snapshot_parts, ignore_index=True)
    snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(snapshots_path, index=False)
    record(snapshots_path, len(snapshots))
    if prediction_ids:
        parts = []
        for run in batch.runs:
            if run.config.config_id in prediction_ids:
                part = run.evaluation.per_game.copy()
                part.insert(0, "config_id", run.config.config_id)
                parts.append(part)
        predictions = pd.concat(parts, ignore_index=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(predictions_path, index=False)
        record(predictions_path, len(predictions))
    metadata = {
        "experiment_id": spec.experiment_id,
        "stage": stage,
        "completed": True,
        "created_at": datetime.now(UTC).isoformat(),
        **asdict(provenance),
        "experiment_sha256": spec.sha256,
        "seed": None,
        "randomness": "None: serial deterministic Elo and BFGS MLE",
        "conversion_input": "elo_diff / 400, without HFA",
        "configurations": len(batch.runs),
        "fold_rows": len(folds),
        "seconds": {r.config.config_id: r.seconds for r in batch.runs},
        "comparison": {
            "all_winners": batch.comparison.all_winners,
            "last5_winners": batch.comparison.last5_winners,
            "disagreement": batch.comparison.disagreement,
            "requires_decision": batch.comparison.requires_decision,
        },
        "files": files,
    }
    manifest = directory / "metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify_calibration(manifest: Path) -> dict[str, Any]:
    """Verify every manifest checksum, including derived artifacts kept outside Git."""
    metadata: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("completed") is not True:
        raise ValueError("Experiment save is incomplete")
    for name, expected in metadata["files"].items():
        path = manifest.parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("Experiment artifact checksum mismatch")
    return metadata
