"""EXP-003 publication: immutable manifests and derived-only per-game predictions."""

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.context_experiment import ContextExperimentSpec, ContextRun, context_comparisons
from ligat.evaluate.goal_backtest import GoalEvaluationResult


@dataclass(frozen=True)
class ContextProvenance:
    """Exact fixture/contrast input and uncommitted source fingerprints, plus base revision."""

    commit: str
    input_sha256: str
    source_sha256: dict[str, str]


def _coefficient_rows(run: ContextRun) -> list[dict[str, object]]:
    records = []
    for row in run.fits.to_dict("records"):
        if run.configuration.reference == "b0":
            continue
        if run.configuration.route == "P":
            labels = ["c", "gamma", "a", "d", "b_elo", *run.configuration.columns]
            coefficients = row["coefficients"]
        else:
            labels = (
                ["b_elo"]
                if run.configuration.route == "O" or run.configuration.reference == "elo"
                else []
            ) + list(run.configuration.columns)
            coefficients = [row["beta"], *row["additional_betas"]]
            labels += ["theta1", "delta"]
            coefficients += [row["theta1"], row["delta"]]
        for label, value in zip(labels, coefficients, strict=True):
            value = float(value)
            records.append(
                {
                    "config_id": run.config_id,
                    "fold": row["fold"],
                    "season_id": row["season_id"],
                    "coefficient": label,
                    "value": value,
                    "sign": "positive" if value > 0 else "negative" if value < 0 else "zero",
                }
            )
    return records


def save_context_experiment(
    runs: Sequence[ContextRun],
    spec: ContextExperimentSpec,
    provenance: ContextProvenance,
    *,
    stage: Literal["pilot", "full"],
) -> Path:
    """Save only the supplied runs; pilot scores are written but never returned or printed.

    Both stages persist fold metrics, signed fits, convergence and paired tables.
    All per-game/per-team predictions go directly to derived storage; the original
    manifest verifier follows their explicit relative paths and verifies all files.
    """
    ids = {r.config_id for r in runs}
    expected_pilot = set(spec.raw["pilot"]["configurations"])
    if stage not in ("pilot", "full") or not runs or len(ids) != len(runs):
        raise ValueError("Distinct explicitly evaluated EXP-003 configurations required")
    if stage == "pilot" and ids != expected_pilot:
        raise ValueError("Pilot requires exactly its four declared configurations")
    if stage == "full":
        from ligat.evaluate.context_experiment import context_configurations

        if ids != {c.config_id for c in context_configurations()}:
            raise ValueError("Full publication requires all 33 declared configurations")
    for run in runs:
        expected = (
            spec.raw["pilot"]["validation_seasons"]
            if stage == "pilot"
            else [s for s in range(11, 26) if not run.configuration.sensitivity or s != 11]
        )
        if run.evaluation.per_fold.season_id.tolist() != expected:
            raise ValueError("Saved folds differ from the predeclared run specification")
        for fit in run.fits.itertuples():
            training = tuple(
                s
                for s in range(9, int(fit.season_id))
                if not run.configuration.sensitivity or s != 11
            )
            if tuple(fit.training_seasons) != training:
                raise ValueError("Saved fit includes an undeclared training season")
    hashes = [provenance.input_sha256, *provenance.source_sha256.values()]
    if (
        len(provenance.commit) != 40
        or not provenance.source_sha256
        or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)
    ):
        raise ValueError("Full commit, input and source checksums required")
    directory = Path("reports/experiments/EXP-003") / stage
    derived = Path("data/derived/exp003") / stage
    if directory.exists() or derived.exists():
        raise FileExistsError("Refusing to overwrite existing EXP-003 output")
    comparisons, direct = context_comparisons(runs)
    directory.mkdir(parents=True, exist_ok=False)
    derived.mkdir(parents=True, exist_ok=False)
    files = {}

    def record(path: Path, rows: int | None = None) -> None:
        files[Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": rows,
        }

    path = directory / "experiment.yaml"
    path.write_text(spec.source_text, encoding="utf-8", newline="\n")
    record(path)
    path = directory / "configurations.json"
    path.write_text(
        json.dumps([{**asdict(r.configuration), "config_id": r.config_id} for r in runs], indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    record(path, len(runs))
    scores, fits, predictions, teams = [], [], [], []
    for run in runs:
        scores.append(run.evaluation.per_fold.assign(config_id=run.config_id))
        fits.append(run.fits.assign(config_id=run.config_id))
        predictions.append(run.evaluation.per_game.assign(config_id=run.config_id))
        if isinstance(run.evaluation, GoalEvaluationResult):
            teams.append(run.evaluation.per_team.assign(config_id=run.config_id))
    for name, parts in (("fold_scores", scores), ("fits", fits)):
        frame = pd.concat(parts, ignore_index=True)
        path = directory / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        record(path, len(frame))
    coefficients = pd.DataFrame([row for run in runs for row in _coefficient_rows(run)])
    path = directory / "coefficients.parquet"
    coefficients.to_parquet(path, index=False)
    record(path, len(coefficients))
    convergence = pd.concat(fits, ignore_index=True).loc[
        :,
        [
            "config_id",
            "fold",
            "season_id",
            "training_seasons",
            "converged",
            "iterations",
            "accepted_by_gradient_check",
            "max_abs_gradient",
            "elo_coefficient",
        ],
    ]
    path = directory / "convergence.csv"
    convergence.to_csv(path, index=False, lineterminator="\n")
    record(path, len(convergence))
    for name, frame in (("comparison", comparisons), ("direct_comparison", direct)):
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        record(path, len(frame))
    for name, parts in (("predictions", predictions), ("team_predictions", teams)):
        if parts:
            frame = pd.concat(parts, ignore_index=True)
            path = derived / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            record(path, len(frame))
    metadata = {
        "experiment_id": "EXP-003",
        "stage": stage,
        "completed": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **asdict(provenance),
        "experiment_sha256": spec.sha256,
        "inputs": spec.raw["inputs"],
        "seed": None,
        "randomness": "None: deterministic unweighted BFGS MLE",
        "training_start": 9,
        "half_life": None,
        "loso": False,
        "pilot_blind": stage == "pilot",
        "configurations": len(runs),
        "fold_rows": sum(len(r.fits) for r in runs),
        "validation_seasons": {r.config_id: r.evaluation.per_fold.season_id.tolist() for r in runs},
        "seconds": {r.config_id: r.seconds for r in runs},
        "coefficient_order": {
            "P": ["c", "gamma", "a", "d", "b_elo", "context coefficients"],
            "ordered": "beta first, additional_betas, theta1, delta; theta2=theta1+exp(delta)",
        },
        "paired_difference": (
            "Reference/default loss minus candidate/alternative; positive favors right side"
        ),
        "last5_note": (
            "Full/sensitivity: seasons 21–25. Pilot is incomplete and never used for decisions."
        ),
        "selection": "No model selected; predeclared paired statistics require Amit review",
        "files": files,
    }
    manifest = directory / "metadata.json"
    manifest.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_calibration(manifest)
    return manifest
