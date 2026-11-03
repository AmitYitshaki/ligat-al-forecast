"""EXP-002 manifests/checksums; pilot publication deliberately contains no scores."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.form_experiment import FormExperimentSpec, FormRun, compare_form_candidates
from ligat.evaluate.goal_backtest import GoalEvaluationResult


@dataclass(frozen=True)
class FormExperimentProvenance:
    """Projected fixture input and working-source hashes, in addition to frozen artifacts."""

    commit: str
    input_sha256: str
    source_sha256: dict[str, str]


def save_form_experiment(
    runs: Sequence[FormRun],
    spec: FormExperimentSpec,
    provenance: FormExperimentProvenance,
    *,
    directory: Path,
    stage: Literal["pilot", "full"],
) -> Path:
    """Save a new run only; pilot omits losses, rankings, predictions and coefficients.

    Full output is supported for later B4c, but is never implicitly evaluated.
    The manifest is written last; the EXP-001 checksum verifier is reused unchanged.
    """
    root = Path("reports/experiments/EXP-002").resolve()
    if not directory.resolve().is_relative_to(root) or stage not in ("pilot", "full"):
        raise ValueError("EXP-002 results must stay under reports/experiments/EXP-002")
    hashes = [provenance.input_sha256, *provenance.source_sha256.values()]
    if (
        len(provenance.commit) != 40
        or not provenance.source_sha256
        or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)
    ):
        raise ValueError("Full commit, input and source checksums required")
    if not runs or len({r.config_id for r in runs}) != len(runs):
        raise ValueError("Nonempty distinct runs required")
    if stage == "pilot" and any(
        r.candidate is not None and r.candidate not in spec.pilot for r in runs
    ):
        raise ValueError("Pilot may contain only predeclared pilot candidates and references")
    directory.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}

    def record(path: Path, rows: int | None = None) -> None:
        files[path.relative_to(directory).as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": rows,
        }

    config = directory / "experiment.yaml"
    config.write_text(spec.source_text, encoding="utf-8", newline="\n")
    record(config)
    identities = directory / "configurations.json"
    identities.write_text(
        json.dumps(
            [
                {
                    "config_id": r.config_id,
                    "route": r.route,
                    "with_elo": r.with_elo,
                    "candidate": None if r.candidate is None else asdict(r.candidate),
                }
                for r in runs
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    record(identities, len(runs))
    diagnostics = []
    shapes = {}
    for r in runs:
        safe = [
            "fold",
            "season_id",
            "training_seasons",
            "converged",
            "iterations",
            "accepted_by_gradient_check",
            "max_abs_gradient",
        ]
        part = r.fits.loc[:, safe].copy()
        part.insert(0, "config_id", r.config_id)
        diagnostics.append(part)
        shapes[r.config_id] = {
            "per_game": list(r.evaluation.per_game.shape),
            "per_fold": list(r.evaluation.per_fold.shape),
            "fits": list(r.fits.shape),
        }
        if isinstance(r.evaluation, GoalEvaluationResult):
            shapes[r.config_id]["per_team"] = list(r.evaluation.per_team.shape)
    diagnostics_frame = pd.concat(diagnostics, ignore_index=True)
    path = directory / "convergence.csv"
    diagnostics_frame.to_csv(path, index=False, lineterminator="\n")
    record(path, len(diagnostics_frame))
    if stage == "full":
        scores = []
        predictions = []
        team_predictions = []
        fits = []
        for r in runs:
            scores.append(r.evaluation.per_fold.assign(config_id=r.config_id))
            predictions.append(r.evaluation.per_game.assign(config_id=r.config_id))
            fits.append(r.fits.assign(config_id=r.config_id))
            if isinstance(r.evaluation, GoalEvaluationResult):
                team_predictions.append(r.evaluation.per_team.assign(config_id=r.config_id))
        for name, parts in [
            ("fold_scores", scores),
            ("predictions", predictions),
            ("fits", fits),
            ("team_predictions", team_predictions),
        ]:
            if parts:
                frame = pd.concat(parts, ignore_index=True)
                path = directory / f"{name}.parquet"
                frame.to_parquet(path, index=False)
                record(path, len(frame))
        for route in ("P", "O"):
            if any(r.route == route and r.candidate is not None and r.with_elo for r in runs):
                comparison = compare_form_candidates(runs, route)
                path = directory / f"comparison_{route}.csv"
                comparison.rankings.to_csv(path, lineterminator="\n")
                record(path, len(comparison.rankings))
    omitted = max(
        (
            float(r.evaluation.per_game.omitted_mass.max())
            for r in runs
            if isinstance(r.evaluation, GoalEvaluationResult)
        ),
        default=0.0,
    )
    metadata = {
        "experiment_id": "EXP-002",
        "stage": stage,
        "completed": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **asdict(provenance),
        "experiment_sha256": spec.sha256,
        "inputs": spec.raw["inputs"],
        "configurations": len(runs),
        "fold_rows": len(diagnostics_frame),
        "shapes": shapes,
        "seconds": {r.config_id: r.seconds for r in runs},
        "max_omitted_mass": omitted,
        "seed": None,
        "randomness": "None: serial deterministic BFGS MLE; no time weights or LOSO",
        "pilot_blind": stage == "pilot",
        "files": files,
    }
    manifest = directory / "metadata.json"
    manifest.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_calibration(manifest)
    return manifest
