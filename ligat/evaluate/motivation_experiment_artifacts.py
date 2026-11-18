"""Immutable EXP-004 artifacts and an explicitly score-free engineering pilot report."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.motivation_experiment import (
    DEEP_SEASONS,
    PILOT_IDS,
    PILOT_SEASONS,
    VALIDATION_SEASONS,
    Metric,
    MotivationExperimentInput,
    MotivationExperimentSpec,
    MotivationRun,
    check_leakage,
    motivation_comparisons,
    motivation_configurations,
    motivation_diagnostics,
    paired_statistics,
)


@dataclass(frozen=True)
class MotivationProvenance:
    """Base commit, actual source bytes (including uncommitted code) and fixture hash."""

    commit: str
    input_sha256: str
    source_sha256: dict[str, str]


def blind_pilot_report(runs: Sequence[MotivationRun]) -> pd.DataFrame:
    """Safe to display: counts, optimizer diagnostics, parameter counts and timings only."""
    if (
        len(runs) != len(PILOT_IDS)
        or {r.config_id for r in runs} != set(PILOT_IDS)
        or any(
            r.stage != "pilot" or tuple(r.evaluation.per_fold.season_id) != PILOT_SEASONS
            for r in runs
        )
    ):
        raise ValueError("Blind report requires the five declared three-fold pilot runs")
    records = []
    for run in runs:
        for row in run.fits.itertuples():
            records.append(
                {
                    "config_id": run.config_id,
                    "season_id": row.season_id,
                    "training_games": row.training_games,
                    "validation_games": row.validation_games,
                    "parameters": 3 + len(run.configuration.columns),
                    "converged": bool(row.converged),
                    "iterations": int(row.iterations),
                    "accepted_by_gradient_check": bool(row.accepted_by_gradient_check),
                    "max_abs_gradient": float(row.max_abs_gradient),
                    "run_seconds": run.seconds,
                    "leakage_checks_passed": True,
                    "audit_required": run.audit_required,
                    "audit_passed": bool(run.audit and run.audit["passed"])
                    if run.audit_required
                    else True,
                }
            )
    return pd.DataFrame(records)


def coefficient_rows(runs: Sequence[MotivationRun]) -> pd.DataFrame:
    """Signed machine-readable coefficients; never include these in blind QA text."""
    rows = []
    for run in runs:
        for fit in run.fits.itertuples():
            labels = ["b_elo", *run.configuration.columns, "theta1", "delta", "theta2"]
            values = [fit.beta, *fit.additional_betas, fit.theta1, fit.delta, fit.theta2]
            for label, value in zip(labels, values, strict=True):
                rows.append(
                    {
                        "config_id": run.config_id,
                        "season_id": fit.season_id,
                        "coefficient": label,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def save_motivation_experiment(
    runs: Sequence[MotivationRun],
    spec: MotivationExperimentSpec,
    prepared: MotivationExperimentInput,
    provenance: MotivationProvenance,
    *,
    stage: Literal["pilot", "full"],
    baseline_verification: dict[str, Any],
) -> Path:
    """Publish exactly the requested stage, preserving every file under data/derived.

    Pilot score/fit files are machine data for later audit. Its comparison has no
    selection statistics, and blind_report.csv contains no scores or coefficients.
    Per-game predictions stay inside this experiment directory, not derived inputs.
    """
    expected_ids = (
        set(PILOT_IDS) if stage == "pilot" else {c.config_id for c in motivation_configurations()}
    )
    seasons = PILOT_SEASONS if stage == "pilot" else VALIDATION_SEASONS
    if (
        stage not in ("pilot", "full")
        or len(runs) != len(expected_ids)
        or {r.config_id for r in runs} != expected_ids
    ):
        raise ValueError("Publication requires exactly the declared stage configurations")
    if not baseline_verification.get("passed") or set(
        baseline_verification.get("controls", {})
    ) != {"C0", "C0-hl4"}:
        raise ValueError("Successful baseline reproduction is required before publication")
    if any(r.stage != stage or tuple(r.evaluation.per_fold.season_id) != seasons for r in runs):
        raise ValueError("Publication folds/stages differ from the preregistration")
    by_id = {r.config_id: r for r in runs}
    for run in runs:
        ref = by_id.get(run.configuration.reference_id) if run.config_id != "C0" else None
        required = check_leakage(run.evaluation, ref.evaluation if ref else None)
        if required != run.audit_required or (
            required
            and (not run.audit or not run.audit.get("passed") or not run.audit["timing"]["passed"])
        ):
            raise ValueError("Flagged configurations require complete passing automatic audits")
        if (
            not run.fits.converged.all()
            or not np.isfinite(run.fits.max_abs_gradient.to_numpy()).all()
        ):
            raise ValueError("Cannot publish nonconverged fits")
        for fit in run.fits.itertuples():
            if tuple(fit.training_seasons) != tuple(range(8, int(fit.season_id))):
                raise ValueError("Saved fit includes undeclared training seasons")
    if (
        len(provenance.commit) != 40
        or not provenance.source_sha256
        or any(
            len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in [provenance.input_sha256, *provenance.source_sha256.values()]
        )
    ):
        raise ValueError("Full commit, source and input checksums are required")
    directory = Path("reports/experiments/EXP-004") / stage
    if directory.exists():
        raise FileExistsError("Refusing to overwrite existing EXP-004 output")
    if stage == "full":
        comparison, review = motivation_comparisons(runs)
        subset, effects = motivation_diagnostics(runs, prepared)
    else:
        comparison = pd.DataFrame(
            [
                {
                    "config_id": r.config_id,
                    "reference": r.configuration.reference_id,
                    "status": "not_evaluated_blind_pilot",
                }
                for r in runs
            ]
        )
        review = {
            "selected_configuration": None,
            "status": "Blind engineering pilot; no model comparisons or selection",
        }
    directory.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}

    def record(path: Path, rows: int | None = None) -> None:
        files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": rows}

    def write_json(name: str, value: Any) -> None:
        path = directory / name
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        record(path)

    def write_frame(name: str, frame: pd.DataFrame) -> None:
        path = directory / name
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
        record(path, len(frame))

    path = directory / "experiment.yaml"
    path.write_text(spec.source_text, encoding="utf-8", newline="\n")
    record(path)
    write_json(
        "configurations.json", [{**asdict(r.configuration), "config_id": r.config_id} for r in runs]
    )
    write_frame(
        "fold_scores.parquet",
        pd.concat(
            [r.evaluation.per_fold.assign(config_id=r.config_id) for r in runs], ignore_index=True
        ),
    )
    fits = pd.concat([r.fits.assign(config_id=r.config_id) for r in runs], ignore_index=True)
    write_frame("fits.parquet", fits)
    write_frame("coefficients.parquet", coefficient_rows(runs))
    write_frame(
        "convergence.csv",
        fits[
            [
                "config_id",
                "season_id",
                "training_games",
                "validation_games",
                "training_seasons",
                "converged",
                "iterations",
                "accepted_by_gradient_check",
                "max_abs_gradient",
            ]
        ],
    )
    write_frame(
        "predictions.parquet",
        pd.concat(
            [r.evaluation.per_game.assign(config_id=r.config_id) for r in runs], ignore_index=True
        ),
    )
    write_frame("comparison.csv", comparison)
    write_json("review.json", review)
    audit_records = []
    for run in runs:
        if run.audit is None:
            continue
        audit = run.audit
        prefix = "audit_" + run.config_id
        for name in ("decomposition", "top10", "placebos"):
            write_frame(prefix + "_" + name + ".csv", audit[name])
        summary = {
            "config_id": run.config_id,
            "passed": audit["passed"],
            "zero_column_max_error": audit["zero_column_max_error"],
            "timing_passed": audit["timing"]["passed"],
            "seconds": audit["seconds"],
            **audit["placebo_summary"],
        }
        write_json(prefix + ".json", summary)
        audit_records.append(summary)
    if audit_records:
        write_frame("audit_summary.csv", pd.DataFrame(audit_records))
        write_json("timing_audit.json", next(r.audit["timing"] for r in runs if r.audit))
    if stage == "pilot":
        write_frame("blind_report.csv", blind_pilot_report(runs))
    else:
        write_frame("involved_games.csv", subset)
        write_frame("expected_points.csv", effects)
        summary_rows = []
        for run in runs:
            folds = run.evaluation.per_fold.set_index("season_id")
            row = {"config_id": run.config_id, "audit_required": run.audit_required}
            for scope, seasons in (("all15", VALIDATION_SEASONS), ("deep", DEEP_SEASONS)):
                for metric in ("log_loss", "rps"):
                    row[scope + "_" + metric] = float(
                        np.average(
                            folds.loc[list(seasons), metric],
                            weights=folds.loc[list(seasons), "games"],
                        )
                    )
            metrics: tuple[Metric, ...] = ("log_loss", "rps")
            for metric in metrics:
                ref = by_id[run.configuration.reference_id]
                s = paired_statistics(run.evaluation, ref.evaluation, metric, DEEP_SEASONS)
                row.update({metric + "_" + k: v for k, v in s.items()})
                row[metric + "_passes_rule8"] = (
                    s["mean_improvement"] > s["se"] and s["last5_folds_better"] >= 4
                )
            summary_rows.append(row)
        write_frame("summary.csv", pd.DataFrame(summary_rows))
        write_frame(
            "shallow_fold.csv",
            pd.concat(
                [
                    r.evaluation.per_fold.loc[r.evaluation.per_fold.season_id.eq(11)].assign(
                        config_id=r.config_id
                    )
                    for r in runs
                ],
                ignore_index=True,
            ),
        )
    metadata = {
        "experiment_id": "EXP-004",
        "stage": stage,
        "completed": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **asdict(provenance),
        "experiment_sha256": spec.sha256,
        "inputs": spec.raw["inputs"],
        "baseline_verification": baseline_verification,
        "seed": None,
        "randomness": "None: deterministic ordinal maximum likelihood",
        "training_start": 8,
        "loso": False,
        "pilot_blind": stage == "pilot",
        "warmup_note": (
            "2006/07-2008/09 importance uses the first fold converter fitted on those seasons; "
            "these seasons are training-only, never validation."
        ),
        "coefficient_order": (
            "beta=Elo/400, additional_betas in configuration.columns order, "
            "theta1, delta; theta2=theta1+exp(delta)"
        ),
        "paired_difference": (
            "Matching-decay C0 loss minus representation loss; positive is improvement"
        ),
        "selection": "Proposals only; Amit approves the decision",
        "configurations": len(runs),
        "fold_rows": len(fits),
        "seconds": {r.config_id: r.seconds for r in runs},
        "audit_configurations": [r.config_id for r in runs if r.audit_required],
        "audit_settings": spec.raw["audit"],
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
