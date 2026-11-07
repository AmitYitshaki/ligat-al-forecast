"""Checksummed, non-overwriting form and selected-Elo artifacts in derived storage."""

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pandas.testing import assert_frame_equal  # type: ignore[import-untyped]

from ligat.analysis.prepare import AnalysisTables
from ligat.features.elo import EloParams, EloResult
from ligat.features.form import FormResult
from ligat.features.registry import FORM_COLUMNS, validate_feature_columns
from ligat.features.table_state import game_rows
from ligat.simulate.rules import SeasonRules

FORM_VERSION = "1"
E0050_PARAMS = EloParams(
    k=10,
    home_advantage=30,
    season_regression=0.33,
    use_goal_multiplier=True,
    promoted_initialization="fixed_1450",
    dead_game_discount=1,
)


def form_input_fingerprint(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules], neutral: pd.DataFrame
) -> str:
    """Hash projected results, the early sporting-rank rules and explicit neutral input."""
    rows = game_rows(tables)
    digest = hashlib.sha256(pd.util.hash_pandas_object(rows, index=False).values.tobytes())
    selected = sorted(int(s) for s in rows.season_id.unique())
    digest.update(json.dumps({s: asdict(rules[s]) for s in selected}, sort_keys=True).encode())
    digest.update(
        neutral.sort_index()
        .sort_index(axis=1)
        .to_json(orient="split", double_precision=15)
        .encode()
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class FormProvenance:
    """Full inputs and uncommitted-source hashes; a Git revision alone is insufficient."""

    commit: str
    input_sha256: str
    source_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        hashes = [self.input_sha256, *self.source_sha256.values()]
        if (
            not self.commit
            or not self.source_sha256
            or any(
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in hashes
            )
        ):
            raise ValueError("Commit, full input SHA-256 and source SHA-256 hashes are required")


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    # Pandas maps missing early rho_raw values to JSON null, never nonstandard NaN.
    result: list[dict[str, object]] = json.loads(
        frame.to_json(orient="records", double_precision=15)
    )
    return result


def _publish(
    frame: pd.DataFrame,
    target: Path,
    metadata: dict[str, object],
    block: str,
    *,
    validate_columns: Callable[[pd.DataFrame], None] | None = None,
) -> tuple[Path, Path]:
    if not target.resolve().is_relative_to(Path("data/derived").resolve()):
        raise ValueError("Artifacts must stay inside data/derived")
    sidecar = target.with_suffix(".metadata.json")
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite {target} or its metadata")
    if validate_columns is None:
        validate_feature_columns(frame, block)
    else:
        validate_columns(frame)
    if frame.index.names != ["game_id", "team_id"] or not frame.index.is_unique:
        raise ValueError("Expected unique (game_id, team_id) artifact index")
    sizes = frame.groupby(level="game_id").size()
    if frame.empty or not sizes.eq(2).all():
        raise ValueError("Expected two team rows per game")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{uuid4().hex}.parquet"
    metadata_temp = temporary.with_suffix(".json")
    try:
        frame.to_parquet(temporary, index=True)
        assert_frame_equal(frame, pd.read_parquet(temporary))
        complete = {
            **metadata,
            "artifact_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "rows": len(frame),
            "columns": list(frame.columns),
            "index_columns": list(frame.index.names),
            "as_of_rule": "strictly earlier Asia/Jerusalem calendar dates; no same-day results",
        }
        metadata_temp.write_text(
            json.dumps(complete, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        # Exclusive links publish complete files without an overwrite race. A
        # sidecar checksum also detects interruption between these two operations.
        os.link(temporary, target)
        os.link(metadata_temp, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return target, sidecar


def save_form(
    result: FormResult,
    provenance: FormProvenance,
    *,
    directory: Path = Path("data/derived/form"),
) -> tuple[Path, Path]:
    """Save exactly the registered 66 candidates and their complete opening audit."""
    frame = result.features
    if tuple(frame.columns) != FORM_COLUMNS or not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("Form artifact requires exactly 66 finite candidate columns")
    metadata: dict[str, object] = {
        "component": "form",
        "version": FORM_VERSION,
        "commit": provenance.commit,
        "input_sha256": provenance.input_sha256,
        "source_sha256": dict(provenance.source_sha256),
        "seasons": sorted(int(s) for s in result.openings.season_id.unique()),
        "rho_method": "Unweighted team-season OLS through origin, clipped to [0,1]",
        "league_and_promoted_means": "Game weighted, separately for all/home/away",
        "training_end": None,
        "season_8_exception": (
            "Only opening values use full season-8 league means (training only); "
            "w/EWMA/std update causally, prev/hist remain neutral. Injectable neutral input."
        ),
        "coefficients": _records(result.coefficients),
        "openings": _records(result.openings),
        "fallback_counts": _records(result.fallback_counts),
        "fallback_count_unit": "Unique (game_id, team_id), union over metrics and strata",
    }
    return _publish(frame, directory / f"form_v{FORM_VERSION}.parquet", metadata, "form")


def save_e0050(
    result: EloResult,
    provenance: FormProvenance,
    *,
    table_state_sha256: str,
    directory: Path = Path("data/derived/elo"),
) -> tuple[Path, Path]:
    """Publish the existing Elo engine's E0050 prematch rows, without fitting anything."""
    if result.config_id != E0050_PARAMS.config_id:
        raise ValueError("Only the selected E0050 configuration may be saved here")
    if len(table_state_sha256) != 64:
        raise ValueError("Full table-state checksum required")
    metadata: dict[str, object] = {
        "component": "elo",
        "experiment_configuration": "E0050",
        "config_id": result.config_id,
        "params": asdict(E0050_PARAMS),
        "commit": provenance.commit,
        "input_sha256": provenance.input_sha256,
        "source_sha256": dict(provenance.source_sha256),
        "table_state_sha256": table_state_sha256,
        "training_end": None,
        "seasons": sorted(int(s) for s in result.prematch.season_id.unique()),
    }
    return _publish(result.prematch, directory / "elo_E0050_prematch.parquet", metadata, "elo")
