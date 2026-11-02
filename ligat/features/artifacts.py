"""Versioned derived table-state artifacts, isolated from immutable clean inputs."""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import AnalysisTables
from ligat.features.registry import validate_feature_columns
from ligat.features.table_state import game_rows
from ligat.simulate.rules import SeasonRules

TABLE_STATE_VERSION = "2"


def input_fingerprint(tables: AnalysisTables, rules: Mapping[int, SeasonRules]) -> str:
    """Hash only the actual projected result/schedule inputs and their season rules."""
    games = game_rows(tables)
    selected = sorted(int(s) for s in games.season_id.unique())
    digest = hashlib.sha256(pd.util.hash_pandas_object(games, index=False).values.tobytes())
    digest.update(
        json.dumps({sid: asdict(rules[sid]) for sid in selected}, sort_keys=True).encode()
    )
    return digest.hexdigest()


def save_table_state(
    frame: pd.DataFrame,
    *,
    input_sha256: str,
    commit: str,
    directory: Path = Path("data/derived/prematch"),
) -> tuple[Path, Path]:
    """Write a Parquet and checksummed sidecar only inside the derived data layer.

    No fitting takes place in this block, so training_end is explicitly null.
    Pair publication is not transactional: a sidecar checksum detects interruption.
    """
    root = Path("data/derived").resolve()
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Table-state artifacts must stay inside data/derived")
    validate_feature_columns(frame, "table_state")
    if frame.index.names != ["game_id", "team_id"] or not frame.index.is_unique:
        raise ValueError("Expected unique (game_id, team_id) artifact index")
    if not commit or len(input_sha256) != 64:
        raise ValueError("Commit and full input SHA-256 are required")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"table_state_v{TABLE_STATE_VERSION}.parquet"
    sidecar = target.with_suffix(".metadata.json")
    temporary = directory / f".{uuid4().hex}.parquet"
    metadata_temp = temporary.with_suffix(".json")
    try:
        frame.to_parquet(temporary, index=True)
        metadata = {
            "component": "table_state",
            "version": TABLE_STATE_VERSION,
            "changes_from_version_1": [
                "QA 1: Unreachable target gaps are NA after the playoff split.",
                "QA 2: places_decided and nothing_to_play include place lines (ה-ת10).",
            ],
            "commit": commit,
            "input_sha256": input_sha256,
            "artifact_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
            "created_at": datetime.now(UTC).isoformat(),
            "training_end": None,
            "as_of_rule": "strictly earlier Asia/Jerusalem calendar dates, same season",
            "rows": len(frame),
            "index_columns": list(frame.index.names),
            "columns": list(frame.columns),
            "seasons": sorted(int(s) for s in frame.season_id.unique()),
            "table_basis": "Sporting points, no deductions or excluded technical matches",
        }
        metadata_temp.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        os.replace(metadata_temp, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return target, sidecar
