"""Immutable season-context publishing using the existing derived artifact contract."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ligat.analysis.prepare import AnalysisTables
from ligat.features.form_artifacts import FormProvenance, _publish, _records
from ligat.features.registry import SEASON_CONTEXT_COLUMNS
from ligat.features.season_context import (
    PROMOTED_POSITION_PCT,
    SeasonContextResult,
    season_value_inputs,
)
from ligat.features.table_state import game_rows
from ligat.simulate.rules import SeasonRules

SEASON_CONTEXT_VERSION = "1"


def season_context_input_fingerprint(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules]
) -> str:
    """Cover projected results, source value fields and rules; exclude official standings."""
    rows = game_rows(tables)
    digest = hashlib.sha256(
        rows.to_json(orient="split", double_precision=15, date_format="iso").encode()
    )
    digest.update(season_value_inputs(tables).to_json(orient="split", double_precision=15).encode())
    seasons = sorted(int(s) for s in rows.season_id.unique())
    digest.update(json.dumps({s: asdict(rules[s]) for s in seasons}, sort_keys=True).encode())
    return digest.hexdigest()


def save_season_context(
    result: SeasonContextResult,
    provenance: FormProvenance,
    *,
    directory: Path = Path("data/derived/prematch"),
) -> tuple[Path, Path]:
    """Publish v1 and its full audit; the shared provenance type also serves Elo artifacts."""
    frame = result.features
    if tuple(frame.columns) != SEASON_CONTEXT_COLUMNS:
        raise ValueError("Season context requires exactly its nine registered columns")
    values = frame.loc[:, SEASON_CONTEXT_COLUMNS[:5]]
    known = frame.prev_season_known
    if (
        not values.loc[~known].isna().all().all()
        or not np.isfinite(values.loc[known].to_numpy(float)).all()
    ):
        raise ValueError("Only season 8 may have missing previous-season context")
    metadata: dict[str, object] = {
        "component": "season_context",
        "version": SEASON_CONTEXT_VERSION,
        "commit": provenance.commit,
        "input_sha256": provenance.input_sha256,
        "source_sha256": dict(provenance.source_sha256),
        "training_end": None,
        "seasons": sorted(int(s) for s in result.audit.season_id.unique()),
        "season_8_exception": (
            "No prior season: five value columns stay NaN; no neutral replacement."
        ),
        "value_imputation": (
            "Quantile 0.30 of the lower n//2 available prior-league shares; no renormalization"
        ),
        "promoted_position_pct": PROMOTED_POSITION_PCT,
        "promoted_position_mean": (
            "Prior promoted team-season rank/n mean; "
            "0.75 with fewer than four prior events; audit only"
        ),
        "audit": _records(result.audit),
    }
    return _publish(
        frame,
        directory / f"season_context_v{SEASON_CONTEXT_VERSION}.parquet",
        metadata,
        "season_context",
    )
