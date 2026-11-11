"""Versioned, non-overwriting publication of the unified M0 prematch table."""

import hashlib
import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from pandas.testing import assert_frame_equal  # type: ignore[import-untyped]

from ligat.features.form_artifacts import FormProvenance, _publish
from ligat.features.pipeline import DEVELOPMENT_SEASONS, PrematchConfig, validate_prematch_frame

PREMATCH_VERSION = "1"


def save_prematch(
    frame: pd.DataFrame,
    spec: PrematchConfig,
    provenance: FormProvenance,
    *,
    directory: Path = Path("data/derived/prematch"),
) -> tuple[Path, Path]:
    """Publish v1 once, recording all four pinned inputs and current working sources."""
    if provenance.input_sha256 != spec.input_fingerprint():
        raise ValueError("Prematch provenance does not identify the configured inputs")
    metadata: dict[str, object] = {
        "component": "prematch",
        "version": PREMATCH_VERSION,
        "commit": provenance.commit,
        "input_sha256": provenance.input_sha256,
        "inputs": spec.input_records(),
        "config_sha256": spec.sha256,
        "source_sha256": dict(provenance.source_sha256),
        "seasons": list(DEVELOPMENT_SEASONS),
        "training_end": None,
        "assembly": "Frozen blocks only; no feature block recomputation, model or target",
        "decision": "ה-ת14; T-010 §9; C4",
        "season_8_exception": "Prior-season numeric values and all four contrasts remain NaN",
    }
    return _publish(
        frame,
        directory / f"prematch_v{PREMATCH_VERSION}.parquet",
        metadata,
        "prematch",
        validate_columns=validate_prematch_frame,
    )


def ensure_prematch(
    frame: pd.DataFrame,
    spec: PrematchConfig,
    provenance: FormProvenance,
    *,
    directory: Path = Path("data/derived/prematch"),
) -> bool:
    """Save a missing pair, or validate identical existing output; True means created.

    A partial pair, tampering, changed provenance inputs or changed values fails;
    updating an immutable version requires an explicitly new version instead.
    """
    target = directory / f"prematch_v{PREMATCH_VERSION}.parquet"
    if not target.resolve().is_relative_to(Path("data/derived").resolve()):
        raise ValueError("Prematch artifacts must stay inside data/derived")
    sidecar = target.with_suffix(".metadata.json")
    if not target.exists() and not sidecar.exists():
        save_prematch(frame, spec, provenance, directory=directory)
        return True
    if not target.exists() or not sidecar.exists():
        raise ValueError("Prematch version 1 is incomplete; refusing to overwrite either file")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        metadata.get("component") != "prematch"
        or metadata.get("version") != PREMATCH_VERSION
        or metadata.get("artifact_sha256") != hashlib.sha256(target.read_bytes()).hexdigest()
        or metadata.get("input_sha256") != spec.input_fingerprint()
        or metadata.get("inputs") != spec.input_records()
        or metadata.get("rows") != len(frame)
        or metadata.get("columns") != list(frame.columns)
        or metadata.get("index_columns") != list(frame.index.names)
        or metadata.get("seasons") != list(DEVELOPMENT_SEASONS)
    ):
        raise ValueError("Prematch version 1 metadata/checksum mismatch; a new version is required")
    saved = pd.read_parquet(target)
    validate_prematch_frame(saved)
    validate_prematch_frame(frame)
    try:
        assert_frame_equal(frame, saved, check_exact=True)
    except AssertionError as exc:
        raise ValueError(
            "Prematch differs from saved version 1; a new version is required"
        ) from exc
    return False
