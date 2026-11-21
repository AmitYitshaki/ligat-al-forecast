"""Pinned v1/form loading and exclusive publication of the independent prematch v2."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import yaml
from pandas.testing import assert_frame_equal  # type: ignore[import-untyped]

from ligat.features.form_artifacts import FormProvenance, _publish
from ligat.features.pipeline import (
    DEVELOPMENT_SEASONS,
    PREMATCH_COLUMNS,
    PREMATCH_ROWS,
    FrozenPrematchInput,
    validate_prematch_frame,
)
from ligat.features.prematch_v2 import PREMATCH_V2_COLUMNS, validate_prematch_v2_frame
from ligat.features.registry import FORM_COLUMNS

PREMATCH_V2_VERSION = "2"
_INPUT_COLUMNS = {"prematch": PREMATCH_COLUMNS, "form": FORM_COLUMNS}


@dataclass(frozen=True)
class PrematchV2Config:
    """The two explicit input artifacts and the fingerprint of their YAML declaration."""

    inputs: dict[str, FrozenPrematchInput]
    sha256: str

    def input_records(self) -> dict[str, dict[str, str]]:
        """Keep both artifact and sidecar checksums in the output provenance."""
        return {name: self.inputs[name].record() for name in _INPUT_COLUMNS}

    def input_fingerprint(self) -> str:
        """Identify both complete input declarations independently of YAML comments."""
        return hashlib.sha256(json.dumps(self.input_records(), sort_keys=True).encode()).hexdigest()


def load_prematch_v2_config(path: Path = Path("config/prematch_v2.yaml")) -> PrematchV2Config:
    """Read only YAML, rejecting missing inputs, incomplete hashes and extra settings."""
    content = path.read_bytes()
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid prematch v2 YAML") from exc
    if not isinstance(raw, dict) or set(raw) != {"inputs"}:
        raise ValueError("Prematch v2 config requires only inputs")
    entries = raw["inputs"]
    if not isinstance(entries, dict) or set(entries) != set(_INPUT_COLUMNS):
        raise ValueError("Prematch v2 requires exactly prematch and form inputs")
    inputs = {}
    for name, item in entries.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "metadata_sha256"}:
            raise ValueError(f"Prematch v2 {name}: path and both SHA-256 values are required")
        if not isinstance(item["path"], str) or not item["path"].endswith(".parquet"):
            raise ValueError(f"Prematch v2 {name}: expected a Parquet path")
        for key in ("sha256", "metadata_sha256"):
            value = item[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"Prematch v2 {name}: full SHA-256 values are required")
        inputs[name] = FrozenPrematchInput(
            Path(item["path"]), item["sha256"], item["metadata_sha256"]
        )
    return PrematchV2Config(inputs, hashlib.sha256(content).hexdigest())


def _verify_inputs(spec: PrematchV2Config) -> None:
    if set(spec.inputs) != set(_INPUT_COLUMNS):
        raise ValueError("Prematch v2 requires exactly prematch and form inputs")
    for name, item in spec.inputs.items():
        if not item.path.resolve().is_relative_to(Path("data/derived").resolve()):
            raise ValueError("Prematch v2 input must stay inside data/derived")
        sidecar = item.path.with_suffix(".metadata.json")
        for path, digest in ((item.path, item.sha256), (sidecar, item.metadata_sha256)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Prematch v2 {name}: checksum mismatch")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            not isinstance(meta, dict)
            or meta.get("component") != name
            or meta.get("version") != "1"
            or meta.get("artifact_sha256") != item.sha256
            or meta.get("rows") != PREMATCH_ROWS
            or meta.get("columns") != list(_INPUT_COLUMNS[name])
            or meta.get("index_columns") != ["game_id", "team_id"]
            or meta.get("seasons") != list(DEVELOPMENT_SEASONS)
        ):
            raise ValueError(f"Prematch v2 {name}: metadata mismatch")


def load_prematch_v2_inputs(
    spec: PrematchV2Config, *, seasons: Sequence[int]
) -> dict[str, pd.DataFrame]:
    """Verify every checksum before loading, explicitly accepting only seasons 8-25.

    Form has no season column: select its game IDs from the filtered v1 frame,
    after verifying its sidecar declares exactly the development seasons.
    """
    if tuple(seasons) != DEVELOPMENT_SEASONS:
        raise ValueError("Prematch v2 requires explicit seasons 8-25")
    _verify_inputs(spec)
    prematch = pd.read_parquet(
        spec.inputs["prematch"].path, filters=[("season_id", "in", list(seasons))]
    )
    validate_prematch_frame(prematch)
    form = pd.read_parquet(
        spec.inputs["form"].path,
        filters=[("game_id", "in", prematch.index.get_level_values("game_id").unique().tolist())],
    )
    if tuple(form.columns) != FORM_COLUMNS:
        raise ValueError("Prematch v2 form columns differ from the frozen contract")
    return {"prematch": prematch, "form": form}


def save_prematch_v2(
    frame: pd.DataFrame,
    spec: PrematchV2Config,
    provenance: FormProvenance,
    *,
    directory: Path = Path("data/derived/prematch"),
) -> tuple[Path, Path]:
    """Publish the complete v2 once; either existing output file prevents replacement."""
    _verify_inputs(spec)
    if provenance.input_sha256 != spec.input_fingerprint():
        raise ValueError("Prematch v2 provenance differs from the configured inputs")
    metadata: dict[str, object] = {
        "component": "prematch",
        "version": PREMATCH_V2_VERSION,
        "commit": provenance.commit,
        "input_sha256": provenance.input_sha256,
        "inputs": spec.input_records(),
        "config_sha256": spec.sha256,
        "source_sha256": dict(provenance.source_sha256),
        "seasons": list(DEVELOPMENT_SEASONS),
        "training_end": None,
        "column_count": len(PREMATCH_V2_COLUMNS),
        "decision": "H-SH21; H-SH24; H-SH25; Q26; Q31; T-014 A2",
        "assembly": "51 unchanged v1 columns, 5 form copies, 10 opponent mirrors, gd_per_game",
        "zero_games": "gd_per_game is NaN; original points_per_game is unchanged",
    }
    return _publish(
        frame,
        directory / "prematch_v2.parquet",
        metadata,
        "prematch",
        validate_columns=validate_prematch_v2_frame,
    )


def ensure_prematch_v2(
    frame: pd.DataFrame,
    spec: PrematchV2Config,
    provenance: FormProvenance,
    *,
    directory: Path = Path("data/derived/prematch"),
) -> bool:
    """Create the missing pair, or verify existing bytes and values without rewriting."""
    target = directory / "prematch_v2.parquet"
    if not target.resolve().is_relative_to(Path("data/derived").resolve()):
        raise ValueError("Prematch v2 artifacts must stay inside data/derived")
    if provenance.input_sha256 != spec.input_fingerprint():
        raise ValueError("Prematch v2 provenance differs from the configured inputs")
    _verify_inputs(spec)
    sidecar = target.with_suffix(".metadata.json")
    if not target.exists() and not sidecar.exists():
        save_prematch_v2(frame, spec, provenance, directory=directory)
        return True
    if not target.exists() or not sidecar.exists():
        raise ValueError("Prematch v2 is incomplete; refusing to overwrite either file")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        meta.get("component") != "prematch"
        or meta.get("version") != PREMATCH_V2_VERSION
        or meta.get("artifact_sha256") != hashlib.sha256(target.read_bytes()).hexdigest()
        or meta.get("input_sha256") != spec.input_fingerprint()
        or meta.get("inputs") != spec.input_records()
        or meta.get("config_sha256") != spec.sha256
        or meta.get("rows") != len(frame)
        or meta.get("columns") != list(PREMATCH_V2_COLUMNS)
        or meta.get("column_count") != 67
        or meta.get("index_columns") != list(frame.index.names)
        or meta.get("seasons") != list(DEVELOPMENT_SEASONS)
    ):
        raise ValueError("Prematch v2 metadata/checksum mismatch; refusing to overwrite")
    saved = pd.read_parquet(target, filters=[("season_id", "in", list(DEVELOPMENT_SEASONS))])
    validate_prematch_v2_frame(saved)
    validate_prematch_v2_frame(frame)
    try:
        assert_frame_equal(frame, saved, check_exact=True)
    except AssertionError as exc:
        raise ValueError(
            "Prematch v2 differs from the saved version; refusing to overwrite"
        ) from exc
    return False
