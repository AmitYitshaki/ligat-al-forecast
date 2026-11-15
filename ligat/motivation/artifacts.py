"""Filtered inputs and immutable publication of the M1 motivation artifact."""

import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import yaml
from pandas.testing import assert_frame_equal  # type: ignore[import-untyped]

from ligat.data.repository import Repository
from ligat.features.form_artifacts import _publish
from ligat.models.ordered_logit import OrderedFit
from ligat.motivation.pipeline import (
    MOTIVATION_COLUMNS,
    build_motivation,
    validate_motivation_frame,
)
from ligat.simulate.leverage import SIMULATOR_VERSION, conversion_fits
from ligat.simulate.rules import SeasonRules, load_season_rules
from ligat.simulate.snapshot import DEVELOPMENT_SEASONS

MOTIVATION_VERSION = "1"
INPUT_NAMES = frozenset(("matches", "elo", "fold_scores", "prematch", "rules"))
SOURCE_FILES = (
    "ligat/cli.py",
    "ligat/motivation/pipeline.py",
    "ligat/motivation/artifacts.py",
    "ligat/simulate/snapshot.py",
    "ligat/simulate/engine.py",
    "ligat/simulate/leverage.py",
    "ligat/simulate/rules.py",
    "ligat/features/table_state.py",
    "ligat/models/ordered_logit.py",
    "ligat/features/form_artifacts.py",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotivationConfig:
    """Explicit inputs, simulation budget and destinations; no fitted parameters."""

    seed: int
    paths: int
    seasons: tuple[int, ...]
    inputs: Mapping[str, Path]
    output: Path
    checkpoints: Path
    sha256: str


@dataclass(frozen=True)
class MotivationInputs:
    """Already filtered sporting rows and frozen calibration inputs."""

    rows: pd.DataFrame
    elo: pd.DataFrame
    fits: Mapping[int, OrderedFit]
    rules: Mapping[int, SeasonRules]
    prematch: pd.DataFrame


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_motivation_config(path: Path = Path("config/motivation.yaml")) -> MotivationConfig:
    """Require the full authorized 8-25 build and destinations outside raw/clean storage."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "seed",
        "N",
        "seasons",
        "inputs",
        "output",
        "checkpoints",
    }:
        raise ValueError("Invalid motivation config fields")
    if (
        type(raw["seed"]) is not int
        or raw["seed"] < 0
        or type(raw["N"]) is not int
        or raw["N"] < 2
        or raw["seasons"] != list(range(8, 26))
    ):
        raise ValueError("Require nonnegative seed, N >= 2 and exactly development seasons 8-25")
    if not isinstance(raw["inputs"], dict) or set(raw["inputs"]) != INPUT_NAMES:
        raise ValueError("Motivation requires matches, elo, fold_scores, prematch and rules")
    output, checkpoints = Path(raw["output"]), Path(raw["checkpoints"])
    if output.resolve() != Path("data/derived/motivation").resolve():
        raise ValueError("Motivation v1 must be published in data/derived/motivation")
    if not checkpoints.resolve().is_relative_to(Path("data/reports").resolve()):
        raise ValueError("Checkpoints must stay in data/reports")
    return MotivationConfig(
        raw["seed"],
        raw["N"],
        tuple(raw["seasons"]),
        {name: Path(p) for name, p in raw["inputs"].items()},
        output,
        checkpoints,
        _hash(path),
    )


def match_participants(matches: pd.DataFrame) -> pd.DataFrame:
    """Project approved sporting matches into paired rows, with no official table input."""
    if matches.empty or not set(matches.season_id) <= DEVELOPMENT_SEASONS:
        raise ValueError("Match inputs require development seasons 8-25")
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        part = matches[
            [
                "game_id",
                "season_id",
                "date",
                "round",
                "stage",
                f"{side}_team_id",
                f"{other}_team_id",
                f"{side}_goals",
                f"{other}_goals",
            ]
        ].rename(
            columns={
                f"{side}_team_id": "team_id",
                f"{other}_team_id": "opponent_id",
                f"{side}_goals": "goals_scored",
                f"{other}_goals": "goals_conceded",
            }
        )
        part["is_home"] = side == "home"
        part["as_of"] = part.date.dt.normalize()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def load_motivation_inputs(spec: MotivationConfig) -> MotivationInputs:
    """Push development filters into every Parquet read; reject held-out inputs."""
    if not set(spec.seasons) <= DEVELOPMENT_SEASONS:
        raise ValueError("Only development seasons 8-25 may be loaded")
    filters = [("season_id", "in", list(range(8, 26)))]
    if spec.inputs["matches"].name != "matches.parquet":
        raise ValueError("Expected the approved clean matches dataset")
    matches = Repository(spec.inputs["matches"].parent).read_filtered("matches", filters)
    elo = pd.read_parquet(spec.inputs["elo"], filters=filters)
    prematch = pd.read_parquet(spec.inputs["prematch"], filters=filters)
    folds = pd.read_csv(spec.inputs["fold_scores"], float_precision="round_trip")
    for frame in (matches, elo, prematch, folds):
        if frame.empty or not set(frame.season_id) <= DEVELOPMENT_SEASONS:
            raise ValueError("Loaded data contain a non-development season")
    if len(prematch) != 8630 or set(prematch.season_id) != DEVELOPMENT_SEASONS:
        raise ValueError("Frozen prematch must cover all 8630 development rows")
    return MotivationInputs(
        match_participants(matches),
        elo,
        conversion_fits(folds),
        load_season_rules(spec.inputs["rules"]),
        prematch,
    )


def input_hashes(spec: MotivationConfig) -> dict[str, str]:
    """Fingerprint immutable input bytes, including rules and the copied M0 flags."""
    return {name: _hash(path) for name, path in spec.inputs.items()}


def source_hashes() -> dict[str, str]:
    """Identify actual working-source bytes even before this stage is committed."""
    root = Path(__file__).resolve().parents[2]
    return {name: _hash(root / name) for name in SOURCE_FILES}


def validate_existing(
    spec: MotivationConfig,
    prematch: pd.DataFrame,
    inputs: Mapping[str, str],
    sources: Mapping[str, str],
) -> pd.DataFrame:
    """Verify both immutable files and their current inputs without rerunning simulation."""
    target = spec.output / "motivation_v1.parquet"
    sidecar = target.with_suffix(".metadata.json")
    if not target.exists() or not sidecar.exists():
        raise ValueError("Motivation version 1 is incomplete; refusing to overwrite")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        metadata.get("component") != "motivation"
        or metadata.get("version") != MOTIVATION_VERSION
        or metadata.get("simulator_version") != SIMULATOR_VERSION
        or metadata.get("artifact_sha256") != _hash(target)
        or metadata.get("inputs_sha256") != dict(inputs)
        or metadata.get("source_sha256") != dict(sources)
        or metadata.get("config_sha256") != spec.sha256
        or metadata.get("seed") != spec.seed
        or metadata.get("N") != spec.paths
        or metadata.get("seasons") != list(spec.seasons)
        or metadata.get("columns") != list(MOTIVATION_COLUMNS)
        or metadata.get("index_columns") != ["game_id", "team_id"]
        or metadata.get("rows") != len(prematch)
    ):
        raise ValueError("Motivation metadata/checksum mismatch; a new version is required")
    saved = pd.read_parquet(target, filters=[("season_id", "in", list(spec.seasons))])
    validate_motivation_frame(saved)
    expected = prematch.sort_index()
    if not saved.index.equals(expected.index):
        raise ValueError("Motivation identities do not match frozen prematch")
    assert_frame_equal(saved[["nothing_to_play"]], expected[["nothing_to_play"]], check_exact=True)
    return saved


def ensure_motivation(spec: MotivationConfig) -> bool:
    """Build and exclusively publish a missing v1, or verify an unchanged existing pair."""
    inputs, sources = input_hashes(spec), source_hashes()
    loaded = load_motivation_inputs(spec)
    target = spec.output / "motivation_v1.parquet"
    if target.exists() or target.with_suffix(".metadata.json").exists():
        validate_existing(spec, loaded.prematch, inputs, sources)
        return False

    def checkpoint(sid: int, frame: pd.DataFrame, audit: Mapping[str, object]) -> None:
        spec.checkpoints.mkdir(parents=True, exist_ok=True)
        path = spec.checkpoints / f"motivation_season_{sid}.parquet"
        sidecar = path.with_suffix(".metadata.json")
        if path.exists() or sidecar.exists():
            if not path.exists() or not sidecar.exists():
                raise ValueError("Incomplete motivation checkpoint")
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                previous.get("inputs_sha256") != inputs
                or previous.get("source_sha256") != sources
                or previous.get("seed") != spec.seed
                or previous.get("N") != spec.paths
                or previous.get("artifact_sha256") != _hash(path)
            ):
                raise ValueError("Checkpoint has different inputs, code or parameters")
            assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
        else:
            frame.to_parquet(path, index=True)
            assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
            sidecar.write_text(
                json.dumps(
                    {
                        **audit,
                        "inputs_sha256": inputs,
                        "source_sha256": sources,
                        "artifact_sha256": _hash(path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        LOGGER.info("CHECKPOINT season=%s rows=%s seconds=%.2f", sid, len(frame), audit["seconds"])

    result = build_motivation(
        loaded.rows,
        loaded.elo,
        loaded.fits,
        loaded.rules,
        loaded.prematch,
        seasons=spec.seasons,
        paths=spec.paths,
        seed=spec.seed,
        checkpoint=checkpoint,
    )
    if input_hashes(spec) != inputs or source_hashes() != sources:
        raise ValueError("Inputs or implementation changed during motivation construction")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata: dict[str, object] = {
        "component": "motivation",
        "version": MOTIVATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "seed": spec.seed,
        "N": spec.paths,
        "commit": commit,
        "source_sha256": sources,
        "inputs_sha256": inputs,
        "config_sha256": spec.sha256,
        "seasons": list(spec.seasons),
        "seconds": result.seconds,
        "season_audits": list(result.seasons),
        "training_end": None,
        "warmup_exception": (
            "Seasons 8-10 use season-11 conversion and 8-10 exact scores; training only"
        ),
        "leverage": (
            "max over title, cut, survival of max(P(target|win)-P(target|loss),0); "
            "survival_direct is a stored component only; paired CRN; draw is diagnostic"
        ),
        "commit_note": ("Git HEAD at build; source_sha256 identifies the exact source files"),
    }
    _publish(
        result.features, target, metadata, "motivation", validate_columns=validate_motivation_frame
    )
    saved = validate_existing(spec, loaded.prematch, inputs, sources)
    assert_frame_equal(result.features, saved, check_exact=True)
    LOGGER.info("COMPLETE rows=%s seconds=%.2f", len(saved), result.seconds)
    return True
