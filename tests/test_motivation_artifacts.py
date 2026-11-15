"""Same-build synthetic checks, immutable M1 publication and real artifact validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal
from test_table_state import block, synthetic_tables
from typer.testing import CliRunner

from ligat.cli import app
from ligat.features.table_state import game_rows
from ligat.models.ordered_logit import OrderedFit
from ligat.motivation.artifacts import (
    MotivationConfig,
    MotivationInputs,
    ensure_motivation,
    input_hashes,
    load_motivation_config,
    source_hashes,
    validate_existing,
)
from ligat.motivation.pipeline import build_motivation, validate_motivation_frame


@pytest.fixture
def synthetic_inputs():
    tables = synthetic_tables()
    rows = game_rows(tables)
    prematch = block().compute(tables)
    elo = (
        rows[["game_id", "team_id", "season_id", "as_of"]]
        .assign(elo_pre=1500.0)
        .set_index(["game_id", "team_id"])
    )
    fit = OrderedFit(2.0, -0.8, 0.2, 0.0, True, 0, False, 0.0)
    return MotivationInputs(rows, elo, {8: fit}, block().rules, prematch)


def build(inputs, **kwargs):
    return build_motivation(
        inputs.rows,
        inputs.elo,
        inputs.fits,
        inputs.rules,
        inputs.prematch,
        seasons=(8,),
        paths=64,
        **kwargs,
    )


def test_same_build_on_small_input_copies_flags_and_keeps_inputs_unchanged(synthetic_inputs):
    inputs = synthetic_inputs
    originals = [f.copy(deep=True) for f in (inputs.rows, inputs.elo, inputs.prematch)]
    checkpoints = []
    result = build(
        inputs, checkpoint=lambda sid, frame, audit: checkpoints.append((sid, frame, audit))
    )
    assert len(result.features) == 12
    assert result.features.N.eq(64).all()
    assert not result.features.before_split.any()
    validate_motivation_frame(result.features)
    assert_frame_equal(result.features[["nothing_to_play"]], inputs.prematch[["nothing_to_play"]])
    assert checkpoints[0][0] == 8
    assert checkpoints[0][2]["conservation_checked_paths"] == 6 * 3 * 64
    assert len(result.seasons) == 1 and result.seconds > 0
    for actual, original in zip((inputs.rows, inputs.elo, inputs.prematch), originals, strict=True):
        assert_frame_equal(actual, original, check_exact=True)


def test_raw_negative_effect_is_retained_and_only_score_is_clipped(synthetic_inputs):
    frame = build(synthetic_inputs).features.copy()
    key = frame.index[0]
    frame.loc[
        key, ["title_p_win", "title_p_draw", "title_p_loss", "title_lev_raw", "title_lev"]
    ] = [0.0, 0.1, 0.25, -0.25, 0.0]
    frame["leverage_max"] = frame[
        [f"{t}_lev" for t in ("title", "cut", "survival", "survival_direct")]
    ].max(axis=1)
    validate_motivation_frame(frame)
    assert frame.loc[key, "title_lev_raw"] == -0.25
    assert frame.loc[key, "title_lev"] == 0


@pytest.mark.parametrize("column", ["title_p_win", "N", "nothing_to_play", "survival_mc_se"])
def test_missing_mandatory_values_fail(synthetic_inputs, column):
    frame = build(synthetic_inputs).features.copy()
    frame[column] = np.nan
    with pytest.raises(ValueError):
        validate_motivation_frame(frame)


@pytest.fixture
def publication(tmp_path, monkeypatch, synthetic_inputs):
    monkeypatch.chdir(tmp_path)
    spec = MotivationConfig(
        14112026,
        64,
        (8,),
        {},
        Path("data/derived/motivation"),
        Path("data/reports/checkpoints"),
        "a" * 64,
    )
    monkeypatch.setattr(
        "ligat.motivation.artifacts.load_motivation_inputs", lambda spec: synthetic_inputs
    )
    monkeypatch.setattr(
        "ligat.motivation.artifacts.input_hashes", lambda spec: {"matches": "b" * 64}
    )
    monkeypatch.setattr("ligat.motivation.artifacts.source_hashes", lambda: {"code": "c" * 64})
    monkeypatch.setattr(
        "ligat.motivation.artifacts.subprocess.check_output", lambda *a, **k: "d" * 40
    )
    return spec


def test_publication_roundtrip_and_existing_version_preserve_bytes_and_mtimes(publication):
    spec = publication
    assert ensure_motivation(spec)
    paths = list(Path("data/derived/motivation").iterdir())
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    assert not ensure_motivation(spec)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths} == before
    metadata = json.loads(Path("data/derived/motivation/motivation_v1.metadata.json").read_text())
    assert metadata["N"] == 64 and metadata["seed"] == 14112026
    assert metadata["simulator_version"] == "1" and metadata["source_sha256"]
    assert metadata["warmup_exception"]


@pytest.mark.parametrize("bad", ["partial", "tampered"])
def test_existing_partial_or_changed_artifact_fails_without_overwrite(publication, bad):
    assert ensure_motivation(publication)
    target = Path("data/derived/motivation/motivation_v1.parquet")
    if bad == "partial":
        target.with_suffix(".metadata.json").unlink()
    else:
        target.write_bytes(target.read_bytes() + b"changed")
    before = target.read_bytes()
    with pytest.raises(ValueError, match="incomplete|checksum"):
        ensure_motivation(publication)
    assert target.read_bytes() == before


def test_cli_uses_same_builder_then_validates_existing_version(publication, monkeypatch):
    config = Path("motivation.yaml")
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "ligat.motivation.artifacts.load_motivation_config", lambda path: publication
    )
    runner = CliRunner()
    first = runner.invoke(app, ["run", "motivation", "--config", str(config)])
    assert first.exit_code == 0, first.output
    assert "created" in first.output
    second = runner.invoke(app, ["run", "motivation", "--config", str(config)])
    assert second.exit_code == 0, second.output
    assert "up to date" in second.output


def test_config_rejects_heldout_seasons_before_loading(tmp_path):
    raw = yaml.safe_load(Path("config/motivation.yaml").read_text())
    raw["seasons"] = list(range(8, 27))
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="development seasons"):
        load_motivation_config(path)


def test_built_real_artifact_has_exact_coverage_and_frozen_flags(real_tables):
    target = Path("data/derived/motivation/motivation_v1.parquet")
    if not target.exists():
        pytest.skip("Local motivation_v1 has not been built yet")
    spec = load_motivation_config()
    prematch = pd.read_parquet(
        "data/derived/prematch/prematch_v1.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    frame = validate_existing(spec, prematch, input_hashes(spec), source_hashes())
    assert len(frame) == 8630 == len(real_tables.team_matches)
    assert frame.N.eq(2000).all()
    assert set(frame.season_id) == set(range(8, 26))
    assert frame.loc[frame.nothing_to_play, "leverage_max"].eq(0).all()
    validate_motivation_frame(frame)
