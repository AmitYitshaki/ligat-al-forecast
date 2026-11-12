"""Frozen-block assembly, exact pairing, immutable publication and CLI failure gates."""

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from ligat.analysis.prepare import AnalysisTables
from ligat.cli import app
from ligat.evaluate.context_experiment import ContextInputs, prepare_context_experiment
from ligat.features.form_artifacts import FormProvenance
from ligat.features.pipeline import (
    DEVELOPMENT_SEASONS,
    ELO_COLUMNS,
    PREMATCH_COLUMNS,
    PREVIOUS_VALUE_COLUMNS,
    SELECTED_CONTEXT_COLUMNS,
    SELECTED_FORM_COLUMNS,
    TABLE_COLUMNS,
    build_prematch,
    load_prematch_config,
    load_prematch_inputs,
    validate_prematch_frame,
)
from ligat.features.prematch_artifacts import ensure_prematch, save_prematch
from ligat.features.registry import (
    FEATURES,
    FORM_COLUMNS,
    PREMATCH_CONTRAST_COLUMNS,
    PREMATCH_OPPONENT_COLUMNS,
    SEASON_CONTEXT_COLUMNS,
    validate_prematch_feature_columns,
)


@pytest.fixture
def blocks():
    games = np.repeat(np.arange(4315), 2)
    home = np.tile([True, False], 4315)
    index = pd.MultiIndex.from_arrays([games, np.where(home, 1, 2)], names=["game_id", "team_id"])
    season = np.repeat(8 + np.arange(4315) % 18, 2)
    table = pd.DataFrame(0, index=index, columns=TABLE_COLUMNS)
    table["season_id"] = pd.array(season, dtype="Int64")
    table["date"] = (
        pd.to_datetime(pd.Series(season + 1998).astype(str) + "-08-01")
        .dt.tz_localize("Asia/Jerusalem")
        .to_numpy()
    )
    table["as_of"] = table.date.dt.normalize()
    table["stage"] = pd.Categorical(["regular"] * len(index))
    table["is_home"] = pd.array(home, dtype="boolean")
    table["opponent_id"] = pd.array(np.where(home, 2, 1), dtype="Int64")
    for name in [
        "title_decided",
        "cut_decided",
        "relegation_decided",
        "places_decided",
        "nothing_to_play",
    ]:
        table[name] = False
    table["gap_to_cut"] = np.where(home, np.nan, 0.0)
    elo = pd.DataFrame(
        {
            "season_id": season,
            "as_of": table.as_of,
            "elo_pre": np.where(home, 1500.0, 1450.0),
            "elo_opponent_pre": np.where(home, 1450.0, 1500.0),
            "elo_diff": np.where(home, 50.0, -50.0),
            "elo_expected": 0.5,
        },
        index=index,
    )
    form = pd.DataFrame(
        {c: 1.0 + i / 100 + np.where(home, 0.2, -0.2) for i, c in enumerate(FORM_COLUMNS)},
        index=index,
    )
    context = pd.DataFrame(
        {c: np.where(home, 0.8, 1.2) for c in SEASON_CONTEXT_COLUMNS[:5]}, index=index
    )
    context["prev_position_pct"] = np.where(home, 0.25, 0.75)
    context["prev_ppm"] = np.where(home, 1.8, 1.1)
    context["prev_gd_per_game"] = np.where(home, 0.5, -0.25)
    context["promoted_in"] = False
    context["squad_value_imputed"] = False
    context["prev_season_known"] = season != 8
    context["league_size"] = 12
    context.loc[season == 8, list(SEASON_CONTEXT_COLUMNS[:5])] = np.nan
    return {"table_state": table, "elo": elo, "form": form, "season_context": context}


def write_inputs(directory, frames):
    config = {"inputs": {}}
    for name, frame in frames.items():
        path = directory / "data/derived" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {
            "component": name,
            "version": "2" if name == "table_state" else "1",
            "artifact_sha256": digest,
            "rows": len(frame),
            "columns": list(frame.columns),
            "index_columns": list(frame.index.names),
            "seasons": list(DEVELOPMENT_SEASONS),
        }
        sidecar = path.with_suffix(".metadata.json")
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        config["inputs"][name] = {
            "path": path.relative_to(directory).as_posix(),
            "sha256": digest,
            "metadata_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        }
    target = directory / "prematch.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    return target


def provenance(spec):
    return FormProvenance("a" * 40, spec.input_fingerprint(), {"pipeline.py": "b" * 64})


def test_pure_union_reconstructs_exact_blocks_and_hand_calculated_contrasts(blocks):
    before = {name: frame.copy(deep=True) for name, frame in blocks.items()}
    result = build_prematch(**blocks)
    assert result.shape == (8630, 51) and tuple(result.columns) == PREMATCH_COLUMNS
    assert result.index.is_unique and result.groupby(level="game_id").size().eq(2).all()
    for name, columns in [
        ("table_state", TABLE_COLUMNS),
        ("elo", ELO_COLUMNS),
        ("form", SELECTED_FORM_COLUMNS),
        ("season_context", SELECTED_CONTEXT_COLUMNS),
    ]:
        assert_frame_equal(
            result.loc[:, list(columns)], blocks[name].loc[:, list(columns)], check_exact=True
        )
    sample = result.loc[(1, 1)]  # season 9, home perspective
    assert sample.value_avg_log_ratio == pytest.approx(np.log(0.8 / 1.2), abs=1e-12)
    assert sample.position_log_ratio == pytest.approx(np.log(0.25 / 0.75), abs=1e-12)
    assert sample.prev_ppm_diff == pytest.approx(0.7, abs=1e-12)
    assert sample.prev_gd_diff == pytest.approx(0.75, abs=1e-12)
    assert sample.opp_gf_venue_ewma5 == blocks["form"].loc[(1, 2), "gf_venue_ewma5"]
    assert sample.opp_ga_venue_ewma5 == blocks["form"].loc[(1, 2), "ga_venue_ewma5"]
    away = result.xs(2, level="team_id")[list(PREMATCH_CONTRAST_COLUMNS)]
    home = result.xs(1, level="team_id")[list(PREMATCH_CONTRAST_COLUMNS)]
    np.testing.assert_allclose(home, -away, atol=1e-12, rtol=0, equal_nan=True)
    for name in before:
        assert_frame_equal(before[name], blocks[name], check_exact=True)


def assert_exp003_contrasts(blocks, audit):
    # Only fixture projection needs goals. Synthetic constants never enter the
    # context contrasts, and this test never calls a model, score or fitting API.
    identity = blocks["table_state"].assign(goals_scored=1, goals_conceded=1)
    tables = AnalysisTables(
        pd.DataFrame(), identity, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    inputs = ContextInputs(blocks["season_context"], blocks["elo"], blocks["form"], audit)
    reference = prepare_context_experiment(tables, inputs).games.set_index("game_id")
    result = build_prematch(**blocks)
    home = result.loc[result.is_home].reset_index().set_index("game_id").reindex(reference.index)
    np.testing.assert_allclose(
        home[list(PREMATCH_CONTRAST_COLUMNS)],
        reference[["z_value_avg", "z_position", "z_ppm", "z_gd"]],
        atol=1e-12,
        rtol=0,
        equal_nan=True,
    )


def test_four_home_contrasts_reproduce_frozen_exp003_preparation(blocks):
    audit = pd.DataFrame({"season_id": list(DEVELOPMENT_SEASONS), "promoted_position_mean": 0.75})
    assert_exp003_contrasts(blocks, audit)


@pytest.mark.real_data
def test_actual_frozen_inputs_reconstruct_exp003_without_loading_clean_or_fitting():
    spec = load_prematch_config()
    if not all(item.path.exists() for item in spec.inputs.values()):
        pytest.skip("Frozen prematch inputs are local only")
    frames = load_prematch_inputs(spec)
    sidecar = spec.inputs["season_context"].path.with_suffix(".metadata.json")
    audit = pd.DataFrame(json.loads(sidecar.read_text(encoding="utf-8"))["audit"])
    assert_exp003_contrasts(frames, audit)


@pytest.mark.parametrize("season", [26, 27, 28])
def test_holdout_and_live_seasons_are_rejected(blocks, season):
    blocks["table_state"].loc[blocks["table_state"].season_id.eq(8), "season_id"] = season
    with pytest.raises(ValueError, match="development seasons"):
        build_prematch(**blocks)


@pytest.mark.parametrize(
    "bad",
    [
        "rows",
        "keys",
        "duplicate",
        "column",
        "order",
        "opponent",
        "self",
        "home",
        "overlap_season",
        "overlap_date",
    ],
)
def test_structural_mismatches_fail_instead_of_silent_join(blocks, bad):
    if bad == "rows":
        blocks = {name: frame.iloc[2:].copy() for name, frame in blocks.items()}
    if bad == "keys":
        blocks["form"] = blocks["form"].rename(index={1: 99}, level="team_id")
    if bad == "duplicate":
        blocks["form"].index = pd.MultiIndex.from_tuples(
            [blocks["form"].index[1], *blocks["form"].index[1:]], names=["game_id", "team_id"]
        )
    if bad == "column":
        blocks["form"] = blocks["form"].drop(columns="pts_all_w5")
    if bad == "order":
        blocks["form"] = blocks["form"].iloc[::-1]
    if bad == "opponent":
        blocks["table_state"].iloc[0, blocks["table_state"].columns.get_loc("opponent_id")] = 99
    if bad == "self":
        blocks["table_state"].iloc[0, blocks["table_state"].columns.get_loc("opponent_id")] = 1
    if bad == "home":
        blocks["table_state"].loc[(0, 2), "is_home"] = True
    if bad == "overlap_season":
        blocks["elo"].loc[(0, 1), "season_id"] = 9
    if bad == "overlap_date":
        blocks["elo"].loc[(0, 1), "as_of"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Prematch"):
        build_prematch(**blocks)


@pytest.mark.parametrize(
    "bad", ["early_filled", "later_missing", "infinite", "zero_ratio", "known_flag"]
)
def test_missingness_and_log_domains_fail_closed(blocks, bad):
    context = blocks["season_context"]
    if bad == "early_filled":
        context.loc[(0, 1), "prev_ppm"] = 1.0
    if bad == "later_missing":
        context.loc[(1, 1), "prev_ppm"] = np.nan
    if bad == "infinite":
        blocks["form"].loc[(1, 1), "gf_all_w5"] = np.inf
    if bad == "zero_ratio":
        context.loc[(1, 1), "prev_value_share_avg"] = 0.0
    if bad == "known_flag":
        context.loc[(1, 1), "prev_season_known"] = False
    with pytest.raises(ValueError, match="Prematch"):
        build_prematch(**blocks)


@pytest.mark.parametrize(
    "column",
    [
        "referee_id",
        "goals_scored",
        "position_mean",
        "unregistered_feature",
        "prev_value_share_total",
    ],
)
def test_unregistered_or_unselected_output_columns_fail(blocks, column):
    result = build_prematch(**blocks)
    assert "prev_value_share_total" not in result and "goals_scored" not in result
    with pytest.raises(ValueError, match="Unregistered|ordered 51"):
        validate_prematch_frame(result.assign(**{column: 0.0}))
    if column not in FEATURES:
        with pytest.raises(ValueError, match="Unregistered"):
            validate_prematch_feature_columns(result.assign(**{column: 0.0}))


def test_selected_registry_status_and_model_override_remain_explicit():
    assert FEATURES["pts_venue_ewma5"].status == "🔬"
    assert FEATURES["prev_value_share_avg"].model_status == (("M2", "✅"),)
    assert FEATURES["value_avg_log_ratio"].model_status == (("M2", "✅"),)
    assert "prev_value_share_total" in FEATURES
    for name in (*PREMATCH_CONTRAST_COLUMNS, *PREMATCH_OPPONENT_COLUMNS):
        assert FEATURES[name].block == "prematch" and FEATURES[name].status == "🔬"
        assert FEATURES[name].availability == ("S" if name in PREMATCH_CONTRAST_COLUMNS else "D")
        assert FEATURES[name].decision == "ה-ת14; T-010 §9"


@pytest.mark.parametrize("column", ["opp_gf_venue_ewma5", "prev_gd_diff"])
def test_output_validation_detects_corrupted_pair_features(blocks, column):
    frame = build_prematch(**blocks)
    frame.loc[(1, 1), column] += 1.0
    with pytest.raises(ValueError, match="opponent feature|antisymmetric"):
        validate_prematch_frame(frame)


def test_build_is_byte_stable_and_season8_is_not_imputed(blocks):
    a = build_prematch(**blocks)
    b = build_prematch(**blocks)
    buffers = [io.BytesIO(), io.BytesIO()]
    for frame, buffer in zip([a, b], buffers, strict=True):
        frame.to_parquet(buffer)
    assert (
        hashlib.sha256(buffers[0].getvalue()).hexdigest()
        == hashlib.sha256(buffers[1].getvalue()).hexdigest()
    )
    columns = [*PREVIOUS_VALUE_COLUMNS, *PREMATCH_CONTRAST_COLUMNS]
    assert a.loc[a.season_id.eq(8), columns].isna().all().all()
    assert np.isfinite(a.loc[a.season_id.gt(8), columns]).all().all()
    assert a.gap_to_cut.isna().equals(blocks["table_state"].gap_to_cut.isna())


@pytest.mark.parametrize("which", ["parquet", "sidecar"])
def test_changed_input_bytes_fail_before_any_dataframe_read(tmp_path, monkeypatch, blocks, which):
    config = write_inputs(tmp_path, blocks)
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_config(config)
    target = spec.inputs["form"].path
    if which == "sidecar":
        target = target.with_suffix(".metadata.json")
    target.write_bytes(target.read_bytes() + b"changed")
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *a, **k: pytest.fail("DataFrame read before all checksums passed"),
    )
    with pytest.raises(ValueError, match="form: checksum"):
        load_prematch_inputs(spec)


@pytest.mark.parametrize(
    "field,value",
    [("rows", 1), ("seasons", [8, 26]), ("columns", ["wrong"]), ("artifact_sha256", "f" * 64)],
)
def test_rehashed_but_inconsistent_sidecar_is_rejected(tmp_path, monkeypatch, blocks, field, value):
    config = write_inputs(tmp_path, blocks)
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_config(config)
    item = spec.inputs["form"]
    sidecar = item.path.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata))
    spec = replace(
        spec,
        inputs={
            **spec.inputs,
            "form": replace(item, metadata_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest()),
        },
    )
    with pytest.raises(ValueError, match="metadata"):
        load_prematch_inputs(spec)


def test_publish_is_immutable_and_up_to_date_preserves_both_bytes_and_mtimes(
    tmp_path, monkeypatch, blocks
):
    config = write_inputs(tmp_path, blocks)
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_config(config)
    frame = build_prematch(**blocks)
    args = (frame, spec, provenance(spec))
    assert ensure_prematch(*args)
    files = list(Path("data/derived/prematch").iterdir())
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    assert not ensure_prematch(*args)
    with pytest.raises(FileExistsError):
        save_prematch(*args)
    modified = frame.copy()
    modified.loc[(1, 1), "elo_pre"] += 1.0
    with pytest.raises(ValueError, match="new version"):
        ensure_prematch(modified, spec, provenance(spec))
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files} == before
    metadata = json.loads(Path("data/derived/prematch/prematch_v1.metadata.json").read_text())
    assert metadata["version"] == "1" and metadata["inputs"] == spec.input_records()
    assert metadata["training_end"] is None and metadata["source_sha256"]
    assert_frame_equal(
        pd.read_parquet("data/derived/prematch/prematch_v1.parquet"), frame, check_exact=True
    )


@pytest.mark.parametrize("bad", ["partial", "tampered"])
def test_partial_or_tampered_saved_version_fails_without_overwrite(
    tmp_path, monkeypatch, blocks, bad
):
    config = write_inputs(tmp_path, blocks)
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_config(config)
    frame = build_prematch(**blocks)
    ensure_prematch(frame, spec, provenance(spec))
    target = Path("data/derived/prematch/prematch_v1.parquet")
    if bad == "partial":
        target.with_suffix(".metadata.json").unlink()
    else:
        target.write_bytes(target.read_bytes() + b"changed")
    original = target.read_bytes()
    with pytest.raises(ValueError, match="incomplete|checksum"):
        ensure_prematch(frame, spec, provenance(spec))
    assert target.read_bytes() == original


def test_run_prematch_cli_build_then_up_to_date_then_failure(tmp_path, monkeypatch, blocks, caplog):
    config = write_inputs(tmp_path, blocks)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ligat.cli.subprocess.check_output", lambda *a, **k: "a" * 40)

    def forbidden(*args, **kwargs):
        pytest.fail("CLI must only assemble the frozen blocks, never recompute them")

    for name in (
        "ligat.features.table_state.TableStateBlock.compute",
        "ligat.features.elo.compute_elo",
        "ligat.features.form.compute_form",
        "ligat.features.season_context.compute_season_context",
    ):
        monkeypatch.setattr(name, forbidden)
    runner = CliRunner()
    first = runner.invoke(app, ["run", "prematch", "--config", str(config)])
    assert first.exit_code == 0, first.output
    assert "created" in first.output
    second = runner.invoke(app, ["run", "prematch", "--config", str(config)])
    assert second.exit_code == 0 and "up to date" in second.output, second.output
    target = Path("data/derived/prematch/prematch_v1.parquet")
    before = target.read_bytes()
    changed = build_prematch(**blocks)
    changed.loc[(1, 1), "elo_pre"] += 1.0
    monkeypatch.setattr("ligat.features.pipeline.build_prematch", lambda **kw: changed)
    different = runner.invoke(app, ["run", "prematch", "--config", str(config)])
    assert different.exit_code != 0 and "new version" in caplog.text
    assert target.read_bytes() == before
    data = Path("data/derived/form.parquet")
    data.write_bytes(b"changed")
    failed = runner.invoke(app, ["run", "prematch", "--config", str(config)])
    assert failed.exit_code != 0 and "checksum mismatch" in caplog.text
    assert target.read_bytes() == before
