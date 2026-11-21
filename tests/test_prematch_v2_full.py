"""Complete v2 structure, registry, immutable publication and chronological causality."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from ligat.cli import app
from ligat.features.form_artifacts import E0050_PARAMS, FormProvenance
from ligat.features.pipeline import (
    DEVELOPMENT_SEASONS,
    PREMATCH_COLUMNS,
    PREMATCH_SCHEMA,
    build_prematch,
)
from ligat.features.prematch_v2 import (
    PREMATCH_V2_ADDITION_COLUMNS,
    PREMATCH_V2_COLUMNS,
    PREMATCH_V2_SCHEMA,
    build_prematch_v2,
    validate_prematch_v2_frame,
)
from ligat.features.prematch_v2_artifacts import (
    ensure_prematch_v2,
    load_prematch_v2_config,
    load_prematch_v2_inputs,
    save_prematch_v2,
)
from ligat.features.registry import FEATURES, M2_BASE_COLUMNS
from tests.test_prematch_pipeline import blocks as blocks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def full_sources(blocks):
    """Reuse the existing synthetic 8,630-row v1 contract, with nonzero and zero exposures."""
    prematch = build_prematch(**blocks)
    prematch["games_played"] = np.tile([0, 1, 2, 3, 4], len(prematch) // 5)
    prematch["goal_difference"] = np.tile([0, -2, 3, 1, -5], len(prematch) // 5)
    prematch["points_per_game"] = prematch.games_played.where(prematch.games_played.gt(0), 0) * 0.5
    return {"prematch": prematch, "form": blocks["form"]}


def write_sources(root, sources):
    entries = {}
    for name, frame in sources.items():
        target = root / "data/derived" / f"{name}_v1.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target)
        sidecar = target.with_suffix(".metadata.json")
        sidecar.write_text(
            json.dumps(
                {
                    "component": name,
                    "version": "1",
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "index_columns": list(frame.index.names),
                    "seasons": list(DEVELOPMENT_SEASONS),
                    "artifact_sha256": sha(target),
                }
            ),
            encoding="utf-8",
        )
        entries[name] = {
            "path": target.relative_to(root).as_posix(),
            "sha256": sha(target),
            "metadata_sha256": sha(sidecar),
        }
    config = root / "prematch_v2.yaml"
    config.write_text(yaml.safe_dump({"inputs": entries}), encoding="utf-8")
    return config


def provenance(spec):
    return FormProvenance("test-revision", spec.input_fingerprint(), {"prematch_v2.py": "a" * 64})


def test_complete_structure_ratio_and_exact_v1_preservation(full_sources):
    """All 51 original columns survive exactly, and only zero exposures produce missing ratios."""
    original = {name: frame.copy(deep=True) for name, frame in full_sources.items()}
    result = build_prematch_v2(**full_sources)
    validate_prematch_v2_frame(result)
    assert result.shape == (8630, 67)
    assert tuple(result.columns) == PREMATCH_V2_COLUMNS
    assert len(PREMATCH_V2_ADDITION_COLUMNS) == 16
    assert result.groupby(level="game_id").size().eq(2).all()
    zero = result.games_played.eq(0)
    assert result.gd_per_game.isna().equals(zero)
    np.testing.assert_array_equal(
        result.loc[~zero, "gd_per_game"],
        (result.goal_difference / result.games_played).loc[~zero],
    )
    assert not np.isinf(result.gd_per_game).any()
    assert result.loc[zero, "points_per_game"].eq(0).all()
    assert_frame_equal(result.loc[:, PREMATCH_COLUMNS], original["prematch"], check_exact=True)
    for name in original:
        assert_frame_equal(full_sources[name], original[name], check_exact=True)


def test_full_builder_preserves_v1_row_order(full_sources):
    """Key-based additions do not reorder the original 51-column frame."""
    shuffled = full_sources["prematch"].sample(frac=1, random_state=19)
    result = build_prematch_v2(shuffled, full_sources["form"].iloc[::-1])
    assert_frame_equal(result.loc[:, PREMATCH_COLUMNS], shuffled, check_exact=True)


@pytest.mark.parametrize(
    "bad", ["rows", "order", "unknown", "ratio", "mirror", "infinite", "missing_form"]
)
def test_complete_validation_fails_closed(full_sources, bad):
    """Damaged full frames cannot be published despite having plausible values elsewhere."""
    frame = build_prematch_v2(**full_sources)
    if bad == "rows":
        frame = frame.iloc[2:]
    elif bad == "order":
        frame = frame.iloc[:, ::-1]
    elif bad == "unknown":
        frame["unregistered_feature"] = 0
    elif bad == "ratio":
        frame.iloc[1, frame.columns.get_loc("gd_per_game")] += 1
    elif bad == "mirror":
        frame.iloc[0, frame.columns.get_loc("opp_points_per_game")] += 1
    elif bad == "infinite":
        frame.iloc[0, frame.columns.get_loc("gd_per_game")] = np.inf
    else:
        frame.iloc[0, frame.columns.get_loc("gf_all_w5")] = np.nan
    with pytest.raises(ValueError):
        validate_prematch_v2_frame(frame)


@pytest.mark.parametrize("season", [26, 27, 28])
def test_holdouts_are_rejected(full_sources, season):
    """A caller cannot include a held-out or live season in v2 assembly."""
    full_sources["prematch"]["season_id"] = season
    with pytest.raises(ValueError, match="development seasons"):
        build_prematch_v2(**full_sources)


def test_v1_schema_and_explicit_m2_base_are_independent():
    """Registry defaults cannot silently expand the ten preregistered M2 input columns."""
    assert tuple(PREMATCH_SCHEMA.columns) == PREMATCH_COLUMNS and len(PREMATCH_COLUMNS) == 51
    assert tuple(PREMATCH_V2_SCHEMA.columns) == PREMATCH_V2_COLUMNS
    assert M2_BASE_COLUMNS == (
        "elo_pre",
        "elo_diff",
        "is_home",
        "gf_venue_ewma5",
        "opp_ga_venue_ewma5",
        "value_avg_log_ratio",
        "stage",
        "rounds_to_stage_end",
        "promoted_in",
        "opp_promoted_in",
    )
    for column in M2_BASE_COLUMNS:
        spec = FEATURES[column]
        assert dict(spec.model_status).get("M2", spec.status) == "✅"
    for column in PREMATCH_V2_ADDITION_COLUMNS:
        spec = FEATURES[column]
        assert dict(spec.model_status).get("M2", spec.status) == (
            "✅" if column == "opp_promoted_in" else "🔬"
        )
    assert "elo_expected" not in M2_BASE_COLUMNS and "elo_opponent_pre" not in M2_BASE_COLUMNS


@pytest.mark.parametrize(
    "column,roles",
    [
        ("elo_pre", {"M1": "🔬", "M4": "🔬"}),
        ("elo_diff", {"M1": "🔬", "M4": "🔬"}),
        ("gf_venue_ewma5", {"M4": "🔬"}),
        ("opp_ga_venue_ewma5", {"M4": "🔬"}),
        ("promoted_in", {"M4": "✅", "M5": "✅"}),
    ],
)
def test_existing_other_model_statuses_are_preserved(column, roles):
    """Each updated existing feature retains its previous roles outside M2."""
    spec = FEATURES[column]
    assert set(spec.models) - {"M2"} == set(roles)
    for model, status in roles.items():
        assert dict(spec.model_status).get(model, spec.status) == status


@pytest.mark.parametrize("source", ["prematch", "form"])
@pytest.mark.parametrize("which", ["parquet", "sidecar"])
def test_all_checksums_precede_every_parquet_read(
    tmp_path, monkeypatch, full_sources, source, which
):
    """Either input or sidecar hash failure prevents even the first DataFrame load."""
    config = write_sources(tmp_path, full_sources)
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_v2_config(config)
    target = spec.inputs[source].path
    if which == "sidecar":
        target = target.with_suffix(".metadata.json")
    target.write_bytes(target.read_bytes() + b"tampered")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Premature Parquet read"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_prematch_v2_inputs(spec, seasons=DEVELOPMENT_SEASONS)


def test_bad_configured_hash_and_seasons_fail_before_loading(tmp_path, monkeypatch, full_sources):
    """A valid-looking wrong YAML checksum and a forbidden season scope both fail closed."""
    config = write_sources(tmp_path, full_sources)
    raw = yaml.safe_load(config.read_text())
    raw["inputs"]["form"]["sha256"] = "f" * 64
    config.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_v2_config(config)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Premature Parquet read"))
    with pytest.raises(ValueError, match="checksum"):
        load_prematch_v2_inputs(spec, seasons=DEVELOPMENT_SEASONS)
    with pytest.raises(ValueError, match="explicit seasons"):
        load_prematch_v2_inputs(spec, seasons=range(8, 28))


def test_publication_roundtrip_idempotence_and_v1_bytes(tmp_path, monkeypatch, full_sources):
    """Publishing and verifying v2 leaves both frozen v1 files byte-for-byte untouched."""
    config = write_sources(tmp_path, full_sources)
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_v2_config(config)
    inputs = load_prematch_v2_inputs(spec, seasons=DEVELOPMENT_SEASONS)
    original = spec.inputs["prematch"].path
    before_v1 = sha(original), sha(original.with_suffix(".metadata.json"))
    frame = build_prematch_v2(**inputs)
    assert ensure_prematch_v2(frame, spec, provenance(spec))
    target = Path("data/derived/prematch/prematch_v2.parquet")
    sidecar = target.with_suffix(".metadata.json")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (target, sidecar)}
    assert not ensure_prematch_v2(frame, spec, provenance(spec))
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (target, sidecar)}
    assert before_v1 == (sha(original), sha(original.with_suffix(".metadata.json")))
    assert_frame_equal(pd.read_parquet(target), frame, check_exact=True)
    meta = json.loads(sidecar.read_text())
    assert meta["version"] == "2" and meta["component"] == "prematch"
    assert meta["column_count"] == 67 and meta["rows"] == 8630
    assert meta["artifact_sha256"] == sha(target)
    assert meta["inputs"] == spec.input_records()
    assert "Q26" in meta["decision"] and "Q31" in meta["decision"]
    with pytest.raises(FileExistsError):
        save_prematch_v2(frame, spec, provenance(spec))


@pytest.mark.parametrize("bad", ["partial", "tampered", "different"])
def test_saved_version_cannot_be_replaced(tmp_path, monkeypatch, full_sources, bad):
    """Incomplete, corrupted or different existing output is never overwritten."""
    config = write_sources(tmp_path, full_sources)
    monkeypatch.chdir(tmp_path)
    spec = load_prematch_v2_config(config)
    frame = build_prematch_v2(**full_sources)
    ensure_prematch_v2(frame, spec, provenance(spec))
    target = Path("data/derived/prematch/prematch_v2.parquet")
    if bad == "partial":
        target.with_suffix(".metadata.json").unlink()
    elif bad == "tampered":
        target.write_bytes(target.read_bytes() + b"tampered")
    else:
        frame = frame.copy()
        frame.iloc[0, frame.columns.get_loc("gf_all_w5")] += 1
    original = target.read_bytes()
    with pytest.raises(ValueError, match="incomplete|checksum|differs"):
        ensure_prematch_v2(frame, spec, provenance(spec))
    assert target.read_bytes() == original


def test_cli_version_two_uses_frozen_sources(tmp_path, monkeypatch, full_sources):
    """The explicit v2 CLI publishes once and then verifies without recomputing features."""
    config = write_sources(tmp_path, full_sources)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ligat.cli.subprocess.check_output", lambda *a, **k: "test-revision")
    runner = CliRunner()
    args = ["run", "prematch", "--version", "2", "--config", str(config)]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "created" in first.output
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "up to date" in second.output


def test_score_change_cannot_change_own_67_features_but_changes_later(monkeypatch):
    """Recompute synthetic upstream blocks to prove causality with a nonempty later-game control."""
    from ligat.features.elo import compute_elo
    from ligat.features.form import compute_form
    from ligat.features.season_context import compute_season_context
    from ligat.features.table_state import TableStateBlock
    from tests.test_form import change_score, fixed_neutral, rules, season_fixtures, tables_from

    original = tables_from(season_fixtures(9))
    team_values = original.team_matches.reset_index()[["season_id", "team_id"]].drop_duplicates()
    team_values["avg_value_eur"] = 1_000_000.0 + team_values.team_id * 10_000
    team_values["total_value_eur"] = team_values.avg_value_eur * 20
    original = replace(original, team_seasons=team_values)
    # Only fixture dimensions are reduced; v1's actual arithmetic and schema are reused.
    monkeypatch.setattr("ligat.features.pipeline.PREMATCH_ROWS", len(original.team_matches))
    monkeypatch.setattr("ligat.features.pipeline.DEVELOPMENT_SEASONS", (8, 9))

    def assemble(tables):
        state = TableStateBlock(rules()).compute(tables)
        form = compute_form(tables, rules(), neutral=fixed_neutral()).features
        elo = compute_elo(tables, state.nothing_to_play, E0050_PARAMS).prematch
        context = compute_season_context(tables, rules()).features
        v1 = build_prematch(state, elo, form, context)
        return build_prematch_v2(v1, form)

    before = assemble(original)
    after = assemble(change_score(original, 902, 9))
    assert before.shape[1] == 67
    assert_frame_equal(before.loc[902], after.loc[902], check_exact=True)
    assert before.loc[(903, 1), "gf_all_w5"] != after.loc[(903, 1), "gf_all_w5"]


@pytest.mark.real_data
def test_real_sources_publish_exact_v2_without_modifying_v1(real_tables, tmp_path, monkeypatch):
    """Actual development sources reproduce all copies/mirrors and publish safely in a temp root."""
    assert set(real_tables.team_matches.season_id) == set(DEVELOPMENT_SEASONS)
    spec = load_prematch_v2_config()
    original_v1 = spec.inputs["prematch"].path.resolve()
    before = sha(original_v1), sha(original_v1.with_suffix(".metadata.json"))
    config = Path("config/prematch_v2.yaml").resolve()
    for item in spec.inputs.values():
        for source in (item.path, item.path.with_suffix(".metadata.json")):
            destination = tmp_path / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    monkeypatch.chdir(tmp_path)
    inputs = load_prematch_v2_inputs(load_prematch_v2_config(config), seasons=tuple(range(8, 26)))
    frame = build_prematch_v2(**inputs)
    validate_prematch_v2_frame(frame)
    assert_frame_equal(frame.loc[:, PREMATCH_COLUMNS], inputs["prematch"], check_exact=True)
    assert (
        frame.gd_per_game.isna().sum()
        == frame.groupby("season_id")
        .apply(lambda f: f.index.get_level_values("team_id").nunique(), include_groups=False)
        .sum()
    )
    assert frame.loc[frame.games_played.eq(0), "points_per_game"].eq(0).all()
    for column in PREMATCH_V2_ADDITION_COLUMNS[:-1]:
        assert frame[column].notna().all()
    assert ensure_prematch_v2(frame, spec, provenance(spec))
    saved = pd.read_parquet(
        "data/derived/prematch/prematch_v2.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    assert_frame_equal(saved, frame, check_exact=True)
    assert before == (sha(original_v1), sha(original_v1.with_suffix(".metadata.json")))
