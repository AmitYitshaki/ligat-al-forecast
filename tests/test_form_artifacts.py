"""Immutable derived publishing, provenance and round trips for B4a outputs."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.features.elo import compute_elo
from ligat.features.form import compute_form, neutral_form_values
from ligat.features.form_artifacts import (
    E0050_PARAMS,
    FormProvenance,
    form_input_fingerprint,
    save_e0050,
    save_form,
)
from tests.test_form import change_score, rules, season_fixtures, tables_from


def provenance() -> FormProvenance:
    return FormProvenance("revision", "a" * 64, {"ligat/features/form.py": "b" * 64})


def test_form_round_trip_metadata_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tables = tables_from(season_fixtures(8))
    result = compute_form(tables, rules())
    target, sidecar = save_form(result, provenance())
    assert_frame_equal(pd.read_parquet(target), result.features)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["artifact_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert meta["input_sha256"] == "a" * 64
    assert meta["source_sha256"] == dict(provenance().source_sha256)
    assert len(meta["coefficients"]) == 9 and len(meta["openings"]) == 36
    assert all(r["rho_raw"] is None for r in meta["coefficients"])
    assert "training only" in meta["season_8_exception"]
    assert meta["fallback_counts"] == result.fallback_counts.to_dict("records")
    before = target.read_bytes(), sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        save_form(result, provenance())
    assert (target.read_bytes(), sidecar.read_bytes()) == before
    assert not list(target.parent.glob(".*.parquet"))


def test_e0050_round_trip_without_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tables = tables_from(season_fixtures(8))
    index = tables.team_matches.reset_index().set_index(["game_id", "team_id"]).index
    flags = pd.Series(False, index=index, name="nothing_to_play")
    elo = compute_elo(tables, flags, E0050_PARAMS)
    target, sidecar = save_e0050(elo, provenance(), table_state_sha256="c" * 64)
    assert_frame_equal(pd.read_parquet(target), elo.prematch)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["config_id"] == E0050_PARAMS.config_id
    assert meta["params"]["k"] == 10 and meta["params"]["dead_game_discount"] == 1
    assert meta["table_state_sha256"] == "c" * 64
    assert meta["artifact_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Only the selected"):
        save_e0050(replace(elo, config_id="other"), provenance(), table_state_sha256="c" * 64)
    with pytest.raises(FileExistsError):
        save_e0050(elo, provenance(), table_state_sha256="c" * 64)


@pytest.mark.parametrize(
    "directory", ["data/raw", "data/clean", "elsewhere", "data/derived/../../raw"]
)
def test_cannot_publish_outside_derived(
    directory: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = compute_form(tables_from(season_fixtures(8)), rules())
    with pytest.raises(ValueError, match="inside data/derived"):
        save_form(result, provenance(), directory=Path(directory))
    assert not list(tmp_path.rglob("*.parquet"))


def test_metadata_alone_also_blocks_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = Path("data/derived/form")
    directory.mkdir(parents=True)
    sidecar = directory / "form_v1.metadata.json"
    sidecar.write_text("sentinel", encoding="utf-8")
    result = compute_form(tables_from(season_fixtures(8)), rules())
    with pytest.raises(FileExistsError):
        save_form(result, provenance())
    assert sidecar.read_text() == "sentinel"
    assert not list(directory.glob("*.parquet"))


def test_input_hash_excludes_official_views_but_covers_results_rules_and_neutral() -> None:
    tables = tables_from(season_fixtures(8))
    neutral = neutral_form_values(tables)
    baseline = form_input_fingerprint(tables, rules(), neutral)
    assert baseline == form_input_fingerprint(
        replace(tables, team_seasons=pd.DataFrame({"points_official": [999]})), rules(), neutral
    )
    assert baseline != form_input_fingerprint(change_score(tables, 802, 8), rules(), neutral)
    changed_rules = rules() | {8: replace(rules()[8], points_halved=True)}
    assert baseline != form_input_fingerprint(tables, changed_rules, neutral)
    assert baseline != form_input_fingerprint(tables, rules(), neutral * 2)


def test_incomplete_or_nonfinite_candidates_are_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = compute_form(tables_from(season_fixtures(8)), rules())
    with pytest.raises(ValueError, match="66 finite"):
        save_form(replace(result, features=result.features.iloc[:, :-1]), provenance())
    bad = result.features.copy()
    bad.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError, match="66 finite"):
        save_form(replace(result, features=bad), provenance())


def test_provenance_requires_full_hashes() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        FormProvenance("revision", "short", {"form.py": "b" * 64})
