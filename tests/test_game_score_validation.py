"""Real-page score comparisons must gate publication without network access."""

import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ligat.data.build import SeasonInput, build_datasets
from ligat.data.game_pages import GamePageRepository, parse_game_score
from ligat.data.parse import ParsedRound, parse_round
from ligat.data.repository import Repository

FIXTURES = Path(__file__).parent / "fixtures"


def _inputs(tmp_path: Path, *, documented: bool) -> dict:
    parsed = parse_round((FIXTURES / "parse" / "twelve.xml").read_bytes())
    game = replace(parsed.games[0], game_id=339766, home_goals=3, away_goals=0)
    config = {
        "season_id": 15,
        "label": "2013/14",
        "regular_rounds": 1,
        "n_teams": 12,
        "playoffs": [],
        "points_halved": False,
        "points_win": 3,
        "points_draw": 1,
    }
    season = SeasonInput(
        config,
        tmp_path,
        {"files": {"round_01.xml": {"stages": []}}},
        {1: ParsedRound((game,), parsed.table)},
    )
    content = (FIXTURES / "games" / "339766.html").read_bytes()
    compressed = gzip.compress(content, mtime=0)
    root = tmp_path / "pages"
    directory = root / "15"
    directory.mkdir(parents=True)
    (directory / "339766.html.gz").write_bytes(compressed)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "season_id": 15,
                "status": "completed",
                "files": {
                    "339766.html.gz": {
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "compressed_bytes": len(compressed),
                        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                    }
                },
            }
        )
    )
    seasons = tmp_path / "seasons.yaml"
    seasons.write_text(yaml.safe_dump({"defaults": {}, "seasons": [config]}))
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        yaml.safe_dump(
            {
                "matches": {
                    339766: {
                        "status": "awarded",
                        "action": "drop",
                        "verified": documented,
                    }
                }
            }
        )
    )
    return {
        "games_root": root,
        "seasons_path": seasons,
        "overrides_path": overrides,
        "teams_path": tmp_path / "teams.yaml",
        "season": season,
    }


def test_real_score_orientation_and_halftime_is_not_used() -> None:
    assert parse_game_score((FIXTURES / "games" / "339766.html").read_bytes(), 339766) == (2, 1)
    assert parse_game_score((FIXTURES / "games" / "113478.html").read_bytes(), 113478) == (4, 1)
    with pytest.raises(ValueError, match="identity"):
        parse_game_score((FIXTURES / "games" / "339766.html").read_bytes(), 999)


def test_documented_mismatch_passes_and_is_dropped(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, documented=True)
    season = inputs.pop("season")
    before = {p: p.read_bytes() for p in inputs["games_root"].rglob("*") if p.is_file()}
    with (
        patch("ligat.data.build._load_season", return_value=season),
        patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")),
    ):
        frames = build_datasets(**inputs, repository=Repository(tmp_path / "clean"))
    assert frames["matches"].empty
    assert len(frames["round_tables"]) == 12
    assert before == {p: p.read_bytes() for p in before}


def test_unverified_mismatch_fails_without_replacing_good_output(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, documented=False)
    season = inputs.pop("season")
    repo = Repository(tmp_path / "clean")
    repo.root.mkdir()
    original = b"Existing clean data must remain unchanged"
    (repo.root / "matches.parquet").write_bytes(original)
    with (
        patch("ligat.data.build._load_season", return_value=season),
        pytest.raises(
            ValueError, match=r"Undocumented game score mismatches:[\s\S]*game_id=339766"
        ) as error,
    ):
        build_datasets(**inputs, repository=repo)
    assert "page_home_away=(2, 1)" in str(error.value)
    assert "round_home_away=(3, 0)" in str(error.value)
    assert (repo.root / "matches.parquet").read_bytes() == original
    assert not inputs["teams_path"].exists()


def test_undocumented_mismatch_lists_all_games(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, documented=False)
    season = inputs.pop("season")
    original = season.rounds[1].games[0]
    second = replace(original, game_id=339767)
    season = replace(season, rounds={1: ParsedRound((original, second), season.rounds[1].table)})
    directory = inputs["games_root"] / "15"
    content = gzip.decompress((directory / "339766.html.gz").read_bytes()).replace(
        b"339766", b"339767"
    )
    compressed = gzip.compress(content, mtime=0)
    (directory / "339767.html.gz").write_bytes(compressed)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["339767.html.gz"] = {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    with (
        patch("ligat.data.build._load_season", return_value=season),
        pytest.raises(ValueError) as error,
    ):
        build_datasets(**inputs, repository=Repository(tmp_path / "clean"))
    assert "game_id=339766" in str(error.value) and "game_id=339767" in str(error.value)
    assert not (tmp_path / "clean").exists()


@pytest.mark.parametrize("fault", ["missing", "compressed", "content", "manifest", "gzip"])
def test_page_integrity_is_mandatory(tmp_path: Path, fault: str) -> None:
    inputs = _inputs(tmp_path, documented=True)
    root = inputs["games_root"]
    path = root / "15" / "339766.html.gz"
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "compressed":
        path.write_bytes(path.read_bytes() + b"corruption")
    elif fault == "content":
        manifest["files"][path.name]["sha256"] = "wrong"
    elif fault == "gzip":
        compressed = b"invalid gzip"
        path.write_bytes(compressed)
        manifest["files"][path.name].update(
            compressed_bytes=len(compressed),
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        )
    else:
        manifest["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        GamePageRepository(root).score(15, 339766)
