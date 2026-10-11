"""League rules, manifest integrity, snapshot selection and validated Parquet tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pandera.pandas as pa
import pytest
import yaml
from typer.testing import CliRunner

from ligat.cli import app
from ligat.data.build import (
    SeasonInput,
    _groups,
    _load_overrides,
    _table_rows,
    build_datasets,
    latest_completed_snapshot,
    model_points,
)
from ligat.data.parse import ParsedRound, TableRow, parse_round
from ligat.data.repository import Repository
from ligat.data.schemas import make_frame

FIXTURES = Path(__file__).parent / "fixtures" / "parse"


@pytest.mark.parametrize(
    "halved,after,expected",
    [
        (True, True, 42),
        (True, False, 80),
        (False, True, 80),
        (False, False, 80),
    ],
)
def test_half_only_regular_points(halved: bool, after: bool, expected: int) -> None:
    # 77 regular points -> 39 carried points, plus 3 earned in playoffs.
    assert model_points(80, 77, halved=halved, after_regular=after) == expected


def test_verified_overrides_only() -> None:
    overrides = _load_overrides(Path("config/match_overrides.yaml"))
    assert overrides == {
        211602: "awarded",
        240618: "awarded",
        388645: "void",
        525468: "awarded",
        732553: "awarded",
        732312: "awarded",
        954863: "void",
        1060422: "awarded",
    }
    assert 732462 not in overrides  # Real 3:0 victory, not a technical result.


def test_build_drops_exactly_eight_games_and_preserves_void_points(tmp_path: Path) -> None:
    parsed = parse_round((FIXTURES / "void.xml").read_bytes())
    void = next(game for game in parsed.games if game.game_id == 388645)
    dropped = set(_load_overrides(Path("config/match_overrides.yaml")))
    assert dropped == {211602, 240618, 388645, 525468, 732553, 732312, 954863, 1060422}
    games = tuple(replace(void, game_id=game_id) for game_id in sorted(dropped | {999999}))
    config = {
        "season_id": 16,
        "label": "2014/15",
        "regular_rounds": 7,
        "n_teams": 14,
        "playoffs": [],
        "points_halved": False,
        "points_win": 3,
        "points_draw": 1,
    }
    season = SeasonInput(
        config,
        tmp_path,
        {"files": {"round_07.xml": {"stages": []}}},
        {7: ParsedRound(games, parsed.table)},
    )
    seasons_path = tmp_path / "seasons.yaml"
    seasons_path.write_text(yaml.safe_dump({"defaults": {}, "seasons": [config]}))
    baseline = make_frame("round_tables", _table_rows(season, [(7, 2176, 1061)]))
    with patch("ligat.data.build._load_season", return_value=season):
        frames = build_datasets(
            seasons_path=seasons_path,
            teams_path=tmp_path / "teams.yaml",
            repository=Repository(tmp_path / "clean"),
        )
    assert set(frames["matches"].game_id) == {999999}
    assert set(frames["matches"].status) == {"played"}
    assert len(games) - len(frames["matches"]) == 8
    pd.testing.assert_frame_equal(frames["round_tables"], baseline)
    table = frames["round_tables"].set_index("team_id")
    assert table.loc[2176, "points_model"] == 10
    assert table.loc[1061, "points_model"] == 15


def test_void_no_point_and_deduction_not_in_model() -> None:
    row = TableRow(2176, "Team", 1, 7, 3, 2, 2, 6, 6, 10)
    season = SeasonInput(
        {
            "season_id": 16,
            "regular_rounds": 7,
            "points_win": 3,
            "points_draw": 1,
            "points_halved": False,
        },
        Path("unused"),
        {"files": {"round_07.xml": {"stages": []}}},
        {7: ParsedRound((), (row,))},
    )
    table = _table_rows(season, [(7, 2176, 1061)])[0]
    assert table["points_model"] == 10
    assert table["deduction"] == 0
    assert _table_rows(season, [])[0]["points_model"] == 11
    punished = TableRow(2176, "Team", 1, 7, 3, 2, 2, 6, 6, 5)
    season = SeasonInput(
        season.config, season.directory, season.manifest, {7: ParsedRound((), (punished,))}
    )
    table = _table_rows(season, [(7, 2176, 1061)])[0]
    assert (table["points_model"], table["points_official"], table["deduction"]) == (10, 5, -5)


@pytest.mark.parametrize(
    "sid,baseline,playoff,halved,sizes",
    [
        (11, "halved_baseline", "sixteen_halved", True, {"upper": 6, "middle": 4, "lower": 6}),
        (13, "whole_baseline", "sixteen_whole", False, {"upper": 8, "lower": 8}),
    ],
)
def test_real_playoff_groups_and_point_carry(
    sid: int, baseline: str, playoff: str, halved: bool, sizes: dict[str, int]
) -> None:
    season = SeasonInput(
        {
            "season_id": sid,
            "regular_rounds": 30,
            "points_win": 3,
            "points_draw": 1,
            "points_halved": halved,
            "n_teams": 16,
            "playoffs": list(sizes),
            "playoff_groups": sizes,
        },
        Path("unused"),
        {"files": {"round_30.xml": {"stages": []}, "round_31.xml": {"stages": []}}},
        {
            30: parse_round((FIXTURES / f"{baseline}.xml").read_bytes()),
            31: parse_round((FIXTURES / f"{playoff}.xml").read_bytes()),
        },
    )
    groups = _groups(season)
    assert {name: list(groups.values()).count(name) for name in sizes} == sizes
    rows = _table_rows(season, [])
    if halved:
        haifa = next(row for row in rows if row["round"] == 31 and row["team_id"] == 1005)
        assert (haifa["points_model"], haifa["points_official"]) == (42, 42)
    else:
        for row in rows:
            assert row["points_model"] == 3 * row["won"] + row["drawn"]


def _collection(directory: Path, sid: int, files: dict[int, str]) -> None:
    directory.mkdir(parents=True)
    entries = {}
    for round_id, fixture in files.items():
        content = (FIXTURES / f"{fixture}.xml").read_bytes()
        name = f"round_{round_id:02d}.xml"
        (directory / name).write_bytes(content)
        entries[name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "downloaded_at": "2026-10-11T08:23:28+00:00",
            "stages": [{"name": "מבחן עליה" if fixture == "promotion" else "סבב 1"}],
        }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "season_id": sid,
                "status": "completed",
                "completed_at": "2026-10-11T08:24:28+00:00",
                "expected_rounds": sorted(files),
                "files": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "input"
    _collection(raw / "8", 8, {1: "twelve", 2: "promotion"})
    _collection(raw / "28" / "snapshot" / "completed", 28, {6: "mixed", 26: "future"})
    failed = raw / "28" / "snapshot" / "newer_failed"
    failed.mkdir()
    (failed / "manifest.json").write_text(json.dumps({"status": "failed"}))
    config = tmp_path / "seasons.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "defaults": {"points_win": 3, "points_draw": 1},
                "seasons": [
                    {
                        "season_id": 8,
                        "label": "2006/07",
                        "regular_rounds": 1,
                        "n_teams": 12,
                        "playoffs": [],
                        "points_halved": False,
                    },
                    {
                        "season_id": 28,
                        "label": "2026/27",
                        "regular_rounds": 26,
                        "n_teams": 14,
                        "playoffs": ["upper", "lower"],
                        "points_halved": False,
                        "live": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {"raw_root": raw, "seasons_path": config, "teams_path": tmp_path / "teams.yaml"}


def test_build_offline_and_repository_roundtrip(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    repo = Repository(tmp_path / "clean")
    before = {p: p.read_bytes() for p in inputs["raw_root"].rglob("*") if p.is_file()}
    with patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")
    ):
        frames = build_datasets(**inputs, repository=repo)
    assert len(frames["matches"]) == 11
    assert len(frames["fixtures"]) == 9
    assert len(frames["fixtures"].query("`round` == 6")) == 2
    assert 238521 not in set(frames["matches"].game_id)  # Promotion test excluded by stage name.
    assert set(frames["final_tables"].season_id) == {8}
    assert frames["final_tables"].promoted_in.isna().all()
    assert frames["final_tables"].relegated.isna().all()
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, repo.read(name))
    assert before == {p: p.read_bytes() for p in before}
    teams = yaml.safe_load(inputs["teams_path"].read_text(encoding="utf-8"))
    team = next(team for team in teams["teams"] if team["team_id"] == 1064)
    assert 'מכבי  פ"ת' in team["aliases"]
    team["name_en"] = "Manual enrichment"
    inputs["teams_path"].write_text(yaml.safe_dump(teams, allow_unicode=True), encoding="utf-8")
    build_datasets(**inputs, repository=repo)
    teams = yaml.safe_load(inputs["teams_path"].read_text(encoding="utf-8"))
    assert next(t for t in teams["teams"] if t["team_id"] == 1064)["name_en"] == "Manual enrichment"


def test_integrity_failure_does_not_publish(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    (inputs["raw_root"] / "8" / "round_01.xml").write_bytes(b"truncated")
    clean = tmp_path / "clean"
    with pytest.raises(ValueError, match="integrity mismatch"):
        build_datasets(**inputs, repository=Repository(clean))
    assert not clean.exists()
    assert not inputs["teams_path"].exists()


def test_snapshot_requires_completed_status(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = inputs["raw_root"] / "28" / "snapshot"
    assert latest_completed_snapshot(root).name == "completed"
    with pytest.raises(ValueError, match="No completed snapshot"):
        latest_completed_snapshot(tmp_path / "empty")


def test_repository_validates_reads_and_preserves_good_file(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "clean")
    frame = make_frame("fixtures", [])
    repo.write("fixtures", frame)
    original = (repo.root / "fixtures.parquet").read_bytes()
    with pytest.raises(pa.errors.SchemaErrors):
        repo.write("fixtures", frame.drop(columns="stage"))
    assert (repo.root / "fixtures.parquet").read_bytes() == original
    with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            repo.write("fixtures", frame)
    assert (repo.root / "fixtures.parquet").read_bytes() == original
    assert not list(repo.root.glob(".*.parquet"))
    frame.drop(columns="stage").to_parquet(repo.root / "fixtures.parquet")
    with pytest.raises(pa.errors.SchemaErrors):
        repo.read("fixtures")


def test_build_cli(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "build",
            "--raw-root",
            str(inputs["raw_root"]),
            "--clean-root",
            str(tmp_path / "clean"),
            "--seasons-config",
            str(inputs["seasons_path"]),
            "--teams-config",
            str(inputs["teams_path"]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "clean" / "matches.parquet").exists()


def test_repository_rejects_raw_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="raw storage"):
        Repository(tmp_path / "data" / "raw")


def test_build_rejects_outputs_inside_custom_raw_root(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    repo = Repository(inputs["raw_root"] / "output")
    with pytest.raises(ValueError, match="raw storage"):
        build_datasets(**inputs, repository=repo)
    inputs["teams_path"] = inputs["raw_root"] / "teams.yaml"
    with pytest.raises(ValueError, match="raw storage"):
        build_datasets(**inputs, repository=Repository(tmp_path / "clean"))


def test_schema_rejects_same_team_and_duplicate_match() -> None:
    game = parse_round((FIXTURES / "twelve.xml").read_bytes()).games[0]
    record = {
        "game_id": game.game_id,
        "season_id": 8,
        "round": 1,
        "stage": "regular",
        "season": "2006/07",
        "date": game.date,
        "home_team_id": game.home_team_id,
        "away_team_id": game.home_team_id,
        "home_goals": 4,
        "away_goals": 1,
        "venue_raw": game.venue_raw,
        "venue": game.venue,
        "is_home_venue": True,
        "status": "played",
        "source": "ifa",
    }
    with pytest.raises(pa.errors.SchemaErrors):
        make_frame("matches", [record])
    record["away_team_id"] = game.away_team_id
    with pytest.raises(pa.errors.SchemaErrors):
        make_frame("matches", [record, record])
