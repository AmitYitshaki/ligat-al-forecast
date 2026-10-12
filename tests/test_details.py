"""Offline golden pages and explicit missing-data/event semantics."""

import gzip
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandera.pandas as pa
import pytest

from ligat.data.details import cross_check, load_exceptions, parse_game, parse_minute
from ligat.data.game_pages import GamePageRepository
from ligat.data.schemas import make_frame

FIXTURES = Path(__file__).parent / "fixtures" / "details"


@pytest.mark.parametrize("game_id", [113478, 211600, 211773, 240575, 240618, 1120561])
def test_golden_page(game_id: int) -> None:
    expected = json.loads((FIXTURES / f"{game_id}.json").read_text(encoding="utf-8"))
    parsed = parse_game(
        gzip.decompress((FIXTURES / f"{game_id}.html.gz").read_bytes()),
        game_id,
        expected["home_id"],
        expected["away_id"],
    )
    actual = json.loads(json.dumps(asdict(parsed)))
    assert actual == {key: expected[key] for key in actual}
    assert cross_check(parsed, expected["home_id"], expected["away_id"], parsed.final_score) == []
    for name in ("goals", "red_cards", "coaches"):
        make_frame(name, getattr(parsed, name))
    make_frame("game_details", [parsed.details])


def test_own_goal_credited_to_opponent() -> None:
    page = gzip.decompress((FIXTURES / "240618.html.gz").read_bytes())
    game = parse_game(page, 240618, 1198, 2171)
    own = [row for row in game.goals if row["is_own_goal"]]
    assert [(row["team_id"], row["minute"]) for row in own] == [(1198, 26)]
    assert game.coaches[0]["coach_id"] is None  # Source sentinel -1, real coach name.
    assert game.coaches[0]["coach_name"]


def test_unknown_red_minute_preserves_second_yellow() -> None:
    page = gzip.decompress((FIXTURES / "113478.html.gz").read_bytes())
    page = page.replace(b">85<", b">-1<")
    game = parse_game(page, 113478, 2266, 2173)
    assert game.red_cards[-1]["minute"] is None
    assert game.red_cards[-1]["is_second_yellow"]
    frame = make_frame("red_cards", game.red_cards)
    assert frame.minute.isna().sum() == 1


def test_explicit_stoppage_time_counts_in_first_half() -> None:
    page = gzip.decompress((FIXTURES / "113478.html.gz").read_bytes())
    game = parse_game(page, 113478, 2266, 2173)
    goal = game.goals[0] | {"minute": 45, "added_time": 2}
    game = replace(game, goals=[goal, *game.goals[1:]])
    assert parse_minute("45 + 2'") == (45, 2)
    assert cross_check(game, 2266, 2173, (4, 1)) == []


def test_cross_checks_report_counts_and_flags() -> None:
    page = gzip.decompress((FIXTURES / "113478.html.gz").read_bytes())
    game = parse_game(page, 113478, 2266, 2173)
    game = replace(game, goals=[], details=game.details | {"has_lineups": True})
    assert {row["kind"] for row in cross_check(game, 2266, 2173, (4, 1))} == {
        "goals_total",
        "halftime_goals",
        "has_lineups",
    }


def test_missing_page_has_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="game_id=123.*ligat scrape games --live"):
        GamePageRepository(tmp_path).read(28, 123)


@pytest.mark.parametrize(
    "game_id,kind", [(113512, "halftime_timeline_conflict"), (864982, "goals_total")]
)
def test_real_source_discrepancies_are_not_silently_corrected(game_id: int, kind: str) -> None:
    ids = json.loads((FIXTURES / f"{game_id}.source.json").read_text())
    page = gzip.decompress((FIXTURES / f"{game_id}.html.gz").read_bytes())
    game = parse_game(page, game_id, ids["home_id"], ids["away_id"])
    assert {
        row["kind"] for row in cross_check(game, ids["home_id"], ids["away_id"], game.final_score)
    } == {kind}


@pytest.mark.parametrize("game_id", [113512, 864982])
def test_documented_exceptions_keep_headers_and_other_checks(game_id: int) -> None:
    ids = json.loads((FIXTURES / f"{game_id}.source.json").read_text())
    page = gzip.decompress((FIXTURES / f"{game_id}.html.gz").read_bytes())
    reason = load_exceptions(Path("config/game_detail_exceptions.yaml"))[game_id]
    game = parse_game(page, game_id, ids["home_id"], ids["away_id"], exception=reason)
    assert cross_check(game, ids["home_id"], ids["away_id"], game.final_score) == []
    if game_id == 113512:
        assert game.details["ht_conflict"] and game.details["goals_available"]
        assert (game.details["ht_home_goals"], game.details["ht_away_goals"]) == (0, 1)
        assert game.goals and all(row["half"] is None for row in game.goals)
        assert not any(row["is_first_half_stoppage"] for row in game.goals)
    else:
        assert not game.details["goals_available"] and not game.goals
    assert "final_score" in {
        issue["kind"] for issue in cross_check(game, ids["home_id"], ids["away_id"], (99, 99))
    }


def test_running_stoppage_minutes_keep_source_values() -> None:
    page = gzip.decompress((FIXTURES / "113478.html.gz").read_bytes()).replace(b" 3,", b" 47,")
    game = parse_game(page, 113478, 2266, 2173)
    first = next(row for row in game.goals if row["minute"] == 47)
    assert first["half"] == 1 and first["is_first_half_stoppage"]
    assert first["added_time"] is None
    assert cross_check(game, 2266, 2173, (4, 1)) == []


def test_minute_45_can_belong_to_second_half_by_header() -> None:
    ids = json.loads((FIXTURES / "428714.source.json").read_text())
    page = gzip.decompress((FIXTURES / "428714.html.gz").read_bytes())
    game = parse_game(page, 428714, ids["home_id"], ids["away_id"])
    goal = next(row for row in game.goals if row["minute"] == 45)
    assert goal["half"] == 2 and not goal["is_first_half_stoppage"]
    assert cross_check(game, ids["home_id"], ids["away_id"], game.final_score) == []


def test_real_first_half_stoppage_page_268007() -> None:
    page = gzip.decompress((FIXTURES / "268007.html.gz").read_bytes())
    game = parse_game(page, 268007, 1005, 1198)
    assert game.final_score == (3, 0)
    assert (game.details["ht_home_goals"], game.details["ht_away_goals"]) == (1, 0)
    assert [(g["minute"], g["half"], g["is_first_half_stoppage"]) for g in game.goals] == [
        (46, 1, True),
        (80, 2, False),
        (85, 2, False),
    ]
    assert cross_check(game, 1005, 1198, (3, 0)) == []


@pytest.mark.parametrize("minute,conflict", [(60, False), (61, True)])
def test_first_half_limit_detects_undocumented_conflict(minute: int, conflict: bool) -> None:
    page = gzip.decompress((FIXTURES / "268007.html.gz").read_bytes())
    page = page.replace(b" 46,", f" {minute},".encode())
    game = parse_game(page, 268007, 1005, 1198)
    issues = cross_check(game, 1005, 1198, (3, 0))
    assert bool(issues) == conflict
    if conflict:
        assert issues[0]["kind"] == "halftime_timeline_conflict"


def test_goal_header_is_authoritative_for_own_goal_credit() -> None:
    page = gzip.decompress((FIXTURES / "240618.html.gz").read_bytes())
    page = page.replace(b"node owngoal team2", b"node goal team1")
    game = parse_game(page, 240618, 1198, 2171)
    assert [(g["team_id"], g["minute"]) for g in game.goals if g["is_own_goal"]] == [(1198, 26)]


def test_unassignable_halftime_fails() -> None:
    page = gzip.decompress((FIXTURES / "113478.html.gz").read_bytes())
    game = parse_game(page, 113478, 2266, 2173)
    game = replace(game, details=game.details | {"ht_home_goals": 5})
    assert "halftime_goals" in {issue["kind"] for issue in cross_check(game, 2266, 2173, (4, 1))}


def test_exception_list_is_exact_and_verified(tmp_path: Path) -> None:
    exceptions = load_exceptions(Path("config/game_detail_exceptions.yaml"))
    assert {
        gid for gid, reason in exceptions.items() if reason == "halftime_timeline_conflict"
    } == {
        113512,
        113560,
        140304,
        168624,
        211641,
        339816,
        388670,
        388687,
        634560,
        634579,
        635121,
        683148,
        732630,
    }
    assert exceptions[864982] == "unreliable_timeline_scorers" and len(exceptions) == 14
    path = tmp_path / "bad.yaml"
    path.write_text("games: {1: {reason: halftime_timeline_conflict, verified: false}}")
    with pytest.raises(ValueError, match="Unverified"):
        load_exceptions(path)


@pytest.mark.parametrize("changes", [{"half": 3}, {"is_first_half_stoppage": True}])
def test_goal_schema_rejects_invalid_half_and_stoppage(changes: dict) -> None:
    page = gzip.decompress((FIXTURES / "113478.html.gz").read_bytes())
    game = parse_game(page, 113478, 2266, 2173)
    with pytest.raises(pa.errors.SchemaErrors):
        make_frame("goals", [game.goals[0] | changes])
