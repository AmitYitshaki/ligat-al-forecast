"""Offline golden pages and explicit missing-data/event semantics."""

import gzip
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ligat.data.details import cross_check, parse_game, parse_minute
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


@pytest.mark.parametrize("game_id,kind", [(113512, "halftime_goals"), (864982, "goals_total")])
def test_real_source_discrepancies_are_not_silently_corrected(game_id: int, kind: str) -> None:
    ids = json.loads((FIXTURES / f"{game_id}.source.json").read_text())
    page = gzip.decompress((FIXTURES / f"{game_id}.html.gz").read_bytes())
    game = parse_game(page, game_id, ids["home_id"], ids["away_id"])
    assert {
        row["kind"] for row in cross_check(game, ids["home_id"], ids["away_id"], game.final_score)
    } == {kind}
