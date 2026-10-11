"""Offline golden parsing and explicit checks against independently read source values."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ligat.data.parse import clean_team_name, clean_venue, parse_round, parse_score

FIXTURES = Path(__file__).parent / "fixtures" / "parse"


@pytest.mark.parametrize(
    "name",
    [
        "twelve",
        "sixteen_halved",
        "sixteen_whole",
        "fourteen",
        "mixed",
        "future",
        "void",
        "promotion",
        "halved_baseline",
        "whole_baseline",
    ],
)
def test_golden_round(name: str) -> None:
    parsed = parse_round((FIXTURES / f"{name}.xml").read_bytes())
    actual = json.loads(json.dumps(asdict(parsed), default=str, ensure_ascii=False))
    assert actual == json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_source_ids_and_inverted_goals() -> None:
    parsed = parse_round((FIXTURES / "twelve.xml").read_bytes())
    first = parsed.games[0]
    assert (first.game_id, first.home_team_id, first.away_team_id) == (113478, 2266, 2173)
    assert (first.home_goals, first.away_goals) == (4, 1)
    assert str(first.date.tzinfo) == "Asia/Jerusalem"
    assert (first.date.hour, first.date.minute) == (17, 30)
    kfar_saba = next(row for row in parsed.table if row.team_id == 2266)
    assert (kfar_saba.goals_for, kfar_saba.goals_against) == (4, 1)


def test_mixed_round_keeps_individual_future_games() -> None:
    games = parse_round((FIXTURES / "mixed.xml").read_bytes()).games
    assert sum(game.home_goals is not None for game in games) == 5
    assert sum(game.home_goals is None for game in games) == 2
    assert all(game.game_id is None for game in games if game.home_goals is None)


def test_names_containing_hyphen_are_preserved() -> None:
    games = parse_round((FIXTURES / "sixteen_halved.xml").read_bytes()).games
    assert games[0].away_name == 'בית"ר י-ם'


@pytest.mark.parametrize("raw", ["(סגור)בלומפילד", "(סגור) בלומפילד", "בלומפילד"])
def test_clean_venue(raw: str) -> None:
    assert clean_venue(raw) == "בלומפילד"


def test_normalize_name_without_changing_identity() -> None:
    assert clean_team_name('מכבי  פ"ת / לא פעיל') == 'מכבי פ"ת'


def test_score_rejects_non_score() -> None:
    assert parse_score("0 - 3") == (3, 0)
    with pytest.raises(ValueError, match="Invalid score"):
        parse_score("Temporary error")


@pytest.mark.parametrize("content", [b"<html>Error</html>", b"<Response><HtmlData /></Response>"])
def test_missing_source_html_is_rejected(content: bytes) -> None:
    with pytest.raises(ValueError, match="HtmlData"):
        parse_round(content)
