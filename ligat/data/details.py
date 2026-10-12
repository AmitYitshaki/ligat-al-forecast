"""Pure parsing and cross-validation of static IFA game details; never accesses the network."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup, SoupStrainer  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ParsedGame:
    """One page's source facts and event rows before schema validation."""

    details: dict[str, Any]
    goals: list[dict[str, Any]]
    red_cards: list[dict[str, Any]]
    coaches: list[dict[str, Any]]
    final_score: tuple[int, int]
    lineup_counts: tuple[int, int]
    timeline_nodes: int


def parse_minute(value: str) -> tuple[int, int | None]:
    """Preserve explicit stoppage time separately; a plain minute has null added_time."""
    value = value.replace("′", "").replace("'", "").strip()
    match = re.fullmatch(r"(\d+)\s*(?:\+\s*(\d+))?", value)
    if match is None:
        raise ValueError(f"Invalid event minute: {value!r}")
    return int(match[1]), int(match[2]) if match[2] is not None else None


def parse_game(content: bytes, game_id: int, home_id: int, away_id: int) -> ParsedGame:
    """Read header, timeline, lineups and coaches without inferring missing source facts."""
    soup = BeautifulSoup(content, "lxml", parse_only=SoupStrainer("section", id=f"gfd_{game_id}"))
    section = soup.select_one(f"section.game-details#gfd_{game_id}")
    if section is None:
        raise ValueError(f"Incorrect game page identity: game_id={game_id}")
    for label in section.select(".sr-only"):
        label.decompose()

    def score(selector: str) -> tuple[int, int] | None:
        node = section.select_one(selector)
        if node is None:
            return None
        match = re.search(r"(\d+)\s*:\s*(\d+)", node.get_text(" ", strip=True))
        if match is None:
            raise ValueError(f"Invalid score in {selector}: game_id={game_id}")
        return int(match[1]), int(match[2])

    final = score(".game-time .total")
    if final is None:
        raise ValueError(f"Missing final score: game_id={game_id}")
    halftime = score(".game-time .result-half")
    goals: list[dict[str, Any]] = []
    reds: list[dict[str, Any]] = []
    nodes = section.select(".timeline .node")
    for node in nodes:
        classes = set(node.get_attribute_list("class"))
        goal = bool(classes & {"goal", "penalty", "owngoal"})
        red = bool(classes & {"red", "yellow2"})
        if not goal and not red:
            continue
        if ("team1" in classes) == ("team2" in classes):
            raise ValueError(f"Missing/ambiguous event team: game_id={game_id}")
        team = home_id if "team1" in classes else away_id
        minute_node = node.select_one(".minute div")
        if minute_node is None:
            raise ValueError(f"Missing event minute: game_id={game_id}")
        minute_text = minute_node.get_text(" ", strip=True)
        minute, added = (
            (None, None) if red and not goal and minute_text == "-1" else parse_minute(minute_text)
        )
        if goal:
            own = "owngoal" in classes
            goals.append(
                {
                    "game_id": game_id,
                    "team_id": (away_id if team == home_id else home_id) if own else team,
                    "minute": minute,
                    "added_time": added,
                    "is_penalty": "penalty" in classes,
                    "is_own_goal": own,
                }
            )
        if red:
            reds.append(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "minute": minute,
                    "is_second_yellow": "yellow2" in classes,
                }
            )
    coaches = []
    for side, team in (("HOME", home_id), ("GUEST", away_id)):
        container = section.select_one(f'.players-cont [aria-labelledby="GAME_COACH_{side}"]')
        name_node = container.select_one(".name") if container is not None else None
        name = name_node.get_text(" ", strip=True) if name_node is not None else None
        anchor = container.select_one("a[href]") if container is not None else None
        query = parse_qs(urlparse(str(anchor["href"])).query) if anchor is not None else {}
        coach_id = int(query["coach_id"][0]) if "coach_id" in query else None
        coaches.append(
            {
                "game_id": game_id,
                "team_id": team,
                "coach_id": coach_id if coach_id is not None and coach_id > 0 else None,
                "coach_name": name or None,
            }
        )
    referee = None
    for judge in section.select(".judge .player"):
        role = judge.select_one(".position")
        judge_name = judge.select_one(".name")
        if role is not None and role.get_text(strip=True) == "שופט ראשי" and judge_name is not None:
            referee = judge_name.get_text(" ", strip=True) or None
    lineup_counts = tuple(
        len(section.select(f".players-cont .{side}.Active a[href]")) for side in ("home", "guest")
    )
    details = {
        "game_id": game_id,
        "ht_home_goals": halftime[0] if halftime is not None else None,
        "ht_away_goals": halftime[1] if halftime is not None else None,
        "referee": referee,
        "has_timeline": bool(nodes),
        "has_lineups": all(lineup_counts),
        "has_coaches": all(row["coach_name"] is not None for row in coaches),
    }
    return ParsedGame(
        details, goals, reds, coaches, final, (lineup_counts[0], lineup_counts[1]), len(nodes)
    )


def cross_check(
    game: ParsedGame, home_id: int, away_id: int, expected: tuple[int, int]
) -> list[dict[str, Any]]:
    """Return every discrepancy so the builder can report and gate publication."""
    issues = []
    totals = Counter(row["team_id"] for row in game.goals)
    halves = Counter(row["team_id"] for row in game.goals if row["minute"] <= 45)
    actual = totals[home_id], totals[away_id]
    halftime = game.details["ht_home_goals"], game.details["ht_away_goals"]
    for kind, found, wanted in (
        ("final_score", game.final_score, expected),
        ("goals_total", actual, expected),
        ("halftime_goals", (halves[home_id], halves[away_id]), halftime),
    ):
        if found != wanted:
            issues.append(
                {
                    "game_id": game.details["game_id"],
                    "kind": kind,
                    "actual": found,
                    "expected": wanted,
                }
            )
    for flag, expected_flag in (
        ("has_coaches", all(row["coach_name"] is not None for row in game.coaches)),
        ("has_timeline", bool(game.timeline_nodes)),
        ("has_lineups", all(game.lineup_counts)),
    ):
        if game.details[flag] != expected_flag:
            issues.append(
                {
                    "game_id": game.details["game_id"],
                    "kind": flag,
                    "actual": game.details[flag],
                    "expected": expected_flag,
                }
            )
    return issues
