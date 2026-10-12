"""Pure parsing and cross-validation of static IFA game details; never accesses the network."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
from bs4 import BeautifulSoup, SoupStrainer  # type: ignore[attr-defined]

from ligat.data.game_pages import game_section


def load_exceptions(path: Path) -> dict[int, str]:
    """Only explicitly verified, recognized exceptions can change availability."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["games"]
    result = {}
    for game_id, entry in entries.items():
        reason = entry.get("type", entry["reason"])
        if entry.get("verified") is not True or reason not in {
            "halftime_timeline_conflict",
            "unreliable_timeline_scorers",
        }:
            raise ValueError(f"Unverified or unknown game-detail exception: game_id={game_id}")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError(f"Missing game-detail exception explanation: game_id={game_id}")
        result[int(game_id)] = reason
    return result


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


def parse_game(
    content: bytes, game_id: int, home_id: int, away_id: int, *, exception: str | None = None
) -> ParsedGame:
    """Read header, timeline, lineups and coaches without inferring missing source facts."""
    soup = BeautifulSoup(
        game_section(content, game_id),
        "lxml",
        parse_only=SoupStrainer("section", id=f"gfd_{game_id}"),
    )
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
    if exception not in {None, "halftime_timeline_conflict", "unreliable_timeline_scorers"}:
        raise ValueError(f"Unknown game-detail exception: {exception}")
    ht_conflict = exception == "halftime_timeline_conflict"
    goals_available = exception != "unreliable_timeline_scorers"
    goals: list[dict[str, Any]] = []
    if goals_available:
        for side, team in (("team-home", home_id), ("team-guest", away_id)):
            goalers = section.select_one(f".{side} .goalers")
            if goalers is None:
                continue
            for label in goalers.select("h4"):
                label.decompose()
            text = goalers.get_text(" ", strip=True)
            for entry in text.split(",") if text else []:
                match = re.search(r"\s(\d+(?:\s*\+\s*\d+)?)\s*((?:\([פע]\)\s*)*)$", entry)
                if match is None:
                    raise ValueError(f"Invalid header scorer: game_id={game_id}, entry={entry!r}")
                goal_minute, goal_added = parse_minute(match[1])
                goals.append(
                    {
                        "game_id": game_id,
                        "team_id": team,
                        "minute": goal_minute,
                        "added_time": goal_added,
                        "is_penalty": "(פ)" in match[2],
                        "is_own_goal": "(ע)" in match[2],
                        "half": None,
                        "is_first_half_stoppage": False,
                    }
                )
        # Header sides already credit own goals to the beneficiary; do not flip again.
        goals.sort(key=lambda row: (row["minute"], row["added_time"] or 0, row["team_id"]))
    reds: list[dict[str, Any]] = []
    nodes = section.select(".timeline .node")
    for node in nodes:
        classes = set(node.get_attribute_list("class"))
        red = bool(classes & {"red", "yellow2"})
        if not red:
            continue
        if ("team1" in classes) == ("team2" in classes):
            raise ValueError(f"Missing/ambiguous event team: game_id={game_id}")
        team = home_id if "team1" in classes else away_id
        minute_node = node.select_one(".minute div")
        if minute_node is None:
            raise ValueError(f"Missing event minute: game_id={game_id}")
        minute_text = minute_node.get_text(" ", strip=True)
        minute, added = (None, None) if minute_text == "-1" else parse_minute(minute_text)
        if red:
            reds.append(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "minute": minute,
                    "is_second_yellow": "yellow2" in classes,
                }
            )
    if halftime is not None and not ht_conflict:
        for team, count in ((home_id, halftime[0]), (away_id, halftime[1])):
            team_goals = sorted(
                (row for row in goals if row["team_id"] == team),
                key=lambda row: (row["minute"], row["added_time"] or 0),
            )
            for index, row in enumerate(team_goals):
                row["half"] = 1 if index < count else 2
                row["is_first_half_stoppage"] = row["half"] == 1 and row["minute"] > 45
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
        "ht_conflict": ht_conflict,
        "goals_available": goals_available,
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
    halves = Counter(row["team_id"] for row in game.goals if row["half"] == 1)
    actual = totals[home_id], totals[away_id]
    halftime = game.details["ht_home_goals"], game.details["ht_away_goals"]
    for kind, found, wanted in (
        ("final_score", game.final_score, expected),
        ("goals_total", actual, expected),
        ("halftime_goals", (halves[home_id], halves[away_id]), halftime),
    ):
        if kind != "final_score" and not game.details["goals_available"]:
            continue
        if kind == "halftime_goals" and game.details["ht_conflict"]:
            continue
        if found != wanted:
            issues.append(
                {
                    "game_id": game.details["game_id"],
                    "kind": kind,
                    "actual": found,
                    "expected": wanted,
                }
            )
    if game.details["goals_available"] and not game.details["ht_conflict"]:
        conflicting_goals = [
            row
            for row in game.goals
            if (row["half"] == 2 and row["minute"] < 45)
            or (row["half"] == 1 and row["minute"] > 60)
        ]
        if conflicting_goals:
            issues.append(
                {
                    "game_id": game.details["game_id"],
                    "kind": "halftime_timeline_conflict",
                    "actual": conflicting_goals,
                    "expected": (
                        "First half at minute 60 or earlier; second half at minute 45 or later"
                    ),
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
