"""Pure parsing of saved IFA XML; never accesses the network or changes raw files."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

TIMEZONE = ZoneInfo("Asia/Jerusalem")


@dataclass(frozen=True)
class GameRow:
    """A league fixture, with an optional result and source game identifier."""

    game_id: int | None
    date: datetime
    home_team_id: int
    away_team_id: int
    home_name: str
    away_name: str
    home_goals: int | None
    away_goals: int | None
    venue_raw: str
    venue: str


@dataclass(frozen=True)
class TableRow:
    """Official cumulative counters for one team, before model point calculation."""

    team_id: int
    name: str
    position: int
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    points_official: int


@dataclass(frozen=True)
class ParsedRound:
    """All games and official standings in one saved round response."""

    games: tuple[GameRow, ...]
    table: tuple[TableRow, ...]


def clean_venue(value: str) -> str:
    """Remove the closed-venue prefix, including when no space follows it."""
    return re.sub(r"^\s*\(סגור\)\s*", "", value).strip()


def clean_team_name(value: str) -> str:
    """Normalize display names only; team identity always comes from source IDs."""
    return " ".join(re.sub(r"\s*/\s*לא פעיל\s*$", "", value).split())


def parse_score(value: str) -> tuple[int, int]:
    """Return home, away goals from the source's away-first display."""
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if match is None:
        raise ValueError(f"Invalid score: {value!r}")
    away, home = map(int, match.groups())
    return home, away


def parse_round(content: bytes) -> ParsedRound:
    """Extract played/future games and standings, rejecting malformed source rows."""
    root = ET.fromstring(content)
    fragments = [node.text for node in root.iter() if node.tag.split("}")[-1] == "HtmlData"]
    if len(fragments) != 1 or not fragments[0]:
        raise ValueError("Expected exactly one nonempty HtmlData element")
    soup = BeautifulSoup(fragments[0], "lxml")
    for label in soup.select(".sr-only"):
        label.decompose()
    games: list[GameRow] = []
    table: list[TableRow] = []
    for row in soup.select(".table_row"):
        cells = [cell.get_text(" ", strip=True) for cell in row.select(".table_col")]
        href = str(row.get("href", ""))
        query = parse_qs(urlparse(href).query)
        if row.has_attr("data-team1") or row.has_attr("data-team2"):
            if len(cells) != 5:
                raise ValueError(f"Expected five game cells: {cells!r}")
            names = [name.get_text(strip=True) for name in row.select(".team-name-text")]
            if len(names) != 2:
                raise ValueError("Expected two team-name spans")
            names[0] = re.sub(r"\s+-\s*$", "", names[0]).strip()
            game_id = int(query["game_id"][0]) if "game_id" in query else None
            goals = parse_score(cells[4]) if cells[4] else (None, None)
            if cells[4] and game_id is None:
                raise ValueError("Played game has no game_id")
            time = cells[3] if re.fullmatch(r"\d{2}:\d{2}", cells[3]) else "00:00"
            if cells[4] and time == "00:00" and cells[3] != "00:00":
                raise ValueError("Played game has unknown kickoff time")
            date = datetime.strptime(f"{cells[0]} {time}", "%d/%m/%Y %H:%M").replace(
                tzinfo=TIMEZONE
            )
            games.append(
                GameRow(
                    game_id,
                    date,
                    int(str(row["data-team1"])),
                    int(str(row["data-team2"])),
                    names[0],
                    names[1],
                    goals[0],
                    goals[1],
                    cells[2],
                    clean_venue(cells[2]),
                )
            )
        elif "team_id" in query:
            if len(cells) != 8:
                raise ValueError(f"Expected eight standing cells: {cells!r}")
            goals_for, goals_against = parse_score(cells[6])
            table.append(
                TableRow(
                    int(query["team_id"][0]),
                    cells[1],
                    int(cells[0]),
                    int(cells[2]),
                    int(cells[3]),
                    int(cells[4]),
                    int(cells[5]),
                    goals_for,
                    goals_against,
                    int(cells[7]),
                )
            )
    return ParsedRound(tuple(games), tuple(table))
