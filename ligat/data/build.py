"""Offline orchestration from immutable IFA responses to clean league datasets."""

import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]
import yaml

from ligat.data.details import cross_check, load_exceptions, parse_game
from ligat.data.game_pages import GamePageRepository
from ligat.data.parse import ParsedRound, TableRow, clean_team_name, parse_round
from ligat.data.repository import Repository
from ligat.data.schemas import make_frame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeasonInput:
    """A completed collection, with provenance and parsed rounds."""

    config: dict[str, Any]
    directory: Path
    manifest: dict[str, Any]
    rounds: dict[int, ParsedRound]


def latest_completed_snapshot(directory: Path) -> Path:
    """Ignore failed/running snapshots, selecting by manifest completion time."""
    completed: list[tuple[datetime, Path]] = []
    for manifest_path in directory.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            completed.append(
                (datetime.fromisoformat(manifest["completed_at"]), manifest_path.parent)
            )
    if not completed:
        raise ValueError(f"No completed snapshot in {directory}")
    return max(completed, key=lambda item: (item[0], str(item[1])))[1]


def _load_season(raw_root: Path, config: dict[str, Any]) -> SeasonInput:
    directory = raw_root / str(config["season_id"])
    if config.get("live", False):
        directory = latest_completed_snapshot(directory / "snapshot")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest["season_id"] != config["season_id"]:
        raise ValueError(f"Incomplete or mismatched season manifest: {directory}")
    rounds: dict[int, ParsedRound] = {}
    for round_id in manifest["expected_rounds"]:
        name = f"round_{round_id:02d}.xml"
        content = (directory / name).read_bytes()
        metadata = manifest["files"][name]
        if (
            len(content) != metadata["bytes"]
            or hashlib.sha256(content).hexdigest() != metadata["sha256"]
        ):
            raise ValueError(f"Raw file integrity mismatch: {directory / name}")
        rounds[round_id] = parse_round(content)
    return SeasonInput(config, directory, manifest, rounds)


def _is_promotion(season: SeasonInput, round_id: int) -> bool:
    stages = season.manifest["files"][f"round_{round_id:02d}.xml"]["stages"]
    return any("מבחן" in stage["name"] for stage in stages)


def _groups(season: SeasonInput) -> dict[int, str]:
    config = season.config
    if not config["playoffs"]:
        return {}
    regular = season.rounds[config["regular_rounds"]].table
    if not regular and config.get("live", False):
        return {}  # Current regular season has not ended yet.
    if len(regular) != config["n_teams"]:
        raise ValueError(f"Missing regular-season standings for {config['season_id']}")
    sizes = config["playoff_groups"]
    if sum(sizes.values()) != config["n_teams"] or set(sizes) != set(config["playoffs"]):
        raise ValueError(f"Invalid playoff group sizes for {config['season_id']}")
    groups = {}
    for row in regular:
        boundary = 0
        for stage in config["playoffs"]:
            boundary += sizes[stage]
            if row.position <= boundary:
                groups[row.team_id] = stage
                break
    return groups


def _stage(season: SeasonInput, round_id: int, home: int, away: int, groups: dict[int, str]) -> str:
    if round_id <= season.config["regular_rounds"]:
        return "regular"
    if home not in groups or away not in groups or groups[home] != groups[away]:
        raise ValueError(f"Invalid playoff opponents: {season.config['season_id']}/{round_id}")
    return groups[home]


def _load_overrides(path: Path) -> dict[int, str]:
    """Return verified status overrides for compatibility with source validation."""
    return _load_override_policy(path)[0]


def _load_override_policy(path: Path) -> tuple[dict[int, str], set[int], set[int]]:
    """Separate status, removal and deferred point corrections explicitly."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    result: dict[int, str] = {}
    dropped: set[int] = set()
    deferred_points: set[int] = set()
    for game_id, override in document["matches"].items():
        if override["status"] not in ("played", "awarded", "void"):
            raise ValueError(f"Unknown status override for {game_id}")
        if not isinstance(override["verified"], bool):
            raise ValueError(f"Override verified flag must be boolean: {game_id}")
        action = override.get("action", "keep")
        if action not in ("keep", "drop"):
            raise ValueError(f"Unknown override action for {game_id}")
        adjust_points = override.get("adjust_points", True)
        if not isinstance(adjust_points, bool):
            raise ValueError(f"Override adjust_points flag must be boolean: {game_id}")
        if override["verified"]:
            result[int(game_id)] = override["status"]
            if action == "drop":
                dropped.add(int(game_id))
            if not adjust_points:
                deferred_points.add(int(game_id))
    return result, dropped, deferred_points


def model_points(total: int, regular_total: int, *, after_regular: bool, halved: bool) -> int:
    """Halve only points carried into playoffs; playoff points remain whole."""
    return (regular_total + 1) // 2 + total - regular_total if halved and after_regular else total


def _table_rows(season: SeasonInput, voids: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    config = season.config
    end = config["regular_rounds"]

    def points(row: TableRow, round_id: int) -> int:
        # IFA counts the article-64 0:0 as a draw in W/D/L, but awards no point.
        no_points = sum(r <= round_id and row.team_id in (h, a) for r, h, a in voids)
        return int(config["points_win"]) * row.won + int(config["points_draw"]) * (
            row.drawn - no_points
        )

    baseline = {row.team_id: points(row, end) for row in season.rounds[end].table}
    records: list[dict[str, Any]] = []
    for round_id, parsed in sorted(season.rounds.items()):
        if _is_promotion(season, round_id):
            continue
        for row in parsed.table:
            total = points(row, round_id)
            modeled = model_points(
                total,
                baseline.get(row.team_id, 0),
                after_regular=round_id > end,
                halved=config["points_halved"],
            )
            record = asdict(row)
            del record["name"]
            record.update(
                season_id=config["season_id"],
                round=round_id,
                points_model=modeled,
                deduction=row.points_official - modeled,
            )
            records.append(record)
    return records


def _atomic_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_datasets(
    *,
    raw_root: Path = Path("data/raw/ifa/rounds"),
    games_root: Path = Path("data/raw/ifa/games"),
    seasons_path: Path = Path("config/seasons.yaml"),
    overrides_path: Path = Path("config/match_overrides.yaml"),
    exceptions_path: Path = Path("config/game_detail_exceptions.yaml"),
    validate_only: bool = False,
    teams_path: Path = Path("config/teams.yaml"),
    repository: Repository | None = None,
) -> dict[str, pd.DataFrame]:
    """Build eight validated datasets and a team alias draft, entirely offline."""
    if any(teams_path.resolve().is_relative_to(root.resolve()) for root in (raw_root, games_root)):
        raise ValueError("Team draft cannot write into raw storage")
    repo = repository if repository is not None else Repository()
    if any(repo.root.resolve().is_relative_to(root.resolve()) for root in (raw_root, games_root)):
        raise ValueError("Clean repository cannot write into raw storage")
    document = yaml.safe_load(seasons_path.read_text(encoding="utf-8"))
    seasons = [
        _load_season(raw_root, document["defaults"] | config)
        for config in sorted(document["seasons"], key=lambda item: item["season_id"])
    ]
    overrides, dropped, deferred_points = _load_override_policy(overrides_path)
    exceptions = load_exceptions(exceptions_path)
    exception_notes = yaml.safe_load(exceptions_path.read_text(encoding="utf-8"))["games"]
    pages = GamePageRepository(games_root)
    mismatches = []
    for season in seasons:
        for round_id, parsed in sorted(season.rounds.items()):
            if _is_promotion(season, round_id):
                continue
            for game in parsed.games:
                if game.game_id is None:
                    continue
                page_score = pages.score(season.config["season_id"], game.game_id)
                round_score = (game.home_goals, game.away_goals)
                if page_score != round_score and game.game_id not in overrides:
                    mismatches.append(
                        f"game_id={game.game_id}, season_id={season.config['season_id']}, "
                        f"round={round_id}, page_home_away={page_score}, "
                        f"round_home_away={round_score}"
                    )
    if mismatches:
        raise ValueError("Undocumented game score mismatches:\n" + "\n".join(mismatches))
    matches: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    aliases: dict[int, set[str]] = defaultdict(set)
    latest_names: dict[int, str] = {}
    groups_by_season: dict[int, dict[int, str]] = {}
    rosters: dict[int, set[int]] = {}

    def observe(team_id: int, name: str) -> None:
        aliases[team_id].add(name)
        latest_names[team_id] = clean_team_name(name)

    for season in seasons:
        config = season.config
        sid = config["season_id"]
        groups = _groups(season)
        groups_by_season[sid] = groups
        rosters[sid] = {row.team_id for parsed in season.rounds.values() for row in parsed.table}
        voids: list[tuple[int, int, int]] = []
        for round_id, parsed in sorted(season.rounds.items()):
            for row in parsed.table:
                observe(row.team_id, row.name)
            for game in parsed.games:
                observe(game.home_team_id, game.home_name)
                observe(game.away_team_id, game.away_name)
                if _is_promotion(season, round_id):
                    continue
                shared = {
                    "season_id": sid,
                    "round": round_id,
                    "stage": _stage(season, round_id, game.home_team_id, game.away_team_id, groups),
                    "date": game.date,
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                    "venue": game.venue,
                }
                if game.home_goals is None:
                    if config.get("live", False):
                        metadata = season.manifest["files"][f"round_{round_id:02d}.xml"]
                        fixtures.append(
                            shared
                            | {"scraped_at": datetime.fromisoformat(metadata["downloaded_at"])}
                        )
                    else:
                        raise ValueError(f"Unplayed historical game: {sid}/{round_id}")
                else:
                    status = overrides.get(game.game_id or 0, "played")
                    matches.append(
                        shared
                        | {
                            "game_id": game.game_id,
                            "season": config["label"],
                            "home_goals": game.home_goals,
                            "away_goals": game.away_goals,
                            "venue_raw": game.venue_raw,
                            "status": status,
                            "source": "ifa",
                        }
                    )
                    if status == "void" and game.game_id not in deferred_points:
                        voids.append((round_id, game.home_team_id, game.away_team_id))
        tables.extend(_table_rows(season, voids))

    venues: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for match in matches:
        if match["venue"]:
            venues[match["season_id"], match["home_team_id"]][match["venue"]] += 1
    for match in matches:
        counts = venues[match["season_id"], match["home_team_id"]]
        canonical = min(counts, key=lambda venue: (-counts[venue], venue)) if counts else None
        match["is_home_venue"] = match["venue"] == canonical

    for season in seasons:
        config = season.config
        if config.get("live", False):
            continue  # Partial current standings are never presented as final tables.
        sid = config["season_id"]
        rows = [row for row in tables if row["season_id"] == sid]
        last_round = max(row["round"] for row in rows)
        regular = {
            row["team_id"]: row["points_model"]
            for row in rows
            if row["round"] == config["regular_rounds"]
        }
        for final_row in rows:
            if final_row["round"] != last_round:
                continue
            team_id = final_row["team_id"]
            finals.append(
                final_row
                | {
                    "final_stage": groups_by_season[sid].get(team_id, "regular"),
                    "relegated": team_id not in rosters[sid + 1] if sid + 1 in rosters else None,
                    "promoted_in": team_id not in rosters[sid - 1] if sid - 1 in rosters else None,
                    "regular_season_points": regular[team_id],
                }
            )

    # Interpret administrative results for standings before removing match records.
    matches = [match for match in matches if match["game_id"] not in dropped]
    frames = {
        name: make_frame(name, records)
        for name, records in (
            ("matches", matches),
            ("fixtures", fixtures),
            ("round_tables", tables),
            ("final_tables", finals),
        )
    }
    detail_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("game_details", "goals", "red_cards", "coaches")
    }
    issues = []
    warnings: list[dict[str, Any]] = []
    season_labels = {season.config["season_id"]: season.config["label"] for season in seasons}
    for index, match in enumerate(matches, start=1):
        context = {
            "season_id": match["season_id"],
            "season": season_labels[match["season_id"]],
        }
        content = pages.read(match["season_id"], match["game_id"])
        try:
            parsed_game = parse_game(
                content,
                match["game_id"],
                match["home_team_id"],
                match["away_team_id"],
                exception=exceptions.get(match["game_id"]),
            )
        except ValueError as exc:
            issues.append(
                context
                | {
                    "game_id": match["game_id"],
                    "kind": "page_format",
                    "actual": str(exc),
                    "expected": "Valid game page",
                }
            )
            continue
        if index % 500 == 0:
            LOGGER.info("Validated game details: %d/%d", index, len(matches))
        issues.extend(
            context | issue
            for issue in cross_check(
                parsed_game,
                match["home_team_id"],
                match["away_team_id"],
                (match["home_goals"], match["away_goals"]),
            )
        )
        detail_rows["game_details"].append(parsed_game.details)
        detail_rows["goals"].extend(parsed_game.goals)
        detail_rows["red_cards"].extend(parsed_game.red_cards)
        warnings.extend(
            {
                **context,
                "game_id": match["game_id"],
                "kind": "unknown_red_card_minute",
                "team_id": row["team_id"],
            }
            for row in parsed_game.red_cards
            if row["minute"] is None
        )
        detail_rows["coaches"].extend(parsed_game.coaches)
    report_path = repo.root.parent / "reports" / "game_details_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(
            {
                "checked_games": len(matches),
                "validate_only": validate_only,
                "issues": issues,
                "warnings": warnings,
                "documented_exceptions": [
                    {
                        "game_id": match["game_id"],
                        "season_id": match["season_id"],
                        "season": match["season"],
                        "type": exceptions[match["game_id"]],
                        "reason": exception_notes[match["game_id"]]["reason"],
                    }
                    for match in matches
                    if match["game_id"] in exceptions
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    if warnings:
        LOGGER.warning(
            "Retained %d red cards with unknown minutes; report: %s", len(warnings), report_path
        )
    if issues:
        raise ValueError(
            f"Game details validation failed: {len(issues)} undocumented discrepancies; "
            f"game_id={sorted({issue['game_id'] for issue in issues})}; report: {report_path}"
        )
    frames.update({name: make_frame(name, rows) for name, rows in detail_rows.items()})
    if validate_only:
        LOGGER.info("Validation passed; no Parquet or team draft published (--validate-only)")
        return frames
    for name, frame in frames.items():
        repo.write(name, frame)
        LOGGER.info("Built %s: %d rows", name, len(frame))
    # Preserve manually enriched fields and previous aliases on subsequent builds.
    existing = yaml.safe_load(teams_path.read_text(encoding="utf-8")) if teams_path.exists() else {}
    previous = {item["team_id"]: item for item in (existing or {}).get("teams", [])}
    teams = []
    for team_id in sorted(aliases):
        old = previous.get(team_id, {})
        teams.append(
            {
                "team_id": team_id,
                "name_he": old.get("name_he", latest_names[team_id]),
                "name_en": old.get("name_en"),
                "aliases": sorted(aliases[team_id] | set(old.get("aliases", []))),
                "city": old.get("city"),
            }
        )
    _atomic_yaml(teams_path, {"draft": True, "teams": teams})
    return frames
