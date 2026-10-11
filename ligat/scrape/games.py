"""Sequential, resumable game-page collection sourced only from immutable raw rounds."""

import gzip
import hashlib
import json
import logging
import os
import re
import zlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import yaml
from bs4 import BeautifulSoup

from ligat.data.parse import parse_round
from ligat.scrape.ifa import IFASource, ScrapeConfig, SourceFormatError

LOGGER = logging.getLogger(__name__)
GAME_URL = "https://www.football.org.il/leagues/games/game/"
SAMPLE_COUNT = 10


@dataclass(frozen=True)
class GameRef:
    """Game identity taken directly from the round's source IDs."""

    game_id: int
    season_id: int
    home_team_id: int
    away_team_id: int


def read_game_refs(
    raw_root: Path, seasons: Iterable[int], *, live_season: int = 28
) -> tuple[GameRef, ...]:
    """Read complete verified collections; include technical results, exclude promotion tests."""
    refs: dict[int, GameRef] = {}
    for sid in sorted(set(seasons)):
        directory = raw_root / str(sid)
        if sid == live_season:
            completed = []
            for path in (directory / "snapshot").glob("*/manifest.json"):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("status") == "completed":
                    completed.append(
                        (datetime.fromisoformat(manifest["completed_at"]), str(path.parent))
                    )
            if not completed:
                raise SourceFormatError(f"No completed raw snapshot for season {sid}")
            directory = Path(max(completed)[1])
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("season_id") != sid:
            raise SourceFormatError(f"Incomplete or mismatched raw season {sid}")
        for round_id in manifest["expected_rounds"]:
            name = f"round_{round_id:02d}.xml"
            metadata = manifest["files"][name]
            content = (directory / name).read_bytes()
            if (
                len(content) != metadata["bytes"]
                or hashlib.sha256(content).hexdigest() != metadata["sha256"]
            ):
                raise SourceFormatError(f"Raw round integrity mismatch: {directory / name}")
            if any("מבחן" in stage["name"] for stage in metadata["stages"]):
                continue
            for row in parse_round(content).games:
                if row.game_id is None:
                    continue
                ref = GameRef(row.game_id, sid, row.home_team_id, row.away_team_id)
                if ref.game_id in refs and refs[ref.game_id] != ref:
                    raise SourceFormatError(f"Conflicting source identity for game {ref.game_id}")
                refs[ref.game_id] = ref
    return tuple(sorted(refs.values(), key=lambda ref: (ref.season_id, ref.game_id)))


def select_sample(games: tuple[GameRef, ...], seasons: list[int]) -> tuple[GameRef, ...]:
    """Keep stage A available while using the same raw-only source as full collection."""
    if len(seasons) != SAMPLE_COUNT or len(set(seasons)) != SAMPLE_COUNT:
        raise ValueError("Stage A requires exactly ten distinct sample seasons")
    selected = []
    for sid in seasons:
        candidates = [game for game in games if game.season_id == sid]
        if not candidates:
            raise ValueError(f"No game IDs for sample season {sid}")
        selected.append(min(candidates, key=lambda ref: ref.game_id))
    return tuple(selected)


def inspect_game_page(content: bytes, game: GameRef) -> dict[str, Any]:
    """Check static page identity and section presence; event parsing belongs to T-005."""
    soup = BeautifulSoup(content, "lxml")
    if soup.html is None or soup.body is None:
        raise SourceFormatError("Game response is not a complete HTML document")
    detail = soup.select_one(f"section.game-details#gfd_{game.game_id}")
    if detail is None:
        raise SourceFormatError(f"Game page identity mismatch: {game.game_id}")
    if "פרטי המשחק" not in detail.get_text(" ", strip=True):
        raise SourceFormatError("Game page is missing game-details title")
    for selector, expected in [
        (".team-home", game.home_team_id),
        (".team-guest", game.away_team_id),
    ]:
        anchor = detail.select_one(f".teams-names {selector} a[href]")
        query = parse_qs(urlparse(str(anchor["href"])).query) if anchor is not None else {}
        if query.get("team_id") != [str(expected)] or query.get("season_id") != [
            str(game.season_id)
        ]:
            raise SourceFormatError(f"Game page is missing expected team links: {game.game_id}")
    canonical_ids = set()
    for node in soup.select('link[rel="canonical"], meta[property="og:url"]'):
        url = str(node.get("href", node.get("content", "")))
        canonical_ids.update(parse_qs(urlparse(url).query).get("game_id", []))
    if canonical_ids and canonical_ids != {str(game.game_id)}:
        raise SourceFormatError(f"Game page identity mismatch: {game.game_id}")
    for label in detail.select(".sr-only"):
        label.decompose()
    total = detail.select_one(".game-time .total")
    halftime = detail.select_one(".game-time .result-half")
    date = detail.select_one(".game-time .date")
    has_result = total is not None and bool(re.search(r"\d+\s*:\s*\d+", total.get_text()))
    has_halftime = halftime is not None and bool(re.search(r"\d+\s*:\s*\d+", halftime.get_text()))
    timeline = detail.select_one(".timeline")
    has_timeline = timeline is not None and bool(timeline.get_text(strip=True))
    if not has_result or date is None:
        raise SourceFormatError(f"Game page is missing result/date: {game.game_id}")
    lineups = all(
        detail.select_one(f".players-cont .{side}.Active a[href]") is not None
        for side in ("home", "guest")
    )
    coaches = all(
        detail.select_one(f'.players-cont [aria-labelledby="GAME_COACH_{side}"] a[href]')
        is not None
        for side in ("HOME", "GUEST")
    )
    return {
        "has_halftime": has_halftime,
        "has_timeline": has_timeline,
        "has_lineups": lineups,
        "has_coaches": coaches,
        "game_id_checked": True,
        "teams_and_season_checked": True,
        "canonical_game_id_checked": bool(canonical_ids),
    }


class GamePageSource:
    """Collect sequentially using the shared throttled, blocking HTTP transport."""

    def __init__(self, http: IFASource, raw_root: Path) -> None:
        self.http = http
        self.raw_root = raw_root

    def collect_sample(self, games: tuple[GameRef, ...]) -> list[dict[str, Any]]:
        """Collect a ten-game sample with the same storage and validation rules."""
        if len(games) != SAMPLE_COUNT or len({g.game_id for g in games}) != SAMPLE_COUNT:
            raise ValueError("Stage A accepts exactly ten distinct games")
        return self.collect(games, mode="sample")

    def collect(self, games: tuple[GameRef, ...], *, mode: str) -> list[dict[str, Any]]:
        """Resume each season, marking completed only after every requested page validates."""
        grouped: dict[int, list[GameRef]] = defaultdict(list)
        for game in games:
            grouped[game.season_id].append(game)
        result = []
        for sid, season_games in sorted(grouped.items()):
            directory = self.raw_root / str(sid)
            directory.mkdir(parents=True, exist_ok=True)
            manifest_path = directory / "manifest.json"
            manifest = self._manifest(sid)
            manifest.update(
                status="in_progress",
                mode=mode,
                started_at=datetime.now(UTC).isoformat(),
                expected_game_ids=sorted({g.game_id for g in season_games}),
            )
            manifest.pop("error", None)
            manifest.pop("completed_at", None)
            IFASource._write_manifest(manifest_path, manifest)
            try:
                for game in season_games:
                    result.append(self._collect(game, manifest))
            except BaseException as exc:
                manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                IFASource._write_manifest(manifest_path, manifest)
                raise
            manifest.update(status="completed", completed_at=datetime.now(UTC).isoformat())
            IFASource._write_manifest(manifest_path, manifest)
        return result

    def _manifest(self, sid: int) -> dict[str, Any]:
        path = self.raw_root / str(sid) / "manifest.json"
        manifest: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"season_id": sid, "files": {}}
        )
        if manifest.get("season_id") != sid:
            raise SourceFormatError("Mismatched game-page manifest")
        return manifest

    def _collect(self, game: GameRef, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        directory = self.raw_root / str(game.season_id)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest(game.season_id) if manifest is None else manifest
        path = directory / f"{game.game_id}.html.gz"
        legacy = directory / f"{game.game_id}.html"
        params = {"game_id": game.game_id}
        previous = manifest["files"].get(path.name)
        cached = path.exists()
        status: int | None = None
        downloaded_at = datetime.now(UTC).isoformat()
        if cached:
            try:
                content = gzip.decompress(path.read_bytes())
            except (OSError, EOFError, zlib.error) as exc:
                raise SourceFormatError(f"Cached gzip is corrupt: {path}") from exc
            downloaded_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        elif legacy.exists():
            # Preserve stage-A originals, publishing an additional compressed representation.
            content = legacy.read_bytes()
            previous = manifest["files"].get(legacy.name)
            if previous is not None:
                self._check_digest(content, previous, legacy)
                status = previous.get("http_status")
            inspect_game_page(content, game)
            self._publish(path, content, game, manifest, status)
            cached = True
            downloaded_at = datetime.fromtimestamp(legacy.stat().st_mtime, UTC).isoformat()
        else:
            response = self.http.fetch(GAME_URL, params)
            content = response.content
            status = response.status_code
            self._publish(path, content, game, manifest, status)
        if previous is not None:
            self._check_digest(content, previous, path)
        sections = inspect_game_page(content, game)
        metadata = {
            **asdict(game),
            "url": GAME_URL,
            "params": params,
            "downloaded_at": previous.get("downloaded_at", downloaded_at)
            if previous
            else downloaded_at,
            "http_status": previous.get("http_status", status) if previous else status,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "compressed_bytes": path.stat().st_size,
            "compressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sections": sections,
        }
        if cached and previous and "compressed_sha256" in previous:
            if metadata["compressed_sha256"] != previous["compressed_sha256"]:
                raise SourceFormatError(f"Cached compressed hash mismatch: {path}")
        manifest["files"][path.name] = metadata
        IFASource._write_manifest(directory / "manifest.json", manifest)
        LOGGER.info(
            "Game %s, season %s: %s",
            game.game_id,
            game.season_id,
            "cached" if cached else "downloaded",
        )
        return metadata | {"path": str(path), "cached": cached}

    @staticmethod
    def _check_digest(content: bytes, metadata: dict[str, Any], path: Path) -> None:
        if metadata["sha256"] != hashlib.sha256(content).hexdigest() or metadata["bytes"] != len(
            content
        ):
            raise SourceFormatError(f"Cached game page hash mismatch: {path}")

    def _publish(
        self,
        path: Path,
        content: bytes,
        game: GameRef,
        manifest: dict[str, Any],
        status: int | None,
    ) -> None:
        temporary = NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        )
        staging = Path(temporary.name)
        try:
            with temporary as stream:
                stream.write(gzip.compress(content, mtime=0))
                stream.flush()
                os.fsync(stream.fileno())
            decoded = gzip.decompress(staging.read_bytes())
            if decoded != content:
                raise SourceFormatError("Gzip round-trip changed response content")
            try:
                inspect_game_page(decoded, game)
            except SourceFormatError as exc:
                rejected = path.parent / "rejected"
                rejected.mkdir(exist_ok=True)
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
                archive = rejected / f"{stamp}_{uuid4().hex}_{path.name}"
                os.link(staging, archive)
                manifest.setdefault("rejected", []).append(
                    {
                        "path": archive.relative_to(path.parent).as_posix(),
                        "game_id": game.game_id,
                        "http_status": status,
                        "rejected_at": datetime.now(UTC).isoformat(),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "error": str(exc),
                    }
                )
                IFASource._write_manifest(path.parent / "manifest.json", manifest)
                raise
            os.link(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def collection_report(raw_root: Path, games: tuple[GameRef, ...]) -> list[dict[str, Any]]:
    """Verify saved pages against raw round identities without any network request."""
    grouped: dict[int, list[GameRef]] = defaultdict(list)
    for game in games:
        grouped[game.season_id].append(game)
    result = []
    for sid, refs in sorted(grouped.items()):
        directory = raw_root / str(sid)
        manifest_path = directory / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        valid, missing, invalid = 0, [], []
        for game in refs:
            path = directory / f"{game.game_id}.html.gz"
            if not path.exists():
                missing.append(game.game_id)
                continue
            try:
                content = gzip.decompress(path.read_bytes())
                metadata = manifest["files"][path.name]
                GamePageSource._check_digest(content, metadata, path)
                if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["compressed_sha256"]:
                    raise SourceFormatError("Compressed hash mismatch")
                inspect_game_page(content, game)
                valid += 1
            except (ValueError, OSError, EOFError, KeyError, zlib.error):
                invalid.append(game.game_id)
        result.append(
            {
                "season_id": sid,
                "expected": len(refs),
                "valid": valid,
                "missing": missing,
                "invalid": invalid,
                "rejected": len(list((directory / "rejected").glob("*"))),
                "status": manifest.get("status", "missing"),
            }
        )
    return result


def run_games(
    scrape_config: ScrapeConfig,
    pages_config: Path,
    *,
    seasons: tuple[int, ...] | None = None,
    live: bool = False,
    sample: bool = False,
    report_only: bool = False,
) -> list[dict[str, Any]]:
    """Resolve all source IDs before constructing a session or downloading pages."""
    document = yaml.safe_load(pages_config.read_text(encoding="utf-8"))
    if sum((seasons is not None, live, sample)) != 1:
        raise ValueError("Choose exactly one of --seasons, --live or --sample")
    requested = (
        (scrape_config.live_season_id,)
        if live
        else (tuple(document["sample_seasons"]) if sample else seasons)
    )
    assert requested is not None
    if any(
        not scrape_config.first_season_id <= sid <= scrape_config.live_season_id
        for sid in requested
    ):
        raise ValueError("Season outside configured source range")
    games = read_game_refs(
        scrape_config.raw_root, requested, live_season=scrape_config.live_season_id
    )
    if sample:
        games = select_sample(games, document["sample_seasons"])
    root = Path(document["raw_root"])
    if not report_only:
        http = IFASource(scrape_config)
        try:
            GamePageSource(http, root).collect(
                games, mode="live" if live else "sample" if sample else "history"
            )
        finally:
            http.close()
    return collection_report(root, games)
