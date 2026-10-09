"""Sequential IFA collection with immutable responses and resumable manifests."""

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import requests
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ligat.scrape.base import ScrapeResult

LOGGER = logging.getLogger(__name__)
LEAGUE_URL = "https://www.football.org.il/leagues/league/"
ROUND_URL = "https://www.football.org.il/Components.asmx/League_AllTables"


class ScrapeConfig(BaseModel):
    """Validate request limits and collection paths before accessing the network."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_root: Path
    league_id: int = Field(gt=0)
    first_season_id: int = Field(gt=0)
    live_season_id: int = Field(gt=0)
    delay_seconds: float = Field(ge=2, allow_inf_nan=False)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_attempts: int = Field(ge=1, le=3)
    backoff_seconds: float = Field(gt=0, allow_inf_nan=False)
    user_agent: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_seasons(self) -> "ScrapeConfig":
        """Reject an inverted season range."""
        if self.first_season_id >= self.live_season_id:
            raise ValueError("first_season_id must precede live_season_id")
        return self


def load_scrape_config(path: Path) -> ScrapeConfig:
    """Load the explicitly selected YAML scraper configuration."""
    return ScrapeConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Transport(Protocol):
    """Permit offline tests to replace the requests Session."""

    def get(
        self,
        url: str,
        *,
        params: dict[str, int],
        headers: dict[str, str],
        timeout: float,
        allow_redirects: bool,
    ) -> requests.Response:
        """Return a response without implicitly following redirects."""
        ...


class SourceFormatError(ValueError):
    """The downloaded source does not match the expected document format."""


class AccessBlockedError(RuntimeError):
    """HTTP 403 stops all further requests through this source instance."""


@dataclass(frozen=True)
class Stage:
    """Preserve the source's stage name and published round boundaries."""

    box: str
    name: str
    min_round: int
    max_round: int


def parse_stages(content: bytes) -> tuple[Stage, ...]:
    """Read ddlBoxes only; interpreting standings and match scores belongs to T-003."""
    soup = BeautifulSoup(content, "lxml")
    select = soup.find("select", id="ddlBoxes")
    if select is None:
        raise SourceFormatError("League page is missing select#ddlBoxes")
    stages = []
    for option in select.find_all("option"):
        minimum = option.get("data-min-round")
        maximum = option.get("data-max-round")
        # A select may have a placeholder or an all-stages entry without a round range.
        if minimum is None and maximum is None:
            continue
        try:
            start, end = int(str(minimum)), int(str(maximum))
        except ValueError as exc:
            raise SourceFormatError("Invalid ddlBoxes round bounds") from exc
        name = option.get_text(" ", strip=True)
        if not name or start < 1 or end < start:
            raise SourceFormatError("Invalid stage name or round range")
        stages.append(Stage(str(option.get("value", "")), name, start, end))
    if not stages:
        raise SourceFormatError("League page has no published stage ranges")
    return tuple(stages)


def validate_round(content: bytes) -> None:
    """Reject broken XML or a missing/empty HtmlData without parsing the match rows."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise SourceFormatError("Round response is not XML") from exc
    fields = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "HtmlData"]
    if len(fields) != 1 or not (fields[0].text or "").strip():
        raise SourceFormatError("Published round has missing or empty HtmlData")


class IFASource:
    """Collect every published round, including future fixtures and promotion playoffs."""

    def __init__(
        self,
        config: ScrapeConfig,
        *,
        session: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.config = config
        self._owned_session = requests.Session() if session is None else None
        if self._owned_session is not None:
            self._session: Transport = self._owned_session
        else:
            assert session is not None
            self._session = session
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self._last_finished: float | None = None
        self._blocked = False

    def close(self) -> None:
        """Close only the HTTP session owned by this instance."""
        if self._owned_session is not None:
            self._owned_session.close()

    def scrape_history(self, seasons: Iterable[int]) -> list[ScrapeResult]:
        """Resume completed-season downloads without replacing any raw response."""
        requested = tuple(dict.fromkeys(seasons))
        for season in requested:
            if not self.config.first_season_id <= season < self.config.live_season_id:
                raise ValueError("history accepts completed seasons only")
        return [self._collect(season, "history") for season in requested]

    def scrape_live(self) -> ScrapeResult:
        """Download all published current-season rounds into a fresh snapshot."""
        return self._collect(self.config.live_season_id, "live")

    def _request(self, url: str, params: dict[str, int]) -> requests.Response:
        if self._blocked:
            raise AccessBlockedError("Source stopped after HTTP 403")
        for attempt in range(self.config.max_attempts):
            if self._last_finished is not None:
                remaining = self.config.delay_seconds - (self._monotonic() - self._last_finished)
                if remaining > 0:
                    self._sleep(remaining)
            try:
                response = self._session.get(
                    url,
                    params=params,
                    headers={"User-Agent": self.config.user_agent},
                    timeout=self.config.timeout_seconds,
                    allow_redirects=False,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt + 1 == self.config.max_attempts:
                    raise
            else:
                if response.status_code == 403:
                    self._blocked = True
                    raise AccessBlockedError(f"HTTP 403: {url}")
                if not 500 <= response.status_code < 600:
                    response.raise_for_status()
                    if not 200 <= response.status_code < 300:
                        raise SourceFormatError(f"Unexpected HTTP {response.status_code}: {url}")
                    return response
                if attempt + 1 == self.config.max_attempts:
                    response.raise_for_status()
            finally:
                self._last_finished = self._monotonic()
            pause = self.config.backoff_seconds * (2**attempt)
            LOGGER.warning("Retrying %s after attempt %s in %.1fs", url, attempt + 1, pause)
            self._sleep(pause)
        raise RuntimeError("Request attempts exhausted")

    def _collect(self, season: int, mode: Literal["history", "live"]) -> ScrapeResult:
        timestamp = self._now().astimezone(UTC)
        directory = self.config.raw_root / str(season)
        if mode == "live":
            directory = directory / "snapshot" / timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
            directory.mkdir(parents=True, exist_ok=False)
        else:
            directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        manifest: dict[str, Any] = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {
                "season_id": season,
                "mode": mode,
                "started_at": timestamp.isoformat(),
                "files": {},
            }
        )
        manifest["status"] = "in_progress"
        manifest.pop("error", None)
        manifest.pop("completed_at", None)
        downloaded, cached = 0, 0
        try:
            league_path = directory / "league.html"
            used_cache = self._save_resource(league_path, LEAGUE_URL, season, None, manifest)
            cached += int(used_cache)
            downloaded += int(not used_cache)
            stages = parse_stages(league_path.read_bytes())
            manifest["stages"] = [asdict(stage) for stage in stages]
            # Upper/lower playoff ranges overlap: fetch each round once with box=0.
            rounds = sorted(
                {r for stage in stages for r in range(stage.min_round, stage.max_round + 1)}
            )
            manifest["expected_rounds"] = rounds
            self._write_manifest(manifest_path, manifest)
            for round_id in rounds:
                path = directory / f"round_{round_id:02d}.xml"
                used_cache = self._save_resource(path, ROUND_URL, season, round_id, manifest)
                cached += int(used_cache)
                downloaded += int(not used_cache)
                manifest["files"][path.name]["stages"] = [
                    asdict(stage)
                    for stage in stages
                    if stage.min_round <= round_id <= stage.max_round
                ]
                # Save the response and provenance even if format validation fails.
                self._write_manifest(manifest_path, manifest)
                validate_round(path.read_bytes())
            manifest["status"] = "completed"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            self._write_manifest(manifest_path, manifest)
            raise
        manifest.pop("error", None)
        manifest["completed_at"] = self._now().astimezone(UTC).isoformat()
        self._write_manifest(manifest_path, manifest)
        LOGGER.info(
            "Season %s: %s downloaded, %s cached (%s)", season, downloaded, cached, directory
        )
        return ScrapeResult(season, directory, downloaded, cached)

    def _save_resource(
        self,
        path: Path,
        url: str,
        season: int,
        round_id: int | None,
        manifest: dict[str, Any],
    ) -> bool:
        params = {"league_id": self.config.league_id, "season_id": season}
        if round_id is not None:
            params.update(box=0, round_id=round_id)
        cached = path.exists()
        status: int | None = None
        if not cached:
            response = self._request(url, params)
            with path.open("xb") as stream:
                stream.write(response.content)
            status = response.status_code
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        files = manifest["files"]
        if path.name in files:
            if files[path.name]["sha256"] != digest:
                raise SourceFormatError(f"Cached response hash mismatch: {path}")
        else:
            files[path.name] = {
                "url": url,
                "params": params,
                "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                "http_status": status,
                "bytes": len(content),
                "sha256": digest,
                "recovered_from_cache": cached,
            }
        self._write_manifest(path.parent / "manifest.json", manifest)
        return cached

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
