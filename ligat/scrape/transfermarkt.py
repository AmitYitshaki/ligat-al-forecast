"""Sequential, immutable season-page collection with immediate blocking stops."""

import gzip
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

from ligat.data.squad_values import parse_squad_page, read_squad_page
from ligat.scrape.ifa import AccessBlockedError, Transport

USER_AGENT = "LigatAlForecast/0.1 (research; https://github.com/AmitYitshaki/ligat-al-forecast)"
URL = "https://www.transfermarkt.com/ligat-haal/startseite/wettbewerb/ISR1/plus/"


def _publish(path: Path, content: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(content, indent=2), encoding="utf-8")
    temporary.replace(path)


class TransfermarktSource:
    """Keep each response immutable; never retry blocks, CAPTCHAs or bad formats."""

    def __init__(
        self,
        root: Path = Path("data/raw/transfermarkt/ISR1"),
        *,
        delay_seconds: float = 3,
        session: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 3 <= delay_seconds < float("inf"):
            raise ValueError("Transfermarkt delay must be at least 3 seconds")
        self.root, self.delay, self.sleep = root, delay_seconds, sleep
        self.session = session
        self.blocked = False
        self.requested = False

    def collect(self, years: Iterable[int] = range(2006, 2027)) -> dict[str, Any]:
        """Resume validated pages; publish only after parsing and league-total checks."""
        if self.blocked:
            raise AccessBlockedError("Transfermarkt stopped after blocking response")
        requested = tuple(years)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(y not in range(2006, 2027) for y in requested)
        ):
            raise ValueError("Use distinct saison_id values from 2006 through 2026")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": {}}
        manifest.update(status="in_progress", requested_seasons=list(requested))
        try:
            for year in requested:
                target = self.root / f"{year}.html.gz"
                if target.exists():
                    if str(year) not in manifest["files"]:
                        # Recover publication interrupted before its manifest update.
                        compressed = target.read_bytes()
                        content = gzip.decompress(compressed)
                        parse_squad_page(content, year)
                        manifest["files"][str(year)] = self._metadata(
                            year, content, compressed, None
                        ) | {"recovered_from_cache": True}
                    parse_squad_page(
                        read_squad_page(self.root, year, manifest["files"][str(year)]), year
                    )
                else:
                    self._download(year, target, manifest)
                _manifest(path, manifest)
            manifest.update(status="completed", completed_at=datetime.now(UTC).isoformat())
            manifest.pop("error", None)
        except Exception as exc:
            manifest.update(status="blocked" if self.blocked else "failed", error=str(exc))
            _manifest(path, manifest)
            raise
        _manifest(path, manifest)
        return manifest

    @staticmethod
    def _metadata(
        year: int, content: bytes, compressed: bytes, status: int | None
    ) -> dict[str, Any]:
        return {
            "url": f"{URL}?saison_id={year}",
            "downloaded_at": datetime.now(UTC).isoformat(),
            "http_status": status,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "compressed_bytes": len(compressed),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "user_agent": USER_AGENT,
        }

    def _download(self, year: int, target: Path, manifest: dict[str, Any]) -> None:
        if self.requested:
            self.sleep(self.delay)
        self.requested = True
        if self.session is None:
            # Each request has a fresh cookie jar; no browser headers or cookie reuse.
            with requests.Session() as session:
                response = session.get(
                    URL,
                    params={"saison_id": year},
                    headers={"User-Agent": USER_AGENT},
                    timeout=60,
                    allow_redirects=False,
                )
        else:
            response = self.session.get(
                URL,
                params={"saison_id": year},
                headers={"User-Agent": USER_AGENT},
                timeout=60,
                allow_redirects=False,
            )
        content = response.content
        compressed = gzip.compress(content, mtime=0)
        metadata = self._metadata(year, content, compressed, response.status_code)
        soup = BeautifulSoup(content, "lxml", from_encoding="utf-8")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        self.blocked = response.status_code in (403, 429) or bool(
            re.search(r"captcha|just a moment|access denied|verify.*human", title, re.I)
            or soup.select_one("#challenge-form, .g-recaptcha, #cf-challenge-running")
        )
        try:
            if self.blocked:
                raise AccessBlockedError(f"STOP Transfermarkt HTTP {response.status_code}: {title}")
            if response.status_code != 200:
                raise ValueError(f"Unexpected Transfermarkt HTTP {response.status_code}")
            parse_squad_page(content, year)
        except (AccessBlockedError, ValueError) as exc:
            rejected = self.root / "rejected"
            rejected.mkdir(exist_ok=True)
            archive = rejected / f"{year}_{uuid4().hex}.html.gz"
            _publish(archive, compressed)
            manifest.setdefault("rejected", []).append(
                metadata | {"file": str(archive), "error": str(exc)}
            )
            raise
        _publish(target, compressed)
        manifest["files"][str(year)] = metadata
