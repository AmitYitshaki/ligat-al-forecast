"""Read immutable game-page scores for offline build cross-checks."""

import gzip
import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any

# SoupStrainer is a public bs4 re-export omitted from its explicit typing exports.
from bs4 import BeautifulSoup, SoupStrainer  # type: ignore[attr-defined]


def parse_game_score(content: bytes, game_id: int) -> tuple[int, int]:
    """Return home, away goals from the game header, never the halftime result."""
    soup = BeautifulSoup(content, "lxml", parse_only=SoupStrainer("section", id=f"gfd_{game_id}"))
    section = soup.select_one(f"section.game-details#gfd_{game_id}")
    total = section.select_one(".game-time .total") if section is not None else None
    if total is None:
        raise ValueError(f"Missing game-page total or incorrect identity: game_id={game_id}")
    for label in total.select(".sr-only"):
        label.decompose()
    score = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", total.get_text(" ", strip=True))
    if score is None:
        raise ValueError(f"Invalid game-page total: game_id={game_id}")
    return int(score[1]), int(score[2])


class GamePageRepository:
    """Validate both gzip and original bytes against T-004 manifests, read-only."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._manifests: dict[int, dict[str, Any]] = {}

    def score(self, season_id: int, game_id: int) -> tuple[int, int]:
        """Load a saved score without network access or modifying raw storage."""
        directory = self.root / str(season_id)
        if season_id not in self._manifests:
            self._manifests[season_id] = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
        manifest = self._manifests[season_id]
        if manifest.get("status") != "completed" or manifest.get("season_id") != season_id:
            raise ValueError(f"Incomplete or mismatched game-page manifest: season_id={season_id}")
        name = f"{game_id}.html.gz"
        metadata = manifest["files"][name]
        compressed = (directory / name).read_bytes()
        if (
            len(compressed) != metadata["compressed_bytes"]
            or hashlib.sha256(compressed).hexdigest() != metadata["compressed_sha256"]
        ):
            raise ValueError(f"Game-page gzip integrity mismatch: game_id={game_id}")
        try:
            content = gzip.decompress(compressed)
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"Invalid game-page gzip: game_id={game_id}") from exc
        if (
            len(content) != metadata["bytes"]
            or hashlib.sha256(content).hexdigest() != metadata["sha256"]
        ):
            raise ValueError(f"Game-page content integrity mismatch: game_id={game_id}")
        return parse_game_score(content, game_id)
