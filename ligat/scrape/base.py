"""Contracts for collecting source data without interpreting match results."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ScrapeResult:
    """Describe a season collection and its immutable raw output directory."""

    season_id: int
    directory: Path
    downloaded: int
    cached: int


class Source(Protocol):
    """Expose historical collection and timestamped live collection."""

    def scrape_history(self, seasons: Iterable[int]) -> list[ScrapeResult]:
        """Collect completed seasons, reusing previously saved responses."""
        ...

    def scrape_live(self) -> ScrapeResult:
        """Collect the current season into a new snapshot."""
        ...
