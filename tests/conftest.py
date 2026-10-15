"""Session-scoped local analysis inputs and reusable heavy views; CI uses synthetic data."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ligat.analysis.league_stats import match_flow, season_course
from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables, load_analysis_tables
from ligat.data.repository import Repository


@pytest.fixture(scope="session")
def real_tables() -> Iterator[AnalysisTables]:
    """Load once with read guards; memoize pure heavy views only for this exact input object.

    Synthetic tests and modified copies always execute the original implementations.
    Each cached signature is computed by the original function, and returned by copy.
    """
    if not Path("data/clean/squad_values.parquet").exists():
        pytest.skip("Local clean data are deliberately not committed")
    repo = Repository()
    original_read = repo.read_filtered
    calls = []

    def filtered(name: str, filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
        calls.append((name, filters))
        for column, _, values in filters:
            if column == "season_id":
                assert all(7 <= season <= 25 for season in values)
        return original_read(name, filters)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(repo, "read_filtered", filtered)
        patch.setattr(repo, "read", lambda _: pytest.fail("Unfiltered read"))
        tables = load_analysis_tables(repo, seasons=DEVELOPMENT_SEASONS)
    assert calls
    original_timeline = match_flow.game_timelines
    original_standings = season_course.standings_by_match_number
    timelines: dict[tuple[int, ...], pd.DataFrame] = {}
    standings: dict[tuple[tuple[int, ...], str], pd.DataFrame] = {}

    def timeline(source: AnalysisTables, seasons: Any = DEVELOPMENT_SEASONS) -> pd.DataFrame:
        selected = list(seasons)
        if source is not tables:
            return original_timeline(source, selected)
        key = tuple(selected)
        if key not in timelines:
            timelines[key] = original_timeline(source, selected)
        return timelines[key].copy(deep=True)

    def standing(
        source: AnalysisTables, seasons: Any = DEVELOPMENT_SEASONS, stage: str = "regular"
    ) -> pd.DataFrame:
        selected = list(seasons)
        if source is not tables:
            return original_standings(source, selected, stage)
        key = (tuple(selected), stage)
        if key not in standings:
            standings[key] = original_standings(source, selected, stage)
        return standings[key].copy(deep=True)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(match_flow, "game_timelines", timeline)
        patch.setattr(season_course, "standings_by_match_number", standing)
        yield tables


@pytest.fixture(scope="session")
def real_timelines(real_tables: AnalysisTables) -> pd.DataFrame:
    return match_flow.game_timelines(real_tables)


@pytest.fixture(scope="session")
def real_standings(real_tables: AnalysisTables) -> pd.DataFrame:
    return season_course.standings_by_match_number(real_tables)
