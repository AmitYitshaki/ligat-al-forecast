"""Session-scoped local analysis inputs and reusable heavy views; CI uses synthetic data."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ligat.analysis import league_stats
from ligat.analysis.league_stats import course_review, match_flow, season_course
from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables, load_analysis_tables
from ligat.data.repository import Repository


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark local-data tests using the transitive fixture closure and frozen M0 inputs."""
    data_fixtures = {"real_tables", "real_timelines", "real_standings", "frozen"}
    for item in items:
        if data_fixtures.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.real_data)


def _dtype_fingerprint(dtype: Any) -> tuple[Any, ...]:
    if isinstance(dtype, pd.CategoricalDtype):
        return repr(dtype), tuple(dtype.categories.tolist()), dtype.ordered
    return (repr(dtype),)


def _frame_fingerprint(frame: pd.DataFrame) -> tuple[Any, ...]:
    return (
        frame.shape,
        pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes(),
        type(frame.columns),
        tuple(frame.columns),
        tuple(frame.columns.names),
        _dtype_fingerprint(frame.columns.dtype),
        tuple(_dtype_fingerprint(dtype) for dtype in frame.dtypes),
        type(frame.index),
        tuple(frame.index.names),
        tuple(
            _dtype_fingerprint(frame.index.get_level_values(level).dtype)
            for level in range(frame.index.nlevels)
        ),
    )


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
        for column, operator, values in filters:
            if column == "season_id":
                assert operator == "in"
                assert set(values) <= set(range(7, 26))
        frame = original_read(name, filters)
        if "season_id" in frame:
            assert not frame.season_id.isin([26, 27, 28]).any()
        return frame

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(repo, "read_filtered", filtered)
        patch.setattr(repo, "read", lambda _: pytest.fail("Unfiltered read"))
        tables = load_analysis_tables(repo, seasons=DEVELOPMENT_SEASONS)
    assert calls
    snapshots = {
        field.name: (
            _frame_fingerprint(getattr(tables, field.name)),
            deepcopy(getattr(tables, field.name).attrs),
        )
        for field in fields(tables)
    }
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
        patch.setattr(course_review, "standings_by_match_number", standing)
        patch.setattr(league_stats, "game_timelines", timeline)
        patch.setattr(league_stats, "standings_by_match_number", standing)
        try:
            yield tables
        finally:
            for name, (fingerprint, attrs) in snapshots.items():
                frame = getattr(tables, name)
                try:
                    assert _frame_fingerprint(frame) == fingerprint
                    assert frame.attrs == attrs
                except Exception as error:
                    raise AssertionError(f"Shared real_tables.{name} was mutated") from error


@pytest.fixture(scope="session")
def real_timelines(real_tables: AnalysisTables) -> pd.DataFrame:
    return match_flow.game_timelines(real_tables)


@pytest.fixture(scope="session")
def real_standings(real_tables: AnalysisTables) -> pd.DataFrame:
    return season_course.standings_by_match_number(real_tables)
