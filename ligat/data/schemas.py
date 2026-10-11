"""Pandera contracts used at both sides of the clean-data repository."""

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa

STAGES = ("regular", "upper", "middle", "lower")
STATUSES = ("played", "awarded", "void")


def _integer(*, positive: bool = False) -> pa.Column:
    return pa.Column("int64", pa.Check.ge(1 if positive else 0))


def _stage() -> pa.Column:
    return pa.Column("category", pa.Check.isin(STAGES))


def _date() -> pa.Column:
    return pa.Column(pd.DatetimeTZDtype(unit="ns", tz="Asia/Jerusalem"))


def _opponents(frame: Any) -> Any:
    return frame.home_team_id != frame.away_team_id


GAME_COLUMNS = {
    "season_id": _integer(positive=True),
    "round": _integer(positive=True),
    "stage": _stage(),
    "date": _date(),
    "home_team_id": _integer(positive=True),
    "away_team_id": _integer(positive=True),
}
TABLE_COLUMNS = {
    "season_id": _integer(positive=True),
    "round": _integer(positive=True),
    "team_id": _integer(positive=True),
    "position": _integer(positive=True),
    **{
        name: _integer()
        for name in ("played", "won", "drawn", "lost", "goals_for", "goals_against", "points_model")
    },
    "points_official": pa.Column("int64"),
    "deduction": pa.Column("int64"),
}


def _table_checks() -> list[pa.Check]:
    return [
        pa.Check(lambda f: f.played == f.won + f.drawn + f.lost),
        pa.Check(lambda f: f.deduction == f.points_official - f.points_model),
    ]


SCHEMAS = {
    "matches": pa.DataFrameSchema(
        {
            "game_id": _integer(positive=True),
            **GAME_COLUMNS,
            "season": pa.Column(str),
            "home_goals": _integer(),
            "away_goals": _integer(),
            "venue_raw": pa.Column(str),
            "venue": pa.Column(str),
            "is_home_venue": pa.Column(bool),
            "status": pa.Column("category", pa.Check.isin(STATUSES)),
            "source": pa.Column(str),
        },
        checks=pa.Check(_opponents),
        unique=["game_id"],
        strict=True,
    ),
    "fixtures": pa.DataFrameSchema(
        {**GAME_COLUMNS, "venue": pa.Column(str), "scraped_at": _date()},
        checks=pa.Check(_opponents),
        unique=["season_id", "round", "home_team_id", "away_team_id"],
        strict=True,
    ),
    "round_tables": pa.DataFrameSchema(
        TABLE_COLUMNS,
        checks=_table_checks(),
        unique=["season_id", "round", "team_id"],
        strict=True,
    ),
    "final_tables": pa.DataFrameSchema(
        {
            **TABLE_COLUMNS,
            "final_stage": _stage(),
            "relegated": pa.Column("boolean", nullable=True),
            "promoted_in": pa.Column("boolean", nullable=True),
            "regular_season_points": _integer(),
        },
        checks=_table_checks(),
        unique=["season_id", "team_id"],
        strict=True,
    ),
}


def make_frame(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build consistently typed frames, including empty future/final datasets."""
    schema = SCHEMAS[name]
    frame = pd.DataFrame(rows, columns=list(schema.columns))
    for column, spec in schema.columns.items():
        dtype = str(spec.dtype)
        if dtype.startswith("datetime64"):
            frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert("Asia/Jerusalem")
            frame[column] = frame[column].astype(pd.DatetimeTZDtype("ns", "Asia/Jerusalem"))
        elif dtype == "category":
            categories = STATUSES if column == "status" else STAGES
            # Nonempty dictionaries survive an empty Parquet round-trip as category dtype.
            frame[column] = frame[column].astype(pd.CategoricalDtype(categories))
        elif dtype == "str":
            frame[column] = frame[column].astype("str")
        else:
            frame[column] = frame[column].astype(dtype)
    schema.validate(frame, lazy=True)
    return frame
