"""Explicit allowlist for derived features; unknown output columns fail closed."""

from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]

Status = Literal["✅", "🔬"]
Availability = Literal["D", "S", "K"]


@dataclass(frozen=True)
class FeatureSpec:
    """Reviewable feature provenance, simulation availability and model roles."""

    block: str
    status: Status
    availability: Availability
    models: tuple[str, ...]
    decision: str
    model_status: tuple[tuple[str, Status], ...] = ()


IDENTIFICATION_COLUMNS = frozenset(
    {"game_id", "team_id", "season_id", "date", "as_of", "round", "opponent_id"}
)
TABLE_STATE_COLUMNS = (
    "games_played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
    "points_per_game",
    "position",
    "group_position",
    "position_pct",
    "gap_to_leader",
    "gap_to_cut",
    "gap_to_safety",
    "games_remaining_stage",
    "games_behind",
    "title_decided",
    "cut_decided",
    "relegation_decided",
)
FEATURES: dict[str, FeatureSpec] = {
    column: FeatureSpec("table_state", "🔬", "D", ("M1", "M2", "M4", "M5"), "T-010 §3; ה-ס3, ה-ס7")
    for column in TABLE_STATE_COLUMNS
}
FEATURES.update(
    {
        "places_decided": replace(FEATURES["title_decided"], decision="ה-ת10"),
        "stage": FeatureSpec(
            "table_state",
            "✅",
            "K",
            ("M1", "M2", "M4", "M5"),
            "ה-ס10; historical schedule known at split, Amit 2026-11-01",
        ),
        "rounds_to_stage_end": FeatureSpec(
            "table_state",
            "✅",
            "K",
            ("M1", "M2", "M4", "M5"),
            "ה-ס11; remaining stage games AFTER target game",
        ),
        "is_home": FeatureSpec("table_state", "✅", "K", ("M2", "M4"), "FEATURES.md M0: home/away"),
        "nothing_to_play": FeatureSpec(
            "table_state",
            "✅",
            "D",
            ("M1", "M2", "M4"),
            "ה-ת3, ה-ת4; conservative mathematical proof",
            (("M2", "🔬"),),
        ),
    }
)


def validate_feature_columns(frame: pd.DataFrame, block: str) -> None:
    """Reject every unregistered column, including rejected in-match/context features."""
    columns = set(frame.columns) | {name for name in frame.index.names if name is not None}
    unknown = columns - IDENTIFICATION_COLUMNS - FEATURES.keys()
    wrong_block = {
        column for column in columns & FEATURES.keys() if FEATURES[column].block != block
    }
    if unknown or wrong_block:
        raise ValueError(f"Unregistered features for {block}: {sorted(unknown | wrong_block)}")
