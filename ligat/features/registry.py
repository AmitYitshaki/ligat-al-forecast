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
FEATURES.update(
    {
        column: FeatureSpec("elo", "🔬", "D", ("M1", "M2", "M4"), "ה-ת2; ה-ת11; T-010 §8.1")
        for column in ("elo_pre", "elo_opponent_pre", "elo_diff", "elo_expected")
    }
)

FORM_CANDIDATES = (
    "w5",
    "w10",
    "w20",
    "ewma5",
    "ewma10",
    "ewma20",
    "std_m3",
    "std_m6",
    "std_m10",
    "prev",
    "hist",
)
FORM_COLUMNS = tuple(
    f"{metric}_{variant}_{candidate}"
    for metric in ("gf", "ga", "pts")
    for variant in ("all", "venue")
    for candidate in FORM_CANDIDATES
)
FEATURES.update(
    {
        column: FeatureSpec(
            "form",
            "🔬",
            "D",
            ("M2", "M4"),
            "ה-ת12; T-010 §8.4; season 8 league-neutral prior uses its full season (training only)",
        )
        for column in FORM_COLUMNS
    }
)


SEASON_CONTEXT_COLUMNS = (
    "prev_value_share_avg",
    "prev_value_share_total",
    "prev_position_pct",
    "prev_ppm",
    "prev_gd_per_game",
    "promoted_in",
    "squad_value_imputed",
    "prev_season_known",
    "league_size",
)
FEATURES.update(
    {
        column: FeatureSpec(
            "season_context",
            "🔬" if column in SEASON_CONTEXT_COLUMNS[:5] else "✅",
            "K" if column == "league_size" else "S",
            ("M2", "M4", "M5"),
            "ה-ת13; ה-ס1; ה-ס5; T-010 §9",
        )
        for column in SEASON_CONTEXT_COLUMNS
    }
)

FEATURES["prev_value_share_avg"] = replace(
    FEATURES["prev_value_share_avg"],
    model_status=(("M2", "✅"),),
    decision=FEATURES["prev_value_share_avg"].decision + "; ה-ת14",
)
PREMATCH_CONTRAST_COLUMNS = (
    "value_avg_log_ratio",
    "position_log_ratio",
    "prev_ppm_diff",
    "prev_gd_diff",
)
PREMATCH_OPPONENT_COLUMNS = ("opp_gf_venue_ewma5", "opp_ga_venue_ewma5")
FEATURES.update(
    {
        column: FeatureSpec(
            "prematch",
            "🔬",
            "S" if column in PREMATCH_CONTRAST_COLUMNS else "D",
            ("M2", "M4", "M5") if column in PREMATCH_CONTRAST_COLUMNS else ("M2", "M4"),
            "ה-ת14; T-010 §9",
            (("M2", "✅"),) if column == "value_avg_log_ratio" else (),
        )
        for column in (*PREMATCH_OPPONENT_COLUMNS, *PREMATCH_CONTRAST_COLUMNS)
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


def validate_prematch_feature_columns(frame: pd.DataFrame) -> None:
    """Require registration across the frozen blocks and the added prematch features."""
    columns = set(frame.columns) | {name for name in frame.index.names if name is not None}
    unknown = columns - IDENTIFICATION_COLUMNS - FEATURES.keys()
    if unknown:
        raise ValueError(f"Unregistered features for prematch: {sorted(unknown)}")
