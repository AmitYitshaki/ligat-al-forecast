"""Pure prematch v2 assembly from frozen v1 and form, with a separate 67-column schema."""

from copy import deepcopy

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa
from numpy.typing import NDArray

from ligat.features.pipeline import (
    DEVELOPMENT_SEASONS,
    PREMATCH_COLUMNS,
    PREMATCH_SCHEMA,
    PREVIOUS_VALUE_COLUMNS,
    _opponents,
    _validate_pairing,
    _validate_previous_values,
    validate_prematch_frame,
)
from ligat.features.registry import PREMATCH_CONTRAST_COLUMNS, validate_prematch_feature_columns

FORM_COPY_COLUMNS = (
    "gf_all_w5",
    "gf_venue_w5",
    "gf_all_std_m6",
    "gf_all_prev",
    "gf_all_hist",
)
FORM_MIRROR_COLUMNS = (
    "ga_all_w5",
    "ga_venue_w5",
    "ga_all_std_m6",
    "ga_all_prev",
    "ga_all_hist",
    "pts_venue_ewma5",
)
PREMATCH_MIRROR_COLUMNS = (
    "points_per_game",
    "position_pct",
    "promoted_in",
    "nothing_to_play",
)
_PAIRING_COLUMNS = ("season_id", "date", "as_of", "is_home", "opponent_id")
PREMATCH_V2_ADDITION_COLUMNS = (
    *FORM_COPY_COLUMNS,
    *("opp_" + c for c in (*FORM_MIRROR_COLUMNS, *PREMATCH_MIRROR_COLUMNS)),
    "gd_per_game",
)
PREMATCH_V2_COLUMNS = (*PREMATCH_COLUMNS, *PREMATCH_V2_ADDITION_COLUMNS)
PREMATCH_V2_SCHEMA = pa.DataFrameSchema(
    {
        **deepcopy(PREMATCH_SCHEMA.columns),
        **{c: pa.Column(float) for c in FORM_COPY_COLUMNS},
        **{"opp_" + c: pa.Column(float) for c in FORM_MIRROR_COLUMNS},
        **{
            "opp_" + c: pa.Column(
                PREMATCH_SCHEMA.columns[c].dtype,
                nullable=PREMATCH_SCHEMA.columns[c].nullable,
            )
            for c in PREMATCH_MIRROR_COLUMNS
        },
        "gd_per_game": pa.Column(float, nullable=True),
    },
    index=deepcopy(PREMATCH_SCHEMA.index),
    strict=True,
    ordered=True,
    coerce=False,
)


def _check_frame(frame: pd.DataFrame, name: str, columns: tuple[str, ...]) -> None:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != ["game_id", "team_id"]:
        raise ValueError(f"{name} requires a (game_id, team_id) MultiIndex")
    if frame.empty or frame.index.to_frame(index=False).isna().any().any():
        raise ValueError(f"{name} keys must be nonempty and contain no missing values")
    if not frame.index.is_unique:
        raise ValueError(f"{name} contains duplicate (game_id, team_id) keys")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} contains duplicate column names")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing source columns: {missing}")


def _check_numeric(frame: pd.DataFrame, columns: tuple[str, ...], *, nullable: bool) -> None:
    for column in columns:
        source = frame[column]
        if not pd.api.types.is_numeric_dtype(source.dtype) or pd.api.types.is_bool_dtype(
            source.dtype
        ):
            raise ValueError(f"Source {column} must be numeric")
        values = source.to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values) | (np.isnan(values) if nullable else False)
        if not valid.all():
            raise ValueError(f"Source {column} has forbidden missing or nonfinite values")


def _prepare_sources(
    prematch: pd.DataFrame, form: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _check_frame(prematch, "Prematch", (*_PAIRING_COLUMNS, *PREMATCH_MIRROR_COLUMNS))
    _check_frame(form, "Form", (*FORM_COPY_COLUMNS, *FORM_MIRROR_COLUMNS))
    if len(prematch) != len(form) or not prematch.index.isin(form.index).all():
        raise ValueError("Form keys must match prematch keys exactly")
    if not prematch.groupby(level="game_id").size().eq(2).all():
        raise ValueError("Prematch requires exactly two team rows per game")
    if prematch.loc[:, _PAIRING_COLUMNS].isna().any().any():
        raise ValueError("Prematch identity columns must not contain missing values")
    if not prematch.season_id.isin(DEVELOPMENT_SEASONS).all():
        raise ValueError("Prematch additions require development seasons 8-25")
    _check_numeric(form, (*FORM_COPY_COLUMNS, *FORM_MIRROR_COLUMNS), nullable=False)
    _check_numeric(prematch, ("points_per_game", "position_pct"), nullable=True)
    for column in ("promoted_in", "nothing_to_play"):
        if not pd.api.types.is_bool_dtype(prematch[column].dtype):
            raise ValueError(f"Source {column} must be boolean")
    if prematch.promoted_in.isna().any():
        raise ValueError("Source promoted_in must not contain missing values")
    ordered = prematch.sort_index()
    return ordered, form.reindex(ordered.index), _validate_pairing(ordered)


def build_prematch_v2_additions(prematch: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Return only five form copies and ten mirrors, sorted by (game_id, team_id).

    Frozen prematch/form frames may arrive in different row orders. All selection
    uses keys, and each venue value comes from the opponent's own home/away row.
    Dtypes and permitted table-state nulls are retained; form and promoted flags
    must be complete, as in v1. This validates the needed source projection, not
    v1's full 8,630-row artifact contract, so small synthetic fixtures are valid.
    The result contains only copies and mirrors; build_prematch_v2 adds the ratio.
    """
    ordered, own_form, opponent_prematch = _prepare_sources(prematch, form)
    opponent_form = _opponents(own_form, ordered)
    return pd.concat(
        [
            own_form.loc[:, FORM_COPY_COLUMNS],
            opponent_form.loc[:, FORM_MIRROR_COLUMNS].add_prefix("opp_"),
            opponent_prematch.loc[:, PREMATCH_MIRROR_COLUMNS].add_prefix("opp_"),
        ],
        axis=1,
    ).copy(deep=True)


def _gd_per_game(frame: pd.DataFrame) -> NDArray[np.float64]:
    _check_numeric(frame, ("goal_difference", "games_played"), nullable=False)
    played = frame.games_played.to_numpy(dtype=float)
    difference = frame.goal_difference.to_numpy(dtype=float)
    if (played < 0).any() or (played % 1 != 0).any():
        raise ValueError("games_played must contain nonnegative integers")
    return np.asarray(
        np.divide(difference, played, out=np.full(len(frame), np.nan), where=played > 0),
        dtype=float,
    )


def _validate_v2(frame: pd.DataFrame, *, full_development: bool) -> None:
    validate_prematch_feature_columns(frame)
    if tuple(frame.columns) != PREMATCH_V2_COLUMNS:
        raise ValueError("Prematch v2 requires exactly its ordered 67 columns")
    try:
        PREMATCH_V2_SCHEMA.validate(frame, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        raise ValueError(f"Prematch v2 schema validation failed: {exc}") from exc
    if full_development:
        validate_prematch_frame(frame.loc[:, PREMATCH_COLUMNS])
    _validate_previous_values(frame, (*PREVIOUS_VALUE_COLUMNS, *PREMATCH_CONTRAST_COLUMNS))
    opponent = _validate_pairing(frame)
    _check_numeric(
        frame, (*FORM_COPY_COLUMNS, *("opp_" + c for c in FORM_MIRROR_COLUMNS)), nullable=False
    )
    for column in (*PREMATCH_MIRROR_COLUMNS, "pts_venue_ewma5"):
        if not frame["opp_" + column].equals(opponent[column].rename("opp_" + column)):
            raise ValueError(f"Prematch v2 opponent feature mismatch: {column}")
    if not np.array_equal(
        frame.gd_per_game.to_numpy(dtype=float, na_value=np.nan),
        _gd_per_game(frame),
        equal_nan=True,
    ):
        raise ValueError("Prematch v2 gd_per_game differs from its source ratio")


def validate_prematch_v2_frame(frame: pd.DataFrame) -> None:
    """Require the complete 8,630-row, development-only v2 artifact before publication."""
    _validate_v2(frame, full_development=True)


def build_prematch_v2(prematch: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Preserve all 51 v1 columns and their row order, appending sixteen additions.

    Pure assembly also accepts small synthetic sources; publication separately
    requires the full development contract. No existing value is filled or cast.
    """
    if tuple(prematch.columns) != PREMATCH_COLUMNS:
        raise ValueError("Prematch v2 requires the original ordered 51 v1 columns")
    additions = build_prematch_v2_additions(prematch, form)
    result = prematch.copy(deep=True).join(additions, validate="one_to_one", sort=False)
    result["gd_per_game"] = _gd_per_game(result)
    _validate_v2(result, full_development=False)
    return result
