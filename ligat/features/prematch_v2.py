"""Pure copies and opponent mirrors for prematch v2; no computation or publication."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.features.pipeline import DEVELOPMENT_SEASONS, _opponents, _validate_pairing

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
    The result is not a published v2: gd_per_game and registry updates belong to A2.
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
