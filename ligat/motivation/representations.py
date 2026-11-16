"""The preregistered EXP-004 representations, shared by review and later fitting."""

from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.simulate.leverage import SCORE_TARGETS


def _score(frame: pd.DataFrame) -> pd.Series:
    score = frame.leverage_max.astype(float)
    if score.isna().any() or not score.between(0, 1).all() or frame.nothing_to_play.isna().any():
        raise ValueError("Representations need finite importance in [0,1] and known flags")
    return score


def level_l2(frame: pd.DataFrame) -> pd.Series:
    """L2: zero for a mathematically dead team, one otherwise."""
    return (~frame.nothing_to_play.astype(bool)).astype(float).rename("L2")


def level_l3(frame: pd.DataFrame) -> pd.Series:
    """L3: dead=0; alive with importance <=0.01=0.5; remaining alive=1."""
    score = _score(frame)
    return pd.Series(
        np.where(frame.nothing_to_play, 0.0, np.where(score.le(0.01), 0.5, 1.0)),
        index=frame.index,
        name="L3",
    )


def level_l5(frame: pd.DataFrame) -> pd.Series:
    """L5: dead=0; alive intervals <=.01, <=.05, <=.2, >.2 map to .25,.5,.75,1."""
    score = _score(frame)
    values = np.select(
        [score.le(0.01), score.le(0.05), score.le(0.2)], [0.25, 0.5, 0.75], default=1.0
    )
    return pd.Series(np.where(frame.nothing_to_play, 0.0, values), index=frame.index, name="L5")


def level_lc(frame: pd.DataFrame) -> pd.Series:
    """Lc: continuous importance, with the dead-team convention fixed at zero."""
    return _score(frame).where(~frame.nothing_to_play, 0.0).rename("Lc")


def components(frame: pd.DataFrame) -> pd.DataFrame:
    """Comp: title/cut/survival; an inapplicable cut contributes zero, not an imputation.

    The artifact retains NaN for an absent objective. The numeric representation
    uses zero for that objective and for dead rows; direct survival is excluded.
    """
    values = (
        frame[[f"{t}_lev" for t in SCORE_TARGETS]]
        .copy()
        .rename(columns={f"{t}_lev": t for t in SCORE_TARGETS})
    )
    values["cut"] = values.cut.fillna(0.0)
    if not np.isfinite(values.to_numpy(float)).all() or not values.ge(0).all().all():
        raise ValueError("Components require known nonnegative applicable importance")
    values.loc[frame.nothing_to_play.astype(bool), :] = 0.0
    return values


def game_contrast(
    frame: pd.DataFrame, level: str, form: Literal["diff", "sides"] = "diff"
) -> pd.DataFrame:
    """Return one row per fixture: home-minus-away or explicitly ordered home/away sides.

    Swapping the declared venue sides negates diff and swaps sides. Comp has only
    the three-difference form preregistered in EXP-004.
    """
    scalar = {"L2": level_l2, "L3": level_l3, "L5": level_l5, "Lc": level_lc}
    if form not in ("diff", "sides") or (level == "Comp" and form != "diff"):
        raise ValueError("Unknown form; Comp is preregistered only as diff")
    if level == "Comp":
        values = components(frame)
    elif level in scalar:
        values = scalar[level](frame).to_frame("m")
    else:
        raise ValueError("Unknown motivation representation")
    rows = frame.reset_index()
    if (
        rows.duplicated(["game_id", "team_id"]).any()
        or not rows.groupby("game_id").size().eq(2).all()
    ):
        raise ValueError("Expected unique paired team-game rows")
    data = rows[["game_id", "team_id", "opponent_id", "is_home"]].join(
        values.reset_index(drop=True)
    )
    home = data.loc[data.is_home.eq(True)].set_index("game_id")
    away = data.loc[data.is_home.eq(False)].set_index("game_id").reindex(home.index)
    if (
        not home.index.is_unique
        or len(home) * 2 != len(data)
        or away.team_id.isna().any()
        or not home.team_id.eq(away.opponent_id).all()
        or not home.opponent_id.eq(away.team_id).all()
    ):
        raise ValueError("Fixture sides do not identify each other's opponent")
    result = pd.DataFrame(index=home.index)
    for column in values:
        if form == "diff":
            result[f"{column}_diff"] = home[column] - away[column]
        else:
            result[f"{column}_home"], result[f"{column}_away"] = home[column], away[column]
    return result.sort_index()
