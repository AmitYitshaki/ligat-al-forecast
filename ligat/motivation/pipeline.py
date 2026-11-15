"""Pure M1 construction; publication and file loading are separate boundaries."""

import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.models.ordered_logit import OrderedFit
from ligat.simulate.leverage import DEFAULT_SEED, TARGETS, SeasonSimulator, empirical_scores
from ligat.simulate.rules import SeasonRules
from ligat.simulate.snapshot import DEVELOPMENT_SEASONS, season_snapshots

IDENTITY_COLUMNS = ("opponent_id", "season_id", "as_of", "is_home", "N", "before_split")
TARGET_COLUMNS = tuple(
    f"{target}_{suffix}"
    for target in TARGETS
    for suffix in ("p_win", "p_draw", "p_loss", "lev_raw", "lev", "mc_se")
)
MOTIVATION_COLUMNS = (*IDENTITY_COLUMNS, *TARGET_COLUMNS, "leverage_max", "nothing_to_play")
Checkpoint = Callable[[int, pd.DataFrame, Mapping[str, object]], None]


@dataclass(frozen=True)
class MotivationResult:
    """Complete features plus per-season elapsed times and lottery counters."""

    features: pd.DataFrame
    seasons: tuple[Mapping[str, object], ...]
    seconds: float


def validate_motivation_frame(frame: pd.DataFrame) -> None:
    """Reject malformed identities, missing applicable estimates or altered raw effects."""
    if (
        tuple(frame.columns) != MOTIVATION_COLUMNS
        or frame.index.names != ["game_id", "team_id"]
        or not frame.index.is_unique
        or frame.empty
        or frame.index.to_frame().isna().any().any()
        or not frame.groupby(level="game_id").size().eq(2).all()
        or not set(frame.season_id) <= DEVELOPMENT_SEASONS
    ):
        raise ValueError("Invalid motivation columns, development seasons or paired identities")
    for name in frame.index.names:
        if frame.index.get_level_values(name).dtype != np.dtype("int64"):
            raise ValueError("Motivation identifiers require canonical int64")
    if (
        frame.loc[:, [*IDENTITY_COLUMNS, "leverage_max", "nothing_to_play"]].isna().any().any()
        or str(frame.as_of.dt.tz) != "Asia/Jerusalem"
        or not frame.as_of.equals(frame.as_of.dt.normalize())
        or not pd.api.types.is_bool_dtype(frame.nothing_to_play.dtype)
        or not pd.api.types.is_bool_dtype(frame.before_split.dtype)
        or not frame.N.ge(2).all()
    ):
        raise ValueError("Missing identity, invalid morning, flags or path counts")
    for target in TARGETS:
        available = pd.Series(True, index=frame.index)
        if target == "cut":
            available = frame.before_split
        elif target == "survival_direct":
            available = frame.season_id.isin([11, 12])
        columns = [
            f"{target}_{suffix}"
            for suffix in ("p_win", "p_draw", "p_loss", "lev_raw", "lev", "mc_se")
        ]
        if not frame.loc[~available, columns].isna().all().all():
            raise ValueError(f"Inapplicable {target} values must be NaN")
        part = frame.loc[available, columns]
        if not np.isfinite(part.to_numpy(float)).all():
            raise ValueError(f"Missing applicable {target} estimates")
        probabilities = part.iloc[:, :3]
        if not ((probabilities >= 0) & (probabilities <= 1)).all().all():
            raise ValueError("Objective probabilities must lie in [0, 1]")
        raw = part[f"{target}_p_win"] - part[f"{target}_p_loss"]
        if (
            not np.array_equal(raw, part[f"{target}_lev_raw"])
            or not np.array_equal(raw.clip(lower=0), part[f"{target}_lev"])
            or not part[f"{target}_mc_se"].ge(0).all()
        ):
            raise ValueError("Raw effects, clipping or paired standard errors are inconsistent")
    maximum = frame[[f"{target}_lev" for target in TARGETS]].max(axis=1)
    if not np.array_equal(maximum, frame.leverage_max) or not maximum.between(0, 1).all():
        raise ValueError("Invalid maximum leverage")
    dead = frame.nothing_to_play
    if frame.loc[dead, "leverage_max"].ne(0).any():
        raise ValueError("HARD STOP: nothing_to_play has nonzero leverage")


def _verify_decided(frame: pd.DataFrame, prematch: pd.DataFrame) -> None:
    for target, flag in (
        ("title", "title_decided"),
        ("cut", "cut_decided"),
        ("survival", "relegation_decided"),
    ):
        if flag not in prematch:
            continue
        selected = prematch[flag].reindex(frame.index) & frame[f"{target}_p_win"].notna()
        part = frame.loc[selected]
        probabilities = part[[f"{target}_p_{x}" for x in ("win", "draw", "loss")]]
        valid = (
            part[f"{target}_lev_raw"].eq(0)
            & probabilities.nunique(axis=1).eq(1)
            & probabilities.iloc[:, 0].isin([0.0, 1.0])
        )
        if not valid.all():
            raise ValueError(
                f"HARD STOP: decided {target}: {part.loc[~valid].to_json(orient='index')}"
            )


def build_motivation(
    rows: pd.DataFrame,
    elo: pd.DataFrame,
    fits: Mapping[int, OrderedFit],
    rules: Mapping[int, SeasonRules],
    prematch: pd.DataFrame,
    *,
    seasons: Sequence[int] = tuple(range(8, 26)),
    paths: int = 2000,
    seed: int = DEFAULT_SEED,
    checkpoint: Checkpoint | None = None,
) -> MotivationResult:
    """Build seasons backwards; frozen flags join one-to-one without overwriting estimates.

    Only seasons 8-25 may enter. Seasons 8-10 deliberately share their in-sample
    conversion and exact-score pool and are reserved for EXP-004 training.
    """
    selected = set(seasons)
    if (
        not selected
        or not selected <= DEVELOPMENT_SEASONS
        or not set(rows.season_id) <= DEVELOPMENT_SEASONS
        or not set(elo.season_id) <= DEVELOPMENT_SEASONS
        or not set(prematch.season_id) <= DEVELOPMENT_SEASONS
    ):
        raise ValueError("Motivation requires development seasons 8-25 only")
    if not prematch.index.is_unique or prematch.index.names != ["game_id", "team_id"]:
        raise ValueError("Prematch requires unique paired identities")
    started = time.perf_counter()
    frames = []
    audits: list[Mapping[str, object]] = []
    for sid in sorted(selected, reverse=True):
        tick = time.perf_counter()
        pools = empirical_scores(rows, sid)
        local = []
        counts: Counter[str] = Counter()
        dates = 0
        for snapshot in season_snapshots(rows, elo, rules, sid):
            result = SeasonSimulator(snapshot, fits[sid], pools).simulate(paths, seed)
            _verify_decided(result.features, prematch)
            local.append(result.features)
            counts.update(result.tie_counts)
            dates += 1
        season = pd.concat(local).sort_index()
        flags = prematch.loc[prematch.season_id.eq(sid), ["nothing_to_play"]]
        if not season.index.equals(flags.sort_index().index):
            raise ValueError(f"Season {sid} does not cover precisely the prematch rows")
        season = season.join(flags, validate="one_to_one").loc[:, MOTIVATION_COLUMNS]
        validate_motivation_frame(season)
        audit = {
            "season_id": sid,
            "N": paths,
            "seed": seed,
            "rows": len(season),
            "dates": dates,
            "seconds": time.perf_counter() - tick,
            **dict(counts),
        }
        if checkpoint is not None:
            checkpoint(sid, season.copy(deep=True), audit)
        frames.append(season)
        audits.append(audit)
    frame = pd.concat(frames).sort_index()
    expected = prematch.loc[prematch.season_id.isin(selected)].sort_index()
    if not frame.index.equals(expected.index):
        raise ValueError("Motivation does not cover the selected frozen prematch rows")
    validate_motivation_frame(frame)
    return MotivationResult(frame, tuple(audits), time.perf_counter() - started)
