"""Retrospective venue/crowd diagnostics; never publish these labels as features."""

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from scipy import stats  # type: ignore[import-untyped]
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS

from .core import _selection, _subset

CONDITIONS = ("NORMAL", "GHOST", "LIMITED", "SHARED", "RELOCATED", "OTHER")
CONDITION_SEED = 20261028


def load_match_conditions(path: Path) -> dict[str, Any]:
    """Load reviewed configuration, validating development limits and local date intervals."""
    with path.open(encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream)
    if config["timezone"] != "Asia/Jerusalem":
        raise ValueError("Condition dates must use Asia/Jerusalem")
    periods = config["crowd_periods"]
    for entry in [*periods, *config["relocations"]]:
        if not entry.get("source") or entry.get("confidence") not in {"low", "medium", "high"}:
            raise ValueError("Every entry needs a source and confidence")
        if "seasons" in entry:
            _selection(entry["seasons"])
        if "start" in entry or "end" in entry:
            start, end = (date.fromisoformat(str(entry[k])) for k in ("start", "end"))
            if start > end or start < date(2006, 7, 1) or end > date(2024, 6, 30):
                raise ValueError("Invalid or non-development condition interval")
            entry.update(start=start, end=end)
    for i, left in enumerate(periods):
        if left["condition"] not in {"GHOST", "LIMITED"}:
            raise ValueError("Unknown crowd condition")
        for right in periods[i + 1 :]:
            if max(left["start"], right["start"]) <= min(left["end"], right["end"]):
                raise ValueError("Crowd periods overlap")
    return config


def classify_conditions(
    matches: pd.DataFrame,
    config: dict[str, Any],
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
) -> pd.DataFrame:
    """Use full-season modal home grounds, then apply the approved exclusive hierarchy."""
    games = _subset(matches, _selection(seasons))
    local = pd.to_datetime(games.date, utc=True).dt.tz_convert(config["timezone"]).dt.date
    if games.venue.isna().any() or local.isna().any():
        raise ValueError("Classification needs known dates and venues")
    modes = games.groupby(["season_id", "home_team_id"], observed=True).venue.agg(
        lambda values: values.mode().iloc[0]
    )
    home = [modes.loc[(s, t)] for s, t in zip(games.season_id, games.home_team_id, strict=True)]
    away = [modes.get((s, t)) for s, t in zip(games.season_id, games.away_team_id, strict=True)]
    games["condition"] = np.where(games.is_home_venue.eq(False), "OTHER", "NORMAL")  # noqa: E712
    for rule in config["relocations"]:
        mask = games.home_team_id.eq(rule["team_id"]) & games.season_id.isin(rule["seasons"])
        if "start" in rule:
            mask &= local.between(rule["start"], rule["end"])
        games.loc[mask & games.venue.ne(rule["original_venue"]), "condition"] = "RELOCATED"
    games.loc[games.venue.eq(home) & games.venue.eq(away), "condition"] = "SHARED"
    for rule in config["crowd_periods"]:
        games.loc[local.between(rule["start"], rule["end"]), "condition"] = rule["condition"]
    return games


def prematch_elo(
    matches: pd.DataFrame, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Chronological Elo snapshots; simultaneous kickoffs all use the same prior state.

    New clubs start at 1450; returning clubs regress 1/3 toward 1500 on their first
    match of each new season. K=20; HFA=50 affects updates only. Goal multipliers
    are 1 (0/1 goal), 1.5 (2 goals), (11+GD)/8 (3+ goals).
    """
    games = _subset(matches, _selection(seasons)).copy()
    games["_time"] = pd.to_datetime(games.date, utc=True)
    if games._time.isna().any():
        raise ValueError("Elo requires kickoff dates")
    games = games.sort_values(["_time", "season_id"], kind="stable")
    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}
    before: dict[Any, tuple[float, float]] = {}
    for _, batch in games.groupby("_time", sort=False):
        increments: dict[int, float] = {}
        for key, game in batch.iterrows():
            clubs = (int(game.home_team_id), int(game.away_team_id))
            for club in clubs:
                if club not in ratings:
                    ratings[club] = 1450.0
                elif last_season[club] != int(game.season_id):
                    ratings[club] = 1500 + (ratings[club] - 1500) * 2 / 3
                last_season[club] = int(game.season_id)
            h, a = (ratings[club] for club in clubs)
            before[key] = h, a
            expected = 1 / (1 + 10 ** (-(h + 50 - a) / 400))
            gd = float(game.home_goals - game.away_goals)
            score = 1.0 if gd > 0 else 0.0 if gd < 0 else 0.5
            magnitude = abs(gd)
            multiplier = 1.0 if magnitude <= 1 else 1.5 if magnitude == 2 else (11 + magnitude) / 8
            delta = 20 * multiplier * (score - expected)
            for club, change in zip(clubs, (delta, -delta), strict=True):
                increments[club] = increments.get(club, 0.0) + change
        for club, change in increments.items():
            ratings[club] += change
    games["elo_home"] = [before[k][0] for k in games.index]
    games["elo_away"] = [before[k][1] for k in games.index]
    games["elo_gap"] = games.elo_home - games.elo_away
    return games.drop(columns="_time").reindex(matches.index.intersection(games.index))


def elo_residuals(
    matches: pd.DataFrame, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """In-sample nuisance regression for EDA adjustment, not forecast validation."""
    games = prematch_elo(matches, seasons)
    x = games.elo_gap.to_numpy(dtype=float) / 100
    design = np.column_stack([np.ones(len(x)), x, x**3])
    games["points_gap"] = np.sign(games.home_goals - games.away_goals) * 3
    beta = np.linalg.lstsq(design, games.points_gap.to_numpy(dtype=float), rcond=None)[0]
    games["residual"] = games.points_gap - design @ beta
    return games


def stratified_effect(
    groups: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    *,
    n_permutations: int = 10000,
    n_bootstrap: int = 5000,
    seed: int = CONDITION_SEED,
) -> dict[str, float]:
    """Fixed condition-count weights; permute ONLY within seasons; BCa within groups.

    Each tuple contains condition and NORMAL residuals from ONE season. SciPy's
    multi-sample BCa uses a separate leave-one-out jackknife for each stratum
    and its sample-size scaling in the acceleration (not a pooled jackknife).
    Confidence intervals condition on the observed seasons and fitted Elo regression.
    """
    if not groups or n_permutations < 1 or n_bootstrap < 2:
        raise ValueError("Nonempty groups and positive resampling counts required")
    if any(min(len(a), len(b)) < 2 for a, b in groups):
        raise ValueError("BCa requires at least two matches in each season/condition stratum")
    # Center each pair together: seasonal offsets cancel in the estimand and
    # should not change BCa tie comparisons through floating-point cancellation.
    groups = [
        (a - np.median(np.concatenate([a, b])), b - np.median(np.concatenate([a, b])))
        for a, b in groups
    ]
    weights = np.array([len(a) for a, _ in groups], dtype=float)
    weights /= weights.sum()
    samples = tuple(array for pair in groups for array in pair)

    def statistic(*arrays: Any, axis: int = -1) -> Any:
        return sum(
            w * (arrays[2 * i].mean(axis=axis) - arrays[2 * i + 1].mean(axis=axis))
            for i, w in enumerate(weights)
        )

    estimate = float(statistic(*samples))
    rng = np.random.default_rng(seed)
    null = np.zeros(n_permutations)
    for weight, (condition, normal) in zip(weights, groups, strict=True):
        pool = np.concatenate([condition, normal])
        for start in range(0, n_permutations, 128):
            end = min(start + 128, n_permutations)
            shuffled = rng.permuted(np.broadcast_to(pool, (end - start, len(pool))), axis=1)
            effect = shuffled[:, : len(condition)].mean(axis=1) - shuffled[
                :, len(condition) :
            ].mean(axis=1)
            null[start:end] += weight * effect
    p = (1 + np.count_nonzero(np.abs(null) >= abs(estimate))) / (n_permutations + 1)
    result = stats.bootstrap(
        samples,
        statistic,
        vectorized=True,
        paired=False,
        method="BCa",
        n_resamples=n_bootstrap,
        batch=128,
        random_state=seed,
    )
    return dict(
        effect=estimate,
        p=float(p),
        ci_lower=float(result.confidence_interval.low),
        ci_upper=float(result.confidence_interval.high),
    )


def condition_comparison(
    matches: pd.DataFrame,
    config: dict[str, Any],
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_permutations: int = 10000,
    n_bootstrap: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare four conditions with NORMAL on the SAME weighted seasonal estimand."""
    selected = _selection(seasons)
    games = elo_residuals(classify_conditions(matches, config, selected), selected)
    rows = []
    for condition in CONDITIONS:
        target = games.loc[games.condition.eq(condition)]
        row: dict[str, Any] = dict(
            condition=condition, n=len(target), raw_gap=target.points_gap.mean()
        )
        if condition not in {"NORMAL", "OTHER"}:
            groups = []
            for season, group in target.groupby("season_id", observed=True):
                normal = games.loc[games.season_id.eq(season) & games.condition.eq("NORMAL")]
                if len(normal):
                    groups.append((group.residual.to_numpy(), normal.residual.to_numpy()))
            row.update(
                n_seasons=len(groups),
                normal_n=sum(len(b) for _, b in groups),
                overlap_n=sum(len(a) for a, _ in groups),
            )
            if groups:
                row.update(
                    stratified_effect(
                        groups, n_permutations=n_permutations, n_bootstrap=n_bootstrap
                    )
                )
        rows.append(row)
    result = pd.DataFrame(rows).set_index("condition")
    tested = result.p.notna()
    # The family is the FOUR prespecified comparisons, even if one is unavailable.
    p_values = result.loc[tested, "p"].to_numpy(dtype=float)
    padded = np.concatenate([p_values, np.ones(4 - len(p_values))])
    result.loc[tested, "p_bonferroni"] = multipletests(padded, method="bonferroni")[1][
        : len(p_values)
    ]
    result.loc[tested, "p_bh"] = multipletests(padded, method="fdr_bh")[1][: len(p_values)]
    return result, games


def home_period_comparison(
    matches: pd.DataFrame,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    normal_only: bool = False,
) -> pd.DataFrame:
    """Equal-weight season means and SE; fixed retrospective cut at 2015/2016."""
    games = _subset(matches, _selection(seasons))
    if normal_only:
        games = games.loc[games.condition.eq("NORMAL")]
    gap = np.sign(games.home_goals - games.away_goals) * 3
    means = gap.groupby(games.season_id).mean()
    rows = []
    for start, end in ((11, 16), (17, 25)):
        data = means.loc[means.index.to_series().between(start, end)]
        rows.append(
            dict(
                start_season=start,
                end_season=end,
                n_seasons=len(data),
                mean_gap=data.mean(),
                se=data.sem(),
                std=data.std(),
            )
        )
    return pd.DataFrame(rows)
