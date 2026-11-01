"""Retrospective chapter-A diagnostics; none publishes prediction features.

Era filters apply to outcomes and adjacent-season pairs. Older development history
may be consulted explicitly for t-2/t-3 diagnostics and promoted-value references.
LOSO deliberately trains on both earlier and later seasons: it is not walk-forward.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import pearsonr, spearmanr  # type: ignore[import-untyped]

from ligat.analysis.league_stats.core import (
    _games,
    _relationship,
    _selection,
    _subset,
    rate_interval,
    team_season_stats,
)
from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables

ERA14_SEASONS = tuple(range(14, 26))
PROMOTED_PERIODS = ((9, 17), (18, 25))


def correlation_evidence(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str]],
    *,
    comparisons: int | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Show complete-pair correlations, ordinary p and a declared Bonferroni family.

    Classical p values assume independent rows; club/season dependence and post-hoc
    selection limit them. They are diagnostics, never an admission rule for features.
    """
    family = len(pairs) if comparisons is None else comparisons
    if not pairs or family < len(pairs) or method not in ("pearson", "spearman"):
        raise ValueError("Declare a valid correlation method and complete comparison family")
    rows = []
    for name, x, y in pairs:
        data = frame[[x, y]].dropna().astype(float)
        r, p = np.nan, np.nan
        if len(data) >= 3 and data[x].nunique() > 1 and data[y].nunique() > 1:
            test = pearsonr if method == "pearson" else spearmanr
            r, p = test(data[x], data[y])
        rows.append(
            dict(
                relationship=name,
                n=len(data),
                correlation=r,
                p_value=p,
                comparisons=family,
                p_bonferroni=min(1.0, family * p) if np.isfinite(p) else np.nan,
            )
        )
    return pd.DataFrame(rows).set_index("relationship")


def adjacent_pairs(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Join by identity and exact t-1 inside the supplied frame, without bridging gaps."""
    data = frame.reset_index()
    previous = data[["season_id", "team_id", *columns]].copy()
    previous["season_id"] += 1
    previous = previous.rename(columns={name: f"{name}_prev" for name in columns})
    return data.merge(previous, on=["season_id", "team_id"], validate="one_to_one").set_index(
        ["season_id", "team_id"]
    )


def era_persistence(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
    *,
    n_bootstrap: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair seasons only within the selected era; bootstrap whole target seasons."""
    frame = team_season_stats(tables, _selection(seasons))
    paired = adjacent_pairs(frame, ["points_per_match", "final_position"])
    pairs = paired.rename(
        columns={
            "points_per_match_prev": "ppm_prev",
            "points_per_match": "ppm_t",
            "final_position_prev": "position_prev",
            "final_position": "position_t",
        }
    )
    pairs["ppm_change"] = pairs.ppm_t - pairs.ppm_prev
    # Keep identifiers here; human season labels belong to ligat.viz.
    pairs["season"] = pairs.index.get_level_values("season_id")
    rows = []
    for name, x, y in (("ppm", "ppm_prev", "ppm_t"), ("position", "position_prev", "position_t")):
        relation = _relationship(pairs, x, y, intervals=True, n_bootstrap=n_bootstrap)
        residual = pairs[y] - relation["intercept"] - relation["slope"] * pairs[x]
        rows.append(
            dict(
                relationship=name,
                **relation,
                residual_sd=residual.std(),
                mean_prev=pairs[x].mean(),
                mean_t=pairs[y].mean(),
            )
        )
    return pairs, pd.DataFrame(rows).set_index("relationship")


def loso_predictions(frame: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    """Fit OLS with intercept excluding every row of each held-out target season.

    Return row-level predictions so isolation and pooled RMSE can be audited.
    Missing predictors are excluded before defining folds; no fitted preprocessing.
    """
    _selection(frame.index.get_level_values("season_id").unique())
    if not predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Predictors must be nonempty and unique")
    complete = frame[["points_per_match", *predictors]].dropna().astype(float)
    if complete.index.get_level_values("season_id").nunique() < 2:
        raise ValueError("LOSO requires at least two complete seasons")
    result = complete[["points_per_match"]].rename(columns={"points_per_match": "observed"})
    result["predicted"] = np.nan
    result["n_train"] = 0
    seasons = complete.index.get_level_values("season_id")
    for season in sorted(seasons.unique()):
        train, test = complete.loc[seasons != season], complete.loc[seasons == season]
        design = np.column_stack([np.ones(len(train)), train[list(predictors)]])
        coefficients = np.linalg.lstsq(design, train.points_per_match, rcond=None)[0]
        test_design = np.column_stack([np.ones(len(test)), test[list(predictors)]])
        result.loc[test.index, "predicted"] = test_design @ coefficients
        result.loc[test.index, "n_train"] = len(train)
    result["error"] = result.observed - result.predicted
    return result


def _ols(frame: pd.DataFrame, predictors: Sequence[str], target: str) -> tuple[np.ndarray, float]:
    data = frame[[*predictors, target]].astype(float)
    x = np.column_stack([np.ones(len(data)), data[list(predictors)]])
    y = data[target].to_numpy()
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    total = float(((y - y.mean()) ** 2).sum())
    return residual, 1 - float((residual**2).sum()) / total if total > 0 else np.nan


def predictor_comparison(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw/log prior value and sporting priors; long-history rows share a sample.

    Older history uses available t-2/t-3 development seasons (mean of one or both),
    matching the explicitly retrospective reference. Current/previous pairs must
    both be inside `seasons`; current-season value never enters any regression.
    """
    selected = _selection(seasons)
    all_stats = team_season_stats(tables, DEVELOPMENT_SEASONS)
    frame = _subset(all_stats, selected)
    pairs = adjacent_pairs(frame, ["points_per_match", "goal_difference_per_match"])
    for lag in (2, 3):
        history = all_stats.goal_difference_per_match.copy()
        history.index = pd.MultiIndex.from_arrays(
            [
                history.index.get_level_values("season_id") + lag,
                history.index.get_level_values("team_id"),
            ],
            names=history.index.names,
        )
        pairs[f"gd_lag{lag}"] = history.reindex(pairs.index)
    pairs["history_gd"] = pairs[["gd_lag2", "gd_lag3"]].mean(axis=1)
    pairs["log_prev_value"] = np.log(
        pairs.prev_season_value_share.where(pairs.prev_season_value_share.gt(0))
    )
    pairs = pairs.dropna(subset=["prev_season_value_share"])
    gd, ppm = "goal_difference_per_match_prev", "points_per_match_prev"
    specifications = {
        "value": ["prev_season_value_share"],
        "log_value": ["log_prev_value"],
        "ppm": [ppm],
        "gd": [gd],
        "gd_value": [gd, "prev_season_value_share"],
        "gd_log_value": [gd, "log_prev_value"],
    }
    rows = []
    for scope, data in (
        ("adjacent", pairs),
        ("history_common", pairs.dropna(subset=["history_gd"])),
    ):
        specs = (
            specifications
            if scope == "adjacent"
            else {
                "gd": [gd],
                "gd_history": [gd, "history_gd"],
                "gd_value": [gd, "prev_season_value_share"],
                "gd_value_history": [gd, "prev_season_value_share", "history_gd"],
            }
        )
        for name, columns in specs.items():
            complete = data.dropna(subset=columns)
            predictions = loso_predictions(complete, columns)
            _, r2 = _ols(complete, columns, "points_per_match")
            rows.append(
                dict(
                    scope=scope,
                    predictor=name,
                    n=len(predictions),
                    r_squared=r2,
                    loso_rmse=np.sqrt(np.mean(predictions.error**2)),
                )
            )
    return pd.DataFrame(rows).set_index(["scope", "predictor"]), pairs


def home_advantage_diagnostics(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
) -> pd.Series:
    """Compare observed dispersion to a pooled independent-game luck benchmark.

    This benchmark is descriptive, not a causal variance decomposition. Actual
    home/away game counts are used, including unequal playoff schedule lengths.
    """
    selected = _selection(seasons)
    frame = team_season_stats(tables, selected)
    pairs = adjacent_pairs(frame, ["home_away_ppm_gap"])
    games = _games(tables, selected)
    variance = games.points_earned.astype(float).var(ddof=0)
    counts = games.groupby(["season_id", "team_id", "is_home"]).size().unstack("is_home")
    luck_variance = variance * (1 / counts[True] + 1 / counts[False])
    clubs = frame.groupby("team_id").agg(
        n=("points_per_match", "size"), gap=("home_away_ppm_gap", "mean")
    )
    clubs = clubs.loc[clubs.n.ge(6)]
    return pd.Series(
        dict(
            n=len(frame),
            n_pairs=len(pairs),
            mean_gap=frame.home_away_ppm_gap.mean(),
            persistence=pairs.home_away_ppm_gap.corr(pairs.home_away_ppm_gap_prev),
            observed_sd=frame.home_away_ppm_gap.std(),
            luck_sd=np.sqrt(luck_variance).mean(),
            top3_gap=frame.loc[frame.final_position.le(3), "home_away_ppm_gap"].mean(),
            places4_8_gap=frame.loc[frame.final_position.between(4, 8), "home_away_ppm_gap"].mean(),
            places9_11_gap=frame.loc[
                frame.final_position.between(9, 11), "home_away_ppm_gap"
            ].mean(),
            club_mean_sd=clubs.gap.std(),
            club_luck_sd=np.sqrt((frame.home_away_ppm_gap.var() / clubs.n).mean()),
        )
    )


def style_diagnostics(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Measure strength, winning margins and openness; labels remain retrospective."""
    frame = team_season_stats(tables, _selection(seasons))
    frame["big_win_share"] = frame.big_win_rate / frame.win_rate.where(frame.win_rate.gt(0))
    columns = [
        "points_per_match",
        "goal_difference_per_match",
        "win_rate",
        "big_win_share",
        "narrow_win_rate",
        "btts_rate",
        "over_2_5_rate",
    ]
    residual_r2 = {}
    for metric in ("btts_rate", "over_2_5_rate"):
        residual, r2 = _ols(frame, ["goals_scored_per_match", "goals_conceded_per_match"], metric)
        frame[f"{metric}_residual"] = residual
        columns.append(f"{metric}_residual")
        residual_r2[metric] = r2
    paired = adjacent_pairs(frame, columns)
    residual, base_r2 = _ols(paired, ["goal_difference_per_match_prev"], "points_per_match")
    _, narrow_r2 = _ols(
        paired, ["goal_difference_per_match_prev", "narrow_win_rate_prev"], "points_per_match"
    )
    prior = frame.reindex(
        pd.MultiIndex.from_arrays(
            [
                paired.index.get_level_values("season_id") - 1,
                paired.index.get_level_values("team_id"),
            ],
            names=frame.index.names,
        )
    ).copy()
    prior.index = paired.index
    luck, _ = _ols(prior, ["goal_difference_per_match"], "points_per_match")
    centered = frame[["goals_scored_per_match", "goals_conceded_per_match"]].astype(float)
    centered -= centered.groupby(level="season_id").transform("mean")
    metrics = dict(
        attack_defence_corr=centered.iloc[:, 0].corr(centered.iloc[:, 1]),
        gd_next_corr=paired.goal_difference_per_match_prev.corr(paired.points_per_match),
        ppm_next_corr=paired.points_per_match_prev.corr(paired.points_per_match),
        big_top3=frame.loc[frame.final_position.le(3), "big_win_rate"].mean(),
        big_bottom3=frame.loc[frame.final_position.ge(frame.n_teams - 2), "big_win_rate"].mean(),
        narrow_luck_corr=paired.narrow_win_rate_prev.corr(pd.Series(luck, index=paired.index)),
        gd_next_r2=base_r2,
        gd_narrow_next_r2=narrow_r2,
        league_ppm=frame.points_per_match.mean(),
    )
    rows = []
    for metric in columns:
        p = frame[metric].mean()
        rows.append(
            dict(
                metric=metric,
                league_mean=p,
                persistence=paired[metric].corr(paired[f"{metric}_prev"]),
                observed_sd=frame[metric].std(),
                luck_sd=np.sqrt(p * (1 - p) / 36) if metric.endswith("rate") else np.nan,
                r_squared=residual_r2.get(metric, np.nan),
            )
        )
    return pd.Series(metrics), pd.DataFrame(rows).set_index("metric")


def promoted_rule_comparison(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare old median and quantile-30 without modifying the production imputation.

    Shares are not normalized by league size, matching the agreed rule. Only
    development t-1 reference data are used; actual t values are diagnostic targets.
    Periods 9–17 / 18–25 are the predeclared descriptive stability split.
    """
    selected = _selection(seasons)
    context = sorted(set(selected) | {s - 1 for s in selected if s - 1 in DEVELOPMENT_SEASONS})
    frame = _subset(tables.team_seasons, context)
    benchmarks = {}
    for season, group in frame.groupby(level="season_id"):
        shares = group.value_share_of_league.dropna().sort_values()
        bottom = shares.iloc[: len(shares) // 2]
        benchmarks[season] = (bottom.median(), bottom.quantile(0.30))
    entrants = _subset(frame, selected)
    entrants = entrants.loc[
        entrants.promoted_in & (entrants.index.get_level_values("season_id") > 8)
    ]
    rows = []
    for (season, team), row in entrants.iterrows():
        for rule, offset in (("median", 0), ("q30", 1)):
            benchmark = benchmarks.get(season - 1, (np.nan, np.nan))[offset]
            rows.append(
                dict(
                    season_id=season,
                    team_id=team,
                    rule=rule,
                    team_name=row.team_name,
                    season=season,
                    value_share_of_league=row.value_share_of_league,
                    imputed_share=benchmark,
                    diff=row.value_share_of_league - benchmark,
                )
            )
    detail = pd.DataFrame(rows).set_index(["season_id", "team_id", "rule"])
    numeric = ["value_share_of_league", "imputed_share", "diff"]
    detail[numeric] = detail[numeric].apply(pd.to_numeric).astype(float)
    detail["ratio"] = detail.value_share_of_league / detail.imputed_share.where(
        detail.imputed_share.gt(0)
    )
    summaries = []
    for period, lo, hi in (
        ("all", min(selected), max(selected)),
        *((f"{lo}–{hi}", lo, hi) for lo, hi in PROMOTED_PERIODS),
    ):
        data = detail.loc[
            detail.index.get_level_values("season_id").to_series(index=detail.index).between(lo, hi)
        ]
        for rule, group in data.groupby(level="rule"):
            complete = group.dropna(subset=["diff"])
            rate, low, high = rate_interval(int(complete["diff"].lt(0).sum()), len(complete))
            summaries.append(
                dict(
                    period=period,
                    rule=rule,
                    n=len(complete),
                    fraction_below=rate,
                    median_diff_pp=100 * complete["diff"].median(),
                    mae_pp=100 * complete["diff"].abs().mean(),
                    ci_lower=low,
                    ci_upper=high,
                )
            )
    return detail, pd.DataFrame(summaries).set_index(["period", "rule"])


def promoted_performance(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
) -> pd.DataFrame:
    """Summarize sporting performance with official relegation/position context."""
    frame = team_season_stats(tables, _selection(seasons))
    rows = []
    for promoted, group in frame.groupby("promoted_in"):
        rows.append(
            dict(
                promoted_in=bool(promoted),
                n=len(group),
                points_per_match=group.points_per_match.mean(),
                goal_difference_per_match=group.goal_difference_per_match.mean(),
                std_ppm=group.points_per_match.std(),
                relegations=int(group.relegated.sum()),
                best_final_position=group.final_position.min(),
            )
        )
    return pd.DataFrame(rows).set_index("promoted_in")
