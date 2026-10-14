"""Development-only descriptive season statistics; never features or trained forecasts."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from statsmodels.stats.proportion import proportion_confint  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables

BOOTSTRAP_SEED = 20261014
MEAN_BOOTSTRAP_DRAWS = 2000
RELATIONSHIP_BOOTSTRAP_DRAWS = 1000

SEASON_COLUMNS = [
    "team_name",
    "final_position",
    "final_stage",
    "matches_played",
    "wins",
    "draws",
    "losses",
    "points_from_matches",
    "points_per_match",
    "goals_scored_per_match",
    "goals_conceded_per_match",
    "home_points_per_match",
    "away_points_per_match",
    "clean_sheet_rate",
    "failed_to_score_rate",
    "longest_unbeaten_run",
    "longest_winless_run",
    "relegated",
    "promoted_in",
    "value_share_of_league",
    "prev_season_value_share",
    "draw_rate",
    "squad_size",
]
RATE_FLAGS = {
    "win_rate": "win",
    "draw_rate": "draw",
    "loss_rate": "loss",
    "home_win_rate": "home_win",
    "away_win_rate": "away_win",
    "big_win_rate": "big_win",
    "narrow_win_rate": "narrow_win",
    "heavy_loss_rate": "heavy_loss",
    "btts_rate": "btts",
    "over_2_5_rate": "over_2_5",
}


def _regular_points(tables: AnalysisTables, selected: list[int]) -> pd.DataFrame:
    games = _subset(tables.team_matches, selected).reset_index()
    return (
        games.loc[games.stage.eq("regular")]
        .groupby(["season_id", "team_id"], observed=True)
        .agg(points=("points_earned", "sum"), games=("game_id", "size"))
    )


def _upper_boundary(regular: pd.DataFrame, k: int) -> pd.DataFrame:
    if not 0 < k < len(regular):
        raise ValueError("Upper playoff size must leave teams on both sides of the boundary")
    # Points alone define the cutoff; team_id only makes tied rows reproducible.
    return (
        regular.sort_index()
        .sort_values("points", ascending=False, kind="stable")
        .iloc[k - 1 : k + 1]
    )


def season_thresholds(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
) -> pd.DataFrame:
    """Describe observed objectives using full sporting points and official outcome labels.

    Pass upper_sizes loaded from config/seasons.yaml outside this pure calculation.
    Upper points are at the regular-stage boundary, other points at season end.
    Ties use the mean games played by tied boundary teams for the descriptive ppm.
    """
    selected = _selection(seasons)
    teams = _subset(tables.team_seasons, selected)
    regular = _regular_points(tables, selected)
    eras = _subset(tables.team_matches, selected).groupby("season_id", observed=True).era.first()
    rows = []
    for season, group in teams.groupby(level="season_id", observed=True):
        if season not in upper_sizes:
            raise ValueError(f"Missing playoff configuration for season_id={season}")
        champion = group.loc[group.final_position.eq(1)].iloc[0]
        runner = group.loc[group.final_position.eq(2)].iloc[0]
        survivors = group.loc[~group.relegated]
        relegated = group.loc[group.relegated]
        row: dict[str, Any] = dict(
            season_id=season,
            era=eras.loc[season],
            n_teams=len(group),
            games_per_team=group.matches_played.mean(),
            games_per_team_min=group.matches_played.min(),
            games_per_team_max=group.matches_played.max(),
            n_relegated=int(group.relegated.sum()),
        )
        for label, candidates, points in (
            ("champion", group.loc[group.final_position.eq(1)], champion.points_from_matches),
            ("runner_up", group.loc[group.final_position.eq(2)], runner.points_from_matches),
            ("safety", survivors, survivors.points_from_matches.min()),
            ("relegated_max", relegated, relegated.points_from_matches.max()),
        ):
            tied = candidates.loc[candidates.points_from_matches.eq(points)]
            games = tied.matches_played.mean()
            row.update(
                {f"{label}_points": points, f"{label}_ppm": points / games, f"{label}_games": games}
            )
        row["title_margin_points"] = row["champion_points"] - row["runner_up_points"]
        row["title_margin_ppm"] = row["champion_ppm"] - row["runner_up_ppm"]
        row["upper_line_points"] = row["upper_line_ppm"] = row["upper_line_games"] = float("nan")
        k = upper_sizes[int(season)]
        if k is not None:
            season_regular = regular.xs(season, level="season_id")
            points = _upper_boundary(season_regular, k).points.iloc[0]
            games = season_regular.loc[season_regular.points.eq(points), "games"].mean()
            row.update(
                upper_line_points=points, upper_line_ppm=points / games, upper_line_games=games
            )
        row["flag_inverted"] = row["relegated_max_points"] > row["safety_points"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("season_id", drop=True).sort_index()


def critical_ties(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
) -> pd.DataFrame:
    """Count equal sporting points across official title/relegation and regular playoff lines."""
    selected = _selection(seasons)
    thresholds = season_thresholds(tables, selected, upper_sizes=upper_sizes)
    regular = _regular_points(tables, selected)
    teams = _subset(tables.team_seasons, selected)
    rows = []
    for season, group in teams.groupby(level="season_id", observed=True):
        survivor = group.loc[~group.relegated].sort_values("final_position").iloc[-1]
        relegated = group.loc[group.relegated].sort_values("final_position").iloc[0]
        upper: Any = pd.NA
        k = upper_sizes[int(season)]
        if k is not None:
            boundary = _upper_boundary(regular.xs(season, level="season_id"), k)
            upper = boundary.points.iloc[0] == boundary.points.iloc[1]
        rows.append(
            dict(
                season_id=season,
                title_tie=thresholds.loc[season, "title_margin_points"] == 0,
                upper_line_tie=upper,
                relegation_tie=survivor.points_from_matches == relegated.points_from_matches,
            )
        )
    frame = pd.DataFrame(rows).set_index("season_id", drop=True).sort_index()
    return frame.astype("boolean")


def threshold_ranges(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
    era: str = "14_teams",
) -> pd.DataFrame:
    """Summarize observed objective totals within one era, retaining stage-specific game counts."""
    frame = season_thresholds(tables, seasons, upper_sizes=upper_sizes)
    frame = frame.loc[frame.era.eq(era)]
    rows = []
    for objective in ("champion", "upper_line", "safety", "relegated_max"):
        points = frame[f"{objective}_points"].dropna()
        rows.append(
            dict(
                objective=objective,
                n=len(points),
                points_min=points.min(),
                points_median=points.median(),
                points_max=points.max(),
                ppm_mean=frame[f"{objective}_ppm"].mean(),
                games_min=frame[f"{objective}_games"].min(),
                games_max=frame[f"{objective}_games"].max(),
            )
        )
    return pd.DataFrame(rows).set_index("objective", drop=True)


def critical_ties_summary(ties: pd.DataFrame) -> pd.DataFrame:
    """Summarize ties with the applicable-season denominator, not fictitious pre-playoff lines."""
    rows = [
        dict(
            boundary=column,
            n_seasons=len(ties),
            n_applicable=int(ties[column].count()),
            n_ties=int(ties[column].sum()),
        )
        for column in ties
    ]
    return pd.DataFrame(rows).set_index("boundary", drop=True)


def _match_rates(games: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"n": len(games), "goals_per_match": games.total_goals.mean()}
    for label, flag in (
        ("home_win_rate", games.result.eq("home_win")),
        ("draw_rate", games.result.eq("draw")),
        ("away_win_rate", games.result.eq("away_win")),
        ("zero_zero_rate", games.total_goals.eq(0)),
        ("big_margin_rate", (games.home_goals - games.away_goals).abs().ge(3)),
    ):
        row[label], row[f"{label}_ci_lower"], row[f"{label}_ci_upper"] = rate_interval(
            int(flag.sum()), len(games)
        )
    return row


def home_advantage_by_season(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Compare home and away goals/points per match, retaining uncertainty in result rates."""
    matches = _subset(tables.match_table, _selection(seasons))
    rows = []
    for season, games in matches.groupby("season_id", observed=True):
        row = dict(season_id=season, **_match_rates(games))
        row["home_goals_pm"] = games.home_goals.mean()
        row["away_goals_pm"] = games.away_goals.mean()
        row["goals_pm_gap"] = row["home_goals_pm"] - row["away_goals_pm"]
        row["home_ppm"] = 3 * row["home_win_rate"] + row["draw_rate"]
        row["away_ppm"] = 3 * row["away_win_rate"] + row["draw_rate"]
        row["ppm_gap"] = row["home_ppm"] - row["away_ppm"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("season_id", drop=True).sort_index()


def competitive_balance(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Measure dispersion between sporting team rates without assuming official rank orders ppm."""
    teams = _subset(tables.team_seasons, _selection(seasons))
    rows = []
    for season, group in teams.groupby(level="season_id", observed=True):
        champion = group.loc[group.final_position.eq(1), "points_per_match"].iloc[0]
        runner = group.loc[group.final_position.eq(2), "points_per_match"].iloc[0]
        rows.append(
            dict(
                season_id=season,
                std_ppm=group.points_per_match.std(),
                top_bottom_ppm_gap=group.points_per_match.max() - group.points_per_match.min(),
                title_margin_ppm=champion - runner,
            )
        )
    return pd.DataFrame(rows).set_index("season_id", drop=True).sort_index()


def champions_summary(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Count titles by official champion labels; names use the latest selected season."""
    teams = _subset(tables.team_seasons, _selection(seasons)).sort_index().reset_index()
    winners = teams.loc[teams.final_position.eq(1)]
    return (
        winners.groupby("team_id")
        .agg(titles=("season_id", "size"), team_name=("team_name", "last"))
        .sort_values("titles", ascending=False)
    )


def _goals_interval(games: pd.DataFrame, n_bootstrap: int) -> tuple[float, float]:
    # Season-block bootstrap preserves within-season dependence and weights by matches.
    grouped = games.groupby("season_id", observed=True).total_goals.agg(["sum", "count"])
    if len(grouped) < 2:
        return float("nan"), float("nan")
    data = grouped.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = data[rng.integers(len(data), size=(n_bootstrap, len(data)))].sum(axis=1)
    estimates = sampled[:, 0] / sampled[:, 1]
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def era_comparison(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Compare match-weighted era outcomes and season-weighted competitive dispersion."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    matches = _subset(tables.match_table, selected)
    balance = competitive_balance(tables, selected)
    rows = []
    for era, games in matches.groupby("era", observed=True):
        seasons_in_era = games.season_id.unique()
        row = dict(era=era, **_match_rates(games), n_seasons=len(seasons_in_era))
        row["n_matches"] = row.pop("n")
        row["std_ppm"] = balance.loc[seasons_in_era, "std_ppm"].mean()
        row["title_margin_ppm"] = balance.loc[seasons_in_era, "title_margin_ppm"].mean()
        row["goals_per_match_ci_lower"], row["goals_per_match_ci_upper"] = _goals_interval(
            games, n_bootstrap
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("era", drop=True).sort_index()


def calendar_effects(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Describe calendar groups in Israel local time; preserve missing dates as unknown groups."""
    games = _subset(tables.match_table, _selection(seasons))
    _bootstrap_count(n_bootstrap)
    dates = pd.to_datetime(games.date, utc=True).dt.tz_convert("Asia/Jerusalem")
    games["weekday"] = dates.dt.dayofweek.map({5: "Saturday", 6: "Sunday", 0: "Monday"}).fillna(
        "other"
    )
    games.loc[dates.isna(), "weekday"] = "unknown"
    hours = dates.dt.hour
    games["kickoff_hour"] = np.select(
        [hours.lt(18), hours.lt(20), hours.ge(20)],
        ["before_18", "18_to_20", "20_plus"],
        default="unknown",
    )
    games["month"] = dates.dt.month.astype("Int64").astype("string").fillna("unknown")
    rows = []
    for dimension in ("weekday", "kickoff_hour", "month"):
        for level, group in games.groupby(dimension, observed=True):
            row = dict(
                dimension=dimension,
                level=level,
                **_match_rates(group),
                n_seasons=group.season_id.nunique(),
            )
            row["goals_per_match_ci_lower"], row["goals_per_match_ci_upper"] = _goals_interval(
                group, n_bootstrap
            )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["dimension", "level"], drop=True).sort_index()


def postponed_summary(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Compare observed postponed matches with all others; flags are limited to the same stage."""
    games = _subset(tables.match_table, _selection(seasons))
    _bootstrap_count(n_bootstrap)
    rows = []
    for postponed, group in games.groupby("is_postponed", observed=True):
        row = dict(
            is_postponed=postponed, **_match_rates(group), n_seasons=group.season_id.nunique()
        )
        row["goals_per_match_ci_lower"], row["goals_per_match_ci_upper"] = _goals_interval(
            group, n_bootstrap
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("is_postponed", drop=True).sort_index()


def _selection(seasons: Iterable[int]) -> list[int]:
    selected = list(seasons)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(DEVELOPMENT_SEASONS)
    ):
        raise ValueError("League statistics require unique development seasons 8–25")
    return sorted(selected)


def _subset(frame: pd.DataFrame, selected: list[int]) -> pd.DataFrame:
    if "season_id" in frame.index.names:
        return frame.loc[frame.index.get_level_values("season_id").isin(selected)].copy()
    return frame.loc[frame.season_id.isin(selected)].copy()


def rate_interval(count: int, n: int) -> tuple[float, float, float]:
    """Return an event rate and marginal 95% Wilson bounds; no trials means unknown."""
    if n < 0 or count < 0 or count > n:
        raise ValueError("Expected 0 <= count <= n")
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    low, high = proportion_confint(count, n, alpha=0.05, method="wilson")
    return count / n, float(low), float(high)


def _bootstrap_count(n_bootstrap: int) -> None:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")


def _mean_interval(
    values: pd.Series, n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    _bootstrap_count(n_bootstrap)
    data = values.dropna().to_numpy(dtype=float)
    if len(data) < 2:
        return float("nan"), float("nan")
    # Fixed seed; resampling season observations, not treating all matches as independent.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(data, size=(n_bootstrap, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _games(tables: AnalysisTables, selected: list[int]) -> pd.DataFrame:
    games = _subset(tables.team_matches, selected).reset_index()
    for result in ("win", "draw", "loss"):
        games[result] = games.result.eq(result)
    games["home_win"] = games.win.where(games.is_home)
    games["away_win"] = games.win.where(~games.is_home)
    games["big_win"] = games.goal_difference.ge(3)
    games["narrow_win"] = games.goal_difference.eq(1)
    games["heavy_loss"] = games.goal_difference.le(-3)
    games["btts"] = games.goals_scored.gt(0) & games.goals_conceded.gt(0)
    games["over_2_5"] = (games.goals_scored + games.goals_conceded).ge(3)
    return games


def team_season_stats(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Describe each sporting team-season, retaining official positions as context only."""
    selected = _selection(seasons)
    frame = _subset(tables.team_seasons, selected)[SEASON_COLUMNS].copy()
    formats = _subset(tables.season_summary, selected)[["n_teams"]]
    frame = frame.join(formats, on="season_id", validate="many_to_one")
    games = _games(tables, selected)
    eras = games.groupby("season_id", observed=True).era.first()
    frame["era"] = frame.index.get_level_values("season_id").map(eras)
    rows: list[dict[str, Any]] = []
    for (season, team), group in games.groupby(["season_id", "team_id"], observed=True):
        row: dict[str, Any] = {"season_id": season, "team_id": team}
        for rate, flag in RATE_FLAGS.items():
            values = group[flag].dropna()
            n, count = len(values), int(values.sum())
            value, low, high = rate_interval(count, n)
            row.update(
                {rate: value, f"{rate}_n": n, f"{rate}_ci_lower": low, f"{rate}_ci_upper": high}
            )
        rows.append(row)
    rates = pd.DataFrame(rows).set_index(["season_id", "team_id"], drop=True)
    frame = frame.drop(columns="draw_rate").join(rates, validate="one_to_one")
    frame["goal_difference_per_match"] = (
        frame.goals_scored_per_match - frame.goals_conceded_per_match
    )
    frame["home_away_ppm_gap"] = frame.home_points_per_match - frame.away_points_per_match
    return frame.sort_index()


def position_profile(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Compare official finishing places within 12/16/14-team eras using sporting rates."""
    source = team_season_stats(tables, seasons).reset_index()
    rows = []
    metrics = ["points_per_match", "goals_scored_per_match", "goals_conceded_per_match"]
    for (era, position), group in source.groupby(["era", "final_position"], observed=True):
        row = {"era": era, "final_position": position, "n": len(group)}
        for metric in metrics:
            values = group[metric]
            for operation in ("mean", "median", "min", "max"):
                row[f"{metric}_{operation}"] = getattr(values, operation)()
            row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = _mean_interval(
                values, n_bootstrap
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index(["era", "final_position"], drop=True).sort_index()


def season_persistence(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Pair requested seasons with development t-1; returning clubs do not bridge gaps."""
    selected = _selection(seasons)
    context = sorted(set(selected) | {s - 1 for s in selected if s - 1 in DEVELOPMENT_SEASONS})
    frame = team_season_stats(tables, context).reset_index()
    current = frame[["season_id", "team_id", "team_name", "points_per_match", "final_position"]]
    current = current.rename(columns={"points_per_match": "ppm_t", "final_position": "position_t"})
    current = current.loc[current.season_id.isin(selected)]
    previous = frame[
        ["season_id", "team_id", "points_per_match", "final_position", "value_share_of_league"]
    ].copy()
    previous["season_id"] += 1
    previous = previous.rename(
        columns={
            "points_per_match": "ppm_prev",
            "final_position": "position_prev",
            "value_share_of_league": "value_share_prev",
        }
    )
    return current.merge(
        previous, on=["season_id", "team_id"], how="inner", validate="one_to_one"
    ).set_index(["season_id", "team_id"], drop=True)


def _relationship(
    frame: pd.DataFrame,
    x: str,
    y: str,
    intervals: bool = False,
    n_bootstrap: int = RELATIONSHIP_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    complete = frame[[x, y]].dropna().astype(float)
    row: dict[str, Any] = {
        "n": len(complete),
        "pearson": np.nan,
        "spearman": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
    }
    if len(complete) >= 2 and complete[x].nunique() > 1:
        slope = float(
            ((complete[x] - complete[x].mean()) * (complete[y] - complete[y].mean())).sum()
            / ((complete[x] - complete[x].mean()) ** 2).sum()
        )
        row.update(slope=slope, intercept=float(complete[y].mean() - slope * complete[x].mean()))
        if complete[y].nunique() > 1:
            row.update(
                pearson=complete[x].corr(complete[y]),
                spearman=complete[x].corr(complete[y], method="spearman"),
            )
    if intervals:
        _bootstrap_count(n_bootstrap)
        blocks = list(complete.groupby(level="season_id").indices.values())
        row["n_seasons"] = len(blocks)
        draws: dict[str, list[float]] = {metric: [] for metric in ("pearson", "spearman", "slope")}
        if len(blocks) >= 2 and np.isfinite(row["slope"]):
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            for _ in range(n_bootstrap):
                indices = np.concatenate(
                    [blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]
                )
                sample = _relationship(complete.iloc[indices], x, y)
                for metric in draws:
                    if np.isfinite(sample[metric]):
                        draws[metric].append(sample[metric])
        for metric, samples in draws.items():
            bounds = np.quantile(samples, [0.025, 0.975]) if samples else [np.nan, np.nan]
            row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = bounds
    return row


def persistence_summary(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = RELATIONSHIP_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Summarize sporting persistence; a slope alone does not prove perfect prediction."""
    pairs = season_persistence(tables, seasons)
    rows = [
        {
            "relationship": "ppm",
            **_relationship(pairs, "ppm_prev", "ppm_t", intervals=True, n_bootstrap=n_bootstrap),
        },
        {
            "relationship": "position",
            **_relationship(
                pairs, "position_prev", "position_t", intervals=True, n_bootstrap=n_bootstrap
            ),
        },
    ]
    return pd.DataFrame(rows).set_index("relationship", drop=True)


def value_vs_performance(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Describe value/points associations and in-sample residuals, never forecast accuracy."""
    frame = team_season_stats(tables, seasons)[
        [
            "team_name",
            "prev_season_value_share",
            "value_share_of_league",
            "points_per_match",
            "final_position",
            "squad_size",
        ]
    ].copy()
    relation = _relationship(frame, "prev_season_value_share", "points_per_match")
    frame["expected_ppm"] = (
        relation["intercept"] + relation["slope"] * frame.prev_season_value_share
    )
    frame["over_under"] = frame.points_per_match - frame.expected_ppm
    return frame


def value_summary(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = RELATIONSHIP_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Report complete-pair counts for prior/current value and squad-size associations."""
    frame = value_vs_performance(tables, seasons)
    rows = [
        {
            "relationship": name,
            **_relationship(
                frame, name, "points_per_match", intervals=True, n_bootstrap=n_bootstrap
            ),
        }
        for name in ("prev_season_value_share", "value_share_of_league", "squad_size")
    ]
    return pd.DataFrame(rows).set_index("relationship", drop=True)


def promoted_value_check(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Compare actual t value with t-1 bottom-half median; never use the entrant in its reference.

    Only available shares enter a median; odd reference counts use floor(n/2).
    The same-t median excluding promoted clubs is a secondary descriptive comparison.
    """
    selected = _selection(seasons)
    context = sorted(set(selected) | {s - 1 for s in selected if s - 1 in DEVELOPMENT_SEASONS})
    frame = _subset(tables.team_seasons, context)
    benchmarks, same_season = {}, {}
    for season, group in frame.groupby(level="season_id"):
        shares = group.value_share_of_league.dropna().sort_values()
        benchmarks[season] = shares.iloc[: len(shares) // 2].median()
        incumbent = (
            group.loc[~group.promoted_in.fillna(False)].value_share_of_league.dropna().sort_values()
        )
        same_season[season] = incumbent.iloc[: len(incumbent) // 2].median()
    frame = _subset(frame, selected)
    frame = frame.loc[frame.promoted_in.eq(True)].copy()
    frame = frame.loc[frame.index.get_level_values("season_id") > 8]
    frame = frame[["team_name", "value_share_of_league"]]
    frame["imputed_share"] = (frame.index.get_level_values("season_id") - 1).map(benchmarks)
    frame["same_season_share_excl_promoted"] = frame.index.get_level_values("season_id").map(
        same_season
    )
    frame["diff"] = frame.value_share_of_league - frame.imputed_share
    frame["ratio"] = frame.value_share_of_league / frame.imputed_share.where(
        frame.imputed_share.gt(0)
    )
    return frame


def promoted_value_summary(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Summarize complete promoted-club comparisons, explicitly counting missing values."""
    frame = promoted_value_check(tables, seasons)
    complete = frame.dropna(subset=["value_share_of_league", "imputed_share"])
    rate, low, high = rate_interval(int(complete["diff"].lt(0).sum()), len(complete))
    return pd.DataFrame(
        [
            {
                "scope": "promoted",
                "n": len(complete),
                "n_missing": len(frame) - len(complete),
                "median_diff": complete["diff"].median(),
                "fraction_below": rate,
                "ci_lower": low,
                "ci_upper": high,
            }
        ]
    ).set_index("scope", drop=True)


def team_consistency(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Describe equal-weight season consistency and observed promotions, not latent ability."""
    frame = team_season_stats(tables, seasons)
    rows = []
    for team, group in frame.groupby(level="team_id"):
        low, high = _mean_interval(group.points_per_match, n_bootstrap)
        rows.append(
            {
                "team_id": team,
                "team_name": group.team_name.iloc[-1],
                "seasons": len(group),
                "mean_ppm": group.points_per_match.mean(),
                "std_ppm": group.points_per_match.std(ddof=1) if len(group) >= 3 else np.nan,
                "mean_ppm_ci_lower": low,
                "mean_ppm_ci_upper": high,
                "best_final_position": group.final_position.min(),
                "worst_final_position": group.final_position.max(),
                "titles": int(group.final_position.eq(1).sum()),
                "top3_seasons": int(group.final_position.le(3).sum()),
                "relegations": int(group.relegated.sum()),
                "promotions": int(group.promoted_in.sum()),
                "is_yoyo": bool(group.promoted_in.sum() >= 2),
            }
        )
    return pd.DataFrame(rows).set_index("team_id", drop=True).sort_index()


def team_style_profile(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Equal-season BTTS/over-2.5 averages for clubs with at least five observed seasons."""
    frame = team_season_stats(tables, seasons)
    rows = []
    for team, group in frame.groupby(level="team_id"):
        if len(group) < 5:
            continue
        row = {
            "team_id": team,
            "team_name": group.team_name.iloc[-1],
            "n_seasons": len(group),
            "n_matches": int(group.matches_played.sum()),
        }
        for metric in ("btts_rate", "over_2_5_rate"):
            row[metric] = group[metric].mean()
            row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = _mean_interval(
                group[metric], n_bootstrap
            )
        rows.append(row)
    columns = [
        "team_id",
        "team_name",
        "n_seasons",
        "n_matches",
        "btts_rate",
        "btts_rate_ci_lower",
        "btts_rate_ci_upper",
        "over_2_5_rate",
        "over_2_5_rate_ci_lower",
        "over_2_5_rate_ci_upper",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("team_id", drop=True)
