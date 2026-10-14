"""Chapter C: descriptive standings by games played, not prematch features."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables

from .core import (
    MEAN_BOOTSTRAP_DRAWS,
    _bootstrap_count,
    _goals_interval,
    _match_rates,
    _mean_interval,
    _selection,
    _subset,
    rate_interval,
)

STAGES = ("regular", "upper", "middle", "lower")


def _rank_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["points", "goal_difference", "goals_scored"], ascending=False)
    frame["rank"] = np.arange(1, len(frame) + 1, dtype=float)
    frame["rank"] = frame.groupby(["points", "goal_difference", "goals_scored"], observed=True)[
        "rank"
    ].transform("mean")
    return frame


def standings_by_match_number(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    stage: str = "regular",
) -> pd.DataFrame:
    """Compare each team's first k games; carry forward short teams without inventing points.

    stage='all' is the full sporting season, without halving. Equal points/difference/goals
    share an average rank. Other teams' kth games need not have happened on the same date.
    """
    selected = _selection(seasons)
    if stage not in (*STAGES, "all"):
        raise ValueError(f"Unknown stage: {stage}")
    games = _subset(tables.team_matches, selected).reset_index()
    if stage != "all":
        games = games.loc[games.stage.eq(stage)].copy()
    games = games.sort_values(["season_id", "team_id", "date", "game_id"])
    grouped = games.groupby(["season_id", "team_id"], observed=True)
    games["k"] = grouped.cumcount() + 1
    games["points"] = grouped.points_earned.cumsum()
    games["goals_scored"] = grouped.goals_scored.cumsum()
    games["goal_difference"] = grouped.goal_difference.cumsum()
    rows = []
    for season, group in games.groupby("season_id", observed=True):
        teams = sorted(group.team_id.unique())
        maximum = int(group.k.max())
        for k in range(1, maximum + 1):
            snapshot = group.loc[group.k.le(k)].groupby("team_id", observed=True).tail(1)
            snapshot = snapshot.set_index("team_id", drop=True).reindex(teams)
            snapshot = snapshot[["k", "points", "goal_difference", "goals_scored"]].rename(
                columns={"k": "games_played"}
            )
            snapshot["games_short"] = snapshot.games_played.lt(k)
            snapshot = _rank_table(snapshot)
            snapshot["season_id"], snapshot["k"] = season, k
            rows.append(snapshot.reset_index())
    columns = [
        "season_id",
        "k",
        "team_id",
        "points",
        "goal_difference",
        "goals_scored",
        "games_played",
        "games_short",
        "rank",
    ]
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    for column in ("season_id", "k", "team_id"):
        frame[column] = frame[column].astype("Int64")
    return frame.set_index(["season_id", "k", "team_id"], drop=True).sort_index()


def split_movement(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
) -> pd.DataFrame:
    """Compare regular-stage sporting rank with official finishing place after splitting."""
    selected = _selection(seasons)
    observed = set(_subset(tables.team_seasons, selected).index.get_level_values("season_id"))
    if not observed <= set(upper_sizes):
        raise ValueError("Missing playoff configuration")
    standings = standings_by_match_number(tables, selected).reset_index()
    last = standings.groupby("season_id", observed=True).k.transform("max")
    frame = standings.loc[standings.k.eq(last) & standings.season_id.map(upper_sizes).notna()]
    frame = frame.set_index(["season_id", "team_id"], drop=True)
    frame = frame[["rank", "games_short"]].rename(columns={"rank": "rank_at_split"})
    frame = frame.join(
        _subset(tables.team_seasons, selected)[["final_position", "final_stage", "team_name"]],
        validate="one_to_one",
    )
    frame = frame.rename(columns={"final_stage": "group"})
    frame["move"] = frame.rank_at_split - frame.final_position
    frame["points_halved"] = frame.index.get_level_values("season_id").isin([11, 12])
    return frame.sort_index()


def _unique_extreme(group: pd.DataFrame, highest: bool) -> pd.DataFrame:
    target = group["rank"].min() if highest else group["rank"].max()
    candidates = group.loc[group["rank"].eq(target)]
    return candidates if len(candidates) == 1 else candidates.iloc[:0]


def table_settling(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Measure association with the official finish; unique leaders/bottoms only for rates.

    Full ties are excluded from the relevant rate, with explicit denominators and tie counts.
    Upper-line cohorts spanning a full tie at the cutoff are likewise excluded, not broken.
    """
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    observed = set(_subset(tables.team_seasons, selected).index.get_level_values("season_id"))
    if not observed <= set(upper_sizes):
        raise ValueError("Missing playoff configuration")
    standings = standings_by_match_number(tables, selected).reset_index()
    labels = _subset(tables.team_seasons, selected)[["final_position", "relegated", "final_stage"]]
    standings = standings.join(labels, on=["season_id", "team_id"], validate="many_to_one")
    eras = _subset(tables.team_matches, selected).groupby("season_id", observed=True).era.first()
    observations = []
    for (season, k), group in standings.groupby(["season_id", "k"], observed=True):
        row: dict[str, Any] = dict(
            season_id=season,
            k=k,
            era=eras.loc[season],
            spearman=(
                group["rank"].corr(group.final_position, method="spearman")
                if group["rank"].nunique() > 1
                else float("nan")
            ),
            leader=pd.NA,
            bottom=pd.NA,
            upper_n=0,
            upper_success=0,
            upper_tied=False,
        )
        leader, bottom = _unique_extreme(group, True), _unique_extreme(group, False)
        if not leader.empty:
            row["leader"] = bool(leader.final_position.iloc[0] == 1)
        if not bottom.empty:
            row["bottom"] = bool(bottom.relegated.iloc[0])
        size = upper_sizes[int(season)]
        if size is not None:
            upper = group.loc[group["rank"].le(size)]
            if len(upper) == size:
                row["upper_n"] = len(upper)
                row["upper_success"] = int(upper.final_stage.eq("upper").sum())
            else:
                row["upper_tied"] = True
        observations.append(row)
    rows = []
    for (era, k), group in pd.DataFrame(observations).groupby(["era", "k"], observed=True):
        row = dict(
            era=era,
            k=k,
            n_seasons=len(group),
            spearman_mean=group.spearman.mean(),
            spearman_n=int(group.spearman.count()),
        )
        row["spearman_ci_lower"], row["spearman_ci_upper"] = _mean_interval(
            group.spearman, n_bootstrap
        )
        for column, label in (
            ("leader", "leader_won_title_rate"),
            ("bottom", "bottom_relegated_rate"),
        ):
            known = group[column].dropna()
            row[f"{label}_n"] = len(known)
            row[f"{label}_count"] = int(known.sum())
            row[f"{label}_tied"] = len(group) - len(known)
            row[label], row[f"{label}_ci_lower"], row[f"{label}_ci_upper"] = rate_interval(
                int(known.sum()), len(known)
            )
        label = "in_upper_line_stayed_rate"
        row[f"{label}_n"] = int(group.upper_n.sum())
        row[f"{label}_count"] = int(group.upper_success.sum())
        row[f"{label}_tied_seasons"] = int(group.upper_tied.sum())
        row[label], row[f"{label}_ci_lower"], row[f"{label}_ci_upper"] = rate_interval(
            int(group.upper_success.sum()), int(group.upper_n.sum())
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index(["era", "k"], drop=True).sort_index()


def split_summary(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
) -> pd.DataFrame:
    """Describe post-split movement and how often the unique regular-stage leader won."""
    selected = _selection(seasons)
    source = split_movement(tables, selected, upper_sizes=upper_sizes).reset_index()
    leaders = []
    n_tied = 0
    for _, group in source.groupby("season_id", observed=True):
        first = group.loc[group.rank_at_split.eq(group.rank_at_split.min())]
        if len(first) == 1:
            leaders.append(bool(first.final_position.iloc[0] == 1))
        else:
            n_tied += 1
    rate, lower, upper = rate_interval(sum(leaders), len(leaders))
    rows = []
    for name, group in source.groupby("group", observed=True):
        rows.append(
            dict(
                group=name,
                n=len(group),
                move_mean=group.move.mean(),
                move_median=group.move.median(),
                move_min=group.move.min(),
                move_max=group.move.max(),
                improved=int(group.move.gt(0).sum()),
                unchanged=int(group.move.eq(0).sum()),
                declined=int(group.move.lt(0).sum()),
                leader_at_split_won_title_rate=rate,
                leader_at_split_won_title_rate_ci_lower=lower,
                leader_at_split_won_title_rate_ci_upper=upper,
                leader_n=len(leaders),
                leader_won_count=sum(leaders),
                leader_tied_seasons=n_tied,
                n_seasons=source.season_id.nunique(),
                n_halved_seasons=source.loc[source.points_halved, "season_id"].nunique(),
            )
        )
    columns = [
        "group",
        "n",
        "move_mean",
        "move_median",
        "move_min",
        "move_max",
        "improved",
        "unchanged",
        "declined",
        "leader_at_split_won_title_rate",
        "leader_at_split_won_title_rate_ci_lower",
        "leader_at_split_won_title_rate_ci_upper",
        "leader_n",
        "leader_won_count",
        "leader_tied_seasons",
        "n_seasons",
        "n_halved_seasons",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("group", drop=True).sort_index()


def stage_comparison(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Describe stage outcomes, using the same full-game inclusion rules as chapters A/B."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    rows = []
    for stage, games in _subset(tables.match_table, selected).groupby("stage", observed=True):
        row = dict(stage=stage, **_match_rates(games), n_seasons=games.season_id.nunique())
        row["n_matches"] = row.pop("n")
        for rate, flag in (
            ("one_one_rate", games.home_goals.eq(1) & games.away_goals.eq(1)),
            ("btts_rate", games.home_goals.gt(0) & games.away_goals.gt(0)),
        ):
            row[rate], row[f"{rate}_ci_lower"], row[f"{rate}_ci_upper"] = rate_interval(
                int(flag.sum()), len(games)
            )
        row["goals_per_match_ci_lower"], row["goals_per_match_ci_upper"] = _goals_interval(
            games, n_bootstrap
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("stage", drop=True).sort_index()


def playoff_form(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
) -> pd.DataFrame:
    """Compare sporting points rates before/after splitting, without historical point halving."""
    games = _subset(tables.team_matches, _selection(seasons)).reset_index()
    games["period"] = np.where(games.stage.eq("regular"), "regular", "playoff")
    grouped = games.groupby(["season_id", "team_id", "period"], observed=True).agg(
        ppm=("points_earned", "mean"), n=("game_id", "size")
    )
    frame = grouped.unstack("period")
    frame.columns = [f"{metric}_{period}" for metric, period in frame.columns]
    frame = frame.reindex(columns=["ppm_regular", "ppm_playoff", "n_regular", "n_playoff"])
    frame = frame.loc[frame.n_playoff.notna()].copy()
    frame = frame.join(
        _subset(tables.team_seasons, _selection(seasons))[["team_name", "final_stage"]],
        validate="one_to_one",
    )
    frame["ppm_change"] = frame.ppm_playoff - frame.ppm_regular
    return frame.sort_index()


def _pregame_rates(tables: AnalysisTables, selected: list[int]) -> pd.DataFrame:
    # Select each side's own strictly prior games, then retrieve its cumulative kth totals.
    standings = standings_by_match_number(tables, selected, stage="all")
    games = _subset(tables.team_matches, selected).reset_index()
    games = games.sort_values(["season_id", "team_id", "date", "game_id"])
    rows = []
    for (season, team), group in games.groupby(["season_id", "team_id"], observed=True):
        dates = group.date.to_numpy()
        for game in group.itertuples():
            k = int(np.searchsorted(dates, game.date, side="left"))
            ppm: Any = pd.NA if k == 0 else standings.loc[(season, k, team), "points"] / k
            rows.append(dict(game_id=game.game_id, team_id=team, ppm_before=ppm, n_before=k))
    return pd.DataFrame(rows).set_index(["game_id", "team_id"], drop=True)


def season_points_race(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
) -> pd.DataFrame:
    """Present full-season sporting trajectories; regular_games locates the descriptive split."""
    selected = _selection(seasons)
    frame = standings_by_match_number(tables, selected, stage="all")
    names = _subset(tables.team_seasons, selected)[["team_name"]]
    frame = frame.join(names, on=["season_id", "team_id"], validate="many_to_one")
    games = _subset(tables.team_matches, selected).reset_index()
    counts = games.loc[games.stage.eq("regular")].groupby(["season_id", "team_id"]).size()
    return frame.join(
        counts.rename("regular_games"), on=["season_id", "team_id"], validate="many_to_one"
    ).sort_index()


def final_rounds_check(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    last_n: int = 3,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Compare last numbered rounds within each stage; upset needs both strictly prior team rates.

    Upset denominator excludes equal ppm and missing history. Draws with unequal known
    ppm remain eligible non-upsets. This is descriptive, not mathematical motivation.
    """
    selected = _selection(seasons)
    if last_n < 1:
        raise ValueError("last_n must be positive")
    _bootstrap_count(n_bootstrap)
    games = _subset(tables.match_table, selected)
    last_round = games.groupby(["season_id", "stage"], observed=True)["round"].transform("max")
    games["period"] = np.where(games["round"].gt(last_round - last_n), "last", "rest")
    history = _pregame_rates(tables, selected)
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([games.index, games[f"{side}_team_id"]])
        games[f"{side}_ppm_before"] = pd.array(history.ppm_before.reindex(keys), dtype="Float64")
    home, away = games.home_ppm_before, games.away_ppm_before
    eligible = (home.notna() & away.notna() & home.ne(away)).fillna(False)
    games["upset_eligible"] = eligible
    games["upset"] = (
        (home.lt(away) & games.result.eq("home_win"))
        | (away.lt(home) & games.result.eq("away_win"))
    ) & eligible
    rows = []
    for (stage, period), group in games.groupby(["stage", "period"], observed=True):
        row = dict(stage=stage, period=period, **_match_rates(group))
        row["upset_n"] = int(group.upset_eligible.sum())
        row["upset_count"] = int(group.upset.sum())
        row["upset_rate"], row["upset_rate_ci_lower"], row["upset_rate_ci_upper"] = rate_interval(
            row["upset_count"], row["upset_n"]
        )
        row["goals_per_match_ci_lower"], row["goals_per_match_ci_upper"] = _goals_interval(
            group, n_bootstrap
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index(["stage", "period"], drop=True).sort_index()
