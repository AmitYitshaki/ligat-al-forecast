"""Chapter C retrospective diagnostics; sporting snapshots against official outcomes."""

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from statsmodels.stats.contingency_tables import StratifiedTable  # type: ignore[import-untyped]

from ligat.analysis.prepare import AnalysisTables

from .chapter_a import ERA14_SEASONS
from .core import (
    MEAN_BOOTSTRAP_DRAWS,
    _match_rates,
    _mean_interval,
    _selection,
    _subset,
    rate_interval,
)
from .season_course import (
    playoff_deviations,
    split_movement,
    standings_by_match_number,
    table_settling,
)

REGIONS = ((1, 3, "top"), (4, 6, "upper_line"), (7, 10, "middle_table"), (11, 14, "bottom"))


def playoff_noise_diagnostics(
    tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS
) -> pd.DataFrame:
    """Signal spread and IID points noise, without correcting for playoff selection."""
    selected = _selection(seasons)
    deviations = playoff_deviations(tables, selected)
    games = _subset(tables.team_matches, selected)
    regular = games.loc[games.stage.eq("regular")]
    variance = regular.groupby("season_id").points_earned.var()
    rows = []
    for stage, group in deviations.groupby("final_stage", observed=True):
        if stage not in {"upper", "lower"}:
            continue
        noise_variance = (
            group.index.get_level_values("season_id").map(variance).to_numpy()
            / group.n_regular.to_numpy()
        )
        rows.append(
            dict(
                final_stage=stage,
                regular_sd=group.groupby(level="season_id").regular_deviation.std().mean(),
                noise_sd=float(np.sqrt(np.mean(noise_variance))),
                n=len(group),
            )
        )
    return pd.DataFrame(rows).set_index("final_stage")


def random_mean_movement(n: int) -> float:
    """Mean absolute displacement under a uniformly random permutation of n clubs."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Group size must be a positive integer")
    return float((n**2 - 1) / (3 * n))


def position_credits(frame: pd.DataFrame, rank: str = "rank") -> pd.DataFrame:
    """Expand tied average ranks uniformly over ALL occupied slots, preserving unit mass.

    Input contains season_id, k and one row per club. A three-way tie at rank 3
    occupies slots 2/3/4, each with weight 1/3 per club (not just floor/ceil).
    """
    _selection(frame.season_id.unique())
    rows = []
    for _, group in frame.groupby(["season_id", "k", rank], observed=True, sort=False):
        size = len(group)
        start = float(group[rank].iloc[0]) - (size - 1) / 2
        if not np.isclose(start, round(start)):
            raise ValueError("Rank is inconsistent with the size of its tied group")
        for slot in range(int(round(start)), int(round(start)) + size):
            copy = group.copy()
            copy["position"], copy["weight"] = slot, 1 / size
            rows.append(copy)
    return pd.concat(rows, ignore_index=True)


def rank_snapshots(tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS) -> pd.DataFrame:
    """Join sporting regular-stage ranks to official finish labels; not a live as_of view."""
    selected = _selection(seasons)
    snapshots = standings_by_match_number(tables, selected).reset_index()
    labels = _subset(tables.team_seasons, selected)
    return snapshots.join(
        labels[["team_name", "final_position", "final_stage", "relegated"]],
        on=["season_id", "team_id"],
        validate="many_to_one",
    )


def rank_heatmap(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
    *,
    k: int = 10,
    at_split: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """14 x 14 counts with fractional ties; every position has one unit per season."""
    selected = _selection(seasons)
    frame = rank_snapshots(tables, selected)
    if frame.groupby("season_id").team_id.nunique().ne(14).any():
        raise ValueError("This heatmap compares 14-club seasons only")
    target = frame.groupby("season_id").k.transform("max") if at_split else k
    frame = frame.loc[frame.k.eq(target)].copy()
    if frame.empty or frame.season_id.nunique() != len(selected):
        raise ValueError("Checkpoint must exist in every selected season")
    expanded = position_credits(frame)
    matrix = expanded.pivot_table(
        index="position", columns="final_position", values="weight", aggfunc="sum", fill_value=0
    ).reindex(index=range(1, 15), columns=range(1, 15), fill_value=0)
    return matrix.astype(float), expanded


def region_persistence(
    tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS
) -> pd.DataFrame:
    """Fraction of clubs finishing in the same rank region, conditional on current slots."""
    selected = _selection(seasons)
    snapshots = rank_snapshots(tables, selected)
    if snapshots.groupby("season_id").team_id.nunique().ne(14).any():
        raise ValueError("Regions use the 14-club league boundaries")
    frame = position_credits(snapshots.loc[snapshots.k.le(26)])
    rows = []
    for (k, season), group in frame.groupby(["k", "season_id"], observed=True):
        for low, high, name in REGIONS:
            clubs = group.loc[group.position.between(low, high)]
            denom = clubs.weight.sum()
            rows.append(
                dict(
                    k=k,
                    season_id=season,
                    region=name,
                    n=denom,
                    retained=float(
                        clubs.loc[clubs.final_position.between(low, high), "weight"].sum()
                    ),
                    movement=float(
                        (clubs.weight * (clubs["rank"] - clubs.final_position).abs()).sum()
                    ),
                )
            )
    data = pd.DataFrame(rows)
    result = data.groupby(["region", "k"], observed=True).agg(
        n=("n", "sum"),
        retained=("retained", "sum"),
        movement=("movement", "sum"),
        n_seasons=("season_id", "nunique"),
    )
    result["same_region_rate"] = result.retained / result.n
    result["mean_movement"] = result.movement / result.n
    return result


def settling_review(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
) -> pd.DataFrame:
    """Existing benchmark curve, with no season-based intervals below five seasons."""
    selected = _selection(seasons)
    return table_settling(tables, selected, upper_sizes=upper_sizes, min_bootstrap_seasons=5)


def leader_benchmarks(
    tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS
) -> pd.DataFrame:
    """Equal season credit for a tied leader/bottom; chance lines assume 1/14 and 2/14."""
    selected = _selection(seasons)
    frame = rank_snapshots(tables, selected)
    rows = []
    for k in (10, 20):
        snapshot = frame.loc[frame.k.eq(k)]
        for name, highest, label, chance in (
            ("leader", True, "final_position", 1 / 14),
            ("bottom", False, "relegated", 2 / 14),
        ):
            credits = []
            for _, group in snapshot.groupby("season_id", observed=True):
                extreme = group["rank"].min() if highest else group["rank"].max()
                clubs = group.loc[group["rank"].eq(extreme)]
                credits.append(float(clubs[label].eq(1).mean()))
            rows.append(
                dict(
                    benchmark=name,
                    k=k,
                    n_seasons=len(credits),
                    success=sum(credits),
                    rate=np.mean(credits),
                    chance_rate=chance,
                )
            )
    return pd.DataFrame(rows).set_index(["benchmark", "k"])


def split_diagnostics(
    tables: AnalysisTables,
    seasons: Iterable[int] = ERA14_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Movement and random-order benchmark within each playoff; season facts separately."""
    selected = _selection(seasons)
    movement = split_movement(tables, selected, upper_sizes=upper_sizes).reset_index()
    movement = movement.loc[movement.group.isin(["upper", "lower"])]
    rows = []
    for name, group in movement.groupby("group", observed=True):
        counts = group.groupby("season_id").size()
        correlations = [
            g.rank_at_split.corr(g.final_position, method="spearman")
            for _, g in group.groupby("season_id", observed=True)
        ]
        rows.append(
            dict(
                group=name,
                n=len(group),
                unchanged_rate=group.move.eq(0).mean(),
                mean_movement=group.move.abs().mean(),
                random_movement=np.mean([random_mean_movement(int(n)) for n in counts]),
                spearman_mean=np.mean(correlations),
            )
        )
    snapshots = rank_snapshots(tables, selected)
    final = snapshots.loc[snapshots.k.eq(snapshots.groupby("season_id").k.transform("max"))]
    expanded = position_credits(final)
    facts = []
    for label, low, high, event in (
        ("leader_title", 1, 1, "title"),
        ("last_two_relegated", 13, 14, "relegated"),
        ("places_11_12_relegated", 11, 12, "relegated"),
    ):
        clubs = expanded.loc[expanded.position.between(low, high)]
        flag = clubs.final_position.eq(1) if event == "title" else clubs.relegated
        facts.append(
            dict(
                benchmark=label,
                success=float(clubs.loc[flag, "weight"].sum()),
                n=float(clubs.weight.sum()),
            )
        )
    facts_frame = pd.DataFrame(facts).set_index("benchmark")
    facts_frame["rate"] = facts_frame.success / facts_frame.n
    return pd.DataFrame(rows).set_index("group"), facts_frame


def regular_strength_expectation(
    tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS
) -> pd.DataFrame:
    """Goals expected from regular-stage GF/GA; regular calibration is in-sample."""
    selected = _selection(seasons)
    matches = _subset(tables.match_table, selected)
    team_games = _subset(tables.team_matches, selected).reset_index()
    regular = team_games.loc[team_games.stage.eq("regular")]
    rates = regular.groupby(["season_id", "team_id"], observed=True).agg(
        gf=("goals_scored", "mean"), ga=("goals_conceded", "mean")
    )
    league = regular.groupby("season_id").goals_scored.mean()
    if league.le(0).any():
        raise ValueError("Regular scoring mean must be positive")
    games = matches.copy()
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([games.season_id, games[f"{side}_team_id"]])
        for metric in ("gf", "ga"):
            games[f"{side}_{metric}"] = rates[metric].reindex(keys).to_numpy()
    games["expected_goals"] = (
        games.home_gf * games.away_ga + games.away_gf * games.home_ga
    ) / games.season_id.map(league)
    return games


def stage_strength_comparison(
    tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed rates and conditional scoring expectation; paired season effects with SE."""
    selected = _selection(seasons)
    games = regular_strength_expectation(tables, selected)
    late = games.stage.eq("regular") & games["round"].between(20, 26)
    groups = [(name, games.loc[games.stage.eq(name)]) for name in ("regular", "upper", "lower")]
    groups.insert(1, ("regular_late", games.loc[late]))
    rows = []
    for name, group in groups:
        row = dict(
            stage=name,
            **_match_rates(group),
            expected_goals=group.expected_goals.mean(),
            n_seasons=group.season_id.nunique(),
        )
        row["n_matches"] = row.pop("n")
        # Exactly one value per season: whole-season resampling of the mean.
        values = group.groupby("season_id").total_goals.mean()
        low, high = np.nan, np.nan
        if len(values) >= 5:
            low, high = _mean_interval(values, MEAN_BOOTSTRAP_DRAWS)
        row.update(goals_per_match_ci_lower=low, goals_per_match_ci_upper=high)
        row["btts_rate"] = ((group.home_goals > 0) & (group.away_goals > 0)).mean()
        # Wilson for the fourth C4 panel uses the existing shared match summary.
        _, lo, hi = rate_interval(
            int(((group.home_goals > 0) & (group.away_goals > 0)).sum()), len(group)
        )
        row.update(btts_rate_ci_lower=lo, btts_rate_ci_upper=hi)
        rows.append(row)
    by_season = games.groupby(["season_id", "stage"], observed=True).agg(
        observed=("total_goals", "mean"), expected=("expected_goals", "mean")
    )
    effects = []
    for name in ("upper", "lower"):
        data = by_season.xs(name, level="stage")
        for contrast, values in (
            ("excess_strength", data.observed - data.expected),
            ("minus_regular", data.observed - by_season.xs("regular", level="stage").observed),
        ):
            effects.append(
                dict(
                    stage=name,
                    contrast=contrast,
                    mean=values.mean(),
                    se=values.sem(),
                    positive=int(values.gt(0).sum()),
                    n_seasons=len(values),
                )
            )
    early = (
        games.loc[games.stage.eq("regular") & games["round"].le(19)]
        .groupby("season_id")
        .total_goals.mean()
    )
    late_means = games.loc[late].groupby("season_id").total_goals.mean()
    delta = late_means - early
    effects.append(
        dict(
            stage="regular_late",
            contrast="minus_early",
            mean=delta.mean(),
            se=delta.sem(),
            positive=int(delta.gt(0).sum()),
            n_seasons=len(delta),
        )
    )
    return pd.DataFrame(rows).set_index("stage"), pd.DataFrame(effects).set_index(
        ["stage", "contrast"]
    )


def draw_strata(tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS) -> pd.DataFrame:
    """2x2 cells: rows last-three/rest, columns draw/not-draw, strata season x stage."""
    selected = _selection(seasons)
    games = _subset(tables.match_table, selected)
    games = games.loc[games.stage.isin(["regular", "upper", "lower"])].copy()
    end = games.groupby(["season_id", "stage"], observed=True)["round"].transform("max")
    games["last"] = games["round"].gt(end - 3)
    rows = []
    for (season, stage), group in games.groupby(["season_id", "stage"], observed=True):
        last = group.loc[group["last"]]
        rest = group.loc[~group["last"]]
        if not len(last) or not len(rest):
            continue
        a = int(last.result.eq("draw").sum())
        c = int(rest.result.eq("draw").sum())
        rows.append(dict(season_id=season, stage=stage, a=a, b=len(last) - a, c=c, d=len(rest) - c))
    return pd.DataFrame(rows).set_index(["season_id", "stage"])


def mantel_haenszel(cells: pd.DataFrame) -> dict[str, float]:
    """Common draw odds ratio without continuity correction; exact season x stage strata."""
    _selection(cells.index.get_level_values("season_id").unique())
    data = cells[["a", "b", "c", "d"]].to_numpy(dtype=float).reshape(-1, 2, 2).transpose(1, 2, 0)
    model = StratifiedTable(data)
    lower, upper = model.oddsratio_pooled_confint()
    p = float(model.test_null_odds(correction=False).pvalue)
    last_n = float((cells.a + cells.b).sum())
    rest_n = float((cells.c + cells.d).sum())
    return dict(
        odds_ratio=float(model.oddsratio_pooled),
        ci_lower=float(lower),
        ci_upper=float(upper),
        p=p,
        p_bonferroni_12=min(1, 12 * p),
        last_n=last_n,
        rest_n=rest_n,
        last_draw_rate=float(cells.a.sum() / last_n),
        rest_draw_rate=float(cells.c.sum() / rest_n),
        n_strata=float(len(cells)),
    )


def combined_draw_test(
    tables: AnalysisTables, seasons: Iterable[int] = ERA14_SEASONS
) -> pd.DataFrame:
    """Prespecified display of pooled and per-stage MH; choice of this test was post hoc."""
    selected = _selection(seasons)
    cells = draw_strata(tables, selected)
    rows = [dict(stage="all", **mantel_haenszel(cells))]
    for stage in ("regular", "upper", "lower"):
        part = cells.loc[cells.index.get_level_values("stage") == stage]
        if len(part):
            rows.append(dict(stage=stage, **mantel_haenszel(part)))
    result = pd.DataFrame(rows).set_index("stage")
    by_season = cells.groupby("season_id").sum()
    last = by_season.a / (by_season.a + by_season.b)
    rest = by_season.c / (by_season.c + by_season.d)
    result.attrs["same_direction_seasons"] = int(last.lt(rest).sum())
    result.attrs["n_seasons"] = len(by_season)
    return result
