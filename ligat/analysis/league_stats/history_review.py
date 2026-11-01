"""Development-only chapter B diagnostics and explicitly defined uncertainty."""

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables

from .core import (
    _match_rates,
    _selection,
    _subset,
    competitive_balance,
    home_advantage_by_season,
    season_thresholds,
)


def threshold_diagnostics(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    upper_sizes: Mapping[int, int | None],
) -> pd.DataFrame:
    """Mean, sample SD and ordinary linear time-trend p in the 14-club era."""
    frame = season_thresholds(tables, _selection(seasons), upper_sizes=upper_sizes)
    frame = frame.loc[frame.era.eq("14_teams")]
    rows = []
    for key in ("champion", "upper_line", "safety", "relegated_max"):
        data = frame[f"{key}_ppm"].dropna()
        fit = (
            stats.linregress(data.index.to_numpy(dtype=float), data.to_numpy(dtype=float))
            if len(data) > 2
            else None
        )
        rows.append(
            dict(
                objective=key,
                n=len(data),
                mean=data.mean(),
                std=data.std(),
                slope=fit.slope if fit else np.nan,
                p=fit.pvalue if fit else np.nan,
            )
        )
    return pd.DataFrame(rows).set_index("objective")


def home_season_uncertainty(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Independent-match sampling SE of the +3/0/-3 home-minus-away points gap."""
    selected = _selection(seasons)
    frame = home_advantage_by_season(tables, selected)
    games = _subset(tables.match_table, selected)
    gap = np.sign(games.home_goals - games.away_goals) * 3
    frame["ppm_gap_se"] = gap.groupby(games.season_id).sem()
    frame["ppm_gap_ci_lower"] = frame.ppm_gap - stats.norm.ppf(0.975) * frame.ppm_gap_se
    frame["ppm_gap_ci_upper"] = frame.ppm_gap + stats.norm.ppf(0.975) * frame.ppm_gap_se
    return frame


def era_uncertainty(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Match-based goal-mean CI; Student-t CI across seasons for dispersion.

    These are descriptive IID-match intervals for goals, not season-bootstrap CIs.
    The small-era dispersion interval uses t(n_seasons-1), explicitly retaining
    the uncertainty of just three seasons instead of resampling three blocks.
    """
    selected = _selection(seasons)
    games = _subset(tables.match_table, selected)
    balance = competitive_balance(tables, selected)
    rows = []
    for era, group in games.groupby("era", observed=True):
        season_ids = group.season_id.unique()
        row = dict(era=era, **_match_rates(group), n_seasons=len(season_ids))
        row["n_matches"] = row.pop("n")
        row["std_ppm"] = balance.loc[season_ids, "std_ppm"].mean()
        row["title_margin_ppm"] = balance.loc[season_ids, "title_margin_ppm"].mean()
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("era").sort_index()
    for era, group in games.groupby("era", observed=True):
        half = stats.t.ppf(0.975, len(group) - 1) * group.total_goals.sem()
        frame.loc[era, "goals_per_match_ci_lower"] = group.total_goals.mean() - half
        frame.loc[era, "goals_per_match_ci_upper"] = group.total_goals.mean() + half
        values = balance.loc[group.season_id.unique(), "std_ppm"]
        width = stats.t.ppf(0.975, len(values) - 1) * values.sem() if len(values) > 1 else np.nan
        frame.loc[era, "std_ppm_ci_lower"] = (
            max(0, values.mean() - width) if np.isfinite(width) else np.nan
        )
        frame.loc[era, "std_ppm_ci_upper"] = values.mean() + width
    return frame


def seasonal_trend_tests(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Eight prespecified linear trends: four outcomes x all/14-club seasons."""
    selected = _selection(seasons)
    frame = _subset(tables.season_summary, selected)
    rows = []
    for era in ("all", "14_teams"):
        data = frame if era == "all" else frame.loc[frame.n_teams.eq(14)]
        for metric in ("draw_rate", "zero_zero_rate", "penalties_per_match", "red_cards_per_match"):
            valid = data[metric].dropna()
            fit = (
                stats.linregress(valid.index.to_numpy(dtype=float), valid.to_numpy(dtype=float))
                if len(valid) > 2
                else None
            )
            p = float(fit.pvalue) if fit else np.nan
            rows.append(
                dict(
                    era=era,
                    metric=metric,
                    n_seasons=len(valid),
                    slope=fit.slope if fit else np.nan,
                    p=p,
                    p_bonferroni=min(1, 8 * p) if np.isfinite(p) else np.nan,
                )
            )
    return pd.DataFrame(rows).set_index(["era", "metric"])


def balance_decomposition(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Approximate variance decomposition; 'luck' assumes IID points per team-game.

    Uses the mean Var(single-game points)/games_played over clubs in each season.
    Ignores fixture dependence, differing opponents and skill-dependent variance;
    inferred latent dispersion is descriptive, not a causal identification of skill.
    """
    selected = _selection(seasons)
    frame = competitive_balance(tables, selected)
    games = _subset(tables.team_matches, selected)
    teams = _subset(tables.team_seasons, selected)
    for season, group in games.groupby("season_id", observed=True):
        counts = group.reset_index().groupby("team_id").size()
        luck_variance = float((group.points_earned.var() / counts).mean())
        observed = float(frame.loc[season, "std_ppm"])
        frame.loc[season, "luck_std"] = np.sqrt(luck_variance)
        frame.loc[season, "latent_std"] = np.sqrt(max(0, observed**2 - luck_variance))
        frame.loc[season, "luck_fraction"] = luck_variance / observed**2 if observed else np.nan
        frame.loc[season, "n_teams"] = len(teams.xs(season, level="season_id"))
    return frame


def champion_previous_rank(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Official champion's prior SPORTING ppm rank, including prior larger-league eras.

    History is limited to development seasons; a target in 2012/2013 may use
    2011/2012. Ties in prior ppm receive the minimum tied rank (reported explicitly).
    """
    selected = _selection(seasons)
    history = _subset(tables.team_seasons, list(DEVELOPMENT_SEASONS))
    rows = []
    for season in selected:
        if season - 1 not in history.index.get_level_values("season_id"):
            continue
        current = history.xs(season, level="season_id")
        champion = current.loc[current.final_position.eq(1)].iloc[0]
        champion_id = current.loc[current.final_position.eq(1)].index[0]
        previous = history.xs(season - 1, level="season_id")
        ranks = previous.points_per_match.rank(ascending=False, method="min")
        rank = ranks.get(champion_id, np.nan)
        rows.append(
            dict(
                season_id=season,
                team_name=champion.team_name,
                previous_rank=rank,
                previous_available=bool(pd.notna(rank)),
                best_previous=bool(rank == 1),
                top3_previous=bool(rank <= 3),
                n_teams=len(current),
            )
        )
    return pd.DataFrame(rows).set_index("season_id")
