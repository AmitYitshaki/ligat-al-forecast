"""Chapter E: descriptive context and matched comparisons, never causal or prematch models."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import binom, chi2, poisson  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables, teams_overview

from .core import MEAN_BOOTSTRAP_DRAWS, _bootstrap_count, _selection, _subset, rate_interval
from .match_flow import _mean_ci

# City membership is explicit, not inferred from ground-sharing or hosting agreements.
# Resolve these canonical names/aliases against the caller's config/teams.yaml records.
CITY_CLUBS = {
    "tel_aviv": ('מכבי ת"א', 'הפועל ת"א', 'בני יהודה ת"א'),
    "haifa": ("מכבי חיפה", "הפועל חיפה"),
    "jerusalem": ('בית"ר י-ם', "הפועל י-ם"),
    "petah_tikva": ('מכבי פ"ת', 'הפועל פ"ת'),
}
COACH_SEASONS = range(12, 26)


def _limits(rate: float, n: int | Iterable[int], distribution: str) -> pd.DataFrame:
    values = np.asarray([n] if isinstance(n, (int, np.integer)) else list(n), dtype=float)
    if (
        not np.isfinite(rate)
        or rate < 0
        or (distribution == "binomial" and rate > 1)
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 1).any()
        or (values != np.floor(values)).any()
    ):
        raise ValueError("Funnel requires a valid rate and positive integer exposures")
    values = np.unique(values.astype(int))
    rows = []
    for exposure in values:
        quantiles = (
            binom.ppf([0.025, 0.975, 0.001, 0.999], exposure, rate)
            if distribution == "binomial"
            else poisson.ppf([0.025, 0.975, 0.001, 0.999], rate * exposure)
        ) / exposure
        rows.append(
            dict(
                n=exposure,
                lower95=quantiles[0],
                upper95=quantiles[1],
                lower998=quantiles[2],
                upper998=quantiles[3],
            )
        )
    return pd.DataFrame(rows).set_index("n", drop=True)


def funnel_limits_binomial(rate_league: float, n: int | Iterable[int]) -> pd.DataFrame:
    """Exact binomial count quantiles / matches: marginal 95% and 99.8% reference funnels."""
    return _limits(rate_league, n, "binomial")


funnel_limits = funnel_limits_binomial


def funnel_limits_poisson(rate_league: float, n: int | Iterable[int]) -> pd.DataFrame:
    """Exact Poisson(rate*exposure) quantiles / exposure; multiple events per game are allowed."""
    return _limits(rate_league, n, "poisson")


def _rates(games: pd.DataFrame, n_bootstrap: int) -> dict[str, Any]:
    row: dict[str, Any] = dict(n=len(games), n_matches=len(games))
    for label, flag in {
        "home_win_rate": games.home_goals.gt(games.away_goals),
        "draw_rate": games.home_goals.eq(games.away_goals),
        "away_win_rate": games.home_goals.lt(games.away_goals),
        "zero_zero_rate": games.total_goals.eq(0),
    }.items():
        rate, lo, hi = rate_interval(int(flag.sum()), len(games))
        row.update(
            {
                label: rate,
                f"{label}_count": int(flag.sum()),
                f"{label}_ci_lower": lo,
                f"{label}_ci_upper": hi,
            }
        )
    row["goals_per_match"] = games.total_goals.mean()
    row["goals_per_match_ci_lower"], row["goals_per_match_ci_upper"] = _mean_ci(
        games, "total_goals", n_bootstrap
    )
    points = games.assign(
        home_points=3 * games.home_goals.gt(games.away_goals).astype(int)
        + games.home_goals.eq(games.away_goals).astype(int)
    )
    row["home_ppm"] = points.home_points.mean()
    row["home_ppm_ci_lower"], row["home_ppm_ci_upper"] = _mean_ci(
        points, "home_points", n_bootstrap
    )
    return row


def _events(tables: AnalysisTables, selected: list[int], red_cards: pd.DataFrame) -> pd.DataFrame:
    games = _subset(tables.match_table, selected)
    frame = games.copy()
    for label, events in (("red", red_cards), ("penalty", _subset(tables.goal_events, selected))):
        rows = (
            events.loc[events.game_id.isin(games.index)].copy()
            if "game_id" in events
            else (events.reset_index().loc[lambda x: x.game_id.isin(games.index)].copy())
        )
        if label == "penalty":
            rows = rows.loc[rows.is_penalty]
        rows = rows.join(
            games[["home_team_id", "away_team_id"]], on="game_id", validate="many_to_one"
        )
        if not (rows.team_id.eq(rows.home_team_id) | rows.team_id.eq(rows.away_team_id)).all():
            raise ValueError("Context event credited to a nonparticipant")
        for side in ("home", "away"):
            counts = rows.loc[rows.team_id.eq(rows[f"{side}_team_id"])].groupby("game_id").size()
            frame[f"{side}_{label}"] = counts.reindex(frame.index, fill_value=0).astype("Int64")
            if label == "penalty":
                frame.loc[~frame.goals_available, f"{side}_{label}"] = pd.NA
        frame[f"{label}_total"] = frame[f"home_{label}"] + frame[f"away_{label}"]
    return frame


def _add_funnel(
    frame: pd.DataFrame,
    metric: str,
    count: str,
    exposure: str,
    league_rate: float,
    distribution: str,
    min_matches: int,
) -> None:
    valid = frame.loc[frame[exposure].gt(0)]
    fn = funnel_limits_binomial if distribution == "binomial" else funnel_limits_poisson
    bands = fn(league_rate, valid[exposure].astype(int).unique()) if len(valid) else pd.DataFrame()
    for field in ("lower95", "upper95", "lower998", "upper998"):
        frame[f"{metric}_{field}"] = frame[exposure].map(bands[field]) if len(bands) else np.nan
    large = frame.loc[frame.n_matches.ge(min_matches) & frame[exposure].gt(0)]
    expected = large[exposure] * league_rate
    k = len(large)
    phi = (
        float(((large[count] - expected) ** 2 / expected).sum() / (k - 1))
        if k > 1 and league_rate > 0
        else float("nan")
    )
    p = float(chi2.sf(phi * (k - 1), k - 1)) if np.isfinite(phi) else float("nan")
    if distribution != "poisson":
        phi = p = float("nan")
    widened = distribution == "poisson" and phi > 1
    frame.attrs.setdefault("funnel_models", {})[metric] = dict(
        rate=league_rate,
        distribution=distribution,
        phi=phi,
        p_overdispersion=p,
        widened=bool(widened),
        overdispersion_clear=bool(widened and p < 0.05),
        n_entities=k,
        min_matches=min_matches,
    )
    scale = np.sqrt(phi) if widened else 1
    for field in ("lower95", "upper95", "lower998", "upper998"):
        frame[f"{metric}_{field}_expanded"] = (
            league_rate + scale * (frame[f"{metric}_{field}"] - league_rate)
        ).clip(lower=0, upper=1 if distribution == "binomial" else None)


def referee_stats(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    red_cards: pd.DataFrame,
    min_matches: int = 30,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """All named referees, marginal intervals and count funnels; assignment is not randomized."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    if min_matches < 1:
        raise ValueError("min_matches must be positive")
    games = _events(tables, selected, red_cards)
    rows = []
    for referee, group in games.groupby("referee", observed=True):
        row = dict(referee=referee, **_rates(group, n_bootstrap))
        row["n_seasons"] = group.season_id.nunique()
        for label, column in (("red_cards", "red_total"), ("penalty_goals", "penalty_total")):
            valid = group.loc[group[column].notna()]
            row[f"{label}_count"] = int(valid[column].sum())
            row[f"{label}_matches"] = len(valid)
            row[f"{label}_per_match"] = valid[column].mean()
            row[f"{label}_per_match_ci_lower"], row[f"{label}_per_match_ci_upper"] = _mean_ci(
                valid, column, n_bootstrap
            )
        for label, event in (("red", "red"), ("penalty_goal", "penalty")):
            total = int(group[f"{event}_total"].sum())
            rate, lo, hi = rate_interval(int(group[f"home_{event}"].sum()), total)
            row.update(
                {
                    f"home_{label}_share": rate,
                    f"home_{label}_share_n": total,
                    f"home_{label}_share_ci_lower": lo,
                    f"home_{label}_share_ci_upper": hi,
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("referee", drop=True).sort_index()
    frame.attrs.update(
        n_total=len(games),
        n_missing_referee=int(games.referee.isna().sum()),
        min_matches=min_matches,
        n_missing_penalty_matches=int((~games.goals_available).sum()),
    )
    _add_funnel(
        frame,
        "home_win_rate",
        "home_win_rate_count",
        "n_matches",
        float(games.home_goals.gt(games.away_goals).mean()),
        "binomial",
        min_matches,
    )
    for label, column in (("red_cards", "red_total"), ("penalty_goals", "penalty_total")):
        _add_funnel(
            frame,
            f"{label}_per_match",
            f"{label}_count",
            f"{label}_matches",
            float(games[column].mean()),
            "poisson",
            min_matches,
        )
    return frame


def referee_home_bias(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    red_cards: pd.DataFrame,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """League home/away card and scored-penalty counts per eligible match, without causal claims."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    games = _events(tables, selected, red_cards)
    rows = []
    for label in ("red", "penalty"):
        for side in ("home", "away"):
            column = f"{side}_{label}"
            valid = games.loc[games[column].notna()]
            lo, hi = _mean_ci(valid, column, n_bootstrap)
            rows.append(
                dict(
                    event=label,
                    side=side,
                    n_matches=len(valid),
                    n_events=int(valid[column].sum()),
                    per_match=valid[column].mean(),
                    ci_lower=lo,
                    ci_upper=hi,
                )
            )
    return pd.DataFrame(rows).set_index(["event", "side"], drop=True)


def venue_effects(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Borrowed/home grounds among team-seasons observed in both; composition still differs."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    games = _subset(tables.match_table, selected)
    valid = games.loc[games.is_home_venue.notna()].copy()
    both = valid.groupby(["season_id", "home_team_id"]).is_home_venue.transform("nunique").eq(2)
    paired = valid.loc[both]
    rows = [
        dict(is_home_venue=home, **_rates(group, n_bootstrap))
        for home, group in paired.groupby("is_home_venue", observed=True)
    ]
    frame = pd.DataFrame(rows).set_index("is_home_venue", drop=True)
    frame.attrs.update(
        n_total=len(games),
        n_unknown=int(games.is_home_venue.isna().sum()),
        n_borrowed_all=int(games.is_home_venue.eq(False).sum()),
        n_paired_team_seasons=paired.groupby(["season_id", "home_team_id"]).ngroups,
        n_unpaired=len(valid) - len(paired),
    )
    return frame


def venue_summary(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    min_matches: int = 30,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Ground home-win/goal reference funnels, restricted to sufficient observations."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    if min_matches < 1:
        raise ValueError("min_matches must be positive")
    games = _subset(tables.match_table, selected)
    rows = [
        dict(venue=venue, **_rates(group, n_bootstrap), goals_count=int(group.total_goals.sum()))
        for venue, group in games.groupby("venue", observed=True)
        if len(group) >= min_matches
    ]
    frame = pd.DataFrame(rows).set_index("venue", drop=True)
    frame.attrs.update(n_missing_venue=int(games.venue.isna().sum()), min_matches=min_matches)
    _add_funnel(
        frame,
        "home_win_rate",
        "home_win_rate_count",
        "n_matches",
        float(games.home_goals.gt(games.away_goals).mean()),
        "binomial",
        min_matches,
    )
    _add_funnel(
        frame,
        "goals_per_match",
        "goals_count",
        "n_matches",
        float(games.total_goals.mean()),
        "poisson",
        min_matches,
    )
    return frame


def _coach_selection(seasons: Iterable[int]) -> list[int]:
    selected = _selection(seasons)
    if not set(selected) <= set(COACH_SEASONS):
        raise ValueError("Coach comparison requires development seasons 12–25")
    return selected


def _clean_spells(tables: AnalysisTables, selected: list[int]) -> pd.DataFrame:
    spells = tables.coach_spells.reset_index()
    return spells.loc[
        spells.season_id_start.isin(selected) & ~spells.is_left_censored & ~spells.is_flicker
    ].copy()


def coach_change_effect(
    tables: AnalysisTables,
    seasons: Iterable[int] = COACH_SEASONS,
    *,
    window: int = 5,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Before/after coaching change against same-season, same-match-number nearby-form controls.

    Controls exclude self, unknown coaches and every observed spell change inside the
    ten-game window. Treated post-windows must remain within the new observed spell.
    No controls means NA effect, not zero; this comparison cannot establish causality.
    """
    selected = _coach_selection(seasons)
    _bootstrap_count(n_bootstrap)
    if window < 1:
        raise ValueError("window must be positive")
    games = (
        _subset(tables.team_matches, selected)
        .reset_index()
        .sort_values(["season_id", "team_id", "match_number"])
    )
    histories = dict(tuple(games.groupby(["season_id", "team_id"], observed=True)))
    all_spells = tables.coach_spells.reset_index()
    starts = all_spells.loc[
        all_spells.season_id_start.isin(selected) & all_spells.is_mid_season_start
    ]
    clean = _clean_spells(tables, selected)
    candidates = clean.loc[clean.is_mid_season_start]
    reasons: Counter[str] = Counter()
    rows = []
    for spell in candidates.itertuples():
        key = (int(spell.season_id_start), int(spell.team_id))
        history = histories[key]
        position = np.flatnonzero(history.date.eq(spell.first_date).to_numpy())
        if len(position) != 1:
            reasons["unknown_start_game"] += 1
            continue
        boundary = int(position[0])
        before, after = (
            history.iloc[boundary - window : boundary],
            history.iloc[boundary : boundary + window],
        )
        if boundary < window or len(after) != window:
            reasons["insufficient_same_season_games"] += 1
            continue
        next_start = all_spells.loc[
            all_spells.team_id.eq(spell.team_id) & all_spells.first_date.gt(spell.first_date),
            "first_date",
        ].min()
        if pd.notna(next_start) and after.date.ge(next_start).any():
            reasons["new_spell_shorter_than_window"] += 1
            continue
        if (after.coach_id.isna() & after.coach_name.isna()).any():
            reasons["unknown_coach_in_post_window"] += 1
            continue
        ppm_before, ppm_after = (
            float(before.points_earned.mean()),
            float(after.points_earned.mean()),
        )
        control_ids, differences = [], []
        for (season, team), control in histories.items():
            if season != key[0] or team == key[1]:
                continue
            same_number = control.loc[control.match_number.eq(history.iloc[boundary].match_number)]
            if len(same_number) != 1:
                continue
            b = int(np.flatnonzero(control.match_number.eq(same_number.iloc[0].match_number))[0])
            cb, ca = control.iloc[b - window : b], control.iloc[b : b + window]
            if (
                b < window
                or len(ca) != window
                or abs(cb.points_earned.mean() - ppm_before) > 0.3 + 1e-12
            ):
                continue
            combined = pd.concat([cb, ca])
            if (combined.coach_id.isna() & combined.coach_name.isna()).any():
                continue
            changes = starts.loc[starts.team_id.eq(team) & starts.season_id_start.eq(season)]
            if changes.first_date.between(combined.date.iloc[0], combined.date.iloc[-1]).any():
                continue
            control_ids.append(int(team))
            differences.append(float(ca.points_earned.mean() - cb.points_earned.mean()))
        control_diff = float(np.mean(differences)) if differences else float("nan")
        rows.append(
            dict(
                team_id=spell.team_id,
                spell_number=spell.spell_number,
                season_id=key[0],
                first_date=spell.first_date,
                coach_name=spell.coach_name,
                is_left_censored=False,
                is_flicker=False,
                window=window,
                before_game_ids=tuple(before.game_id),
                after_game_ids=tuple(after.game_id),
                n_before=len(before),
                n_after=len(after),
                ppm_before=ppm_before,
                ppm_after=ppm_after,
                diff=ppm_after - ppm_before,
                n_controls=len(control_ids),
                control_team_ids=tuple(control_ids),
                control_diff=control_diff,
                effect=ppm_after - ppm_before - control_diff,
            )
        )
    columns = [
        "team_id",
        "spell_number",
        "season_id",
        "first_date",
        "coach_name",
        "is_left_censored",
        "is_flicker",
        "window",
        "before_game_ids",
        "after_game_ids",
        "n_before",
        "n_after",
        "ppm_before",
        "ppm_after",
        "diff",
        "n_controls",
        "control_team_ids",
        "control_diff",
        "effect",
    ]
    frame = pd.DataFrame(rows, columns=columns).set_index(["team_id", "spell_number"], drop=True)
    frame.attrs.update(n_clean_mid_season=len(candidates), excluded=dict(reasons), window=window)
    return frame


def coach_summary(
    tables: AnalysisTables,
    seasons: Iterable[int] = COACH_SEASONS,
    *,
    window: int = 5,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Separate all eligible changes from matched ones; plot only like-for-like matched means."""
    selected = _coach_selection(seasons)
    changes = coach_change_effect(tables, selected, window=window, n_bootstrap=n_bootstrap)
    rows = []
    for cohort, group in (
        ("all_eligible", changes),
        ("matched", changes.loc[changes.effect.notna()]),
    ):
        row = dict(cohort=cohort, n_changes=len(group), n_seasons=group.season_id.nunique())
        for metric in ("diff", "control_diff", "effect"):
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"mean_{metric}_ci_lower"], row[f"mean_{metric}_ci_upper"] = _mean_ci(
                group, metric, n_bootstrap
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("cohort", drop=True)
    clean = _clean_spells(tables, selected)
    frame.attrs.update(
        changes.attrs,
        median_clean_spell_matches=float(clean.matches.median()),
        changes_by_season=changes.groupby("season_id").size().to_dict(),
    )
    return frame


def derby_and_big_games(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    teams_config: Sequence[Mapping[str, Any]],
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Disjoint city/shared-ground/big/other groups; explicit names resolve through team config."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    names = {
        name: int(team["team_id"])
        for team in teams_config
        for name in [team["name_he"], *team.get("aliases", [])]
    }
    missing = set(name for clubs in CITY_CLUBS.values() for name in clubs) - names.keys()
    if missing:
        raise ValueError(f"City clubs missing from teams_config: {sorted(missing)}")
    city_pairs = {
        tuple(sorted((names[a], names[b]))): city
        for city, clubs in CITY_CLUBS.items()
        for a, b in combinations(clubs, 2)
    }
    games = _subset(tables.match_table, selected)
    overview = teams_overview(tables, selected)
    big_ids = set(overview.head(4).index)
    shared: set[tuple[int, int, int]] = set()
    hosts = games.loc[games.is_home_venue.fillna(False)]
    for (season, _), group in hosts.groupby(["season_id", "venue"], observed=True):
        shared.update(
            (int(season), min(int(a), int(b)), max(int(a), int(b)))
            for a, b in combinations(group.home_team_id.unique(), 2)
        )
    rows, overlaps = [], 0
    for game in games.itertuples():
        pair = tuple(sorted((int(game.home_team_id), int(game.away_team_id))))
        is_big = set(pair) <= big_ids
        if pair in city_pairs:
            label = "city_derby"
            overlaps += int(is_big)
        elif (int(game.season_id), *pair) in shared:
            label = "shared_venue_other"
        elif is_big:
            label = "big_game"
        else:
            label = "other"
        rows.append(dict(game_id=game.Index, game_type=label))
    classified = games.join(pd.DataFrame(rows).set_index("game_id"))
    result = pd.DataFrame(
        [
            dict(
                game_type=label,
                **_rates(group, n_bootstrap),
                pairs=tuple(
                    sorted(
                        {
                            tuple(sorted((int(g.home_team_id), int(g.away_team_id))))
                            for g in group.itertuples()
                        }
                    )
                ),
            )
            for label, group in classified.groupby("game_type", observed=True)
        ]
    )
    result = result.set_index("game_type", drop=True)
    result.attrs.update(
        city_pairs=[dict(team_ids=pair, city=city) for pair, city in city_pairs.items()],
        shared_pairs=sorted(shared),
        big_teams=overview.head(4)
        .reset_index()[["team_id", "team_name", "points_per_match"]]
        .to_dict("records"),
        n_city_also_big=overlaps,
        precedence="city > shared_venue_other > big > other",
    )
    return result


def upset_rates(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Previous-season value ratios/rank gaps only; unknown strength is never a forced weak side."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    games = _subset(tables.match_table, selected)
    seasons_frame = _subset(tables.team_seasons, selected).reset_index()
    # t-1 lookup may use a development season outside a narrow display selection, never holdouts.
    previous = _subset(tables.team_seasons, [s for s in DEVELOPMENT_SEASONS]).reset_index()
    positions = previous.set_index(["season_id", "team_id"]).final_position
    current = seasons_frame.set_index(["season_id", "team_id"])
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([games.season_id, games[f"{side}_team_id"]])
        prior_keys = pd.MultiIndex.from_arrays([games.season_id - 1, games[f"{side}_team_id"]])
        games[f"{side}_share"] = current.prev_season_value_share.reindex(keys).to_numpy(
            dtype=float, na_value=np.nan
        )
        games[f"{side}_rank"] = positions.reindex(prior_keys).to_numpy(dtype=float, na_value=np.nan)
        games[f"{side}_promoted"] = (
            current.promoted_in.reindex(keys).fillna(False).to_numpy(dtype=bool)
        )
    rows, edges = [], []
    for metric, home, away in (
        ("value_ratio", games.home_share, games.away_share),
        ("rank_gap", -games.home_rank, -games.away_rank),
    ):
        valid = home.notna() & away.notna()
        if metric == "value_ratio":
            valid &= home.gt(0) & away.gt(0)
        equal = valid & home.eq(away)
        bins = pd.Series("unknown_both", index=games.index, dtype="string")
        bins.loc[equal] = "equal_strength"
        known = valid & ~equal
        if metric == "value_ratio" and known.any():
            ratios = np.maximum(
                home.loc[known] / away.loc[known], away.loc[known] / home.loc[known]
            )
            quantiles, breaks = pd.qcut(ratios, 5, labels=False, retbins=True, duplicates="drop")
            bins.loc[known] = quantiles.map(lambda x: f"Q{int(x) + 1}" if pd.notna(x) else "Q1")
            edges = breaks.tolist()
        elif metric == "rank_gap":
            gap = (home - away).abs()
            for label, flag in (
                ("gap_1_3", gap.le(3)),
                ("gap_4_7", gap.between(4, 7)),
                ("gap_8_plus", gap.ge(8)),
            ):
                bins.loc[known & flag] = label
        one_missing = home.isna() ^ away.isna()
        veteran_promoted = one_missing & (
            (home.isna() & games.home_promoted) | (away.isna() & games.away_promoted)
        )
        bins.loc[one_missing & ~veteran_promoted] = "missing_source"
        bins.loc[veteran_promoted] = "veteran_vs_promoted"
        orientation = pd.Series("unknown", index=games.index)
        orientation.loc[known & home.lt(away)] = "home_weaker"
        orientation.loc[known & home.gt(away)] = "away_weaker"
        orientation.loc[veteran_promoted & games.home_promoted] = "home_promoted"
        orientation.loc[veteran_promoted & games.away_promoted] = "away_promoted"
        grouped = games.assign(gap_bin=bins, weaker_side=orientation)
        for (label, side), group in grouped.groupby(["gap_bin", "weaker_side"], observed=True):
            row = dict(metric=metric, gap_bin=label, weaker_side=side, **_rates(group, n_bootstrap))
            for name in ("weak_win", "strong_win", "veteran_win", "promoted_win"):
                row[f"{name}_rate"] = row[f"{name}_rate_ci_lower"] = row[
                    f"{name}_rate_ci_upper"
                ] = float("nan")
            if side in ("home_weaker", "away_weaker", "home_promoted", "away_promoted"):
                home_labeled = side.startswith("home")
                prefix = "promoted" if "promoted" in side else "weak"
                opponent = "veteran" if prefix == "promoted" else "strong"
                for dest, origin in (
                    (prefix, "home" if home_labeled else "away"),
                    (opponent, "away" if home_labeled else "home"),
                ):
                    for suffix in ("rate", "rate_ci_lower", "rate_ci_upper"):
                        row[f"{dest}_win_{suffix}"] = row[f"{origin}_win_{suffix}"]
            rows.append(row)
    frame = pd.DataFrame(rows).set_index(["metric", "gap_bin", "weaker_side"], drop=True)
    frame.attrs.update(
        value_ratio_edges=edges,
        n_matches_per_metric=len(games),
        rank_bins="1–3 / 4–7 / 8+ places; full ties separate",
        unknown_policy="home/draw/away only; veteran/promoted explicit; no imputation",
    )
    return frame
