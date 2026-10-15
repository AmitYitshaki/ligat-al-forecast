"""Development-only descriptions of scores and match flow; never prematch features."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import poisson  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, MINUTE_BUCKETS, AnalysisTables

from .core import (
    BOOTSTRAP_SEED,
    MEAN_BOOTSTRAP_DRAWS,
    _bootstrap_count,
    _goals_interval,
    _selection,
    _subset,
    rate_interval,
)

STATES = ("leading", "level", "trailing")
RESULTS = ("win", "draw", "loss")


def _matches(tables: AnalysisTables, seasons: Iterable[int]) -> pd.DataFrame:
    return _subset(tables.match_table, _selection(seasons))


def _metadata(frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    reasons = np.select(
        [games.index == 864982, ~games.goals_available, games.ht_conflict],
        ["unreliable_scorers_864982", "no_goals", "ht_conflict"],
        default="included",
    )
    frame.attrs.update(
        n_total_matches=len(games),
        n_included_matches=int((reasons == "included").sum()),
        n_excluded_matches=int((reasons != "included").sum()),
        exclusions={
            reason: int((reasons == reason).sum())
            for reason in np.unique(reasons)
            if reason != "included"
        },
    )
    return frame


def timeline_exclusions(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Count disjoint exclusions: unreliable 864982, other missing scorers, then HT conflicts."""
    games = _matches(tables, seasons)
    attrs = _metadata(pd.DataFrame(), games).attrs
    counts = attrs["exclusions"]
    return pd.DataFrame(
        [
            (reason, counts.get(reason, 0))
            for reason in ("ht_conflict", "no_goals", "unreliable_scorers_864982")
        ],
        columns=["reason", "n_matches"],
    ).set_index("reason", drop=True)


def _eligible(tables: AnalysisTables, selected: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = _subset(tables.match_table, selected)
    valid = games.loc[~games.ht_conflict & games.goals_available & (games.index != 864982)]
    events = _subset(tables.goal_events, selected).reset_index()
    events = events.loc[events.game_id.isin(valid.index)].copy()
    if events.half.isna().any():
        raise ValueError("Eligible goal timeline has missing half; check source exceptions")
    events = events.sort_values(["game_id", "half", "minute", "goal_number"], kind="stable")
    events["clock"] = np.where(
        events.half.eq(1), events.minute.clip(upper=45), events.minute.clip(lower=45, upper=90)
    )
    return valid, events


def _mean_ci(frame: pd.DataFrame, column: str, n_bootstrap: int) -> tuple[float, float]:
    # Reuse the match-weighted season-block bootstrap with a renamed measurement.
    data = frame[["season_id", column]].rename(columns={column: "total_goals"}).dropna()
    return _goals_interval(data, n_bootstrap)


def _rates(frame: pd.DataFrame, n_bootstrap: int) -> dict[str, Any]:
    row: dict[str, Any] = {"n": len(frame)}
    for outcome in RESULTS:
        count = int(frame.result.eq(outcome).sum())
        rate, low, high = rate_interval(count, len(frame))
        row.update(
            {
                f"{outcome}_n": count,
                f"{outcome}_rate": rate,
                f"{outcome}_rate_ci_lower": low,
                f"{outcome}_rate_ci_upper": high,
            }
        )
    row["ppm"] = frame.points_earned.mean()
    row["ppm_ci_lower"], row["ppm_ci_upper"] = _mean_ci(frame, "points_earned", n_bootstrap)
    return row


def scoreline_distribution(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS, max_goals: int = 8
) -> pd.DataFrame:
    """Observed exact-score mass; tails remain outside the grid, never renormalized away."""
    games = _matches(tables, seasons)
    if max_goals < 0:
        raise ValueError("max_goals must be nonnegative")
    index = pd.Index(range(max_goals + 1), name="home_goals")
    columns = pd.Index(range(max_goals + 1), name="away_goals")
    counts = pd.crosstab(games.home_goals, games.away_goals).reindex(
        index=index, columns=columns, fill_value=0
    )
    frame = counts / len(games)
    n_beyond = int((games.home_goals.gt(max_goals) | games.away_goals.gt(max_goals)).sum())
    intervals = np.array(
        [rate_interval(int(n), len(games))[1:] for n in counts.to_numpy().ravel()]
    ).reshape(max_goals + 1, max_goals + 1, 2)
    frame.attrs.update(
        n_matches=len(games),
        n_beyond=n_beyond,
        counts=counts.to_numpy().tolist(),
        ci_lower=intervals[:, :, 0].tolist(),
        ci_upper=intervals[:, :, 1].tolist(),
    )
    return frame


def top_scorelines(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Ten most common actual results, including any score beyond the displayed grid."""
    games = _matches(tables, seasons)
    frame = games.groupby(["home_goals", "away_goals"], observed=True).size().rename("n")
    frame = (
        frame.reset_index()
        .sort_values(["n", "home_goals", "away_goals"], ascending=[False, True, True])
        .head(10)
    )
    ci = [rate_interval(int(n), len(games)) for n in frame.n]
    frame[["rate", "ci_lower", "ci_upper"]] = ci
    return frame.set_index(["home_goals", "away_goals"], drop=True)


def poisson_reference(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS, max_goals: int = 8
) -> pd.DataFrame:
    """Independent league-average Poisson description; strength heterogeneity is ignored."""
    games = _matches(tables, seasons)
    if max_goals < 0:
        raise ValueError("max_goals must be nonnegative")
    home, away = float(games.home_goals.mean()), float(games.away_goals.mean())
    mass = np.outer(
        poisson.pmf(np.arange(max_goals + 1), home), poisson.pmf(np.arange(max_goals + 1), away)
    )
    frame = pd.DataFrame(
        mass,
        index=pd.Index(range(max_goals + 1), name="home_goals"),
        columns=pd.Index(range(max_goals + 1), name="away_goals"),
    )
    frame.attrs.update(
        lambda_home=home, lambda_away=away, tail_mass=float(1 - mass.sum()), n_matches=len(games)
    )
    return frame


def observed_vs_poisson(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Low scores and all draws against independent Poisson, including its infinite draw tail."""
    selected = _selection(seasons)
    games = _matches(tables, selected)
    reference = poisson_reference(tables, selected)
    home, away = reference.attrs["lambda_home"], reference.attrs["lambda_away"]
    # skellam is undefined at lambda=0; convergent Poisson sum handles synthetic zero means.
    maximum = max(8, int(poisson.ppf(1 - 1e-14, max(home, away))))
    draws = float(
        (
            poisson.pmf(np.arange(maximum + 1), home) * poisson.pmf(np.arange(maximum + 1), away)
        ).sum()
    )
    rows = []
    for label, hg, ag in [
        ("0:0", 0, 0),
        ("1:0", 1, 0),
        ("0:1", 0, 1),
        ("1:1", 1, 1),
        ("2:2", 2, 2),
        ("all_draws", -1, -1),
    ]:
        count = int(
            (
                games.home_goals.eq(games.away_goals)
                if hg < 0
                else games.home_goals.eq(hg) & games.away_goals.eq(ag)
            ).sum()
        )
        observed, lo, hi = rate_interval(count, len(games))
        expected = draws if hg < 0 else float(reference.loc[hg, ag])
        rows.append(
            dict(
                scoreline=label,
                n=len(games),
                count=count,
                observed=observed,
                observed_ci_lower=lo,
                observed_ci_upper=hi,
                expected=expected,
                ratio=observed / expected if expected > 0 else float("nan"),
            )
        )
    return pd.DataFrame(rows).set_index("scoreline", drop=True)


def goal_count_dispersion(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Sample variance/mean and home-away Pearson; bootstrap entire seasons, not matches."""
    games = _matches(tables, seasons)
    _bootstrap_count(n_bootstrap)
    data = games.assign(x=games.home_goals.astype(float), y=games.away_goals.astype(float))
    for name, values in {"xx": data.x**2, "yy": data.y**2, "xy": data.x * data.y}.items():
        data[name] = values
    data["n"] = 1
    blocks = data.groupby("season_id")[["n", "x", "y", "xx", "yy", "xy"]].sum().to_numpy()

    def moments(a: Any) -> Any:
        n, x, y, xx, yy, xy = np.moveaxis(np.asarray(a, dtype=float), -1, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.stack([x / n, y / n, (x + y) / n], axis=-1)
            var = np.stack(
                [
                    (xx - x * x / n) / (n - 1),
                    (yy - y * y / n) / (n - 1),
                    (xx + yy + 2 * xy - (x + y) ** 2 / n) / (n - 1),
                ],
                axis=-1,
            )
            corr = (xy - x * y / n) / np.sqrt((xx - x * x / n) * (yy - y * y / n))
            return means, var, var / means, corr

    means, variance, dispersion, corr = moments(blocks.sum(axis=0))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = blocks[rng.integers(len(blocks), size=(n_bootstrap, len(blocks)))].sum(axis=1)
    sampled = moments(samples)
    rows = []
    for i, name in enumerate(("home_goals", "away_goals", "total_goals")):
        row = dict(
            metric=name,
            n=len(games),
            n_seasons=len(blocks),
            mean=means[i],
            variance=variance[i],
            dispersion=dispersion[i],
            home_away_correlation=corr,
        )
        for metric, values in zip(
            ("mean", "variance", "dispersion", "home_away_correlation"), sampled, strict=True
        ):
            vals = values if metric == "home_away_correlation" else values[:, i]
            vals = vals[np.isfinite(vals)]
            low, high = (
                np.quantile(vals, [0.025, 0.975])
                if len(blocks) >= 2 and len(vals)
                else (float("nan"), float("nan"))
            )
            row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = low, high
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric", drop=True)


def draw_types(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """0:0, 1:1 and 2:2+ as fractions of all matches in each season, not just draws."""
    rows = []
    for season, games in _matches(tables, seasons).groupby("season_id"):
        row = dict(season_id=season, n=len(games))
        for label, flag in {
            "zero_zero": games.home_goals.eq(0) & games.away_goals.eq(0),
            "one_one": games.home_goals.eq(1) & games.away_goals.eq(1),
            "two_two_plus": games.home_goals.eq(games.away_goals) & games.home_goals.ge(2),
        }.items():
            rate, low, high = rate_interval(int(flag.sum()), len(games))
            row.update(
                {
                    f"{label}_n": int(flag.sum()),
                    f"{label}_rate": rate,
                    f"{label}_ci_lower": low,
                    f"{label}_ci_upper": high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("season_id", drop=True)


def _sequenced(events: pd.DataFrame, home: int, away: int) -> list[dict[str, Any]]:
    scores = {home: 0, away: 0}
    sequence = []
    for event in events.itertuples():
        team = int(event.team_id)
        opponent = away if team == home else home
        difference = scores[team] - scores[opponent]
        state = "leading" if difference > 0 else "trailing" if difference < 0 else "level"
        sequence.append(
            dict(
                team_id=team,
                half=event.half,
                minute=event.minute,
                clock=event.clock,
                minute_bucket=event.minute_bucket,
                score_before=state,
            )
        )
        scores[team] += 1
    return sequence


def game_timelines(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Reconstruct 90-minute states in half/minute/goal_number order, including zero durations.

    Added time is clamped to 45/90. Same-minute order is a descriptive convention,
    not evidence of sub-minute ordering. A comeback needs an equalizer within ten
    clamped minutes after a conceded goal that left the team behind.
    """
    selected = _selection(seasons)
    games, goals = _eligible(tables, selected)
    groups = dict(tuple(goals.groupby("game_id", sort=False)))
    rows = []
    for game in games.itertuples():
        home, away = int(game.home_team_id), int(game.away_team_id)
        events = groups.get(game.Index, goals.iloc[:0])
        scores = {home: 0, away: 0}
        minutes = {team: dict.fromkeys(STATES, 0) for team in (home, away)}
        led = dict.fromkeys((home, away), False)
        trailed = led.copy()
        concessions: dict[int, list[int]] = {home: [], away: []}
        exposed = led.copy()
        equalized = led.copy()
        last = 0
        for event in events.itertuples():
            clock = int(event.clock)
            for team, opponent in ((home, away), (away, home)):
                diff = scores[team] - scores[opponent]
                state = "leading" if diff > 0 else "trailing" if diff < 0 else "level"
                minutes[team][state] += clock - last
            scorer = int(event.team_id)
            opponent = away if scorer == home else home
            scores[scorer] += 1
            if scores[scorer] == scores[opponent] and concessions[scorer]:
                equalized[scorer] |= any(clock - t <= 10 for t in concessions[scorer])
            for team, other in ((home, away), (away, home)):
                led[team] |= scores[team] > scores[other]
                trailed[team] |= scores[team] < scores[other]
                if scores[team] >= scores[other]:
                    concessions[team] = []
            if scores[opponent] < scores[scorer]:
                concessions[opponent].append(clock)
                exposed[opponent] = True
            last = clock
        if (scores[home], scores[away]) != (game.home_goals, game.away_goals):
            raise ValueError(f"Timeline score mismatch for game_id={game.Index}")
        for team, opponent in ((home, away), (away, home)):
            diff = scores[team] - scores[opponent]
            state = "leading" if diff > 0 else "trailing" if diff < 0 else "level"
            minutes[team][state] += 90 - last
            points = 3 if diff > 0 else 1 if diff == 0 else 0
            own = events.loc[events.team_id.eq(team)]
            other = events.loc[events.team_id.eq(opponent)]
            rows.append(
                dict(
                    game_id=game.Index,
                    team_id=team,
                    season_id=game.season_id,
                    is_home=team == home,
                    minutes_leading=minutes[team]["leading"],
                    minutes_level=minutes[team]["level"],
                    minutes_trailing=minutes[team]["trailing"],
                    led_at_any_point=led[team],
                    trailed_at_any_point=trailed[team],
                    points_earned=points,
                    points_dropped_from_lead=3 - points if led[team] else pd.NA,
                    points_won_from_behind=points if trailed[team] else pd.NA,
                    equalized_within_10=equalized[team] if exposed[team] else pd.NA,
                    late_goal_for=int((own.half.eq(2) & own.minute.gt(90)).sum()),
                    late_goal_against=int((other.half.eq(2) & other.minute.gt(90)).sum()),
                    first_half_stoppage_goals=int(own.is_first_half_stoppage.sum()),
                )
            )
    frame = pd.DataFrame(rows)
    for column in ("points_dropped_from_lead", "points_won_from_behind"):
        frame[column] = frame[column].astype("Int64")
    frame["equalized_within_10"] = frame.equalized_within_10.astype("boolean")
    return _metadata(
        frame.set_index(["game_id", "team_id"], drop=True), _subset(tables.match_table, selected)
    )


def timeline_team_summary(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Team-season time in each state and sporting points recovered/lost; exact season totals."""
    selected = _selection(seasons)
    timelines = game_timelines(tables, selected)
    frame = timelines.reset_index()
    rows = []
    for (season, team), games in frame.groupby(["season_id", "team_id"]):
        eligible = games.equalized_within_10.dropna()
        rate, lo, hi = rate_interval(int(eligible.sum()), len(eligible))
        rows.append(
            dict(
                season_id=season,
                team_id=team,
                n=len(games),
                ppm=games.points_earned.mean(),
                **{f"minutes_{state}": games[f"minutes_{state}"].mean() for state in STATES},
                points_dropped_from_lead=games.points_dropped_from_lead.sum(min_count=1),
                points_won_from_behind=games.points_won_from_behind.sum(min_count=1),
                n_led=int(games.led_at_any_point.sum()),
                n_trailed=int(games.trailed_at_any_point.sum()),
                equalized_within_10_n=len(eligible),
                equalized_within_10_rate=rate,
                equalized_within_10_ci_lower=lo,
                equalized_within_10_ci_upper=hi,
            )
        )
    result = pd.DataFrame(rows).set_index(["season_id", "team_id"], drop=True)
    result = result.join(_subset(tables.team_seasons, selected)[["team_name"]])
    result.attrs = timelines.attrs.copy()
    return result


def timeline_resilience(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS, *, min_seasons: int = 5
) -> pd.DataFrame:
    """Exact comeback/lost-point totals for established clubs; more matches means more exposure."""
    selected = _selection(seasons)
    if min_seasons < 1:
        raise ValueError("min_seasons must be positive")
    seasons_frame = timeline_team_summary(tables, selected)
    rows = []
    for team, group in seasons_frame.reset_index().groupby("team_id"):
        if len(group) < min_seasons:
            continue
        rows.append(
            dict(
                team_id=team,
                team_name=group.sort_values("season_id").team_name.iloc[-1],
                n_seasons=len(group),
                n=int(group.n.sum()),
                n_led=int(group.n_led.sum()),
                n_trailed=int(group.n_trailed.sum()),
                points_dropped_from_lead=group.points_dropped_from_lead.sum(min_count=1),
                points_won_from_behind=group.points_won_from_behind.sum(min_count=1),
            )
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "team_id",
            "team_name",
            "n_seasons",
            "n",
            "n_led",
            "n_trailed",
            "points_dropped_from_lead",
            "points_won_from_behind",
        ],
    ).set_index("team_id", drop=True)
    result.attrs = seasons_frame.attrs.copy()
    return result


def halftime_transitions(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Conditional W/D/L by halftime state and venue, excluding unreliable halftime timelines."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    valid, _ = _eligible(tables, selected)
    teams = _subset(tables.team_matches, selected).reset_index()
    games = teams.loc[teams.game_id.isin(valid.index) & teams.ht_state.notna()]
    rows = [
        dict(is_home=home, ht_state=state, **_rates(group, n_bootstrap))
        for (home, state), group in games.groupby(["is_home", "ht_state"], observed=True)
    ]
    frame = pd.DataFrame(rows).set_index(["is_home", "ht_state"], drop=True)
    frame.attrs["n_excluded_rows"] = len(teams) - len(games)
    return _metadata(frame, _subset(tables.match_table, selected))


def zero_zero_at_60(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Outcome of games with no goal through clamped minute 60 inclusive."""
    selected = _selection(seasons)
    games, events = _eligible(tables, selected)
    scored = events.loc[events.clock.le(60)].game_id.unique()
    games = games.loc[~games.index.isin(scored)]
    rows = []
    for label, flag in {
        "zero_zero": games.total_goals.eq(0),
        "home_win": games.home_goals.gt(games.away_goals),
        "away_win": games.away_goals.gt(games.home_goals),
        "scoring_draw": games.home_goals.eq(games.away_goals) & games.total_goals.gt(0),
    }.items():
        rate, lo, hi = rate_interval(int(flag.sum()), len(games))
        rows.append(
            dict(
                outcome=label,
                n=len(games),
                count=int(flag.sum()),
                rate=rate,
                ci_lower=lo,
                ci_upper=hi,
            )
        )
    return _metadata(
        pd.DataFrame(rows).set_index("outcome", drop=True), _subset(tables.match_table, selected)
    )


def first_goal_effect(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """First-scorer W/D/L; both sides in same first minute are unknown, not an invented order."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    valid, _ = _eligible(tables, selected)
    teams = _subset(tables.team_matches, selected).reset_index()
    games = teams.loc[teams.game_id.isin(valid.index) & teams.scored_first.notna()]
    rows = [
        dict(is_home=home, scored_first=first, **_rates(group, n_bootstrap))
        for (home, first), group in games.groupby(["is_home", "scored_first"], observed=True)
    ]
    frame = pd.DataFrame(rows).set_index(["is_home", "scored_first"], drop=True)
    frame.attrs["n_zero_zero_or_ambiguous_rows"] = 2 * len(valid) - len(games)
    return _metadata(frame, _subset(tables.match_table, selected))


def first_goal_minute_distribution(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """First-goal bucket as a fraction of eligible games; 0:0 is a separate category."""
    selected = _selection(seasons)
    games, events = _eligible(tables, selected)
    first = events.groupby("game_id", sort=False).head(1)
    counts = first.minute_bucket.value_counts()
    rows = []
    for bucket in [*MINUTE_BUCKETS, "no_goal"]:
        count = (
            int(games.total_goals.eq(0).sum())
            if bucket == "no_goal"
            else int(counts.get(bucket, 0))
        )
        rate, lo, hi = rate_interval(count, len(games))
        rows.append(
            dict(
                minute_bucket=bucket, n=len(games), count=count, rate=rate, ci_lower=lo, ci_upper=hi
            )
        )
    return _metadata(
        pd.DataFrame(rows).set_index("minute_bucket", drop=True),
        _subset(tables.match_table, selected),
    )


def goal_timing(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Scorer's state before each goal, conditional on bucket; deterministic same-minute order."""
    selected = _selection(seasons)
    games, events = _eligible(tables, selected)
    context = games[["home_team_id", "away_team_id"]]
    sequence = []
    for game_id, group in events.groupby("game_id", sort=False):
        game = context.loc[game_id]
        sequence.extend(_sequenced(group, int(game.home_team_id), int(game.away_team_id)))
    frame = pd.DataFrame(sequence, columns=["minute_bucket", "score_before"])
    rows = []
    for bucket in MINUTE_BUCKETS:
        group = frame.loc[frame.minute_bucket.eq(bucket)]
        for state in STATES:
            count = int(group.score_before.eq(state).sum())
            rate, lo, hi = rate_interval(count, len(group))
            rows.append(
                dict(
                    minute_bucket=bucket,
                    score_before=state,
                    n=len(group),
                    count=count,
                    rate=rate,
                    ci_lower=lo,
                    ci_upper=hi,
                )
            )
    return _metadata(
        pd.DataFrame(rows).set_index(["minute_bucket", "score_before"], drop=True),
        _subset(tables.match_table, selected),
    )


def team_late_goals(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS, *, min_seasons: int = 5
) -> pd.DataFrame:
    """Club shares of second-half and minute-76+ goals, excluding first-half stoppage."""
    selected = _selection(seasons)
    if min_seasons < 1:
        raise ValueError("min_seasons must be positive")
    games, goals = _eligible(tables, selected)
    teams = _subset(tables.team_matches, selected).reset_index()
    teams = teams.loc[teams.game_id.isin(games.index)]
    labels = _subset(tables.team_seasons, selected).reset_index().sort_values("season_id")
    rows = []
    for team, group in teams.groupby("team_id"):
        if group.season_id.nunique() < min_seasons:
            continue
        events = goals.loc[goals.team_id.eq(team)]
        row = dict(
            team_id=team,
            team_name=labels.loc[labels.team_id.eq(team)].team_name.iloc[-1],
            n_seasons=group.season_id.nunique(),
            n_matches=len(group),
            n_goals=len(events),
        )
        for name, flag in {
            "second_half_goal_share": events.half.eq(2),
            "late_goal_share": events.half.eq(2) & events.minute.ge(76),
        }.items():
            rate, lo, hi = rate_interval(int(flag.sum()), len(events))
            row.update(
                {
                    name: rate,
                    f"{name}_n": int(flag.sum()),
                    f"{name}_ci_lower": lo,
                    f"{name}_ci_upper": hi,
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("team_id", drop=True)
    return _metadata(frame, _subset(tables.match_table, selected))


def penalty_owngoal_stats(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Scored penalties (not awarded/missed penalties) and own goals credited to beneficiaries."""
    selected = _selection(seasons)
    all_games = _subset(tables.match_table, selected)
    valid = all_games.loc[all_games.goals_available & (all_games.index != 864982)]
    goals = _subset(tables.goal_events, selected).reset_index()
    goals = goals.loc[goals.game_id.isin(valid.index)]
    teams = _subset(tables.team_matches, selected).reset_index()
    teams = teams.loc[teams.game_id.isin(valid.index)]
    rows = []
    for (season, team), group in teams.groupby(["season_id", "team_id"]):
        own = goals.loc[goals.game_id.isin(group.game_id) & goals.team_id.eq(team)]
        against = goals.loc[goals.game_id.isin(group.game_id) & ~goals.team_id.eq(team)]
        row = dict(season_id=season, team_id=team, n=len(group), n_goals=len(own))
        for label, event in (("for", own), ("against", against)):
            for kind, column in (("penalty", "is_penalty"), ("own_goal", "is_own_goal")):
                count = int(event[column].sum())
                row[f"{kind}_{label}_n"] = count
                row[f"{kind}_{label}_per_match"] = count / len(group)
        rate, lo, hi = rate_interval(int(own.is_penalty.sum()), len(own))
        row.update(
            penalty_goal_share=rate, penalty_goal_share_ci_lower=lo, penalty_goal_share_ci_upper=hi
        )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index(["season_id", "team_id"], drop=True)
    frame.attrs.update(
        n_excluded_matches=len(all_games) - len(valid),
        exclusions={"unreliable_or_missing_scorers": len(all_games) - len(valid)},
    )
    return frame


def penalty_owngoal_by_season(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """League totals count each credited goal once, not once for and once against."""
    frame = penalty_owngoal_stats(tables, seasons)
    rows = []
    for season, group in frame.groupby(level="season_id"):
        n = int(group.n.sum()) // 2
        row = dict(season_id=season, n=n, n_goals=int(group.n_goals.sum()))
        for kind in ("penalty", "own_goal"):
            count = int(group[f"{kind}_for_n"].sum())
            # Multiple goals per game are possible: Wilson describes share of goals, not games.
            share, lo, hi = rate_interval(count, row["n_goals"])
            row.update(
                {
                    f"{kind}_n": count,
                    f"{kind}_per_match": count / n,
                    f"{kind}_goal_share": share,
                    f"{kind}_goal_share_ci_lower": lo,
                    f"{kind}_goal_share_ci_upper": hi,
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows).set_index("season_id", drop=True)
    result.attrs = frame.attrs.copy()
    return result


def red_card_stats(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    red_cards: pd.DataFrame,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Long-form card counts, first-card timing and second-yellow share; association, not cause.

    Pass red_cards read_filtered by selected match IDs; this pure function does no I/O.
    Missing card minute stays in counts and second-yellow share, but not the timing split.
    """
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    matches = _subset(tables.match_table, selected)
    teams = _subset(tables.team_matches, selected).reset_index()
    cards = red_cards.loc[red_cards.game_id.isin(matches.index)].copy()
    counts = cards.groupby(["game_id", "team_id"]).size()
    first = cards.groupby(["game_id", "team_id"]).minute.min()
    missing = cards.loc[cards.minute.isna()].groupby(["game_id", "team_id"]).size()
    keys = pd.MultiIndex.from_frame(teams[["game_id", "team_id"]])
    opponent = pd.MultiIndex.from_frame(
        teams[["game_id", "opponent_id"]].rename(columns={"opponent_id": "team_id"})
    )
    teams["red_for"] = counts.reindex(keys, fill_value=0).to_numpy() > 0
    teams["red_against"] = counts.reindex(opponent, fill_value=0).to_numpy() > 0
    teams["first_red_minute"] = first.reindex(keys).to_numpy()
    teams["unknown_red_minute"] = missing.reindex(keys, fill_value=0).to_numpy() > 0
    rows = []
    for (own, other), group in teams.groupby(["red_for", "red_against"]):
        rows.append(
            dict(
                part="cards",
                group=f"{int(own)}:{int(other)}",
                season_id=0,
                **_rates(group, n_bootstrap),
            )
        )
    timed = teams.loc[teams.red_for & ~teams.unknown_red_minute].copy()
    timed["timing"] = np.where(timed.first_red_minute.le(60), "up_to_60", "after_60")
    for timing, group in timed.groupby("timing"):
        rows.append(
            dict(part="first_card_minute", group=timing, season_id=0, **_rates(group, n_bootstrap))
        )
    for season, games in matches.groupby("season_id"):
        events = cards.loc[cards.game_id.isin(games.index)]
        count = int(events.is_second_yellow.sum())
        rate, lo, hi = rate_interval(count, len(events))
        rows.append(
            dict(
                part="second_yellow",
                group="all",
                season_id=season,
                n=len(events),
                second_yellow_n=count,
                second_yellow_share=rate,
                second_yellow_share_ci_lower=lo,
                second_yellow_share_ci_upper=hi,
            )
        )
    frame = pd.DataFrame(rows).set_index(["part", "group", "season_id"], drop=True)
    frame.attrs.update(
        n_cards=len(cards),
        n_missing_minute=int(cards.minute.isna().sum()),
        n_timing_excluded_team_games=int((teams.red_for & teams.unknown_red_minute).sum()),
    )
    return frame
