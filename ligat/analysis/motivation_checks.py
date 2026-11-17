"""Notebook-visible engineering evidence, recomputed from approved frozen inputs."""

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pandas.testing import assert_frame_equal, assert_series_equal  # type: ignore[import-untyped]

from ligat.analysis.motivation_review import MotivationReview
from ligat.analysis.prepare import AnalysisTables
from ligat.features.elo import EloParams, replay_elo_season
from ligat.features.table_state import historical_groups
from ligat.simulate.engine import LeagueEngine
from ligat.simulate.leverage import SCORE_TARGETS, TARGETS, SeasonSimulator, empirical_scores
from ligat.simulate.rules import PlayoffGroup, rank_table, split_playoffs, sporting_table
from ligat.simulate.snapshot import season_snapshots, snapshot_as_of


def choose_july_game(frame: pd.DataFrame) -> tuple[int, pd.Timestamp]:
    """Select the highest-importance July-2020 lower fixture; break ties chronologically."""
    july = frame.loc[
        frame.season_id.eq(21)
        & frame.as_of.dt.year.eq(2020)
        & frame.as_of.dt.month.eq(7)
        & frame.stage.eq("lower")
    ]
    if july.empty:
        raise ValueError("No July 2020 survival example")
    chosen = (
        july.reset_index()
        .sort_values(
            ["leverage_max", "as_of", "game_id", "team_id"], ascending=[False, True, True, True]
        )
        .iloc[0]
    )
    return int(chosen.game_id), pd.Timestamp(chosen.as_of)


def live_example(review: MotivationReview) -> dict[str, Any]:
    """Rerun the focal morning twice and require exact agreement with frozen M1 rows."""
    game, day = choose_july_game(review.frame)
    snapshot = snapshot_as_of(review.participants, review.elo, review.rules, 21, day)
    pools = empirical_scores(review.participants, 21)
    started = time.perf_counter()
    first = SeasonSimulator(snapshot, review.fits[21], pools).simulate(2000, review.seed).features
    second = SeasonSimulator(snapshot, review.fits[21], pools).simulate(2000, review.seed).features
    saved = review.motivation.reindex(first.index).loc[:, first.columns]
    assert_frame_equal(first, saved, check_exact=True)
    assert_frame_equal(first, second, check_exact=True)
    return dict(
        game_id=game,
        as_of=day,
        snapshot=snapshot,
        features=first.loc[[game]],
        artifact_error=0.0,
        repeat_error=0.0,
        seconds=time.perf_counter() - started,
        n=len(first),
        n_games=first.index.get_level_values("game_id").nunique(),
        n_seasons=1,
    )


def tie_example(review: MotivationReview) -> pd.DataFrame:
    """Use real display names for a clearly hypothetical incumbent and two promoted clubs."""
    teams = sorted(review.names)[:3]
    rule = replace(
        review.rules[14],
        n_teams=3,
        groups=(PlayoffGroup("upper", 2, 1), PlayoffGroup("lower", 1, 0)),
    )
    table = pd.DataFrame(
        0,
        index=pd.Index(teams, name="team_id"),
        columns=["points", "goal_difference", "wins", "goals_for"],
    )
    previous = {teams[0]: 1.0}
    lottery = {teams[0]: 0.9, teams[1]: 0.7, teams[2]: 0.2}
    ranked = rank_table(table, rule, previous_positions=previous, lottery_keys=lottery)
    groups = split_playoffs(table, rule, previous_positions=previous, lottery_keys=lottery)
    result = ranked.reset_index()
    result["previous"] = result.team_id.map(previous)
    result["lottery"] = result.team_id.map(lottery)
    result["role"] = ["incumbent" if t in previous else "promoted" for t in result.team_id]
    result["stage"] = result.team_id.map(groups)
    if ranked.index.tolist() != [teams[0], teams[2], teams[1]]:
        raise AssertionError("Incumbent/promoted/lottery example failed")
    return result


def snapshot_diagnostic(review: MotivationReview) -> dict[str, Any]:
    """Check every morning against frozen M0, replay Elo independently, and hide future playoffs."""
    rows = review.participants
    tables = AnalysisTables(
        pd.DataFrame(),
        rows.set_index(["team_id", "game_id"]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    params = EloParams(k=10, home_advantage=30, promoted_initialization="fixed_1450")
    columns = ["points", "position", "goal_difference", "games_played"]
    days = compared = split_seasons = 0
    elo_error = 0.0
    for sid in range(8, 26):
        ratings = (
            review.elo.loc[review.elo.season_id.eq(sid)]
            .reset_index()
            .sort_values(["as_of", "game_id"])
        )
        initial = ratings.drop_duplicates("team_id").set_index("team_id").elo_pre
        replay = replay_elo_season(
            tables, review.prematch.nothing_to_play, season_id=sid, elo_start=initial, params=params
        ).prematch.reset_index()
        frozen = review.table_state.loc[review.table_state.season_id.eq(sid)]
        checked_split = False
        for snapshot in season_snapshots(rows, review.elo, review.rules, sid):
            targets = frozen.loc[frozen.as_of.eq(snapshot.as_of)].reset_index().set_index("team_id")
            assert_frame_equal(
                snapshot.table.reindex(targets.index)[columns], targets[columns], check_exact=True
            )
            future = replay.loc[replay.as_of.ge(snapshot.as_of)].sort_values(["as_of", "game_id"])
            expected = (
                future.drop_duplicates("team_id")
                .set_index("team_id")
                .elo_pre.reindex(snapshot.elo.index)
            )
            assert_series_equal(snapshot.elo, expected, check_exact=False, atol=1e-9, rtol=0)
            error = (snapshot.elo - expected).abs().max()
            elo_error = max(elo_error, float(error))
            if snapshot.phase == "regular":
                assert snapshot.remaining.stage.eq("regular").all()
                assert snapshot.groups is None
            elif not checked_split:
                all_playoffs = set(
                    rows.loc[rows.season_id.eq(sid) & rows.stage.ne("regular"), "game_id"]
                )
                assert all_playoffs <= set(snapshot.remaining.game_id)
                before = snapshot_as_of(
                    rows, review.elo, review.rules, sid, snapshot.as_of - pd.DateOffset(days=1)
                )
                assert all_playoffs.isdisjoint(before.remaining.game_id)
                split_seasons += 1
                checked_split = True
            days += 1
            compared += len(targets)
    assert days == 1668 and compared == 8630
    return dict(
        passed=True,
        n=compared,
        n_games=compared // 2,
        n_seasons=18,
        dates=days,
        max_table_error=0.0,
        max_elo_error=elo_error,
        split_seasons=split_seasons,
    )


def engine_diagnostic(review: MotivationReview) -> dict[str, Any]:
    """Compare 640 ranking cases, real formats/splits and canonical first-day probabilities."""
    checked = 0
    for sid in (8, 10):
        rule = replace(review.rules[sid], n_teams=4, groups=(), regular_rounds=3, relegated=1)
        previous = {1: 2.0, 2: 1.0, 3: np.nan}
        for at_split in (False, True):
            for with_lottery in (False, True):
                rng = np.random.default_rng(1309)
                engine = LeagueEngine(rule, [1, 2, 3, 4], previous)
                state = engine.empty(80)
                fixtures = []
                for home, away in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)):
                    hg, ag = rng.integers(0, 2, size=(2, 80), dtype=np.int16)
                    fixtures.append((home, away, hg, ag))
                    engine.add(state, home, away, hg, ag, regular=True)
                lottery = rng.random((80, 4)) if with_lottery else None
                _, rank = engine.rank(state, at_split=at_split, lottery=lottery)
                for path in range(80):
                    games = []
                    for h, a, hg, ag in fixtures:
                        games += [
                            dict(
                                team_id=h + 1,
                                opponent_id=a + 1,
                                goals_scored=int(hg[path]),
                                goals_conceded=int(ag[path]),
                            ),
                            dict(
                                team_id=a + 1,
                                opponent_id=h + 1,
                                goals_scored=int(ag[path]),
                                goals_conceded=int(hg[path]),
                            ),
                        ]
                    table = pd.DataFrame(
                        dict(
                            points=state["points"][path],
                            goal_difference=state["gf"][path] - state["ga"][path],
                            wins=state["wins"][path],
                            goals_for=state["gf"][path],
                        ),
                        index=[1, 2, 3, 4],
                    )
                    keys = (
                        None
                        if lottery is None
                        else dict(zip([1, 2, 3, 4], lottery[path], strict=True))
                    )
                    canonical = rank_table(
                        table,
                        rule,
                        pd.DataFrame(games),
                        previous_positions=previous if at_split else None,
                        lottery_keys=keys,
                    )
                    np.testing.assert_array_equal(
                        rank[path], canonical.position.reindex(engine.teams)
                    )
                    checked += 1
    rows = review.participants
    for sid in (8, 10, 11, 13, 14):
        games = rows.loc[rows.season_id.eq(sid)]
        rule = review.rules[sid]
        roster = sorted(games.team_id.unique())
        engine = LeagueEngine(rule, roster)
        mapping = historical_groups(games) if rule.groups else None
        groups = (
            None
            if mapping is None
            else np.array(
                [
                    [
                        next(i for i, g in enumerate(rule.groups) if g.name == mapping[t])
                        for t in roster
                    ]
                ]
            )
        )
        actual = engine.totals_frame(engine.from_played(games, 1), groups)
        canonical = sporting_table(games, roster, rule, groups=mapping).reindex(roster)
        for col in actual:
            np.testing.assert_array_equal(actual[col], canonical[col])
    differences: list[tuple[int, int]] = []
    probability_error = 0.0
    for sid in range(11, 26):
        games = rows.loc[rows.season_id.eq(sid)]
        prior = rows.loc[rows.season_id.eq(sid - 1)]
        roster = sorted(games.team_id.unique())
        prior_rule = review.rules[sid - 1]
        previous = sporting_table(
            prior,
            sorted(prior.team_id.unique()),
            prior_rule,
            groups=historical_groups(prior) if prior_rule.groups else None,
        ).position.to_dict()
        regular = games.loc[games.stage.eq("regular")]
        engine = LeagueEngine(review.rules[sid], roster, previous)
        lottery = np.arange(len(roster), dtype=float)[None, :]
        order, _ = engine.rank(engine.from_played(regular, 1), at_split=True, lottery=lottery)
        groups = engine.membership(order)[0]
        mapping = {
            int(t): review.rules[sid].groups[int(groups[i])].name for i, t in enumerate(roster)
        }
        canonical = split_playoffs(
            sporting_table(regular, roster, review.rules[sid]),
            review.rules[sid],
            regular,
            previous_positions=previous,
            lottery_keys=dict(zip(roster, lottery[0], strict=True)),
        )
        assert canonical == mapping
        historical = historical_groups(games)
        differences.extend((sid, int(t)) for t in roster if mapping[t] != historical[t])
        snapshot = next(season_snapshots(rows, review.elo, review.rules, sid))
        simulator = SeasonSimulator(snapshot, review.fits[sid], empirical_scores(rows, sid))
        for fixture in snapshot.remaining.loc[
            snapshot.remaining.as_of.eq(snapshot.as_of)
        ].itertuples():
            expected = review.frame.loc[
                (fixture.game_id, fixture.home_team_id), ["p_home", "p_draw", "p_away"]
            ].to_numpy(float)
            actual = simulator.probabilities(
                engine.lookup[fixture.home_team_id], engine.lookup[fixture.away_team_id]
            )
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
            probability_error = max(probability_error, float(np.abs(actual - expected).max()))
    assert set(differences) == {(12, 2171), (12, 3595)}
    return dict(
        passed=True,
        random_tables=checked,
        replayed_formats=5,
        n=8630,
        n_games=4315,
        n_seasons=18,
        split_differences=[dict(season_id=s, team_name=review.names[t]) for s, t in differences],
        first_day_error=probability_error,
    )


def zero_diagnostic(review: MotivationReview) -> dict[str, Any]:
    """Scan all 1668 mornings at N=128; no target is zeroed by consulting its flag."""
    checks = days = rows = 0
    paths = 0
    for sid in range(8, 26):
        pools = empirical_scores(review.participants, sid)
        for snapshot in season_snapshots(review.participants, review.elo, review.rules, sid):
            result = SeasonSimulator(snapshot, review.fits[sid], pools).simulate(128, review.seed)
            frame = result.features
            flags = review.prematch.reindex(frame.index)
            for target, flag in (
                ("title", "title_decided"),
                ("cut", "cut_decided"),
                ("survival", "relegation_decided"),
            ):
                part = frame.loc[flags[flag] & frame[f"{target}_p_win"].notna()]
                probabilities = part[[f"{target}_p_{x}" for x in ("win", "draw", "loss")]]
                valid = (
                    part[f"{target}_lev_raw"].eq(0)
                    & probabilities.nunique(axis=1).eq(1)
                    & probabilities.iloc[:, 0].isin([0.0, 1.0])
                )
                if not valid.all():
                    raise AssertionError(
                        f"Decided target failed {sid}/{snapshot.as_of}/{target}: "
                        f"{part.loc[~valid].to_dict('index')}"
                    )
                checks += len(part)
            days += 1
            rows += len(frame)
            paths += result.tie_counts["conservation_checked_paths"]
    assert days == 1668 and rows == 8630
    return dict(
        passed=True,
        n=rows,
        n_games=rows // 2,
        n_seasons=18,
        dates=days,
        N=128,
        target_checks=checks,
        conservation_paths=paths,
    )


def pilot_diagnostic(frame: pd.DataFrame, pilot: pd.DataFrame) -> dict[str, Any]:
    """Require exact saved estimates on all fifteen validation seasons, without old A4 files."""
    current = frame.loc[frame.season_id.ge(11)].sort_index()
    reference = pilot.set_index(["game_id", "team_id"]).sort_index()
    assert current.index.equals(reference.index) and len(current) == 7442
    maximum = 0.0
    for target in TARGETS:
        for suffix in ("p_win", "p_draw", "p_loss", "mc_se"):
            key = f"{target}_{suffix}"
            np.testing.assert_array_equal(current[key], reference[key])
            maximum = max(maximum, float((current[key] - reference[key]).abs().max()))
    score = reference[[f"{t}_lev_wl" for t in SCORE_TARGETS]].max(axis=1)
    np.testing.assert_array_equal(current.leverage_max, score)
    return dict(passed=True, max_error=maximum, n=7442, n_games=3721, n_seasons=15)
