"""Canonical-rule equivalence, paired futures, exact-zero targets and pilot regression."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_season_snapshot import morning, snapshot_inputs

from ligat.features.table_state import game_rows, historical_groups
from ligat.models.ordered_logit import OrderedFit
from ligat.simulate.engine import LeagueEngine
from ligat.simulate.leverage import (
    E0050,
    TARGETS,
    SeasonSimulator,
    conversion_fits,
    empirical_scores,
)
from ligat.simulate.rules import load_season_rules, rank_table, split_playoffs, sporting_table
from ligat.simulate.snapshot import season_snapshots, snapshot_as_of


def synthetic_simulator():
    rows, elo, rules = snapshot_inputs()
    snap = snapshot_as_of(rows, elo, rules, 8, morning(1))
    pools = {
        0: np.array([[1, 0], [2, 1]]),
        1: np.array([[0, 0], [1, 1]]),
        2: np.array([[0, 1], [1, 2]]),
    }
    return SeasonSimulator(snap, OrderedFit(2.0, -0.8, 0.2, 0.0, True, 0, False, 0.0), pools)


@pytest.mark.parametrize("sid", [8, 10])
@pytest.mark.parametrize("at_split", [False, True])
@pytest.mark.parametrize("with_lottery", [False, True])
def test_vector_ranks_match_canonical_on_hundreds_of_small_tied_tables(sid, at_split, with_lottery):
    rng = np.random.default_rng(1309)
    rule = replace(load_season_rules()[sid], n_teams=4, groups=(), regular_rounds=3, relegated=1)
    previous = {1: 2.0, 2: 1.0, 3: np.nan}  # Team 4 is also promoted.
    engine = LeagueEngine(rule, [1, 2, 3, 4], previous)
    paths = 80
    state = engine.empty(paths)
    scores = []
    for home, away in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)):
        hg, ag = rng.integers(0, 2, size=(2, paths), dtype=np.int16)
        scores.append((home, away, hg, ag))
        engine.add(state, home, away, hg, ag, regular=True)
    lottery = rng.random((paths, 4)) if with_lottery else None
    _, positions = engine.rank(state, at_split=at_split, lottery=lottery)
    for path in range(paths):
        participants = []
        for home, away, hg, ag in scores:
            participants.extend(
                [
                    dict(
                        team_id=home + 1,
                        opponent_id=away + 1,
                        goals_scored=int(hg[path]),
                        goals_conceded=int(ag[path]),
                    ),
                    dict(
                        team_id=away + 1,
                        opponent_id=home + 1,
                        goals_scored=int(ag[path]),
                        goals_conceded=int(hg[path]),
                    ),
                ]
            )
        games = pd.DataFrame(participants)
        table = pd.DataFrame(
            {
                "points": state["points"][path],
                "goal_difference": state["gf"][path] - state["ga"][path],
                "wins": state["wins"][path],
                "goals_for": state["gf"][path],
            },
            index=[1, 2, 3, 4],
        )
        keys = None if lottery is None else dict(zip([1, 2, 3, 4], lottery[path], strict=True))
        canonical = rank_table(
            table, rule, games, previous_positions=previous if at_split else None, lottery_keys=keys
        )
        np.testing.assert_array_equal(positions[path], canonical.position.reindex(engine.teams))


def test_same_seed_is_exact_other_seed_changes_estimates_and_counts_are_checked():
    simulator = synthetic_simulator()
    first = simulator.simulate(256, 44)
    second = simulator.simulate(256, 44)
    assert_frame_equal(first.features, second.features, check_exact=True)
    assert first.tie_counts == second.tie_counts
    other = simulator.simulate(256, 45)
    assert not first.features.equals(other.features)
    assert first.tie_counts["conservation_checked_paths"] == 2 * 3 * 256
    assert first.features.survival_direct_lev.isna().all()
    assert first.features.leverage_max.between(0, 1).all()
    for target in ("title", "cut", "survival"):
        frame = first.features
        np.testing.assert_array_equal(
            frame[f"{target}_lev_raw"], frame[f"{target}_p_win"] - frame[f"{target}_p_loss"]
        )
        np.testing.assert_array_equal(
            frame[f"{target}_lev"], frame[f"{target}_lev_raw"].clip(lower=0)
        )


def test_playoff_slots_and_no_playoff_seasons():
    simulator = synthetic_simulator()
    assert simulator.playoff_slots() == [(0, 1), (2, 3)]
    snap = simulator.snapshot
    plain = replace(snap, rules=replace(snap.rules, groups=()), groups=None)
    no_playoff = SeasonSimulator(plain, simulator.fit, simulator.pools).simulate(64)
    assert no_playoff.features.cut_p_win.isna().all()
    assert not no_playoff.features.before_split.any()


def test_empirical_training_excludes_validation_scores_and_documents_warmup():
    rows, _, _ = snapshot_inputs()
    mask = rows.game_id.eq(2)
    rows.loc[mask, "goals_scored"] = (~rows.loc[mask, "is_home"]).astype(int)
    rows.loc[mask, "goals_conceded"] = rows.loc[mask, "is_home"].astype(int)
    extras = rows.copy()
    extras["season_id"] = 11
    extras[["goals_scored", "goals_conceded"]] = 88
    combined = pd.concat([rows, extras])
    for sid in (8, 9, 10, 11):
        pools = empirical_scores(combined, sid)
        assert all((p < 88).all() for p in pools.values())


@pytest.mark.parametrize("sid", [8, 10, 11, 13, 14])
def test_true_score_replay_equals_sporting_table_in_every_format(real_tables, sid):
    rows = game_rows(real_tables)
    games = rows.loc[rows.season_id.eq(sid)]
    rule = load_season_rules()[sid]
    roster = sorted(games.team_id.unique())
    engine = LeagueEngine(rule, roster)
    state = engine.from_played(games, 1)
    mapping = historical_groups(games) if rule.groups else None
    groups = (
        None
        if mapping is None
        else np.array(
            [[next(i for i, g in enumerate(rule.groups) if g.name == mapping[t]) for t in roster]]
        )
    )
    actual = engine.totals_frame(state, groups)
    canonical = sporting_table(games, roster, rule, groups=mapping).reindex(roster)
    for column in actual:
        np.testing.assert_array_equal(
            actual[column], canonical[column], err_msg=f"season={sid};column={column}"
        )


def test_every_real_split_matches_canonical_and_only_documented_historical_differences(real_tables):
    rows = game_rows(real_tables)
    rules = load_season_rules()
    mismatches = set()
    for sid in range(11, 26):
        games = rows.loc[rows.season_id.eq(sid)]
        prior = rows.loc[rows.season_id.eq(sid - 1)]
        roster = sorted(games.team_id.unique())
        prior_rule = rules[sid - 1]
        previous = sporting_table(
            prior,
            sorted(prior.team_id.unique()),
            prior_rule,
            groups=historical_groups(prior) if prior_rule.groups else None,
        ).position.to_dict()
        regular = games.loc[games.stage.eq("regular")]
        engine = LeagueEngine(rules[sid], roster, previous)
        state = engine.from_played(regular, 1)
        lottery = np.arange(len(roster), dtype=float)[None, :]
        order, _ = engine.rank(state, at_split=True, lottery=lottery)
        groups = engine.membership(order)[0]
        mapping = {int(t): rules[sid].groups[int(groups[i])].name for i, t in enumerate(roster)}
        canonical = split_playoffs(
            sporting_table(regular, roster, rules[sid]),
            rules[sid],
            regular,
            previous_positions=previous,
            lottery_keys=dict(zip(roster, lottery[0], strict=True)),
        )
        assert mapping == canonical, (sid, mapping, canonical)
        historical = historical_groups(games)
        mismatches.update((sid, int(t)) for t in roster if mapping[t] != historical[t])
    assert mismatches == {(12, 2171), (12, 3595)}


@pytest.fixture(scope="module")
def leverage_inputs(real_tables):
    rows = game_rows(real_tables)
    elo = pd.read_parquet(
        "data/derived/elo/elo_E0050_prematch.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    fits = conversion_fits(
        pd.read_csv(
            "reports/experiments/EXP-001/combined/fold_scores.csv", float_precision="round_trip"
        )
    )
    return rows, elo, fits, load_season_rules()


def test_first_day_probabilities_reproduce_frozen_exp001(leverage_inputs):
    rows, elo, fits, rules = leverage_inputs
    frozen = pd.read_parquet(
        "data/derived/elo/exp001_predictions_combined.parquet",
        filters=[("season_id", "in", list(range(11, 26))), ("config_id", "==", E0050)],
    )
    maximum = 0.0
    for sid in range(11, 26):
        snapshot = next(season_snapshots(rows, elo, rules, sid))
        simulator = SeasonSimulator(snapshot, fits[sid], empirical_scores(rows, sid))
        for game in snapshot.remaining.loc[
            snapshot.remaining.as_of.eq(snapshot.as_of)
        ].itertuples():
            actual = simulator.probabilities(
                simulator.engine.lookup[game.home_team_id],
                simulator.engine.lookup[game.away_team_id],
            )
            saved = frozen.loc[
                frozen.game_id.eq(game.game_id), ["p_home", "p_draw", "p_away"]
            ].to_numpy(float)
            assert saved.shape == (1, 3)
            maximum = max(maximum, float(np.abs(actual - saved[0]).max()))
            np.testing.assert_allclose(actual, saved[0], atol=1e-12, rtol=0)
    print("A3_FIRST_DAY_MAX_PROBABILITY_ERROR", maximum)


@pytest.mark.slow
def test_all_development_dates_have_exact_zero_for_decided_objectives(leverage_inputs):
    rows, elo, fits, rules = leverage_inputs
    frozen = pd.read_parquet(
        "data/derived/prematch/table_state_v2.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    checks = 0
    days = 0
    for sid in range(8, 26):
        pools = empirical_scores(rows, sid)
        for snapshot in season_snapshots(rows, elo, rules, sid):
            frame = SeasonSimulator(snapshot, fits[sid], pools).simulate(128).features
            flags = frozen.reindex(frame.index)
            for target, flag in (
                ("title", "title_decided"),
                ("cut", "cut_decided"),
                ("survival", "relegation_decided"),
            ):
                mask = flags[flag] & frame[f"{target}_p_win"].notna()
                part = frame.loc[mask]
                valid = part[f"{target}_lev_raw"].eq(0)
                probabilities = part[[f"{target}_p_{x}" for x in ("win", "draw", "loss")]]
                valid &= probabilities.nunique(axis=1).eq(1) & probabilities.iloc[:, 0].isin(
                    [0.0, 1.0]
                )
                assert valid.all(), (
                    sid,
                    str(snapshot.as_of),
                    target,
                    part.loc[~valid].to_dict("index"),
                )
                checks += len(part)
            days += 1
        print("A3_ZERO_SEASON", sid, "cumulative_dates", days, "checks", checks, flush=True)
    assert days == 1668


@pytest.mark.slow
def test_season25_matches_pilot_probabilities_and_paired_mcse(leverage_inputs):
    rows, elo, fits, rules = leverage_inputs
    sid = 25
    pools = empirical_scores(rows, sid)
    frames = [
        SeasonSimulator(snap, fits[sid], pools).simulate(2000).features
        for snap in season_snapshots(rows, elo, rules, sid)
    ]
    actual = pd.concat(frames).sort_index()
    frozen = (
        pd.read_parquet(
            "data/reports/m1_pilot/out/S3_leverage.parquet", filters=[("season_id", "==", 25)]
        )
        .set_index(["game_id", "team_id"])
        .sort_index()
    )
    maximum = 0.0
    for target in TARGETS:
        if target == "survival_direct":
            continue
        for suffix in ("p_win", "p_draw", "p_loss", "mc_se"):
            col = f"{target}_{suffix}"
            values, expected = actual[col], frozen[col]
            maximum = max(maximum, float((values - expected).abs().max()))
            np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0, equal_nan=True)
    print("A3_PILOT_SEASON25_MAX_ERROR", maximum, "rows", len(actual), flush=True)
