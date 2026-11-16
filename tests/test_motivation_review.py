"""Hand-derived stratified estimates, season clusters and exhaustive zero cases."""

import numpy as np
import pandas as pd
import pytest

from ligat.analysis.motivation_review import (
    SEASON_BLOCKS,
    alive_contrast,
    block_rows,
    cached_diagnostic,
    closed_game_period_review,
    closed_game_review,
    closed_game_rows,
    compare_tie_rules,
    conditional_summary,
    joint_strata_slopes,
    live_game_rows,
    morning_example,
    naive_paired_se_comparison,
    near_zero_dead_contrast,
    other_only_contrast,
    past_elo_endpoints,
    season_bootstrap_weights,
    select_mc_example,
    stage_residual_review,
    stratified_decomposition,
    typical_difference,
    verify_rounded_claims,
    within_strata_contrast,
    within_strata_slope,
    zero_partition,
)


def contrast_data():
    return pd.DataFrame(
        dict(
            game_id=range(8),
            season_id=[11, 12, 11, 12, 11, 11, 12, 12],
            stratum=["a"] * 4 + ["b"] * 3 + ["c"],
            y=[2.0, 4.0, 0.0, 2.0, 10.0, 4.0, 6.0, 100.0],
            selected=[True, True, False, False, True, False, False, True],
        )
    )


def test_stratified_contrast_matches_hand_result_and_counts_excluded_stratum():
    data = contrast_data()
    result = within_strata_contrast(data, data.selected, "y", repeats=100, seed=8)
    # a: 3-1=2, weight 2; b: 10-5=5, weight 1; c has no comparison.
    assert result["estimate"] == pytest.approx(3.0, abs=1e-12)
    assert (result["n"], result["n_used"], result["n_excluded"]) == (4, 3, 1)


def test_decomposition_reconstructs_contrast_and_stratum_contributions():
    data = contrast_data()
    result, layers = stratified_decomposition(data, data.selected, "y", repeats=100, seed=8)
    old = within_strata_contrast(data, data.selected, "y", repeats=100, seed=8)
    assert result["dead_mean"] == pytest.approx(16 / 3)
    assert result["control_mean"] == pytest.approx(7 / 3)
    assert result["estimate"] == pytest.approx(old["estimate"], abs=1e-12)
    assert result["dead_mean"] - result["control_mean"] == pytest.approx(3, abs=1e-12)
    assert layers.contribution.sum() == pytest.approx(3, abs=1e-12)
    assert result["se"] == pytest.approx(old["se"], abs=1e-12)
    assert result["dead_mean_se"] > 0 and result["control_mean_se"] > 0


def test_sensitivity_excludes_nine_but_keeps_ten_comparison_rows():
    rows = []
    for layer, controls in (("nine", 9), ("ten", 10)):
        for i in range(controls + 1):
            rows.append(
                dict(
                    game_id=len(rows),
                    season_id=11 + i % 2,
                    stratum=layer,
                    y=3.0 if i == 0 else 1.0,
                    selected=i == 0,
                )
            )
    frame = pd.DataFrame(rows)
    result, layers = stratified_decomposition(
        frame, frame.selected, "y", minimum_comparison=10, repeats=20
    )
    assert (result["n_used"], result["n_excluded"]) == (1, 1)
    assert layers.set_index("stratum").included.to_dict() == {"nine": False, "ten": True}
    assert result["estimate"] == 2
    with pytest.raises(ValueError):
        stratified_decomposition(frame, frame.selected, "y", minimum_comparison=9)


def synthetic_closed_games():
    rows = []
    # First fixture is both one-dead and alive-relegation: the first priority wins.
    for game, dead, close, goals, expected, draw in (
        (1, (True, False), (False, True), (1, 1), (1.2, 1.3), 0.3),
        (2, (True, True), (True, True), (2, 1), (1.0, 1.1), 0.25),
        (3, (False, True), (False, False), (0, 0), (0.8, 0.9), 0.4),
        (4, (False, False), (False, False), (3, 1), (1.4, 1.5), 0.2),
    ):
        for side in range(2):
            rows.append(
                dict(
                    game_id=game,
                    team_id=10 + side,
                    opponent_id=11 - side,
                    season_id=11 + game % 2,
                    stratum="regular / 7+",
                    is_home=side == 0,
                    nothing_to_play=dead[side],
                    releg_close=close[side],
                    L2=float(not dead[side]),
                    L3=float(not dead[side]),
                    L5=float(not dead[side]),
                    Lc=0.5 * (not dead[side]),
                    gf=goals[side],
                    ga=goals[1 - side],
                    lambda_home=expected[0],
                    lambda_away=expected[1],
                    p_draw=draw,
                )
            )
    return pd.DataFrame(rows).set_index(["game_id", "team_id"])


def test_closed_game_types_are_exhaustive_disjoint_and_obey_priority():
    games = closed_game_rows(synthetic_closed_games())
    assert len(games) == 4 and games.index.get_level_values("game_id").is_unique
    assert games.game_type.tolist() == ["relegation_alive", "both_dead", "one_dead", "other"]
    assert games.one_dead.sum() == 2


def test_closed_game_goal_and_draw_residuals_match_hand_calculation():
    result = closed_game_review(synthetic_closed_games(), seed=7, repeats=20)
    rows = result["raw"].set_index("game_type")
    assert rows.loc["relegation_alive", "res_total_goals"] == pytest.approx(2 - 2.5)
    assert rows.loc["relegation_alive", "res_draw"] == pytest.approx(1 - 0.3)
    assert rows.loc["both_dead", "res_total_goals"] == pytest.approx(3 - 2.1)
    assert rows.loc["both_dead", "res_draw"] == pytest.approx(0 - 0.25)
    assert result["adjusted"].n.sum() == 6
    assert not result["adjusted"].game_type.eq("other").any()


def test_bottom_line_claim_matches_at_display_rounding():
    result = verify_rounded_claims(
        {"effect": "−0.36", "share": "3.6"}, {"effect": (-0.36333, 2), "share": (3.60118, 1)}
    )
    assert result.matches.all()


def test_bottom_line_claim_fails_on_different_value_or_missing_claim():
    with pytest.raises(AssertionError, match="mismatch"):
        verify_rounded_claims({"effect": "−0.35"}, {"effect": (-0.36333, 2)})
    with pytest.raises(AssertionError, match="Missing"):
        verify_rounded_claims({}, {"effect": (-0.36333, 2)})


def test_other_only_contrast_excludes_other_types_and_unmatched_strata():
    games = pd.DataFrame(
        dict(
            game_id=range(8),
            season_id=[11, 12, 11, 12, 11, 12, 11, 12],
            stratum=["a", "a", "a", "a", "b", "b", "a", "b"],
            game_type=[
                "both_dead",
                "both_dead",
                "other",
                "other",
                "both_dead",
                "one_dead",
                "relegation_alive",
                "one_dead",
            ],
            y=[2.0, 4.0, 0.0, 2.0, 100.0, -999.0, 999.0, -999.0],
        )
    )
    result = other_only_contrast(games, "both_dead", "y", seed=7, repeats=100)
    # Matched a: selected mean 3 minus other mean 1. b has no other fixture.
    assert result["estimate"] == pytest.approx(2.0, abs=1e-12)
    assert (result["n"], result["n_used"], result["n_excluded"]) == (3, 2, 1)
    changed = games.copy()
    changed.loc[changed.game_type.isin(["one_dead", "relegation_alive"]), "y"] = 1e9
    second = other_only_contrast(changed, "both_dead", "y", seed=7, repeats=100)
    assert second == result
    with pytest.raises(ValueError):
        other_only_contrast(games, "other", "y", seed=7)


def test_stage_review_clean_population_and_residuals_match_hand_calculation():
    frame = synthetic_closed_games()
    frame["stage"] = "regular"
    frame.loc[frame.index.get_level_values("game_id").isin([2, 4]), "stage"] = "upper"
    # Give the sole clean game a recent season, so the recent counts/means are tested too.
    frame.loc[frame.index.get_level_values("game_id") == 4, "season_id"] = 21
    result = stage_residual_review(frame, seed=7, repeats=100).set_index(["stage", "population"])
    upper = result.loc[("upper", "all_games")]
    assert upper.n_games == 2
    assert upper.goals_residual == pytest.approx(((3 - 2.1) + (4 - 2.9)) / 2)
    assert upper.draw_residual == pytest.approx((-0.25 - 0.2) / 2)
    assert upper.last5_n_games == 1
    assert upper.last5_goals_residual == pytest.approx(4 - 2.9)
    assert upper.last5_draw_residual == pytest.approx(-0.2)
    clean = result.loc[("upper", "clean")]
    assert clean.n_games == 1 and clean.goals_residual == pytest.approx(1.1)
    # Regular contains an alive relegation fight and a one-dead game; neither is clean.
    assert result.loc[("regular", "clean"), "n_games"] == 0
    assert pd.isna(result.loc[("middle", "clean"), "goals_residual"])
    assert pd.isna(clean.goals_se)  # One observed season cannot identify between-season SE.


def period_game_data():
    parts = []
    for sid, offset in ((10, 1000), (11, 0), (12, 1), (21, 3), (22, 5)):
        part = synthetic_closed_games().reset_index()
        part["game_id"] += sid * 100
        part["season_id"] = sid
        part["gf"] += offset
        part["ga"] += offset
        parts.append(part)
    return pd.concat(parts, ignore_index=True).set_index(["game_id", "team_id"])


def test_period_residuals_exclude_other_years_and_match_hand_means():
    frame = period_game_data()
    result = closed_game_period_review(frame, seed=8, repeats=100).set_index(
        ["period", "game_type"]
    )
    first, recent = result.loc[("first10", "other")], result.loc[("last5", "other")]
    assert first.n_games == recent.n_games == 2
    assert first.actual_goals == pytest.approx((4 + 6) / 2)
    assert first.expected_goals == pytest.approx(2.9)
    assert first.res_total_goals == pytest.approx(2.1)
    assert first.res_draw == pytest.approx(-0.2)
    assert recent.actual_goals == pytest.approx((10 + 14) / 2)
    assert recent.res_total_goals == pytest.approx(9.1)
    changed = frame.copy()
    changed.loc[changed.season_id.ge(21), "gf"] += 100
    second = closed_game_period_review(changed, seed=8, repeats=100).set_index(
        ["period", "game_type"]
    )
    pd.testing.assert_frame_equal(result.loc["first10"], second.loc["first10"])


def test_period_bootstrap_only_samples_its_declared_seasons(monkeypatch):
    calls = []
    original = season_bootstrap_weights

    def spy(years, repeats, seed):
        expected = tuple(years)
        assert expected in (tuple(range(11, 21)), tuple(range(21, 26)))
        weights = original(years, repeats, seed)
        assert tuple(weights.columns) == expected
        assert weights.sum(axis=1).eq(len(expected)).all()
        calls.append(expected)
        return weights

    monkeypatch.setattr("ligat.analysis.motivation_review.season_bootstrap_weights", spy)
    closed_game_period_review(period_game_data(), seed=8, repeats=100)
    assert calls.count(tuple(range(11, 21))) == calls.count(tuple(range(21, 26))) == 8


@pytest.mark.parametrize("wrong_key", [None, "goals", "draw", "both", "one"])
def test_point_four_period_numbers_match_and_reject_each_wrong_claim(wrong_key):
    claims = {"goals": "0.18", "draw": "12", "both": "0.8", "one": "1.1"}
    values = {
        "goals": (0.177837, 2),
        "draw": (12.287626, 0),
        "both": (0.833435, 1),
        "one": (1.105213, 1),
    }
    if wrong_key is None:
        assert verify_rounded_claims(claims, values).matches.all()
    else:
        claims[wrong_key] = {"goals": "0.19", "draw": "11", "both": "0.9", "one": "1.2"}[wrong_key]
        with pytest.raises(AssertionError, match="mismatch"):
            verify_rounded_claims(claims, values)


@pytest.mark.parametrize("wrong", [None, "0.14", "0.03", "0.13", "0.23"])
def test_new_bottom_line_numbers_include_rounding_and_reject_each_wrong_claim(wrong):
    claims = {"dead": "0.15", "recent_dead": "0.02", "upper": "0.12", "lower": "0.22"}
    values = {
        "dead": (0.1517, 2),
        "recent_dead": (0.0193, 2),
        "upper": (0.118071, 2),
        "lower": (0.223988, 2),
    }
    if wrong is None:
        assert verify_rounded_claims(claims, values).matches.all()
    else:
        key = dict(zip(["0.14", "0.03", "0.13", "0.23"], claims, strict=True))[wrong]
        claims[key] = wrong
        with pytest.raises(AssertionError, match="mismatch"):
            verify_rounded_claims(claims, values)


def test_bootstrap_is_repeatable_and_entire_game_pairs_share_season_weight():
    first = season_bootstrap_weights([11, 12], 100, 55)
    second = season_bootstrap_weights([11, 12], 100, 55)
    pd.testing.assert_frame_equal(first, second)
    assert first.sum(axis=1).eq(2).all()
    seasons = [11, 11, 12, 12]
    row_weights = first.loc[:, seasons].to_numpy()
    np.testing.assert_array_equal(row_weights[:, 0], row_weights[:, 1])
    np.testing.assert_array_equal(row_weights[:, 2], row_weights[:, 3])
    data = contrast_data()
    a = within_strata_contrast(data, data.selected, "y", repeats=100, seed=55)
    b = within_strata_contrast(data, data.selected, "y", repeats=100, seed=55)
    assert a["se"] == b["se"]


def test_all_fifteen_seasons_and_sixteen_strata_are_counted_without_code_overflow():
    rows = []
    for sid in range(11, 26):
        for layer in range(16):
            for selected in (False, True):
                rows.append(
                    dict(
                        game_id=len(rows),
                        season_id=sid,
                        stratum=f"s{layer}",
                        selected=selected,
                        y=float(sid + 2 * selected),
                    )
                )
    frame = pd.DataFrame(rows)
    result = within_strata_contrast(frame, frame.selected, "y", repeats=10)
    assert result["estimate"] == pytest.approx(2.0, abs=1e-12)
    assert result["n_used"] == 15 * 16


def test_within_strata_slope_matches_hand_centered_covariances():
    frame = pd.DataFrame(
        dict(
            game_id=range(5),
            season_id=[11, 11, 12, 11, 12],
            stratum=["a", "a", "a", "b", "b"],
            x=[0.0, 1.0, 2.0, 1.0, 3.0],
            res_pts=[0.0, 2.0, 4.0, 10.0, 16.0],
        )
    )
    # Within a: xx=2, xy=4; within b: xx=2, xy=6.
    result = within_strata_slope(frame, "x", repeats=100)
    assert result["estimate"] == pytest.approx(2.5, abs=1e-12)
    assert result["n_used"] == 5


def test_zero_decomposition_is_disjoint_exhaustive_and_preserves_mc_residue():
    frame = pd.DataFrame(
        dict(
            nothing_to_play=[False] * 4 + [True],
            leverage_max=[0.0] * 5,
            title_decided=[True, True, False, False, True],
            cut_decided=[True] * 5,
            relegation_decided=[True, False, True, True, True],
            places_decided=[False, True, True, True, True],
            survival_direct_lev=[np.nan, 0.3, np.nan, np.nan, np.nan],
        )
    )
    for target in ("title", "cut", "survival"):
        frame[f"{target}_lev_raw"] = 0.0
        frame[f"{target}_mc_se"] = 0.0
    frame.loc[3, "title_mc_se"] = 0.01
    result = zero_partition(frame)
    assert result.zero_group.tolist() == [
        "places_only",
        "direct_only",
        "practical_zero",
        "other_mc",
    ]
    assert len(result) == 4 and result.index.is_unique


def test_naive_mc_error_matches_independent_binomial_formula():
    frame = pd.DataFrame(dict(game_id=[1], season_id=[11], N=[100]))
    for target in ("title", "cut", "survival"):
        frame[f"{target}_p_win"] = 0.8
        frame[f"{target}_p_loss"] = 0.2
        frame[f"{target}_mc_se"] = 0.02
    result = naive_paired_se_comparison(frame)
    expected = np.sqrt((0.8 * 0.2 + 0.2 * 0.8) / 100) / 0.02
    assert result["median"].to_numpy() == pytest.approx([expected] * 3, abs=1e-12)


def test_diagnostic_cache_requires_matching_inputs_and_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return {"passed": True, "n": 4}

    path = tmp_path / "data/reports/T-013/nb04_cache"
    first = cached_diagnostic(path, "test", {"input": "a" * 64}, compute, parameters={"N": 128})
    second = cached_diagnostic(path, "test", {"input": "a" * 64}, compute, parameters={"N": 128})
    assert not first["cache_hit"] and second["cache_hit"] and len(calls) == 1
    assert first["computed_at"] == second["computed_at"]
    cached_diagnostic(path, "test", {"input": "b" * 64}, compute, parameters={"N": 128})
    cached_diagnostic(path, "test", {"input": "b" * 64}, compute, parameters={"N": 256})
    assert len(calls) == 3


def test_alive_contrast_excludes_dead_rows_from_both_sides():
    frame = contrast_data().assign(nothing_to_play=False)
    frame.loc[7, "nothing_to_play"] = True
    frame.loc[7, "stratum"] = "a"
    frame.loc[7, "y"] = 999.0
    result = alive_contrast(frame, frame.selected, "y", repeats=20)
    assert result["estimate"] == pytest.approx(3.0, abs=1e-12)
    assert result["n"] == result["n_used"] == 3


def test_near_zero_vs_dead_excludes_high_importance_alive_rows():
    frame = pd.DataFrame(
        dict(
            game_id=range(7),
            season_id=[11, 12, 11, 12, 11, 12, 11],
            stratum=["a"] * 4 + ["b"] * 3,
            leverage_max=[0.01, 0.005, 0, 0, 0.01, 0, 0.8],
            nothing_to_play=[False, False, True, True, False, True, False],
            y=[4.0, 6.0, 0.0, 2.0, 10.0, 4.0, 999.0],
        )
    )
    # a: near mean 5 minus dead mean 1, weight 2; b: 10-4, weight 1.
    result = near_zero_dead_contrast(frame, "y", repeats=20)
    assert result["estimate"] == pytest.approx(14 / 3, abs=1e-12)
    assert result["n"] == 3


def test_live_game_requires_both_teams_alive():
    frame = pd.DataFrame(
        dict(game_id=[1, 1, 2, 2, 3, 3], nothing_to_play=[False, False, False, True, True, True])
    )
    assert live_game_rows(frame).game_id.tolist() == [1, 1]


def test_joint_slopes_equal_manually_centered_lstsq_and_zero_flag_case():
    frame = pd.DataFrame(
        dict(
            game_id=range(8),
            season_id=[11] * 4 + [12] * 4,
            stratum=["a"] * 4 + ["b"] * 4,
            L2_diff=[1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            Lc_diff=[0.3, 0.1, -0.2, 0.4, 0.2, -0.1, 0.5, -0.3],
            res_pts=[1.0, 0.0, -1.0, 0.2, 3.0, 2.0, 1.0, -1.0],
        )
    )
    columns = ["L2_diff", "Lc_diff", "res_pts"]
    centered = frame[columns] - frame.groupby("stratum")[columns].transform("mean")
    expected = np.linalg.lstsq(centered[columns[:2]], centered.res_pts, rcond=None)[0]
    result = joint_strata_slopes(frame, repeats=100, seed=3)
    np.testing.assert_allclose(result.estimate, expected, atol=1e-12, rtol=0)
    again = joint_strata_slopes(frame, repeats=100, seed=3)
    pd.testing.assert_frame_equal(result, again)
    frame["L2_diff"] = 0.0
    joint = joint_strata_slopes(frame, repeats=100)
    single = within_strata_slope(frame, "Lc_diff", repeats=100)
    assert joint.iloc[1].estimate == pytest.approx(single["estimate"], abs=1e-12)
    assert joint.iloc[1].se == pytest.approx(single["se"], abs=1e-12)
    assert not joint.iloc[0].identifiable and pd.isna(joint.iloc[0].se)


def test_five_season_blocks_are_disjoint_and_bootstrap_cannot_draw_outside_them():
    years = [s for values in SEASON_BLOCKS.values() for s in values]
    assert sorted(years) == list(range(11, 26)) and len(set(years)) == 15
    frame = contrast_data()
    for key, seasons in SEASON_BLOCKS.items():
        assert set(block_rows(frame, key).season_id) <= set(seasons)
        weights = season_bootstrap_weights(seasons, 100, 23)
        assert set(weights.columns) == set(seasons) and weights.sum(axis=1).eq(5).all()
    a = within_strata_contrast(
        frame, frame.selected, "y", years=SEASON_BLOCKS["B1"], seed=23, repeats=100
    )
    b = within_strata_contrast(
        frame, frame.selected, "y", years=SEASON_BLOCKS["B1"], seed=23, repeats=100
    )
    assert a["se"] == b["se"]


def test_typical_nonzero_difference_translates_coefficient_and_se_to_points():
    result = typical_difference(pd.DataFrame(dict(x=[0.0, -0.1, 0.2, -0.4, 0.0])), "x", 2.0, 0.5)
    assert result["n_nonzero"] == 3
    assert result["median_diff"] == pytest.approx(0.2)
    assert result["p90_diff"] == pytest.approx(0.36)
    assert result["median_effect"] == pytest.approx(0.4)
    assert result["p90_effect"] == pytest.approx(0.72)
    assert result["median_effect_se"] == pytest.approx(0.1)


def mc_candidates():
    frame = pd.DataFrame(
        dict(
            game_id=[1, 2, 3, 4],
            team_id=[30, 20, 10, 1],
            season_id=[11] * 4,
            as_of=pd.to_datetime(
                ["2010-04-02", "2010-04-01", "2010-04-01", "2010-03-01"], utc=True
            ),
            leverage_max=[0.3, 0.3, 0.3, 0.3],
            title_lev=[0.0] * 4,
            cut_lev=[0.0] * 4,
            survival_lev=[0.3] * 4,
            survival_p_win=[0.7, 0.7, 0.7, 1.0],
            survival_p_draw=[0.5] * 4,
            survival_p_loss=[0.4] * 4,
            survival_mc_se=[0.01] * 4,
            goals_scored=[3, 2, 1, 0],
            res_pts=[1.0, 2.0, 3.0, 4.0],
        )
    )
    return frame.set_index(["game_id", "team_id"])


def test_mc_selection_uses_date_then_team_and_is_invariant_to_outcomes_and_row_order():
    frame = mc_candidates()
    assert select_mc_example(frame) == (3, 10)
    altered = frame.sample(frac=1, random_state=8)
    altered["goals_scored"] = [99, -5, 80, 23]
    altered["res_pts"] = [-90.0, 22.0, 0.0, 99.0]
    assert select_mc_example(altered) == (3, 10)


def test_finished_team_elo_endpoint_uses_only_games_strictly_before_morning():
    from test_season_snapshot import morning, snapshot_inputs

    rows, elo, _ = snapshot_inputs()
    elo["elo_pre"] = 1500.0
    elo["nothing_to_play"] = False
    expected = past_elo_endpoints(rows, elo, morning(5), 8)
    delta = 10 * (1 - 1 / (1 + 10 ** (-30 / 400)))
    assert expected.loc[1] == pytest.approx(1500 + delta, abs=1e-12)
    changed = rows.copy()
    changed.loc[changed.as_of.ge(morning(5)), ["goals_scored", "goals_conceded"]] = 99
    changed_elo = elo.copy()
    changed_elo.loc[changed_elo.as_of.ge(morning(5)), "elo_pre"] = 9999.0
    pd.testing.assert_series_equal(
        expected, past_elo_endpoints(changed, changed_elo, morning(5), 8)
    )


def test_additional_canonical_tiebreaks_resolve_old_full_ties():
    from dataclasses import replace

    from ligat.simulate.engine import LeagueEngine
    from ligat.simulate.rules import load_season_rules

    rules = load_season_rules()
    old = replace(rules[8], n_teams=2, groups=())
    new = replace(old, tiebreak_rules=rules[10].tiebreak_rules)
    state = LeagueEngine(old, [1, 2]).empty(1)
    state["wins"][0] = [2, 1]
    result = compare_tie_rules(state, [1, 2], old, new, np.array([[0.2, 0.7]]))
    assert result["old_tied_pairs"] == result["resolved_pairs"] == 1
    assert result["new_tied_pairs"] == 0


def test_descriptive_summary_rule_has_fixed_se_boundaries_and_keeps_flag_as_baseline():
    keys = [
        "dead",
        "alive_L5_0.25",
        "alive_L5_1",
        "joint_Lc",
        "alive_Comp_survival",
        "alive_releg_close",
    ]
    for estimate, expected in [
        (1.0, "no"),
        (1.5, "unclear"),
        (2.0, "unclear"),
        (2.1, "yes"),
        (-2.1, "yes"),
    ]:
        frame = pd.DataFrame(
            [
                dict(
                    comparison_id=k,
                    representation="original",
                    metric="res_pts",
                    estimate=estimate,
                    se=1.0,
                    median_effect=estimate / 10,
                    median_effect_se=0.1,
                )
                for k in keys
            ]
        )
        summary = conditional_summary(frame)
        assert summary.loc[summary.representation.ne("L2"), "beyond_flag"].eq(expected).all()
        assert summary.loc[summary.representation.eq("L2"), "beyond_flag"].eq("no").all()


def test_morning_display_joins_rounds_only_for_visible_fixtures():
    from types import SimpleNamespace

    from test_season_snapshot import morning, snapshot_inputs

    from ligat.simulate.snapshot import snapshot_as_of

    rows, elo, rules = snapshot_inputs()
    snapshot = snapshot_as_of(rows, elo, rules, 8, morning(1))
    result = morning_example(SimpleNamespace(participants=rows, prematch=elo), snapshot)
    schedule = result["schedule"]
    assert schedule.groupby("stage")["round"].nunique().le(2).all()
    assert set(schedule.game_id) <= set(snapshot.remaining.game_id)
    assert schedule.stage.eq("regular").all()
    assert len(schedule) + int(result["other_schedule"].other_games.iloc[0]) == len(
        snapshot.remaining
    )
