"""Hand arithmetic, season openings and calendar causality of the 66 form candidates."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ligat.analysis.prepare import AnalysisTables
from ligat.features.base import FeatureBlock
from ligat.features.form import (
    FormBlock,
    _rho,
    compute_form,
    form_opening_values,
    neutral_form_values,
)
from ligat.features.registry import FEATURES, FORM_COLUMNS, validate_feature_columns
from ligat.simulate.rules import SeasonRules


def tables_from(fixtures: list[tuple[int, int, int, int, str, int, int]]) -> AnalysisTables:
    rows = []
    for season, game, home, away, date, gf, ga in fixtures:
        for team, opponent, scored, conceded, is_home in (
            (home, away, gf, ga, True),
            (away, home, ga, gf, False),
        ):
            rows.append(
                dict(
                    game_id=game,
                    team_id=team,
                    opponent_id=opponent,
                    season_id=season,
                    date=pd.Timestamp(date, tz="Asia/Jerusalem"),
                    round=1,
                    stage="regular",
                    is_home=is_home,
                    goals_scored=scored,
                    goals_conceded=conceded,
                )
            )
    return AnalysisTables(
        pd.DataFrame(),
        pd.DataFrame(rows).set_index(["team_id", "game_id"]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def season_fixtures(last: int = 11) -> list[tuple[int, int, int, int, str, int, int]]:
    """Four-club double round robins, two newcomers yearly and a two-year return."""
    rosters = {8: [1, 2, 3, 4], 9: [1, 2, 5, 6], 10: [1, 2, 7, 8], 11: [1, 2, 3, 4]}
    rosters.update({s: [1, 2, 2 * s, 2 * s + 1] for s in range(12, 26)})
    fixtures = []
    for season in range(8, last + 1):
        roster = rosters[season]
        for i, home in enumerate(roster):
            for j, away in enumerate(roster):
                if i != j:
                    day = i * 4 + j + 1
                    fixtures.append(
                        (
                            season,
                            season * 100 + day,
                            home,
                            away,
                            f"{1998 + season}-08-{day:02}",
                            4 - i,
                            3 - j,
                        )
                    )
    return fixtures


def rules() -> dict[int, SeasonRules]:
    return {
        s: SeasonRules(s, 4, 6, (), False, 1, None, ("points", "goal_difference", "goals_for"))
        for s in range(8, 26)
    }


def fixed_neutral() -> pd.DataFrame:
    return pd.DataFrame(
        [[2.0, 1.0, 1.5], [3.0, 0.5, 2.0], [1.0, 1.5, 1.0]],
        index=pd.Index(["all", "home", "away"], name="stratum"),
        columns=["gf", "ga", "pts"],
    )


def change_score(tables: AnalysisTables, game: int, home_goals: int) -> AnalysisTables:
    rows = tables.team_matches.copy()
    game_ids = rows.index.get_level_values("game_id")
    rows.loc[(game_ids == game) & rows.is_home, "goals_scored"] = home_goals
    rows.loc[(game_ids == game) & ~rows.is_home, "goals_conceded"] = home_goals
    return replace(tables, team_matches=rows)


def test_hand_windows_ewma_and_season_average() -> None:
    tables = tables_from(
        [
            (8, 1, 1, 2, "2006-08-01", 1, 0),
            (8, 2, 2, 1, "2006-08-02", 0, 3),
            (8, 3, 1, 2, "2006-08-03", 2, 1),
            (8, 4, 2, 1, "2006-08-04", 0, 0),
        ]
    )
    result = compute_form(tables, {}, neutral=fixed_neutral())
    f = result.features
    assert f.loc[(4, 1), "gf_all_w10"] == pytest.approx((1 + 3 + 2 + 7 * 2) / 10)
    alpha = 1 - 0.5 ** (1 / 5)
    assert f.loc[(3, 1), "gf_all_ewma5"] == pytest.approx(
        alpha * 3 + (1 - alpha) * (alpha * 1 + (1 - alpha) * 2)
    )
    assert f.loc[(4, 1), "gf_all_std_m6"] == pytest.approx((6 + 6 * 2) / 9)
    assert f.loc[(3, 1), "gf_venue_w5"] == pytest.approx((1 + 4 * 3) / 5)
    assert f.loc[(4, 1), "gf_venue_w5"] == pytest.approx((3 + 4 * 1) / 5)
    assert f.loc[(1, 1), "gf_all_w5"] == 2
    assert f.gf_all_prev.eq(2).all() and f.gf_all_hist.eq(2).all()
    assert not f.gf_all_w5.eq(2).all()
    assert f.loc[(1, 1), "pts_all_w5"] == 1.5
    assert f.loc[(2, 1), "pts_all_w5"] == pytest.approx((3 + 4 * 1.5) / 5)
    assert tuple(f.columns) == FORM_COLUMNS and len(FORM_COLUMNS) == 66
    block: FeatureBlock = FormBlock({}, fixed_neutral())
    assert_frame_equal(block.compute(tables), f)


def test_window_forgets_prior_and_resets_in_summer() -> None:
    tables = tables_from(season_fixtures(9))
    result = compute_form(tables, rules(), neutral=fixed_neutral())
    rows = tables.team_matches.reset_index()
    games = rows.loc[rows.team_id.eq(1) & rows.season_id.eq(8)].sort_values("date")
    last = int(games.iloc[-1].game_id)
    assert result.features.loc[(last, 1), "gf_all_w5"] == pytest.approx(
        games.iloc[:-1].goals_scored.mean()
    )
    first9 = int(
        rows.loc[rows.season_id.eq(9) & rows.team_id.eq(1)].sort_values("date").iloc[0].game_id
    )
    prior = result.openings.query(
        "season_id == 9 and team_id == 1 and metric == 'gf' and stratum == 'all'"
    ).iloc[0]
    assert result.features.loc[(first9, 1), "gf_all_w5"] == pytest.approx(prior.prior)
    assert prior.prior == pytest.approx(prior.league_mean + 0.5 * (prior.prev - prior.league_mean))


@pytest.mark.parametrize("season", range(8, 26))
def test_future_and_same_day_results_cannot_change_morning_features(season: int) -> None:
    fixtures = season_fixtures(season)
    own = [f for f in fixtures if f[0] == season and (f[2] == 1 or f[3] == 1)]
    target, other = own[1], own[2]
    # Deliberately let one team play twice on the same Israeli calendar day.
    fixtures = [(*f[:4], target[4] + " 23:30", *f[5:]) if f[1] == other[1] else f for f in fixtures]
    tables = tables_from(fixtures)
    before = compute_form(tables, rules(), neutral=fixed_neutral()).features
    changed = change_score(tables, other[1], 8)
    after = compute_form(changed, rules(), neutral=fixed_neutral()).features
    target_day = pd.Timestamp(target[4], tz="Asia/Jerusalem").normalize()
    rows = tables.team_matches.reset_index()
    protected = (
        rows.loc[rows.date.dt.normalize().le(target_day)].set_index(["game_id", "team_id"]).index
    )
    assert_frame_equal(before.loc[protected], after.loc[protected])
    assert not before.equals(after)
    current = compute_form(
        change_score(tables, target[1], 7), rules(), neutral=fixed_neutral()
    ).features
    assert_series_equal(before.loc[(target[1], 1)], current.loc[(target[1], 1)])


def test_opening_coefficients_and_promoted_values_use_completed_seasons_only() -> None:
    tables = tables_from(season_fixtures())
    before = form_opening_values(tables, rules(), neutral=fixed_neutral())
    after = form_opening_values(change_score(tables, 1102, 9), rules(), neutral=fixed_neutral())
    assert_frame_equal(before, after)
    shuffled = replace(tables, team_matches=tables.team_matches.sample(frac=1, random_state=42))
    assert_frame_equal(
        compute_form(tables, rules(), neutral=fixed_neutral()).features,
        compute_form(shuffled, rules(), neutral=fixed_neutral()).features,
    )


def test_promoted_bottom_three_then_historical_pool_and_return_after_two_years() -> None:
    tables = tables_from(season_fixtures())
    openings = form_opening_values(tables, rules(), neutral=fixed_neutral())
    rows = tables.team_matches.reset_index()
    # Club 1 dominates; 2, 3, 4 are the sporting bottom three in season 8.
    expected9 = rows.loc[rows.season_id.eq(8) & rows.team_id.isin([2, 3, 4])].goals_scored.mean()
    newcomer = openings.query(
        "season_id == 9 and team_id == 5 and metric == 'gf' and stratum == 'all'"
    ).iloc[0]
    assert newcomer.prior == pytest.approx(expected9)
    assert newcomer.used_promoted_fallback and newcomer.is_promoted
    pool = rows.loc[
        (rows.season_id.eq(9) & rows.team_id.isin([5, 6]))
        | (rows.season_id.eq(10) & rows.team_id.isin([7, 8]))
    ]
    for stratum in ("all", "home", "away"):
        part = pool if stratum == "all" else pool.loc[pool.is_home.eq(stratum == "home")]
        returned = openings.query(
            "season_id == 11 and team_id == 3 and metric == 'gf' and stratum == @stratum"
        ).iloc[0]
        assert returned.prior == pytest.approx(part.goals_scored.mean())
        assert returned.prev == returned["hist"] == returned.prior
        assert returned.is_promoted and not returned.used_promoted_fallback


def test_hist_is_game_weighted_and_prev_is_raw() -> None:
    fixtures = season_fixtures(10)
    # Give the retained club extra matches in season 9 to distinguish game and season means.
    fixtures.append((9, 999, 1, 2, "2007-08-25", 6, 0))
    tables = tables_from(fixtures)
    openings = form_opening_values(tables, rules(), neutral=fixed_neutral())
    opening = openings.query(
        "season_id == 10 and team_id == 1 and metric == 'gf' and stratum == 'all'"
    ).iloc[0]
    rows = tables.team_matches.reset_index().query("team_id == 1 and season_id < 10")
    assert opening["hist"] == pytest.approx(rows.goals_scored.mean())
    assert opening.prev == pytest.approx(rows.loc[rows.season_id.eq(9)].goals_scored.mean())
    assert opening.prev != opening.prior


def test_promoted_pool_and_league_means_are_game_weighted_by_role() -> None:
    fixtures = season_fixtures()
    fixtures.append((9, 999, 5, 6, "2007-08-25", 6, 0))
    # Stop before season 11 to inspect the changed historical pool without
    # triggering the separately tested late-rho guard on artificial score changes.
    tables = tables_from([f for f in fixtures if f[0] <= 10])
    openings = form_opening_values(tables, rules())
    rows = tables.team_matches.reset_index()
    for stratum in ("all", "home", "away"):
        role_rows = rows if stratum == "all" else rows.loc[rows.is_home.eq(stratum == "home")]
        record = openings.query(
            "season_id == 10 and team_id == 7 and metric == 'gf' and stratum == @stratum"
        ).iloc[0]
        assert record.league_mean == pytest.approx(
            role_rows.loc[role_rows.season_id.eq(9)].goals_scored.mean()
        )
    # By season 11, four completed newcomer events exist; make their game counts unequal.
    # All scores unchanged except an extra 0:0, which keeps the artificial rho inside [0,1].
    fixtures = season_fixtures() + [(9, 999, 5, 6, "2007-08-25", 0, 0)]
    tables = tables_from(fixtures)
    openings = form_opening_values(tables, rules())
    record = openings.query(
        "season_id == 11 and team_id == 3 and metric == 'gf' and stratum == 'all'"
    ).iloc[0]
    rows = tables.team_matches.reset_index()
    pool = rows.loc[
        (rows.season_id.eq(9) & rows.team_id.isin([5, 6]))
        | (rows.season_id.eq(10) & rows.team_id.isin([7, 8]))
    ]
    assert record.prior == pytest.approx(pool.goals_scored.mean())


def test_playoff_results_are_included_in_current_season_windows() -> None:
    tables = tables_from(season_fixtures(8))
    rows = tables.team_matches.copy()
    rows.loc[rows.index.get_level_values("game_id") == 802, "stage"] = "upper"
    changed = replace(tables, team_matches=rows)
    assert_frame_equal(
        compute_form(tables, rules()).features, compute_form(changed, rules()).features
    )


def test_neutral_exception_is_isolated_from_later_seasons() -> None:
    tables = tables_from(season_fixtures(10))
    neutral = neutral_form_values(tables)
    changed = change_score(tables, 802, 7)
    assert not neutral.equals(neutral_form_values(changed))
    original = compute_form(tables, rules(), neutral=neutral).features
    injected = compute_form(tables, rules(), neutral=neutral * 2).features
    rows = tables.team_matches.reset_index()
    later = rows.loc[rows.season_id.gt(8)].set_index(["game_id", "team_id"]).index
    assert_frame_equal(original.loc[later], injected.loc[later])
    # A real season-8 result change legitimately changes X8 and L8 for season 9.
    altered = form_opening_values(changed, rules(), neutral=neutral)
    r = altered.query(
        "season_id == 9 and team_id == 1 and metric == 'gf' and stratum == 'all'"
    ).iloc[0]
    changed_rows = changed.team_matches.reset_index().query("season_id == 8")
    league = changed_rows.goals_scored.mean()
    club = changed_rows.loc[changed_rows.team_id.eq(1)].goals_scored.mean()
    assert r.prior == pytest.approx(league + 0.5 * (club - league))


def test_rho_origin_regression_unweighted_and_early_fallback() -> None:
    x, y = np.array([-2.0, -1.0, 2.0]), np.array([-1.0, 0.0, 1.0])
    assert _rho(x, y, 2, 11)["rho"] == pytest.approx(4 / 9)
    early = _rho(x, y, 1, 10)
    assert early["rho"] == 0.5 and early["rho_fallback"]
    assert early["rho_raw"] == pytest.approx(4 / 9)
    assert _rho(np.ones(3), np.ones(3), 2, 10)["rho_reason"] == "zero_variation"


@pytest.mark.parametrize("multiplier", [-1.0, 2.0])
def test_rho_clips_only_in_early_training_seasons(multiplier: float) -> None:
    x = np.array([-1.0, 1.0])
    early = _rho(x, multiplier * x, 2, 10)
    assert early["rho_clipped"] and early["rho_raw"] == multiplier
    assert early["rho"] == max(0, min(1, multiplier))
    with pytest.raises(ValueError, match="clipping"):
        _rho(x, multiplier * x, 2, 11)


def test_unexpected_late_rho_fallback_stops() -> None:
    with pytest.raises(ValueError, match="fallback"):
        _rho(np.zeros(3), np.zeros(3), 2, 11)


def test_registry_rejects_unknown_and_rejected_features() -> None:
    frame = compute_form(tables_from(season_fixtures(8)), rules()).features
    for column in FORM_COLUMNS:
        spec = FEATURES[column]
        assert (spec.block, spec.status, spec.availability) == ("form", "🔬", "D")
        assert "ה-ת12" in spec.decision and "season 8" in spec.decision
    for column in ("unknown_form", "referee", "score_state"):
        with pytest.raises(ValueError, match="Unregistered"):
            validate_feature_columns(frame.assign(**{column: 0}), "form")


def test_no_official_or_other_analysis_view_is_used() -> None:
    tables = tables_from(season_fixtures(9))
    poison = pd.DataFrame({"position": [-999], "points_official": [999999]})
    altered = replace(tables, match_table=poison, team_seasons=poison, season_summary=poison)
    assert_frame_equal(
        compute_form(tables, rules()).features, compute_form(altered, rules()).features
    )


def test_fallback_counts_count_team_game_rows_once() -> None:
    result = compute_form(tables_from(season_fixtures()), rules())
    counts = result.fallback_counts.set_index("season_id")
    assert counts.loc[8, "neutral_rows"] == 24
    assert counts.loc[9, "rho_fallback_rows"] == 12
    assert counts.loc[9, "promoted_fallback_rows"] == 12
    assert counts.loc[11].sum() == 0
    assert len(result.coefficients) == 4 * 9


def test_fail_closed_on_non_development_and_overlapping_seasons() -> None:
    fixtures = season_fixtures(9)
    fixtures[-1] = (*fixtures[-1][:4], "2006-07-01", *fixtures[-1][5:])
    with pytest.raises(ValueError, match="nonoverlapping"):
        compute_form(tables_from(fixtures), rules())
    with pytest.raises(ValueError, match="development"):
        compute_form(tables_from([(26, 1, 1, 2, "2024-08-01", 0, 0)]), rules())
