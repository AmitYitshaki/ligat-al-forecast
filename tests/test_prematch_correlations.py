"""Independent arithmetic, paired targets, fixed quintiles and immediate leakage stops."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.prematch_correlations import (
    CONTRAST_ELO_COLUMNS,
    TARGET_COLUMNS,
    LeakageDetectedError,
    contrast_elo_correlations,
    correlation_sanity_checks,
    feature_catalog,
    feature_outcome_chi2,
    feature_pair_correlations,
    feature_target_correlations,
    game_targets,
)


@pytest.fixture
def sample():
    rng = np.random.default_rng(29)
    games = 360
    home = np.tile([True, False], games)
    ids = np.repeat(np.arange(games), 2)
    index = pd.MultiIndex.from_arrays([ids, np.where(home, 1, 2)], names=["game_id", "team_id"])
    season = np.repeat(np.repeat([8, 9, 20, 21, 22, 25], 60), 2)
    frame = pd.DataFrame(
        {"season_id": season, "is_home": home, "opponent_id": np.where(home, 2, 1)}, index=index
    )
    frame["stage"] = pd.Categorical(
        np.repeat(np.array(["regular", "upper", "middle", "lower"])[np.arange(games) % 4], 2)
    )
    for column in CONTRAST_ELO_COLUMNS:
        frame[column] = np.repeat(rng.normal(0, 1, games), 2) * np.where(home, 1, -1)
    frame["elo_pre"] = 1500 + frame.elo_diff / 2
    frame["elo_opponent_pre"] = 1500 - frame.elo_diff / 2
    frame["gf_venue_ewma5"] = rng.uniform(0.2, 2.5, 2 * games)
    frame["opp_gf_venue_ewma5"] = frame.gf_venue_ewma5.to_numpy().reshape(-1, 2)[:, ::-1].ravel()
    frame["prev_value_share_avg"] = rng.uniform(0.1, 2, 2 * games)
    frame["promoted_in"] = np.where(home, ids % 2 == 0, ids % 5 == 0)
    frame["nothing_to_play"] = np.where(home, ids % 5 == 0, ids % 7 == 0)
    frame["prev_season_known"] = season != 8
    frame["league_size"] = np.where(season < 14, 12, 14)
    frame["games_behind"] = 0  # globally constant, still reported by name
    frame.loc[frame.season_id.eq(8), list(CONTRAST_ELO_COLUMNS[:-1])] = np.nan
    goals_home = np.array([2, 1, 0])[np.arange(games) % 3]
    goals_away = np.array([0, 1, 2])[np.arange(games) % 3]
    rows = frame.reset_index().loc[:, ["game_id", "team_id", "season_id"]]
    rows["goals_scored"] = np.where(home, np.repeat(goals_home, 2), np.repeat(goals_away, 2))
    rows["goals_conceded"] = np.where(home, np.repeat(goals_away, 2), np.repeat(goals_home, 2))
    return frame, rows


def test_targets_are_exactly_keyed_mirrored_and_use_unhalved_310(sample):
    frame, rows = sample
    before = frame.copy(deep=True)
    targets = game_targets(frame, rows.iloc[::-1])
    assert tuple(targets.columns) == TARGET_COLUMNS and targets.index.equals(frame.index)
    assert targets.notna().all().all()
    assert targets.loc[(0, 1), "points_game"] == 3
    assert targets.loc[(0, 2), "points_game"] == 0
    assert targets.loc[(1, 1), "points_game"] == targets.loc[(1, 2), "points_game"] == 1
    assert targets.groupby(level="game_id").points_game.sum().isin([2, 3]).all()
    assert targets.groupby(level="game_id").goal_difference_game.sum().eq(0).all()
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("bad", ["duplicate", "missing", "season", "goals", "negative", "mirror"])
def test_target_join_rejects_incomplete_or_contradictory_results(sample, bad):
    frame, rows = sample
    if bad == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    if bad == "missing":
        rows = rows.iloc[1:]
    if bad == "season":
        rows.loc[0, "season_id"] = 26
    if bad == "goals":
        rows.loc[0, "goals_scored"] = np.nan
    if bad == "negative":
        rows.loc[0, "goals_scored"] = -1
    if bad == "mirror":
        rows.loc[0, "goals_scored"] = 3
    with pytest.raises(ValueError):
        game_targets(frame, rows)


def test_pairwise_complete_counts_and_correlations_match_independent_numpy(sample):
    frame, _ = sample
    frame.loc[(100, 1), "prev_ppm_diff"] = np.nan
    result = feature_pair_correlations(frame)
    row = result.loc[result.feature_a.eq("prev_ppm_diff") & result.feature_b.eq("elo_diff")].iloc[0]
    complete = frame[["prev_ppm_diff", "elo_diff"]].dropna()
    assert row.n == len(complete)
    assert row.pearson == pytest.approx(np.corrcoef(complete.to_numpy().T)[0, 1], abs=1e-12)
    ranks = complete.rank(method="average").to_numpy().T
    assert row.spearman == pytest.approx(np.corrcoef(ranks)[0, 1], abs=1e-12)
    per_season = []
    for season in sorted(frame.season_id.unique()):
        part = frame.loc[frame.season_id.eq(season), ["prev_ppm_diff", "elo_diff"]].dropna()
        if len(part) > 1:
            per_season.append(np.corrcoef(part.to_numpy().T)[0, 1])
    assert row.pearson_min_season == pytest.approx(min(per_season), abs=1e-12)
    assert row.pearson_max_season == pytest.approx(max(per_season), abs=1e-12)
    recent = frame.loc[frame.season_id.ge(21), ["prev_ppm_diff", "elo_diff"]].dropna()
    assert row.pearson_last5 == pytest.approx(np.corrcoef(recent.to_numpy().T)[0, 1], abs=1e-12)
    assert not any("p_value" in c for c in result)


def test_constants_families_and_within_season_constants_are_explicit(sample):
    frame, _ = sample
    catalog = feature_catalog(frame).set_index("feature")
    assert catalog.at["games_behind", "constant"]
    assert not catalog.at["prev_season_known", "constant"]
    assert catalog.at["opp_gf_venue_ewma5", "family"] == "opponent_form"
    assert catalog.at["value_avg_log_ratio", "family"] == "contrast"
    pairs = feature_pair_correlations(frame)
    assert "games_behind" not in set(pairs.feature_a) | set(pairs.feature_b)
    assert not set(["stage", "season_id", "opponent_id"]) & set(catalog.index)
    row = pairs.loc[
        pairs.feature_a.eq("prev_season_known") & pairs.feature_b.eq("league_size")
    ].iloc[0]
    assert pd.isna(row.pearson_min_season) and pd.isna(row.pearson_max_season)


def test_target_correlations_use_available_rows_and_explicit_season_denominator(sample):
    frame, rows = sample
    targets = game_targets(frame, rows)
    result = feature_target_correlations(frame, targets)
    row = result.loc[result.feature.eq("prev_season_known") & result.target.eq("points_game")].iloc[
        0
    ]
    assert row.seasons_counted == 0 and row.seasons_same_sign == 0
    assert pd.isna(row.median_season_pearson)
    row = result.loc[result.feature.eq("prev_ppm_diff") & result.target.eq("goals_scored")].iloc[0]
    mask = frame.prev_ppm_diff.notna()
    expected = np.corrcoef(frame.loc[mask, "prev_ppm_diff"], targets.loc[mask, "goals_scored"])[
        0, 1
    ]
    assert row.pearson == pytest.approx(expected, abs=1e-12)
    assert row.n == mask.sum()
    assert not any("p_value" in c for c in result)


def test_correlation_sanity_matrices_and_repeated_calls_preserve_inputs(sample):
    frame, rows = sample
    before = frame.copy(deep=True)
    targets = game_targets(frame, rows)
    check = correlation_sanity_checks(frame)
    assert check["elo_diff_vs_elo_pre"] > 0 and check["elo_diff_vs_elo_opponent_pre"] < 0
    assert check["max_home_away_difference"] <= 1e-12
    matrices = contrast_elo_correlations(frame)
    assert len(matrices) == 4
    for function in [feature_pair_correlations, feature_target_correlations]:
        args = (frame, targets) if function is feature_target_correlations else (frame,)
        assert_frame_equal(function(*args), function(*args), check_exact=True)
    assert_frame_equal(frame, before, check_exact=True)


def test_pearson_leakage_guard_names_feature_and_target_without_returning_results(sample):
    frame, rows = sample
    targets = game_targets(frame, rows)
    frame["elo_pre"] = targets.goals_scored.astype(float)
    with pytest.raises(LeakageDetectedError) as error:
        feature_target_correlations(frame, targets)
    assert error.value.feature == "elo_pre" and error.value.target == "goals_scored"
    assert error.value.statistic == "pearson"


def manual_chi(counts):
    observed = counts.to_numpy(float)
    observed = observed[:, observed.sum(axis=0) > 0]
    expected = observed.sum(axis=1)[:, None] * observed.sum(axis=0)[None, :] / observed.sum()
    chi = np.sum((observed - expected) ** 2 / expected)
    v = np.sqrt(chi / (observed.sum() * (min(observed.shape) - 1)))
    return chi, v, float(expected.min())


def test_home_only_chi_and_cramers_v_match_hand_calculation_without_yates(sample):
    frame, rows = sample
    frame = frame.loc[frame.index.get_level_values("game_id") < 150].copy()
    rows = rows.loc[rows.game_id.lt(150)]
    targets = game_targets(frame, rows)
    home = frame.loc[frame.is_home]
    frame["promoted_in"] = False
    for difference, wanted in [(2, 30), (0, 25), (-2, 20)]:
        indices = home.index[targets.loc[home.index, "goal_difference_game"].eq(difference)][
            :wanted
        ]
        frame.loc[indices, "promoted_in"] = True
    result = feature_outcome_chi2(frame, targets)
    counts = result.contingencies["promoted_in"]
    expected_chi, expected_v, minimum = manual_chi(counts)
    row = result.statistics.set_index("feature").loc["promoted_in"]
    assert result.home_rows == 150
    assert row.chi2 == pytest.approx(expected_chi, abs=1e-10)
    assert row.cramers_v == pytest.approx(expected_v, abs=1e-10)
    assert row.min_expected == pytest.approx(minimum, abs=1e-10)
    assert row.dof == 2


def test_chi_percentages_baselines_counts_fixed_bins_and_reproducibility(sample):
    frame, rows = sample
    targets = game_targets(frame, rows)
    first = feature_outcome_chi2(frame, targets)
    second = feature_outcome_chi2(frame, targets)
    assert first.home_rows == 360
    assert_frame_equal(first.statistics, second.statistics, check_exact=True)
    assert_frame_equal(first.percentages, second.percentages, check_exact=True)
    assert first.bin_edges == second.bin_edges
    stats = first.statistics.set_index("feature")
    assert {"stage", "away_stage", "away_promoted_in", "away_nothing_to_play"} <= set(stats.index)
    assert_frame_equal(
        first.contingencies["stage"], first.contingencies["away_stage"], check_names=False
    )
    for feature, counts in first.contingencies.items():
        assert counts.to_numpy().sum() == stats.at[feature, "n"]
    percentages = first.percentages
    np.testing.assert_allclose(
        percentages[["H_percent", "D_percent", "A_percent"]].sum(axis=1), 100, atol=1e-10
    )
    np.testing.assert_allclose(
        percentages[["baseline_H_percent", "baseline_D_percent", "baseline_A_percent"]].sum(axis=1),
        100,
        atol=1e-10,
    )
    # Use the full-home boundaries on the last five, never re-fit qcut there.
    home = frame.loc[frame.is_home]
    categories = pd.cut(home.elo_diff, first.bin_edges["elo_diff"], include_lowest=True)
    outcomes = targets.loc[home.index, "goal_difference_game"].map({2: "H", 0: "D", -2: "A"})
    recent = home.season_id.ge(21)
    counts = pd.crosstab(categories.loc[recent], outcomes.loc[recent]).reindex(
        columns=["H", "D", "A"], fill_value=0
    )
    _, v, _ = manual_chi(counts)
    assert stats.at["elo_diff", "cramers_v_last5"] == pytest.approx(v, abs=1e-10)
    # t-1 missingness changes the denominator for both category and baseline rates.
    eligible = percentages.loc[percentages.feature.eq("value_avg_log_ratio")]
    assert eligible.baseline_n.eq(300).all() and eligible.n.sum() == 300


def test_non_testable_chi_features_keep_rows_and_missing_statistics(sample):
    frame, rows = sample
    frame["wins"] = 0
    frame.loc[(10, 1), "wins"] = 1  # qcut collapses to one group
    frame["goals_for"] = np.nan
    targets = game_targets(frame, rows)
    result = feature_outcome_chi2(frame, targets).statistics.set_index("feature")
    assert result.at["is_home", "excluded_reason"] == "constant in home rows"
    assert result.at["wins", "excluded_reason"] == "fewer than 2 categories after qcut"
    assert result.at["goals_for", "excluded_reason"] == "no nonmissing home values"
    assert result.at["is_home", "categories"] == 1 and result.at["wins", "n"] == 360
    for feature in ["is_home", "wins", "goals_for"]:
        assert (
            result.loc[
                feature, ["chi2", "dof", "p_value", "cramers_v", "min_expected", "cramers_v_last5"]
            ]
            .isna()
            .all()
        )


def test_chi_leakage_guard_names_feature_and_ignores_constant_rows(sample):
    frame, rows = sample
    targets = game_targets(frame, rows)
    frame["promoted_in"] = targets.points_game.eq(3)
    with pytest.raises(LeakageDetectedError) as error:
        feature_outcome_chi2(frame, targets)
    assert error.value.feature == "promoted_in" and error.value.statistic == "cramers_v"


@pytest.mark.parametrize("season", [26, 27, 28])
def test_no_holdout_or_live_analysis(sample, season):
    frame, _ = sample
    frame.iloc[0, frame.columns.get_loc("season_id")] = season
    with pytest.raises(ValueError, match="8–25"):
        feature_pair_correlations(frame)


def test_observed_targets_cannot_accidentally_be_included_as_features(sample):
    frame, rows = sample
    targets = game_targets(frame, rows)
    with pytest.raises(ValueError, match="separate"):
        feature_catalog(frame.join(targets))
