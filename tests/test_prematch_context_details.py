"""Saved-artifact accounting and display invariants for notebook 03 R4."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.prematch_context_details import (
    gap_missingness,
    loss_attribution,
    promoted_dispersion,
    promoted_value_example,
    remaining_value_folds,
    robust_value_comparison,
    source_value_medians,
)
from ligat.analysis.prematch_context_review import WORDS
from ligat.viz.prematch_context import context_table


def attribution_inputs():
    identity = pd.DataFrame(
        dict(game_id=[10, 10, 11, 11, 12, 12], team_id=[1, 2, 1, 3, 3, 4], season_id=[11] * 6)
    ).set_index(["game_id", "team_id"])
    predictions = pd.DataFrame(
        [
            dict(game_id=g, season_id=11, config_id=c, log_loss=1 - (d if c.endswith("avg") else 0))
            for g, d in zip([10, 11, 12], [-0.6, -0.4, 0.1], strict=True)
            for c in ("O_elo", "O_value_avg")
        ]
    )
    teams = pd.DataFrame(
        [
            dict(
                game_id=g,
                team_id=t,
                season_id=11,
                config_id=c,
                poisson_nll=1 - (d if c.endswith("avg") else 0),
            )
            for (g, t), d in zip(identity.index, [-0.3, -0.2, -0.1, -0.1, 0.05, 0.05], strict=True)
            for c in ("P_poisson", "P_value_avg")
        ]
    )
    return predictions, teams, identity


def test_loss_attribution_counts_named_derby_once_and_sums_both_poisson_sides():
    predictions, teams, identity = attribution_inputs()
    before = teams.copy(deep=True)
    result = loss_attribution(predictions, teams, identity, {1, 2})
    assert result.games.tolist() == [2, 1, 2, 1]
    assert result.loss_sum.tolist() == pytest.approx([-1, 0.1, -0.7, 0.1])
    assert result.improvement.tolist() == pytest.approx([-0.5, 0.1, -0.35, 0.1])
    assert result.loss_share.iloc[0] > 100
    assert result.groupby("route").loss_share.sum().tolist() == pytest.approx([100, 100])
    assert_frame_equal(teams, before)


def test_loss_attribution_rejects_duplicate_and_unpaired_predictions():
    p, t, identity = attribution_inputs()
    with pytest.raises(ValueError, match="Duplicate"):
        loss_attribution(pd.concat([p, p.iloc[:1]]), t, identity, {1})
    with pytest.raises(ValueError, match="both team losses"):
        loss_attribution(p, t.iloc[1:], identity, {1})
    with pytest.raises(ValueError, match="identical validation fixtures"):
        loss_attribution(p.iloc[1:], t, identity, {1})


def test_context_diagnostics_reject_held_out_seasons_before_any_statistic():
    p, t, identity = attribution_inputs()
    p.loc[0, "season_id"] = 26
    with pytest.raises(ValueError, match="8–25"):
        loss_attribution(p, t, identity, {1})
    with pytest.raises(ValueError, match="8–25"):
        robust_value_comparison(pd.DataFrame({"season_id": [26]}))


def test_robustness_removes_source_season_ten_and_uses_sample_se():
    frame = pd.DataFrame(dict(season_id=[9, 10, 11, 12], improvement=[0.1, 0.8, 0.2, 0.3]))
    rows = robust_value_comparison(frame)
    assert rows.seasons.tolist() == [4, 3]
    assert rows["mean"].tolist() == pytest.approx([0.35, 0.2])
    assert rows.se.iloc[1] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1) / np.sqrt(3))


def test_remaining_fold_mean_keeps_game_weights_instead_of_equal_fold_weights():
    folds = pd.DataFrame(dict(route=["O"] * 3, season_id=[11, 12, 13], improvement=[-1, 0.1, 0.3]))
    counts = pd.DataFrame(dict(config_id=["O_elo"] * 3, season_id=[11, 12, 13], games=[100, 1, 3]))
    row = remaining_value_folds(folds, counts).iloc[0]
    assert row.improvement == pytest.approx(0.25)
    assert row.folds == row.better == 2


def test_promoted_dispersion_uses_observed_events_and_ddof_one():
    frame = pd.DataFrame(
        dict(season_id=[11] * 4, promoted_in=[True, True, True, False], ppm=[0.5, 1, 1.5, 3])
    )
    result = promoted_dispersion(frame)
    assert result.ppm.tolist() == pytest.approx([1, 0.5, 0.5, 1.5])
    assert result.teams.eq(3).all()
    assert result.points_33.tolist() == pytest.approx(result.ppm * 33)


def test_prior_value_quantile_is_from_lower_half_of_prior_league_shares():
    frame = pd.DataFrame(dict(season_id=[24] * 14, avg_value_eur=np.arange(1, 15)))
    row = promoted_value_example(frame).iloc[0]
    assert row.value == pytest.approx(2.8 / 105)
    assert row.clubs == 7


def test_source_median_ignores_missing_club_values_without_filling_them():
    frame = pd.DataFrame(
        dict(season_id=[9, 9, 9, 10, 10], total_value_eur=[1e6, 5e6, np.nan, 2e6, 6e6])
    )
    result = source_value_medians(frame)
    assert result.total_million.tolist() == [3, 4]
    assert result.clubs.tolist() == [2, 2]


def test_gap_counts_separate_all_three_non_split_seasons_from_later_playoffs():
    frame = pd.DataFrame(
        dict(
            season_id=[8, 9, 10, 11],
            gap_to_cut=[np.nan] * 4,
            stage=["regular", "regular", "regular", "upper"],
            gap_to_leader=[0, 0, 0, 0],
            gap_to_safety=[1, 1, 1, np.nan],
        )
    )
    row = gap_missingness(frame).set_index("feature").loc["gap_to_cut"]
    assert row.missing == 4 and row.initial_three == 3 and row.seasons_9_10 == 2


def test_thousandths_are_display_only_and_alternative_heading_is_explicit():
    frame = pd.DataFrame(dict(improvement=[-0.00124], se=[0.0002], **{WORDS: ["בדיקה"]}))
    before = frame.copy(deep=True)
    html = context_table(frame, thousandths=True, alternative=True)._repr_html_()
    assert "−1.24" in html and "0.20" in html and "שיפור החלופה מול ברירת המחדל" in html
    assert_frame_equal(frame, before)
