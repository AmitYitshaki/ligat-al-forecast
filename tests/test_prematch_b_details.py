"""R2 diagnostics preserve paired folds, probability contracts and actual promotion status."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ligat.analysis.prematch_b_details import (
    alignment_trends,
    calibration_error,
    choice_precision,
    configuration_contrast,
    decay_differences,
    decay_profiles,
    draw_peaks,
    gap_case_explanations,
    interaction_table,
    probability_curves,
)
from ligat.analysis.prematch_b_review import ordinal_probabilities
from ligat.viz.prematch_b import alignment_chart, confusion_chart
from ligat.viz.prematch_b_details import decay_chart, interaction_chart, probability_curve_chart


def test_configuration_contrast_lists_only_departures_from_the_reference() -> None:
    row = pd.Series(
        dict(
            k=10,
            home_advantage=30,
            season_regression=0.33,
            use_goal_multiplier=True,
            promoted_initialization="fixed_1450",
            dead_game_discount=1,
            half_life=np.nan,
        )
    )
    assert configuration_contrast(row).startswith("E0050:")
    row["k"], row["half_life"] = 20, 4
    words = configuration_contrast(row)
    assert "K=20" in words and "כל 4 עונות" in words
    assert "1450" not in words and "מחצית" not in words


def test_alignment_trend_excludes_the_first_season_and_keeps_metric_direction() -> None:
    frame = pd.DataFrame(
        dict(
            season_id=[8, 9, 10, 11],
            spearman=[-100, 0.1, 0.2, 0.3],
            exact_percent=[100, 30, 20, 10],
        )
    )
    result = alignment_trends(frame).set_index("metric")
    assert result.seasons.eq(3).all()
    assert result.at["spearman", "correlation"] == pytest.approx(1)
    assert result.at["exact_percent", "correlation"] == pytest.approx(-1)


def test_interaction_averages_the_grid_instead_of_selecting_the_best() -> None:
    rows = [
        dict(
            config_id=f"{k}-{r}-{i}",
            k=k,
            season_regression=r,
            half_life=np.nan,
            log_loss=k + r + i / 1000,
        )
        for k in (10, 20, 30, 40)
        for r in (0.1, 0.16, 0.33, 0.5)
        for i in range(80)
    ]
    frame = pd.DataFrame(rows)
    result = interaction_table(frame)
    assert len(result) == 16 and result.configs.eq(80).all()
    assert result.iloc[0].log_loss == pytest.approx(10.1 + 0.0395)
    assert result.best_in_row.sum() == 4
    chart = interaction_chart(result)
    assert sum("★" in text for row in chart.data[0].text for text in row) == 4
    assert all("★" in row[0] for row in chart.data[0].text)
    with pytest.raises(ValueError, match="1280"):
        interaction_table(frame.iloc[1:])


def test_decay_profiles_use_the_production_age_convention_and_sum_seasons() -> None:
    profiles, totals = decay_profiles()
    assert len(profiles) == 4 * 17
    assert profiles.groupby("decay").tail(1).weight.eq(1).all()
    assert totals.effective_seasons.tolist() == pytest.approx(
        [
            17,
            sum(0.5 ** np.arange(17)),
            sum(0.5 ** (np.arange(17) / 2)),
            sum(0.5 ** (np.arange(17) / 4)),
        ]
    )


def test_decay_se_has_fifteen_fold_replicates_and_ten_paired_bases() -> None:
    configs, scores = [], []
    for base in range(10):
        for h in (0, 1, 2, 4):
            config_id = f"{base}-{h}"
            configs.append(
                dict(
                    config_id=config_id,
                    k=base,
                    home_advantage=30,
                    season_regression=0.33,
                    use_goal_multiplier=True,
                    promoted_initialization="fixed_1450",
                    dead_game_discount=1,
                    half_life=h or np.nan,
                )
            )
            scores.extend(
                dict(
                    config_id=config_id,
                    fold=f,
                    season_id=10 + f,
                    games=100 + f,
                    log_loss=1 + base / 100 + h * f / 100000,
                )
                for f in range(1, 16)
            )
    result = decay_differences(pd.DataFrame(configs), pd.DataFrame(scores))
    f = np.arange(1, 16)
    assert result.difference.iloc[2] == pytest.approx(np.average(2 * f / 100000, weights=100 + f))
    assert result.se.iloc[2] == pytest.approx(np.std(2 * f / 100000, ddof=1) / np.sqrt(15))
    assert result.iloc[0].difference == result.iloc[0].se == 0
    profiles, _ = decay_profiles()
    figure = decay_chart(profiles, result, 0.065)
    assert len(figure.data) == 5 and len(figure.data[-1].error_y.array) == 4
    with pytest.raises(ValueError, match="same 15 folds"):
        decay_differences(pd.DataFrame(configs), pd.DataFrame(scores).iloc[1:])


def test_ece_uses_population_weighted_ten_equal_width_bins() -> None:
    predictions = pd.DataFrame(
        dict(p_home=[1, 0.6, 0.2], p_draw=[0, 0.3, 0.1], p_away=[0, 0.1, 0.7], outcome=[0, 1, 2])
    )
    result = calibration_error(predictions).set_index("outcome")
    assert result.at["בית", "ece"] == pytest.approx(0.8 / 3)
    assert result.games.eq(3).all()


def test_analytic_draw_peak_agrees_with_direct_saved_parameter_conversion() -> None:
    coefficients = pd.DataFrame(
        dict(fold=[15], season_id=[25], beta=[3.0], theta1=[-0.8], theta2=[0.4])
    )
    peak = draw_peaks(coefficients).iloc[0]
    assert peak.width == pytest.approx(1.2)
    assert ordinal_probabilities(peak.peak_gap, 3, -0.8, 0.4)[1] == pytest.approx(
        peak.peak_probability
    )
    assert ordinal_probabilities(peak.peak_gap + 30, 3, -0.8, 0.4)[1] < peak.peak_probability
    with pytest.raises(ValueError, match="positive beta"):
        draw_peaks(coefficients.assign(beta=0))
    curves = probability_curves(SimpleNamespace(tables={"proposal_converter_effect": coefficients}))
    assert np.allclose(curves[["בית", "תיקו", "חוץ"]].sum(axis=1), 1)
    assert curves["בית"].is_monotonic_increasing and curves["חוץ"].is_monotonic_decreasing
    assert len(probability_curve_chart(curves).data) == 3


def test_low_opening_rating_does_not_imply_the_team_was_promoted() -> None:
    gaps = pd.DataFrame(
        dict(
            season_id=[25, 25],
            team_id=[1, 2],
            position=[6, 5],
            elo_rank=[10, 12],
            elo_start=[1450.323295, 1450],
        )
    )
    context = pd.DataFrame(
        dict(season_id=[25, 25], team_id=[1, 2], game_id=[1, 1], promoted_in=[False, True])
    ).set_index(["game_id", "team_id"])
    result = gap_case_explanations(gaps, context)
    assert result.promoted_in.tolist() == [False, True]
    assert "עולה" not in result["הסבר כדורגלני"].iloc[0]
    assert "עולה" in result["הסבר כדורגלני"].iloc[1]
    assert result.crosses_cut.all()


def test_bubbles_count_every_team_season_and_confusion_shares_condition_on_actual_rows() -> None:
    teams = pd.DataFrame(
        dict(
            season_id=[14, 15, 16],
            team_name=["א", "ב", "ג"],
            position=[6, 6, 7],
            elo_rank=[10, 10, 3],
        )
    )
    figure = alignment_chart(teams)
    assert sorted(figure.data[0].marker.size) == [1, 2]
    assert sum(figure.data[0].marker.size) == 3 and len(figure.layout.shapes) == 2
    matrix = pd.DataFrame(
        [[2, 0, 1], [1, 0, 1], [0, 0, 2]],
        index=["בית", "תיקו", "חוץ"],
        columns=["בית", "תיקו", "חוץ"],
    )
    chart = confusion_chart(matrix)
    assert "66.7%" in chart.data[0].text[0, 0]
    precision = choice_precision(matrix).set_index("outcome")
    assert precision.at["בית", "precision"] == pytest.approx(2 / 3)
    assert pd.isna(precision.at["תיקו", "precision"])
