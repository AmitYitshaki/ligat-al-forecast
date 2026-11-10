"""R3 descriptive computations and presentation keep saved numerical inputs intact."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.stats import poisson

from ligat.analysis.prematch_b_details import decay_profiles
from ligat.analysis.prematch_b_review import WORDS
from ligat.analysis.prematch_form_details import (
    coefficient_effects,
    draw_uncertainty,
    implied_home_advantage,
    memory_curves,
    poisson_example,
    venue_prior_calendar,
)
from ligat.evaluate.goal_metrics import poisson_outcomes
from ligat.viz.prematch_b_details import decay_chart
from ligat.viz.prematch_form import form_table
from ligat.viz.presentation import rtl_figure


def test_implied_home_advantage_recovers_the_known_logistic_offset() -> None:
    home = pd.DataFrame(dict(game_id=[1, 2, 3, 4], season_id=21, elo_diff=0.0))
    predictions = home.drop(columns="elo_diff").assign(outcome=[0, 0, 1, 2])
    original = home.copy(deep=True)
    result = implied_home_advantage(home, predictions)
    assert result.iloc[0].actual_score == pytest.approx(0.625)
    assert result.iloc[0].implied_hfa == pytest.approx(400 * np.log10(5 / 3))
    pd.testing.assert_frame_equal(home, original)


def test_implied_home_advantage_rejects_missing_duplicate_and_held_out_games() -> None:
    home = pd.DataFrame(dict(game_id=[1, 2], season_id=21, elo_diff=[-20.0, 20.0]))
    predictions = home.drop(columns="elo_diff").assign(outcome=[0, 2])
    with pytest.raises(ValueError, match="Missing"):
        implied_home_advantage(home.iloc[:1], predictions)
    with pytest.raises(ValueError, match="exactly one"):
        implied_home_advantage(pd.concat([home, home]), predictions)
    with pytest.raises(ValueError, match="development"):
        implied_home_advantage(home.assign(season_id=26), predictions.assign(season_id=26))


def test_poisson_example_is_not_renormalized_after_showing_only_five_scores() -> None:
    result = poisson_example()
    assert result.probability.sum() == pytest.approx(poisson.cdf(4, 1.5))
    assert result.iloc[0].probability == pytest.approx(np.exp(-1.5))
    assert result.iloc[4].nll == pytest.approx(3.056193397915288)
    assert result[WORDS].is_unique
    with pytest.raises(ValueError, match="positive"):
        poisson_example(0)


def test_memory_curves_separate_the_sixth_appearance_from_a_window_cutoff() -> None:
    result = memory_curves()
    sixth = result.loc[result.age.eq(6)].set_index("method")
    assert sixth.at["חלון 5", "weight"] == 0
    assert (sixth.loc[["EWMA 5", "EWMA 10", "EWMA 20"], "weight"] > 0).all()
    ewma = result.loc[result.method.eq("EWMA 5")].set_index("age").weight
    assert ewma.at[6] == pytest.approx(ewma.at[1] / 2)


def test_calendar_prior_uses_saved_season_counts_instead_of_assuming_eighteen() -> None:
    frame = pd.DataFrame(dict(season_id=[14] * 32, team_id=[1] * 15 + [2] * 17, is_home=True))
    result = venue_prior_calendar(frame)
    assert result.iloc[-1].venue_games == 16
    assert result.iloc[-1].approx_rounds == 32
    assert result.iloc[-1].prior_mass == pytest.approx(0.5 ** (16 / 5))
    assert result[WORDS].is_unique


def test_elo_translation_changes_both_goal_rates_in_opposite_directions() -> None:
    review = SimpleNamespace(
        fits=pd.DataFrame(
            [
                dict(
                    config_id="P_ewma5_venue_elo",
                    season_id=9,
                    coefficients=np.array([0, 0, 0, 0, 1.2]),
                    training_seasons=[8],
                )
            ]
        ),
        metadata={"coefficients": [dict(season_id=9, metric="gf", stratum="all", league_mean=1.5)]},
        elo=pd.DataFrame(dict(season_id=[8, 8])),
    )
    result = coefficient_effects(review).iloc[0]
    expected, _ = poisson_outcomes([[1, 1], [np.exp(0.3), np.exp(-0.3)]])
    assert result.elo_rate_percent == pytest.approx(100 * np.expm1(0.3))
    assert result.elo_win_pp == pytest.approx(100 * (expected[1, 0] - expected[0, 0]))
    assert result.home_win_change_pp == 0


def test_draw_rate_se_and_exact_count_noise_use_their_distinct_denominators() -> None:
    rates = np.column_stack([1 + np.arange(9) / 10, np.ones(9)])
    probabilities, _ = poisson_outcomes(rates)
    home = np.array([0, 1, 2, 3, 1, 2, 0, 1, 2])
    away = np.array([0, 0, 2, 1, 1, 0, 1, 0, 2])
    frame = pd.DataFrame(
        dict(
            lambda_home=rates[:, 0],
            lambda_away=rates[:, 1],
            p_home=probabilities[:, 0],
            p_draw=probabilities[:, 1],
            p_away=probabilities[:, 2],
            home_goals=home,
            away_goals=away,
            outcome=np.where(home > away, 0, np.where(home == away, 1, 2)),
        )
    )
    _, strata, exact = draw_uncertainty(frame)
    assert strata.iloc[0].se_pp == pytest.approx(100 * np.sqrt(2 / 27))
    q = np.exp(-rates.sum(axis=1)) / (poisson.cdf(10, rates[:, 0]) * poisson.cdf(10, rates[:, 1]))
    assert exact.iloc[0].expected_count == pytest.approx(q.sum())
    assert exact.iloc[0].count_noise == pytest.approx(np.sqrt(np.sum(q * (1 - q))))
    assert exact.iloc[0].observed_count == 1


def test_candidate_highlight_is_presentation_only() -> None:
    data = pd.DataFrame(
        dict(config_id=["a", "b"], method=["one", "two"], **{WORDS: ["ראשון", "שני"]})
    )
    original = data.copy(deep=True)
    content = form_table(data, columns=["method", WORDS], highlight_id="b").content
    assert "tr:nth-child(2) td{background:#dbeafe" in content
    pd.testing.assert_frame_equal(data, original)


def test_shared_spacing_is_idempotent_without_changing_trace_values() -> None:
    figure = go.Figure(go.Scatter(x=[1, 2], y=[-0.01, 0.02]))
    figure.update_layout(xaxis_title="עונה", yaxis_title="שיעור")
    rtl_figure(figure)
    first = figure.to_plotly_json()
    rtl_figure(figure)
    assert figure.to_plotly_json() == first
    assert figure.layout.yaxis.title.standoff >= 65
    assert figure.layout.xaxis.title.standoff >= 55
    assert list(figure.data[0].y) == [-0.01, 0.02]


def test_decay_colors_pair_each_curve_with_the_same_summary_point() -> None:
    profiles, _ = decay_profiles()
    changes = pd.DataFrame(
        dict(
            difference=[0, 0.001, 0.0001, -0.0001],
            se=[0, 0.0002, 0.0002, 0.0002],
            log_loss=[1.02] * 4,
            decay=["none", "1", "2", "4"],
        )
    )
    figure = decay_chart(profiles, changes, 0.065)
    colors = [trace.line.color for trace in figure.data[:4]]
    assert len(set(colors)) == 4
    assert list(figure.data[4].marker.color) == colors
    np.testing.assert_array_equal(figure.data[4].y, changes.difference)
