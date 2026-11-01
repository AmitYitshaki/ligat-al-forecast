"""Review regressions: fair playoff comparisons, round bins, and held-out score isolation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ligat.analysis.league_stats import season_course
from ligat.analysis.league_stats.poisson_check import (
    poisson_crossfit_comparison,
    poisson_crossfit_predictions,
)
from ligat.analysis.prepare import AnalysisTables


def _tables(games: pd.DataFrame) -> AnalysisTables:
    return AnalysisTables(
        games, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )


def test_playoff_comparison_centres_each_season_and_group(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        dict(
            season_id=season,
            team_id=team,
            final_stage=stage,
            ppm_regular=offset + team,
            ppm_playoff=20 - offset + 2 * team,
        )
        for season, offset in ((12, 3), (13, 7))
        for stage in ("upper", "lower")
        for team in (0, 1, 2)
    ]
    frame = pd.DataFrame(rows).set_index(["season_id", "team_id"])
    monkeypatch.setattr(season_course, "playoff_form", lambda *args: frame.copy())
    source = _tables(pd.DataFrame())
    centred = season_course.playoff_deviations(source)
    assert centred.groupby(["season_id", "final_stage"]).regular_deviation.mean().eq(0).all()
    result = season_course.playoff_persistence(source, n_bootstrap=50)
    assert result.n.eq(6).all() and result.n_seasons.eq(2).all()
    assert result.slope.to_numpy() == pytest.approx([2, 2])
    assert result.pearson.to_numpy() == pytest.approx([1, 1])
    assert result.slope_ci_lower.to_numpy() == pytest.approx([2, 2])
    assert result.slope_ci_upper.to_numpy() == pytest.approx([2, 2])
    with pytest.raises(ValueError):
        season_course.playoff_persistence(source, n_bootstrap=0)


def test_distance_bins_and_wilson_include_zero_and_full_rates() -> None:
    games = pd.DataFrame(
        dict(
            season_id=[12] * 7,
            stage=["regular"] * 7,
            round=[1, 4, 7, 10, 13, 16, 18],
            result=["home_win"] * 5 + ["draw", "draw"],
        )
    )
    original = games.copy(deep=True)
    result = season_course.draw_rates_by_distance(_tables(games))
    regular = result.xs("regular")
    assert list(regular.index) == ["0–2", "3–5", "6–8", "9–11", "12–14", "15+"]
    assert regular.n.sum() == len(games)
    assert regular.loc["0–2", "draw_rate"] == 1
    assert 0 < regular.loc["0–2", "draw_rate_ci_lower"] < 1
    assert regular.loc["15+", "draw_rate"] == 0
    assert regular.loc["15+", "draw_rate_ci_upper"] > 0
    pd.testing.assert_frame_equal(games, original)


@pytest.fixture
def poisson_tables() -> AnalysisTables:
    rng = np.random.default_rng(19)
    games = pd.DataFrame(
        [
            dict(
                game_id=season * 100 + i,
                season_id=season,
                home_team_id=i % 4,
                away_team_id=(i + 1 + i // 4 % 2) % 4,
                home_goals=int(rng.poisson(1.5)),
                away_goals=int(rng.poisson(1)),
            )
            for season in (12, 13)
            for i in range(60)
        ]
    )
    return _tables(games.set_index("game_id"))


def test_crossfit_never_uses_heldout_scores_for_own_lambdas(poisson_tables: AnalysisTables) -> None:
    original = poisson_tables.match_table.copy(deep=True)
    predicted = poisson_crossfit_predictions(poisson_tables)
    changed = original.copy()
    held_out = predicted.index[predicted.fold.eq(0)]
    changed.loc[held_out, ["home_goals", "away_goals"]] = [8, 7]
    altered = poisson_crossfit_predictions(replace(poisson_tables, match_table=changed))
    columns = ["lambda_home", "lambda_away"]
    pd.testing.assert_frame_equal(predicted.loc[held_out, columns], altered.loc[held_out, columns])
    assert (predicted[columns] >= 0).all().all()
    pd.testing.assert_frame_equal(poisson_tables.match_table, original)
    pd.testing.assert_frame_equal(predicted, poisson_crossfit_predictions(poisson_tables))
    shuffled = replace(poisson_tables, match_table=original.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(predicted, poisson_crossfit_predictions(shuffled))
    result = poisson_crossfit_comparison(poisson_tables)
    assert result.loc["all_draws", "observed"] == original.home_goals.eq(original.away_goals).mean()
    assert result.expected.between(0, 1).all()
    assert result.attrs["chronological"] is False


@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
@pytest.mark.parametrize(
    "function",
    [
        season_course.draw_rates_by_distance,
        season_course.playoff_deviations,
        season_course.playoff_persistence,
        poisson_crossfit_predictions,
        poisson_crossfit_comparison,
    ],
)
def test_new_analysis_rejects_non_development(function: object, seasons: list[int]) -> None:
    with pytest.raises(ValueError, match="development"):
        function(_tables(pd.DataFrame()), seasons)  # type: ignore[operator]


def test_crossfit_zero_goals_and_missing_training_club(poisson_tables: AnalysisTables) -> None:
    games = poisson_tables.match_table.copy()
    games[["home_goals", "away_goals"]] = 0
    # A club appearing once is neutral when it is absent from that fold's training half.
    games.iloc[0, games.columns.get_loc("home_team_id")] = 999
    source = replace(poisson_tables, match_table=games)
    result = poisson_crossfit_comparison(source)
    assert result.loc["0:0", "expected"] == 1
    assert result.loc["all_draws", "expected"] == 1
    with pytest.raises(ValueError, match="iterations"):
        poisson_crossfit_predictions(source, iterations=0)


def test_review_table_hides_intervals_and_index_without_changing_values() -> None:
    from bs4 import BeautifulSoup

    from ligat.viz.presentation import display_table

    source = pd.DataFrame({"rate": [0.4], "rate_ci_lower": [0.2], "rate_ci_upper": [0.6]})
    rendered = display_table(
        source,
        {"rate": "שיעור"},
        columns=["rate"],
        percent={"rate"},
        show_intervals=False,
        show_index=False,
    )._repr_html_()
    soup = BeautifulSoup(rendered, "html.parser")
    assert len(soup.select("th")) == len(soup.select("td")) == 1
    assert soup.select_one("td").text == "40.0%"
    assert source.rate_ci_lower.iloc[0] == 0.2


def test_position_chart_orders_all_positions_and_explains_official_penalties() -> None:
    from ligat.viz.league_stats import chapter_a_chart

    source = pd.DataFrame(
        dict(
            season_id=[25, 25, 25],
            team_id=[1, 2, 3],
            team_name=["קבוצה א", "קבוצה ב", "קבוצה ג"],
            era=["14_teams"] * 3,
            final_position=[13, 1, 2],
            points_per_match=[0.9, 2.4, 2.1],
            deduction=[-5, 0, -2],
            season=["2023/24"] * 3,
        )
    ).set_index(["season_id", "team_id"])
    figure = chapter_a_chart("position", source)
    assert figure.layout.xaxis.categoryorder == "array"
    assert list(figure.layout.xaxis.categoryarray) == list(range(1, 15))
    assert (
        "deduction" in figure.data[0].hovertemplate
        or "שינוי נקודות" in figure.data[0].hovertemplate
    )
    assert any(-5 in row for row in figure.data[0].customdata)
    assert any("2023/2024" in row for row in figure.data[0].customdata)


def test_rtl_heatmap_preserves_numeric_text_shape_and_values() -> None:
    import plotly.graph_objects as go

    from ligat.viz.presentation import rtl_figure

    values = np.array([[77.3, 16.5, 6.2], [23.0, 40.0, 37.0]])
    figure = go.Figure(go.Heatmap(z=values, text=values, texttemplate="%{text:.1f}%"))
    rtl_figure(figure)
    assert np.asarray(figure.data[0].text, dtype=float).shape == values.shape
    np.testing.assert_array_equal(np.asarray(figure.data[0].text, dtype=float), values)
    assert "NaN" not in str(figure.data[0].text)
