"""New notebook figures use the approved palette and name clubs at the display boundary."""

import re

import numpy as np
import pandas as pd

from ligat.analysis.motivation_review import distribution_tables
from ligat.viz.motivation import (
    block_estimates,
    comparison_reading,
    decomposition_plot,
    format_date,
    gap_scatter,
    importance_by_remaining,
    importance_histogram,
    normalize_minus,
    residual_forest,
    seasonal_estimates,
    stage_residual_plot,
    table,
    timing_line,
    title_timeline,
    walk_forward,
)


def test_all_motivation_figures_build_on_synthetic_data_and_tables_hide_club_keys():
    frame = pd.DataFrame(
        dict(
            game_id=[1, 1, 2, 2],
            team_id=[1001, 1002, 1001, 1002],
            season_id=[11, 11, 12, 12],
            stage=["regular"] * 4,
            leverage_max=[0.0, 0.2, 0.3, 0.8],
            title_lev=[0.0, 0.2, 0.3, 0.8],
            cut_lev=[0.0] * 4,
            survival_lev=[0.0] * 4,
            games_remaining_stage=[2, 2, 1, 1],
            gap_to_safety=[-1, 1, 0, 0],
            nothing_to_play=[False] * 4,
            title_decided=[False, False, True, True],
            as_of=pd.date_range("2010-01-01", periods=4, tz="Asia/Jerusalem"),
        )
    )
    frame = frame.set_index(["game_id", "team_id"])
    names = {1001: "קבוצה א", 1002: "קבוצה ב"}
    distributions = distribution_tables(frame)
    contrasts = pd.DataFrame(
        [
            dict(
                representation="L2",
                category="dead",
                metric=m,
                method="contrast",
                estimate=0.1,
                se=0.05,
                n=50,
                n_used=48,
                n_excluded=2,
                n_games=45,
                n_seasons=2,
            )
            for m in ("res_pts", "res_gf", "res_ga")
        ]
    )
    seasons = pd.DataFrame(
        [
            dict(
                representation=r,
                category=c,
                metric="res_pts",
                season_id=s,
                estimate=0.1,
                n_used=40,
                n_games=35,
                n_seasons=1,
            )
            for r, c in (("L2", "dead"), ("Lc", "home_minus_away"))
            for s in (11, 12)
        ]
    )
    plan = pd.DataFrame(dict(season_id=[11, 12], fold=[1, 2], deep=[False, True]))
    timeline = title_timeline(
        frame.loc[frame.index.get_level_values("team_id") == 1001].iloc[::-1], "קבוצה א"
    )
    assert list(timeline.data[0].x) == sorted(timeline.data[0].x)
    figures = [
        importance_histogram(frame, distributions["overall"]),
        importance_by_remaining(distributions["distance"]),
        gap_scatter(frame, names=names),
        residual_forest(contrasts),
        seasonal_estimates(seasons),
        walk_forward(plan),
        timeline,
        block_estimates(
            pd.DataFrame(
                [
                    dict(
                        comparison_id=k,
                        metric="res_pts",
                        block=b,
                        estimate=0.1,
                        se=0.05,
                        n=50,
                        n_used=48,
                        n_games=45,
                        n_seasons=5,
                    )
                    for k in ("dead", "all_Lc", "live_Lc")
                    for b in ("B1", "B2", "B3")
                ]
            )
        ),
    ]
    assert all(fig.layout.height in (550, 850, 1150) for fig in figures)
    assert all(fig.layout.font.family == "Arial" for fig in figures)
    output = table(
        frame.assign(n=1, n_games=1, n_seasons=1),
        names=names,
        columns=["team_name", "leverage_max", "n", "n_games", "n_seasons"],
    )
    html = output._repr_html_()
    assert "קבוצה א" in html and "1001" not in html
    assert np.isfinite(distributions["overall"]["mean"]).all()


def test_local_dates_have_unpadded_day_month_and_correct_israel_dst():
    assert format_date(pd.Timestamp("2011-04-17", tz="Asia/Jerusalem")) == "17.4.2011"
    assert (
        format_date(pd.Timestamp("2020-07-02 17:15", tz="UTC"), with_time=True) == "2.7.2020 20:15"
    )
    assert (
        format_date(pd.Timestamp("2020-03-26 22:30", tz="UTC"), with_time=True) == "27.3.2020 00:30"
    )
    assert (
        format_date(pd.Timestamp("2020-03-27 01:30", tz="UTC"), with_time=True) == "27.3.2020 04:30"
    )


def test_generated_numeric_prose_uses_mathematical_minus():
    result = normalize_minus("אפקט -0.36, SE 0.1, סף 1e-9; EXP-004")
    assert "−0.36" in result and "1e−9" in result and "EXP−004" in result
    assert not re.search(r"-\d", result)
    assert not re.search(r"-\d", timing_line(0.2, computed_at="2026-11-15")._repr_html_())


def test_elo_display_is_integer_and_new_plot_has_two_series():
    html = table(
        pd.DataFrame(dict(elo_pre=[1500.49], dead_mean=[-0.125])),
        names={},
        columns=["elo_pre", "dead_mean"],
        labels={"dead_mean": "מתות"},
    )._repr_html_()
    assert ">1500<" in html and "1500.49" not in html and "−0.1250" in html
    frame = pd.DataFrame(
        [
            dict(
                scope=b,
                metric="res_pts",
                dead_mean=-0.1,
                control_mean=0.2,
                dead_mean_se=0.05,
                control_mean_se=0.04,
                n_used=30,
                n_excluded=0,
                n_games=20,
                n_seasons=5,
            )
            for b in ("B1", "B2", "B3")
        ]
    )
    fig = decomposition_plot(frame)
    assert len(fig.data) == 2 and all(len(trace.y) == 3 for trace in fig.data)
    assert len(fig.layout.shapes) == 1


def test_comparison_reading_respects_minimum_coverage_before_size_of_effect():
    assert comparison_reading(18, -10.0, 0.001) == "לא מספיק משחקים להשוואה"
    assert comparison_reading(29, -10.0, 0.001) == "לא מספיק משחקים להשוואה"
    assert "תיאורי בלבד" in comparison_reading(30, -10.0, 0.001)


def test_stage_plot_preserves_missing_cells_and_marks_small_samples():
    frame = pd.DataFrame(
        [
            dict(
                stage=stage,
                population=pop,
                n_games=n,
                n_seasons=2,
                goals_residual=value,
                goals_se=0.1,
            )
            for pop in ("clean", "all_games")
            for stage, n, value in (
                ("regular", 100, 0.01),
                ("upper", 30, 0.1),
                ("lower", 17, 0.2),
                ("middle", 0, np.nan),
            )
        ]
    )
    fig = stage_residual_plot(frame)
    assert len(fig.data) == 2
    assert list(fig.data[0].marker.symbol) == ["circle", "circle", "circle-open", "circle-open"]
    assert pd.isna(fig.data[0].y[-1])
    assert len(fig.layout.shapes) == 1
