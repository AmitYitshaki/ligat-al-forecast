"""Acceptance contracts for the approved notebook-04 walkthrough, using synthetic data."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ligat.analysis import motivation_review as mr
from ligat.analysis import motivation_walkthrough as mw
from ligat.viz.motivation import _finish, normalize_minus
from ligat.viz.motivation_style import two_se


@pytest.mark.parametrize("text", ["בכ-30", "ה-0.36", "−0.20", "EXP-004", "T-013", "E0050"])
def test_minus_preserves_prefixes_and_identifiers(text):
    assert normalize_minus(text) == text


def test_two_se_is_explicit_and_finalization_is_idempotent():
    raw = go.Figure(go.Scatter(x=[1], y=[2], error_y=dict(array=[0.1])))
    with pytest.raises(ValueError, match="two_se"):
        _finish(raw, "raw")
    fig = go.Figure(go.Scatter(x=[1], y=[2], **two_se([0.1])))
    first = _finish(fig, "בכ-30 EXP-004")
    second = _finish(first, "בכ-30 EXP-004")
    np.testing.assert_array_equal(first.data[0].error_y.array, [0.2])
    np.testing.assert_array_equal(second.data[0].error_y.array, [0.2])
    assert "פס = פעמיים SE" in second.layout.legend.title.text
    assert not any(c in second.to_json() for c in ("\u200e", "\u200f"))
    assert "-30" in second.layout.title.text and "EXP-004" in second.layout.title.text
    assert second.layout.xaxis.automargin is None  # no axis title was supplied


def test_survival_profile_keeps_sign_bins_and_quartiles_and_excludes_decided():
    frame = pd.DataFrame(
        dict(
            nothing_to_play=[False] * 8 + [True, False],
            relegation_decided=[False] * 9 + [True],
            gap_to_safety=[-1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
            games_remaining_stage=[1, 1, 2, 3, 4, 6, 7, 8, 1, 1],
            survival_lev=[0.1, 0.3, 0.2, 0.6, 0.4, 0.8, 0.5, 0.9, 99.0, 99.0],
        )
    )
    result = mw.signed_survival_profile(frame).set_index(["gap_to_safety", "remaining_bin"])
    assert len(result) == 4 and result.n.sum() == 8
    row = result.loc[(-1, "≤1")]
    assert (row["median"], row.q25, row.q75) == pytest.approx((0.2, 0.15, 0.25))
    assert result.loc[(1, "7+"), "median"] == pytest.approx(0.7)


def contrast_frame():
    return pd.DataFrame(
        dict(
            game_id=range(8),
            season_id=[11, 12, 11, 12, 11, 11, 12, 12],
            stratum=["a"] * 4 + ["b"] * 3 + ["c"],
            y=[2.0, 4.0, 0.0, 2.0, 10.0, 4.0, 6.0, 100.0],
            selected=[True, True, False, False, True, False, False, True],
        )
    )


def test_default_contrast_preserves_a1f3af7_estimate_and_bootstrap_se():
    # Frozen from the function at a1f3af7, not calculated by the implementation under test.
    frame = contrast_frame()
    actual = mr.within_strata_contrast(frame, frame.selected, "y", repeats=100, seed=8)
    assert actual == pytest.approx(
        dict(
            estimate=3.0, se=0.680092091209163, n=4, n_games=4, n_seasons=2, n_used=3, n_excluded=1
        ),
        abs=1e-12,
    )


def test_contributions_exclude_unmatched_and_reconstruct_total():
    frame = contrast_frame()
    result, layers = mr.stratified_decomposition(frame, frame.selected, "y", repeats=100, seed=8)
    assert layers.contribution.sum() == pytest.approx(result["estimate"], abs=1e-12)
    assert not layers.set_index("stratum").loc["c", "included"]


def test_staircase_uses_all_alive_including_selected_within_each_stratum():
    # stratum a: level .25 mean 2 minus all mean 5 => -3, weight 2;
    # stratum b: level .25 mean 12 minus all mean 10 => +2, weight 1.
    frame = pd.DataFrame(
        dict(
            game_id=range(7),
            season_id=[11, 12, 11, 12, 11, 12, 12],
            stratum=["a"] * 4 + ["b"] * 2 + ["a"],
            nothing_to_play=[False] * 6 + [True],
            L5=[0.25, 0.25, 0.5, 1.0, 0.25, 0.75, 0.0],
            res_pts=[1.0, 3.0, 7.0, 9.0, 12.0, 8.0, 999.0],
            res_ga=[1.0, 3.0, 7.0, 9.0, 12.0, 8.0, 999.0],
        )
    )
    result = mw.alive_staircase(frame, seed=8, repeats=100)
    row = result.loc[result.level.eq(0.25) & result.metric.eq("res_pts")].iloc[0]
    assert row.estimate == pytest.approx(-4 / 3, abs=1e-12)
    assert row.n == 3 and result.loc[result.metric.eq("res_pts"), "n"].sum() == 6


def test_typical_effect_scales_the_same_bootstrap_draws():
    draws = np.array([0.2, 0.5, 0.8])
    gap = 0.05
    slopes = pd.DataFrame(
        dict(
            estimate=[0.5],
            se=[draws.std(ddof=1)],
            median_diff=[gap],
            median_effect=[0.5 * gap],
            median_effect_se=[(draws * gap).std(ddof=1)],
        )
    )
    actual = mw.typical_effects(slopes).iloc[0]
    assert actual.estimate == pytest.approx(0.5 * gap)
    assert actual.se == pytest.approx(draws.std(ddof=1) * gap)


def test_visible_scanner_finds_real_terms_without_rewriting_and_ignores_code_css():
    notebook = {
        "cells": [
            dict(cell_type="markdown", source="A5 survival_direct nan"),
            dict(
                cell_type="code",
                source="A5",
                outputs=[
                    dict(data={"text/html": "<style>A5</style><b>תקין</b>", "text/plain": "A5"})
                ],
            ),
        ]
    }
    assert {v["term"] for v in mw.visible_text_violations(notebook)} == {
        "A5",
        "survival_direct",
        "nan",
    }
    notebook["cells"][0]["source"] = "תיאור תקין EXP-004 בכ-30"
    assert mw.visible_text_violations(notebook) == []
    notebook["cells"].extend(
        [
            dict(cell_type="markdown", source="<h2>6.10 — בדיקה</h2>"),
            dict(
                cell_type="code",
                source="",
                outputs=[
                    dict(
                        data={
                            "text/html": "<table><tr><th>שם טכני</th><th>מה הטענה</th></tr>"
                            "<tr><td>A5</td><td>בדיקה בעברית</td></tr></table>"
                        }
                    )
                ],
            ),
        ]
    )
    assert mw.visible_text_violations(notebook) == []
    notebook["cells"][-1]["outputs"][0]["data"]["text/html"] += "<p>A5</p>"
    assert [v["term"] for v in mw.visible_text_violations(notebook)] == ["A5"]


@pytest.mark.parametrize("extra", [" 17", " (0.2)"])
def test_unregistered_bottom_line_number_fails_even_in_parentheses(extra):
    fragment = '<span data-claim="value">0.20</span>'
    assert mw.verify_bottom_line(fragment, {"value": (0.2, 2)}).matches.all()
    with pytest.raises(AssertionError, match="Unregistered"):
        mw.verify_bottom_line(fragment + extra, {"value": (0.2, 2)})
