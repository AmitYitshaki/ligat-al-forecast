"""Display contracts: bidi islands, unchanged coordinates/data and readable nullable intervals."""

import re
from pathlib import Path

import nbformat
import pandas as pd
import plotly.graph_objects as go
import pytest

from ligat.viz.notebook02 import COLUMN_NAMES, PERCENT_COLUMNS, VALUE_NAMES
from ligat.viz.presentation import display_table, rtl_figure, rtl_text


def test_rtl_numeric_islands_and_figure_coordinates() -> None:
    text = "ניצחון בית 15% (φ=1.26), הפרש −0.41 [-0.5, -0.3]; תיקו (%)"
    protected = rtl_text(text)
    assert "\u202a15%\u202c" in protected
    assert "\u202a(φ=1.26)\u202c" in protected
    assert "\u202a−0.41\u202c" in protected
    assert "\u202a[-0.5, -0.3]\u202c" in protected
    assert "\u202a(%)\u202c" in protected
    assert rtl_text(protected) == protected
    assert rtl_text("95% CI") == "95% CI"
    assert rtl_text("בית %{y:.1f}%<br>n=%{customdata[0]}").count("%{") == 2
    dynamic = rtl_text("%{x}: %{y:.1f}%", force=True)
    assert "\u202b%{x}:" in dynamic and "\u202a%{y:.1f}%\u202c" in dynamic
    figure = go.Figure(
        go.Bar(
            x=["בית", "English"],
            y=[0.15, 0.4],
            name="ניצחון 15%",
            hovertemplate="team_name=%{x}: %{y:.1f}%",
        )
    )
    figure.update_layout(title=text, xaxis_title="תוצאה (%)", legend_title="סוג 1")
    figure.add_annotation(text="φ=1.26 אדומים", x="בית", y=0.15)
    before_x, before_y = tuple(figure.data[0].x), tuple(figure.data[0].y)
    rtl_figure(figure, labels=COLUMN_NAMES)
    assert tuple(figure.data[0].x) == before_x and tuple(figure.data[0].y) == before_y
    assert tuple(figure.layout.xaxis.tickvals) == before_x
    for label in [
        figure.layout.title.text,
        figure.layout.xaxis.title.text,
        figure.layout.legend.title.text,
        figure.data[0].name,
        figure.data[0].hovertemplate,
        figure.layout.annotations[0].text,
        figure.layout.xaxis.ticktext[0],
    ]:
        assert "\u202b" in label
    assert figure.layout.xaxis.ticktext[1] == "English"
    assert "team_name" not in figure.data[0].hovertemplate
    assert "%{x}" in figure.data[0].hovertemplate
    once = figure.to_json()
    assert rtl_figure(figure).to_json() == once


def test_table_percent_interval_na_escaping_and_input_integrity() -> None:
    frame = pd.DataFrame(
        {
            "n": [100, 2],
            "home_win_rate": [0.405, pd.NA],
            "home_win_rate_ci_lower": [0.39, pd.NA],
            "home_win_rate_ci_upper": [0.42, pd.NA],
            "goals_per_match": [2.5, 0],
            "team_name": ["מכבי", "<script>bad</script>"],
        },
        index=pd.Index([True, False], name="is_home"),
    )
    original = frame.copy(deep=True)
    view = display_table(
        frame,
        COLUMN_NAMES,
        columns=["n", "home_win_rate", "goals_per_match", "team_name"],
        percent=PERCENT_COLUMNS,
        values=VALUE_NAMES,
    )._repr_html_()
    assert "40.5%" in view and "39.0–42.0" in view
    assert "ci_lower" not in view and "ci_upper" not in view and "טווח 95%" in view
    assert "2.50" in view and "250.0%" not in view  # Count per game is not a probability.
    assert "NA" in view and "בית" in view and "חוץ" in view
    assert "<script>" not in view and "&lt;script&gt;" in view
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="Missing Hebrew"):
        display_table(pd.DataFrame({"untranslated": [1]}), COLUMN_NAMES)

    # Existing analytical schemas expose several naming conventions for the same CI.
    frame = pd.DataFrame(
        {
            "n": [36],
            "fraction_below": [26 / 36],
            "ci_lower": [0.55],
            "ci_upper": [0.85],
            "median_diff": [-0.0039],
        }
    )
    view = display_table(
        frame, COLUMN_NAMES, percent=PERCENT_COLUMNS | {"median_diff"}
    )._repr_html_()
    assert "72.2%" in view and "55.0–85.0" in view and "-0.4%" in view
    aliases = pd.DataFrame(
        {
            "points_per_match_mean": [2.28],
            "points_per_match_ci_lower": [2.1],
            "points_per_match_ci_upper": [2.4],
        }
    )
    view = display_table(aliases, COLUMN_NAMES)._repr_html_()
    assert "2.10–2.40" in view and view.count("טווח 95%") == 1


def test_notebook02_presentation_structure() -> None:
    notebook = nbformat.read(Path("notebooks/explore/02_league_stats.ipynb"), as_version=4)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert i > 0 and notebook.cells[i - 1].cell_type == "markdown"
            continue
        if cell.cell_type != "markdown":
            continue
        if i + 1 < len(notebook.cells) and notebook.cells[i + 1].cell_type == "code":
            body = cell.source.removeprefix('<div dir="rtl">').strip()
            assert body.startswith("**מה התא עושה:**")
            assert len(body.splitlines()[0].split("**")[-1].split()) <= 12
            assert "**השאלה:**" in body and "**התשובה הכדורגלנית:**" in body
            assert ("**מה לחפש:**" in body) == (".show()" in notebook.cells[i + 1].source)
        # nbstripout renumbers cell IDs in Git; recognize the two long sections by content.
        if re.search(r"^## (תקציר|סיכום המחברת)", cell.source, re.M):
            continue
        limit = 600 if "איך לקרוא את הפרק" in cell.source else 500
        assert len(cell.source) <= limit
        if "איך לקרוא את הפרק" in cell.source:
            assert len(re.findall(r"^- ", cell.source, re.M)) <= 5
