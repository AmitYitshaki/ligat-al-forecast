"""RTL C9–C14 presentation; diverging correlations and zero-based outcome bars."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.prematch_correlations import OutcomeAssociation
from ligat.analysis.prematch_final_review import FEATURE_NAMES, WORDS
from ligat.viz.league_stats import DRAW_COLOR, GOOD_SCALE
from ligat.viz.prematch import STAGES
from ligat.viz.prematch_b import _finish_chart
from ligat.viz.prematch_context import save_context_figures
from ligat.viz.presentation import TableDisplay, display_table

NAMES = {
    "group": "קבוצת עמודות",
    "columns": "עמודות",
    "source": "מקור",
    "feature": "פיצ׳ר",
    "feature_a": "פיצ׳ר א",
    "feature_b": "פיצ׳ר ב",
    "target": "יעד במשחק",
    "family_a": "משפחה א",
    "family_b": "משפחה ב",
    "family": "משפחה",
    "n": "תצפיות",
    "pearson": "Pearson",
    "pearson_last5": "Pearson ב־5 האחרונות",
    "spearman": "Spearman",
    "pearson_min_season": "מינימום בעונה",
    "pearson_max_season": "מקסימום בעונה",
    "median_season_pearson": "חציון בין העונות",
    "seasons_same_sign": "עונות באותו סימן",
    "seasons_counted": "עונות שנמדדו",
    "missing": "חסרים",
    "missing_first": "חסרים ב־2006/2007",
    "missing_later": "חסרים בהמשך",
    "pair": "זוג עמודות",
    "value": "ערך מחושב",
    "expected": "ערך בקרה",
    "passed": "עבר",
    "quantity": "בקרה",
    "games": "משחקים",
    "draw_percent": "תיקו (%)",
    "mean_elo_diff": "פער Elo ממוצע, מנקודת מבט הבית",
    "union_games": "משחקים באיחוד",
    "cramers_v": "Cramér's V",
    "cramers_v_last5": "V ב־5 האחרונות",
    "chi2": "χ²",
    "dof": "דרגות חופש",
    "p_value": "p — תיאורי בלבד",
    "min_expected": "ספירה צפויה מינימלית",
    "expected_below_5": "תא צפוי מתחת ל־5",
    "categories": "קטגוריות",
    "excluded_reason": "סיבת החרגה",
    "artifact": "תוצר / רכיב",
    "status": "תוכן / מעמד",
    "decision": "החלטה",
    "change": "שינוי בין התקופות",
}
FAMILIES = {
    "table": "טבלה",
    "elo": "Elo",
    "form": "כושר",
    "opponent_form": "כושר היריבה",
    "context": "הקשר עונתי",
    "contrast": "הפרשים",
}


def final_table(
    frame: pd.DataFrame, *, columns: list[str] | None = None, words_label: str = WORDS
) -> TableDisplay:
    """The shared numeric/bidi formatter with words beside every displayed method."""
    data = frame.copy()
    if WORDS not in data:
        raise ValueError("Final review tables require explanatory words")
    if "p_value" in data:
        data["p_value"] = data.p_value.map(lambda v: f"{v:.3g}" if pd.notna(v) else "—")
    for column in ("family", "family_a", "family_b"):
        if column in data:
            data[column] = data[column].replace(FAMILIES)
    if "excluded_reason" in data:
        data["excluded_reason"] = data.excluded_reason.replace(
            {"constant in home rows": "קבוע בשורות הבית"}
        )
    chosen = list(data) if columns is None else columns
    precision = dict.fromkeys(chosen, 5)
    for column in (
        "columns",
        "n",
        "missing",
        "missing_first",
        "missing_later",
        "games",
        "union_games",
        "seasons_same_sign",
        "seasons_counted",
        "categories",
        "dof",
    ):
        precision[column] = 0
    return display_table(
        data,
        {c: words_label if c == WORDS else NAMES.get(c, c) for c in chosen},
        columns=chosen,
        show_index=False,
        precision=precision,
        unicode_minus=tuple(chosen),
    )


def example_table(example: pd.DataFrame) -> TableDisplay:
    """Two team rows transposed side by side so mirrored contrasts are easy to inspect."""
    data = example[["feature", "home", "away", WORDS]].copy()
    for index, row in data.iterrows():
        for side in ("home", "away"):
            value = row[side]
            if row.feature in ("date", "as_of"):
                data.at[index, side] = pd.Timestamp(value).strftime("%d.%m.%Y")
            elif row.feature == "stage":
                data.at[index, side] = STAGES[str(value)]
            elif row.feature in ("points", "goals_for", "goals_against"):
                data.at[index, side] = str(int(value))
    return display_table(
        data,
        {
            "feature": "שדה",
            "home": "בית: " + example.home_name.iloc[0],
            "away": "חוץ: " + example.away_name.iloc[0],
            WORDS: "מה זה, במילים",
        },
        show_index=False,
        precision={"home": 6, "away": 6},
        unicode_minus=("home", "away"),
    )


def contrast_heatmap(matrices: dict[str, pd.DataFrame]) -> Any:
    """One signed Pearson scale centered exactly at zero in both time windows."""
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=["כל עונות הפיתוח", "5 האחרונות"], horizontal_spacing=0.18
    )
    for col, key in enumerate(("pearson_all", "pearson_last5"), 1):
        matrix = matrices[key]
        labels = [FEATURE_NAMES[c] for c in matrix]
        figure.add_trace(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=labels,
                y=labels,
                coloraxis="coloraxis",
                texttemplate="%{z:.2f}",
                hovertemplate="%{x}<br>%{y}<br>Pearson: %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        figure.update_yaxes(autorange="reversed", row=1, col=col)
        figure.update_xaxes(tickangle=-25, row=1, col=col)
    figure.update_layout(title="הפרשי t−1 ו-Elo: חפיפה גבוהה נשארת גם בליגה העכשווית")
    _finish_chart(figure, height=720)
    # The shared B finish applies Viridis; the C6b rule explicitly overrides it.
    figure.update_coloraxes(
        colorscale="RdBu", cmin=-1, cmax=1, cmid=0, colorbar=dict(title="Pearson", thickness=18)
    )
    figure.update_layout(margin=dict(l=250, r=110, t=120, b=180))
    return figure


def target_comparison_chart(frame: pd.DataFrame) -> Any:
    """Points, not truncated bars: compare all seasons with the last five."""
    labels = frame.feature.map(FEATURE_NAMES)
    figure = go.Figure()
    for column, name, color in (
        ("pearson", "כל העונות", GOOD_SCALE[7]),
        ("pearson_last5", "5 האחרונות", GOOD_SCALE[3]),
    ):
        figure.add_trace(
            go.Scatter(
                x=frame[column],
                y=labels,
                mode="markers",
                name=name,
                marker=dict(size=13, color=color),
                customdata=frame.feature,
                hovertemplate="%{y}<br>%{customdata}<br>Pearson: %{x:.5f}<extra>"
                + name
                + "</extra>",
            )
        )
    for row, label in zip(frame.itertuples(), labels, strict=True):
        figure.add_annotation(
            x=max(row.pearson, row.pearson_last5) + 0.025,
            y=label,
            text=f"Δ {row.change:+.3f}".replace("-", "−"),
            showarrow=False,
            xanchor="left",
        )
    figure.update_layout(
        title="קשר להפרש השערים במשחק: השווי מתחזק יותר מ-Elo",
        xaxis_title="Pearson — קשר תיאורי, ללא p-value",
    )
    figure.update_xaxes(range=[0, float(frame[["pearson", "pearson_last5"]].max().max()) + 0.13])
    figure.update_yaxes(categoryorder="array", categoryarray=labels.tolist(), autorange="reversed")
    _finish_chart(figure, height=600)
    figure.update_layout(
        margin=dict(l=300, r=110, t=145, b=100), legend=dict(orientation="h", y=1.12, x=0)
    )
    return figure


def trend_description(frame: pd.DataFrame) -> str:
    """Calculate the change and its ratio from full-precision prematch correlations."""
    changes = frame.set_index("feature").change
    value = float(changes["value_avg_log_ratio"])
    expected = float(changes["elo_expected"])
    difference = float(changes["elo_diff"])
    if min(expected, difference) <= 0:
        raise ValueError("Expected positive Elo changes for the declared C11 illustration")
    low, high = sorted([value / expected, value / difference])
    return (
        '<div dir="rtl"><p><b>הסבר נתונים:</b> השווי מתחזק ב־'
        f'<bdi dir="ltr">{value:+.3f}</bdi>, ציפיית Elo ב־'
        f'<bdi dir="ltr">{expected:+.3f}</bdi>, ופער Elo ב־'
        f'<bdi dir="ltr">{difference:+.3f}</bdi>. השווי מתחזק יותר משניהם, פי '
        f'<bdi dir="ltr">{low:.1f}–{high:.1f}</bdi>. '
        "זו התאמה תיאורית ל-EXP-003, לא הוכחה לתוספת חיזוי או לסיבתיות.</p>"
        "<p><b>הסבר כדורגלני:</b> בשנים האחרונות הערכת השוק של מועדון קשורה יותר "
        "לתוצאה על הדשא. אם היא מוסיפה מידע אחרי הדירוג והכושר, זו שאלה לניסוי "
        "ול-ablation של M2; מתאם לבדו אינו מכריע.</p></div>"
    )


def cramers_chart(statistics: pd.DataFrame) -> Any:
    """All calculable features, sorted V, with zero baseline and outlined low-count bars."""
    data = statistics.loc[statistics.excluded_reason.eq("")].sort_values(
        "cramers_v", ascending=True
    )
    labels = data.feature.map(FEATURE_NAMES)
    warnings = data.expected_below_5.fillna(False).astype(bool)
    figure = go.Figure(
        go.Bar(
            x=data.cramers_v,
            y=labels,
            orientation="h",
            marker=dict(
                color=GOOD_SCALE[7],
                line=dict(color="black", width=[2 if w else 0 for w in warnings]),
            ),
            text=[
                f"{v:.3f}" + (" *" if w else "")
                for v, w in zip(data.cramers_v, warnings, strict=True)
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack([data.feature, data.n, data.min_expected]),
            hovertemplate=(
                "%{y}<br>%{customdata[0]}<br>V: %{x:.4f}<br>n: %{customdata[1]}"
                "<br>ספירה צפויה מינימלית: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="קשר לבית/תיקו/חוץ: גודל הקשר קטן; * תא צפוי מתחת ל-5",
        xaxis_title="Cramér's V — שורות בית בלבד",
    )
    figure.update_xaxes(range=[0, float(data.cramers_v.max()) * 1.18])
    figure.update_yaxes(categoryorder="array", categoryarray=labels.tolist())
    _finish_chart(figure, height=1600)
    figure.update_layout(margin=dict(l=355, r=110, t=120, b=90))
    return figure


def outcome_chart(outcomes: OutcomeAssociation, features: tuple[str, ...], title: str) -> Any:
    """Stacked home/draw/away percentages; dashed lines mark full-sample boundaries."""
    titles = ["קבוצת הבית" if not f.startswith("away_") else "הקבוצה האורחת" for f in features]
    figure = make_subplots(
        rows=1, cols=len(features), subplot_titles=titles, horizontal_spacing=0.15
    )
    for col, feature in enumerate(features, 1):
        data = outcomes.percentages.loc[outcomes.percentages.feature.eq(feature)].copy()
        if data.empty:
            raise ValueError("No saved C5 categories for outcome chart")
        categories = data.category.tolist()
        if set(categories) <= {"True", "False"}:
            labels = [
                ("כן" if c == "True" else "לא") + f" (n={n})"
                for c, n in zip(categories, data.n, strict=True)
            ]
        else:
            labels = [f"חמישון {i} (n={n})" for i, n in enumerate(data.n, 1)]
        for outcome, label, color in (
            ("H", "ניצחון בית", GOOD_SCALE[7]),
            ("D", "תיקו", DRAW_COLOR),
            ("A", "ניצחון חוץ", GOOD_SCALE[3]),
        ):
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=data[f"{outcome}_percent"],
                    name=label,
                    legendgroup=outcome,
                    showlegend=col == 1,
                    marker_color=color,
                    text=[f"{p:.1f}%" for p in data[f"{outcome}_percent"]],
                    textposition="inside",
                    insidetextanchor="middle",
                    customdata=np.column_stack([categories, data.n]),
                    hovertemplate="%{x}<br>%{customdata[0]}<br>%{y:.2f}%<extra>"
                    + label
                    + "</extra>",
                ),
                row=1,
                col=col,
            )
        base = data.iloc[0]
        for boundary in (
            base.baseline_H_percent,
            base.baseline_H_percent + base.baseline_D_percent,
        ):
            figure.add_hline(
                y=boundary, line_color="black", line_dash="dot", line_width=1, row=1, col=col
            )
        figure.add_annotation(
            x=0.5,
            y=1.14,
            xref="x domain" if col == 1 else f"x{col} domain",
            yref="y domain" if col == 1 else f"y{col} domain",
            showarrow=False,
            text=(
                f"כלל המדגם (n={base.baseline_n}): בית {base.baseline_H_percent:.1f}% / "
                f"תיקו {base.baseline_D_percent:.1f}% / חוץ {base.baseline_A_percent:.1f}%"
            ),
            font=dict(size=13),
        )
        figure.update_xaxes(
            categoryorder="array",
            categoryarray=labels,
            tickangle=-15 if len(labels) > 2 else 0,
            row=1,
            col=col,
        )
    figure.update_yaxes(range=[0, 100], ticksuffix="%", title_text="שיעור התוצאות")
    figure.update_layout(title=title, barmode="stack")
    _finish_chart(figure, height=720)
    figure.update_layout(
        margin=dict(l=100, r=100, t=205, b=135), legend=dict(orientation="h", y=-0.24, x=0)
    )
    return figure


def save_final_figures(figures: dict[str, Any], directory: Path) -> dict[str, tuple[str, str]]:
    """Use new filenames only so none of the seven frozen C6a exports are touched."""
    if any(not name.startswith("c6b_") for name in figures):
        raise ValueError("Final review figure names require the c6b_ prefix")
    return save_context_figures(figures, directory)
