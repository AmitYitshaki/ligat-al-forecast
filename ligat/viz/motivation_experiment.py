"""Hebrew read-only views of EXP-004; every figure uses the shared RTL policy."""

from html import escape
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from IPython.display import HTML
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.motivation_experiment_review import (
    GROUPS,
    PERIODS,
    SCALARS,
    ExperimentReview,
    configuration_label,
    fold_improvements,
)
from ligat.viz.motivation import _finish
from ligat.viz.motivation_style import two_se
from ligat.viz.seasons import season_label

LABELS = {
    "config_id": "תצורה",
    "reference": "בקרה מזווגת",
    "representation": "ייצוג",
    "form": "צורה",
    "decay": "דעיכה",
    "parameters": "מספר פרמטרים כולל",
    "assumption": "ההנחה הכדורגלנית",
    "season": "עונה",
    "season_id": "מזהה עונה",
    "games": "משחקים",
    "all_games": "כל המשחקים בתקופה",
    "metric": "מדד",
    "scope": "אוכלוסייה",
    "period": "תקופה",
    "group": "סוג המשחק",
    "rank": "מקום בדירוג",
    "mean_improvement": "שיפור ממוצע למשחק",
    "contribution": "תרומה לממוצע הכולל",
    "comparison_rule": "כלל ההשוואה",
    "audit_required": "סימון ה-כ22",
    "mean_gt_se": "ממוצע גדול משגיאת התקן",
    "recent_at_least_four": "לפחות ארבע מהחמש האחרונות",
    "geometric_all15_pct": "הסתברות גאומטרית % — כל הקיפולים",
    "geometric_deep_pct": "הסתברות גאומטרית % — עמוקים",
    "log_loss": "log loss",
    "rps": "RPS",
    "reference_log_loss": "log loss של הבקרה",
    "date": "תאריך",
    "stage": "שלב",
    "home_team_name": "בית",
    "away_team_name": "חוץ",
    "result": "תוצאה: בית:חוץ",
    "m_home": "L3 בית",
    "m_away": "L3 חוץ",
    "actual_probability": "הסתברות לתוצאה בפועל — תצורה",
    "reference_probability": "הסתברות לתוצאה בפועל — בקרה",
    "log_loss_improvement": "שיפור log loss",
    "extreme": "קצה",
    "fit_scope": "התאמה",
    "side": "מגרש",
    "component": "רכיב",
    "expected_points_change": "שינוי נקודות צפויות: חיה למתה",
    "elo_diff": "פער Elo",
    "coefficient": "מקדם",
    "value": "ערך",
    "seconds": "שניות",
    "loss_error": "הפרש מרבי בציונים",
    "coefficient_error": "הפרש מרבי במקדמים",
    "folds": "קיפולים",
    "iterations": "איטרציות",
    "max_abs_gradient": "גרדיאנט מוחלט מרבי",
    "converged": "התכנס",
    "fold": "מספר קיפול",
    "training_games": "משחקי אימון",
    "validation_games": "משחקי ולידציה",
    "training_seasons": "עונות אימון",
    "accepted_by_gradient_check": "אושר לפי גרדיאנט",
    "se": "שגיאת תקן בין קיפולים",
    "passes_rule8": "עבר כלל 8",
    "last5_folds_better": "משתפרים מתוך חמש האחרונות",
    "folds_better": "משתפרים בעמוקים",
    "last5_folds": "מספר הקיפולים האחרונים",
    "last5_improvement": "שיפור בחמש האחרונות",
    "log_loss_sum": "סכום שיפור log loss",
    "rps_sum": "סכום שיפור RPS",
    "log_loss_contribution": "תרומת log loss",
    "rps_contribution": "תרומת RPS",
}
for _metric, _name in (("log_loss", "log loss"), ("rps", "RPS")):
    for _scope, _word in (("all15", "כל הקיפולים"), ("deep", "העמוקים")):
        LABELS[f"{_scope}_{_metric}"] = f"{_name} — {_word}"
    for _field, _word in (
        ("mean_improvement", "ממוצע שיפור"),
        ("se", "שגיאת תקן"),
        ("folds_better", "עמוקים שהשתפרו"),
        ("last5_folds_better", "אחרונים שהשתפרו"),
        ("passes_rule8", "עבר כלל 8"),
    ):
        LABELS[f"{_metric}_{_field}"] = f"{_name} — {_word}"


def table(frame: pd.DataFrame) -> HTML:
    """Scrollable complete tables with Hebrew configuration names and full-year seasons."""
    view = frame.copy()
    if "season_id" in view:
        view.insert(0, "season", [season_label(int(s)) for s in view.season_id])
        view = view.drop(columns="season_id")
    for key in ("config_id", "reference"):
        if key in view:
            view[key] = view[key].map(configuration_label)
    words = {
        **GROUPS,
        **PERIODS,
        "home": "בית",
        "away": "חוץ",
        "last_fold": "קיפול אחרון: 2023/2024",
        "deep_mean": "ממוצע התאמות עמוקות",
        "m": "חשיבות",
        "title": "אליפות",
        "cut": "עליון",
        "survival": "הישארות",
        "regular": "סדירה",
        "upper": "עליון",
        "middle": "אמצעי",
        "lower": "תחתון",
        "all15": "2009/2010–2023/2024",
        "one_dead": "צד מת יחיד",
        "neither_dead": "אין צד מת",
        "none": "אין רכיב מוטיבציה",
    }
    for key in ("group", "period", "scope", "side", "fit_scope", "component", "stage"):
        if key in view:
            view[key] = view[key].map(
                lambda v, key=key: (
                    season_label(int(v)) if key == "scope" and str(v).isdigit() else words.get(v, v)
                )
            )
    if "date" in view:
        view["date"] = pd.to_datetime(view.date).dt.strftime("%d.%m.%Y")
    if "training_seasons" in view:
        view["training_seasons"] = view.training_seasons.map(
            lambda v: ", ".join(
                season_label(int(s)) for s in str(v).strip("()[]").replace(",", " ").split()
            )
        )
    view = view.rename(columns=LABELS)
    html = view.to_html(index=False, escape=True, na_rep="—", float_format=lambda x: f"{x:.6g}")
    html = html.replace("True", "כן").replace("False", "לא")
    return HTML(  # type: ignore[no-untyped-call]
        '<div dir="rtl" style="overflow-x:auto;max-height:800px;overflow-y:auto">' + html + "</div>"
    )


def _finalize(fig: Any, title: str, *, height: int = 650, left: int = 100) -> Any:
    """Apply shared RTL formatting and the required Hebrew two-SE legend."""
    fig = _finish(fig, title, height=height)
    fig.update_layout(margin_l=left)
    if any(t.error_x.array is not None for t in fig.data if hasattr(t, "error_x")):
        fig.update_layout(legend_title_text="הפס: פעמיים שגיאת התקן")
        # Replace the shared legend through rtl_figure.
        from ligat.viz.presentation import rtl_figure

        rtl_figure(fig)
        fig.layout.legend.title.text = fig.layout.legend.title.text.replace("\u200e", "").replace(
            "\u200f", ""
        )
    return fig


def forest(review: ExperimentReview) -> Any:
    """Paired deep-fold mean ± two SE, with filled markers for the first entry condition only."""
    summary = review.frames["summary"].iloc[1:]
    labels = [configuration_label(c) for c in summary.config_id]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["log loss", "RPS"], horizontal_spacing=0.08)
    for col, metric in enumerate(("log_loss", "rps"), 1):
        means, se = summary[f"{metric}_mean_improvement"], summary[f"{metric}_se"]
        fig.add_trace(
            go.Scatter(
                x=1000 * means,
                y=list(range(len(labels))),
                mode="markers",
                marker=dict(
                    symbol=[
                        "circle" if d > s else "circle-open" for d, s in zip(means, se, strict=True)
                    ],
                    size=9,
                ),
                **two_se(1000 * se, axis="x"),
                name="גם סמן מלא צריך ארבע מחמש האחרונות",
                showlegend=col == 1,
                hovertext=labels,
                hovertemplate="%{hovertext}<br>%{x:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=col)
        fig.update_xaxes(title="שיפור × אלף; ימינה טוב", row=1, col=col)
        fig.update_yaxes(
            tickvals=list(range(len(labels))),
            ticktext=labels if col == 1 else [""] * len(labels),
            autorange="reversed",
            row=1,
            col=col,
        )
    return _finalize(fig, "6.3 — השיפור קטן ביחס לפיזור בין עונות", height=1050, left=470)


def heatmap(review: ExperimentReview) -> Any:
    """Exact paired fold differences; shade the shallow fold and outline recent seasons."""
    matrix = fold_improvements(review.frames["fold_scores"])
    values = matrix.to_numpy() * 1000
    bound = float(np.max(np.abs(values)))
    fig = go.Figure(
        go.Heatmap(
            x=list(range(15)),
            y=list(range(20)),
            z=values,
            zmid=0,
            zmin=-bound,
            zmax=bound,
            colorscale="RdBu",
            text=values,
            texttemplate="%{text:.1f}",
            textfont_size=11,
            colorbar=dict(title="שיפור × אלף"),
            hovertemplate="%{z:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(
        tickvals=list(range(15)),
        ticktext=[season_label(int(s)) for s in matrix.columns],
        tickangle=-45,
    )
    fig.update_yaxes(
        tickvals=list(range(20)),
        ticktext=[configuration_label(c) for c in matrix.index],
        autorange="reversed",
    )
    fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="gray", opacity=0.22, line_width=0)
    fig.add_shape(type="rect", x0=9.5, x1=14.5, y0=-0.5, y1=19.5, line=dict(color="black", width=3))
    fig.add_annotation(
        x=0,
        y=1.13,
        xref="x",
        yref="paper",
        text="לא נספר: אימון של שלוש עונות",
        showarrow=False,
        xanchor="left",
    )
    return _finalize(
        fig, "6.4 — השיפור בכל עונת ולידציה; המסגרת: חמש האחרונות", height=1050, left=470
    )


def recent_scatter(review: ExperimentReview) -> Any:
    """Deep versus recent improvement, with separate labels for nearby observations."""
    s = review.frames["summary"].iloc[2:]
    fig = go.Figure()
    # Pixel offsets move only labels; the three nearby observations keep exact coordinates.
    label_offsets = {
        "Lc-diff": (110, 8),
        "Lc-diff-hl4": (65, -28),
        "L2-sides-hl4": (75, 32),
    }
    for i, r in enumerate(s.itertuples()):
        fig.add_trace(
            go.Scatter(
                x=[1000 * r.log_loss_mean_improvement],
                y=[1000 * r.log_loss_last5_improvement],
                name=configuration_label(r.config_id),
                mode="markers" if r.config_id in label_offsets else "markers+text",
                text=None if r.config_id in label_offsets else [r.config_id],
                textposition=["top center" if i % 2 else "bottom center"],
                marker_size=9,
            )
        )
        if r.config_id in label_offsets:
            ax, ay = label_offsets[r.config_id]
            fig.add_annotation(
                x=1000 * r.log_loss_mean_improvement,
                y=1000 * r.log_loss_last5_improvement,
                text=r.config_id,
                ax=ax,
                ay=ay,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor="gray",
                bgcolor="white",
            )
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=0, line_dash="dash")
    fig.update_xaxes(title="ממוצע שיפור בעמוקים × אלף")
    fig.update_yaxes(title="ממוצע שיפור בחמש האחרונות × אלף")
    fig = _finalize(fig, "6.5 — שיפור בעבר אינו מבטיח שיפור בליגה העכשווית", height=1000)
    fig.update_layout(legend=dict(orientation="v", x=1.02, y=1), margin_r=440)
    return fig


def group_contributions(frame: pd.DataFrame) -> Any:
    """Exact loss contributions, not estimates of uncertainty; L2/L3 by game type and period."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=[PERIODS[p] for p in ("first9", "last5")])
    for col, period in enumerate(("first9", "last5"), 1):
        for c in SCALARS[:2]:
            part = (
                frame.loc[
                    frame.period.eq(period) & frame.metric.eq("log_loss") & frame.config_id.eq(c)
                ]
                .set_index("group")
                .reindex(GROUPS)
            )
            fig.add_trace(
                go.Bar(
                    x=[GROUPS[g] for g in part.index],
                    y=1000 * part.contribution,
                    name=configuration_label(c),
                    legendgroup=c,
                    showlegend=col == 1,
                    marker_color="#126e82" if c == "L2-diff" else "#df8c22",
                ),
                row=1,
                col=col,
            )
        fig.add_hline(y=0, line_color="black", row=1, col=col)
        fig.update_yaxes(title="תרומה לשיפור הממוצע × אלף", row=1, col=col)
        fig.update_xaxes(tickangle=-30, row=1, col=col)
    fig.update_layout(barmode="group")
    return _finalize(fig, "7.1 — סוג היריבה מסביר חלק מיתרון שלוש המדרגות", height=800)


def coefficient_paths(review: ExperimentReview) -> Any:
    """Fitted scalar/Comp parameter paths, without implying uncertainty intervals."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["מקדם החשיבות", "רכיבי היעדים"])
    coefs = review.frames["coefficients"]
    for c in (*SCALARS, "Comp"):
        part = coefs.loc[
            coefs.config_id.eq(c) & ~coefs.coefficient.isin(["b_elo", "theta1", "delta", "theta2"])
        ]
        for key, group in part.groupby("coefficient", sort=False):
            target = {
                "Comp_title_diff": "אליפות",
                "Comp_cut_diff": "עליון",
                "Comp_survival_diff": "הישארות",
            }.get(key, "")
            fig.add_trace(
                go.Scatter(
                    x=[season_label(int(s)) for s in group.season_id],
                    y=group.value,
                    mode="lines+markers",
                    name=configuration_label(c) + (": " + target if target else ""),
                ),
                row=1,
                col=2 if c == "Comp" else 1,
            )
    fig.update_xaxes(tickangle=-50)
    fig.update_yaxes(title="מקדם שנלמד; ללא פס אי ודאות")
    return _finalize(fig, "7.4 — מקדם הדגל נחלש; רכיב האליפות של Comp נשאר", height=850)


def placebo(review: ExperimentReview) -> Any:
    """Saved 200 shuffles only; no shuffled models are fit while reading the notebook."""
    frame = review.frames["audit_L2-diff_placebos"]
    observed = review.pilot["audit_L2-diff"]["observed"]
    fig = go.Figure(
        go.Histogram(x=frame.log_loss_improvement, nbinsx=25, name="שיפור בערבובים שנשמרו")
    )
    fig.add_vline(x=observed, line_color="red", line_width=3)
    fig.add_annotation(
        x=observed, y=1.03, xref="x", yref="paper", text="השיפור האמיתי", showarrow=False
    )
    fig.update_xaxes(title="שיפור log loss מול Elo לבדו")
    fig.update_yaxes(title="מספר ערבובים")
    return _finalize(fig, "8.1 — השיפור בריצת הניסיון רחוק מהפלצבו", height=650)


def prose(text: str) -> HTML:
    """Present dynamically calculated prose through an escaped RTL surface."""
    return HTML(  # type: ignore[no-untyped-call]
        '<p dir="rtl" style="line-height:1.8">' + escape(text) + "</p>"
    )
