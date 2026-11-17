"""Reader-facing labels, two-SE graphics and evidence captions for notebook 04."""

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.motivation_walkthrough import PERIODS
from ligat.viz import motivation as mv
from ligat.viz.motivation_style import normalize_minus, two_se
from ligat.viz.presentation import TableDisplay


def caption(
    frame: pd.DataFrame,
    *,
    label: str = "reading_label",
    estimate: str = "estimate",
    se: str = "se",
    football: str = "",
) -> TableDisplay:
    """Compute which displayed two-SE intervals exclude zero instead of copying a conclusion."""
    good = frame[se].notna() & frame[estimate].notna()
    outside = good & (
        (frame[estimate] - 2 * frame[se]).gt(0) | (frame[estimate] + 2 * frame[se]).lt(0)
    )
    yes = "; ".join(normalize_minus(str(x)) for x in frame.loc[outside, label]) or "אין פסים כאלה"
    no = (
        "; ".join(normalize_minus(str(x)) for x in frame.loc[good & ~outside, label])
        or "אין פסים כאלה"
    )
    return TableDisplay(
        '<div dir="rtl"><h3>מה רואים בגרף</h3>'
        f"<p><b>מבט מדעי הנתונים:</b> לא חוצים את האפס: {yes}. "
        f"חוצים את האפס: {no}. פס = פעמיים SE. פס שחוצה אפס אינו הוכחה שאין קשר.</p>"
        f"<p><b>מבט כדורגלני:</b> {football}</p></div>"
    )


def survival_profile(frame: pd.DataFrame) -> Any:
    """Signed safety gap and median/IQR curves; quartiles describe dispersion, not SE."""
    fig = go.Figure()
    for label, color in zip(
        ("≤1", "2–3", "4–6", "7+"), ("#2563EB", "#0891B2", "#9333EA", "#F59E0B"), strict=True
    ):
        p = frame.loc[frame.remaining_bin.astype(str).eq(label)].sort_values("gap_to_safety")
        fig.add_trace(
            go.Scatter(
                x=list(p.gap_to_safety) + list(p.gap_to_safety[::-1]),
                y=list(p.q75) + list(p.q25[::-1]),
                fill="toself",
                fillcolor=color,
                opacity=0.10,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=p.gap_to_safety,
                y=p["median"],
                mode="lines+markers",
                name={
                    "≤1": "נותר משחק אחד לכל היותר",
                    "2–3": "נותרו שניים–שלושה",
                    "4–6": "נותרו ארבעה–שישה",
                    "7+": "נותרו שבעה ומעלה",
                }[label],
                line=dict(color=color),
                customdata=p[["n", "q25", "q75"]],
                hovertemplate="פער %{x}<br>חציון %{y:.2f}<br>שורות %{customdata[0]}"
                "<br>רבעונים %{customdata[1]:.2f}–%{customdata[2]:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="מעל הקו ← פער חתום → מתחת לקו", range=[-12, 15])
    fig.update_yaxes(title="חשיבות להישארות בלבד", range=[0, 1])
    return mv._finish(fig, "אותו פער מקו ההישארות חשוב אחרת כשנותרו פחות משחקים")


def contribution_plot(layers: pd.DataFrame) -> Any:
    """Arithmetic contributions, excluding unmatched strata; no uncertainty bar is implied."""
    fig = make_subplots(
        rows=3, cols=1, subplot_titles=list(PERIODS.values()), vertical_spacing=0.12
    )
    for col, block in enumerate(PERIODS, 1):
        p = layers.loc[layers.scope.eq(block) & layers.included].copy()
        labels = [stratum_words(x) for x in p.stratum]
        colors = ["#F59E0B" if block == "B3" and x == "lower / <=1" else mv.BLUE for x in p.stratum]
        fig.add_trace(
            go.Bar(
                x=p.contribution,
                y=labels,
                orientation="h",
                marker_color=colors,
                showlegend=False,
                customdata=p[["n_dead", "n_control"]],
                hovertemplate="%{y}<br>תרומה %{x:.3f}<br>מתות %{customdata[0]}, "
                "חיות %{customdata[1]}<extra></extra>",
            ),
            row=col,
            col=1,
        )
        fig.add_vline(
            x=float(p.contribution.sum()), line_dash="dash", line_color=mv.GRAY, row=col, col=1
        )
        if col == 3:
            fig.update_xaxes(title_text="תרומה לפער בנקודות", row=col, col=1)
        if block == "B3":
            fig.add_annotation(
                x=-0.31,
                y=labels[list(p.stratum).index("lower / <=1")],
                text="4 שורות חיות בלבד",
                showarrow=True,
                row=col,
                col=1,
            )
    return mv._finish(fig, "תרומות השכבות מסתכמות בפער הכולל; הקו מסמן את הסכום", height=1150)


def stratum_words(value: str) -> str:
    """Translate the fixed analysis stratum without exposing internal slashes."""
    stage, remaining = value.split(" / ")
    return (
        {
            "regular": "עונה סדירה",
            "upper": "פלייאוף עליון",
            "lower": "פלייאוף תחתון",
            "middle": "בית אמצעי",
        }[stage]
        + ", "
        + {
            "<=1": "נותר משחק אחד",
            "2-3": "נותרו שניים–שלושה משחקים",
            "4-6": "נותרו ארבעה–שישה משחקים",
            "7+": "נותרו שבעה משחקים ומעלה",
        }[remaining]
    )


def staircase(frame: pd.DataFrame) -> Any:
    """Living-level residual minus all living rows in its stratum, with two-SE bars."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["נקודות", "ספיגה"], horizontal_spacing=0.16)
    for col, metric in enumerate(("res_pts", "res_ga"), 1):
        p = frame.loc[frame.metric.eq(metric)]
        fig.add_trace(
            go.Scatter(
                x=p.level,
                y=p.estimate,
                **two_se(p.se),
                mode="lines+markers",
                name="אומדנים",
                showlegend=col == 1,
                line=dict(color=mv.BLUE),
                customdata=p[["n", "n_used"]],
                hovertemplate="רמה %{x}<br>אומדן %{y:.3f}<br>שורות %{customdata[0]}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_hline(y=0, line_dash="dash", line_color=mv.GRAY, row=1, col=col)
        fig.update_xaxes(
            title_text="רמת החשיבות בין חיות", tickvals=[0.25, 0.5, 0.75, 1], row=1, col=col
        )
        fig.update_yaxes(title_text="פחות ממוצע כל החיות בשכבה", row=1, col=col)
    return mv._finish(fig, "יותר חשיבות: האם הנקודות עולות והספיגה יורדת?", height=550)


def typical_plot(frame: pd.DataFrame) -> Any:
    """Effects expressed in points at the observed median nonzero gap, including joint terms."""
    fig = go.Figure(
        go.Scatter(
            x=frame.median_effect,
            y=frame.reading_label,
            mode="markers",
            **two_se(frame.median_effect_se, axis="x"),
            name="אומדנים",
            marker=dict(color=mv.BLUE, size=10),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=mv.GRAY)
    fig.update_xaxes(title="נקודות בפער טיפוסי, לעומת הצפי")
    return mv._finish(fig, "שיפוע גדול אינו בהכרח אפקט גדול: נקודות בפער שקורה בפועל", height=850)


def game_flag_plot(blocks: pd.DataFrame, sensitivity: pd.DataFrame) -> Any:
    """Game-level flag beside fixed-threshold row sensitivity; label the distinct estimands."""
    fig = go.Figure()
    for p, label, color in (
        (
            blocks.loc[
                blocks.comparison_id.eq("dead_vs_alive_opponent") & blocks.metric.eq("res_pts")
            ].rename(columns={"block": "scope"}),
            "מתה במשחק מול חיה",
            mv.BLUE,
        ),
        (
            sensitivity.loc[sensitivity.scope.isin(PERIODS) & sensitivity.metric.eq("res_pts")],
            "רגישות: מתה מול השאר, לפחות עשר חיות",
            "#F59E0B",
        ),
    ):
        fig.add_trace(
            go.Scatter(
                x=p.scope.map(PERIODS),
                y=p.estimate,
                **two_se(p.se),
                mode="markers",
                marker=dict(color=color, size=10),
                name=label,
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color=mv.GRAY)
    fig.update_yaxes(title="הפרש שארית נקודות")
    return mv._finish(fig, "הדגל ברמת המשחק נחלש: בחמש האחרונות הפסים חוצים אפס")


def period_plot(frame: pd.DataFrame) -> Any:
    """Two periods, four disjoint game types and separate goals/draw calibration panels."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["שערים מעל הצפוי", "תיקו מעל הצפוי"],
        horizontal_spacing=0.16,
    )
    kinds = {
        "relegation_alive": "מאבק הישארות",
        "both_dead": "שתיהן מתות",
        "one_dead": "צד מת יחיד ללא מאבק הישארות",
        "other": "כל השאר",
    }
    for period, label, color in (
        ("first10", "2009/2010–2018/2019", mv.GRAY),
        ("last5", "2019/2020–2023/2024", mv.BLUE),
    ):
        p = frame.loc[frame.period.eq(period)]
        for col, metric in enumerate(("res_total_goals", "res_draw"), 1):
            fig.add_trace(
                go.Scatter(
                    x=p.game_type.map(kinds),
                    y=p[metric],
                    **two_se(p[f"{metric}_se"]),
                    mode="markers",
                    name=label,
                    legendgroup=period,
                    showlegend=col == 1,
                    marker=dict(
                        color=color,
                        size=9,
                        symbol=["circle-open" if n < 30 else "circle" for n in p.n_games],
                    ),
                    customdata=p[["n_games"]],
                    hovertemplate="%{x}<br>שארית %{y:.3f}"
                    "<br>משחקים %{customdata[0]}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    for col in (1, 2):
        fig.add_hline(y=0, line_dash="dash", line_color=mv.GRAY, row=1, col=col)
    fig.update_yaxes(title_text="שערים למשחק", row=1, col=1)
    fig.update_yaxes(title_text="הפרש הסתברות תיקו", row=1, col=2)
    fig.update_xaxes(tickangle=-25)
    return mv._finish(fig, "משחקי מתות: עודף השערים גדל בחמש האחרונות", height=850)


def stage_plot(frame: pd.DataFrame) -> Any:
    """All/recent clean/all populations; omit middle's tiny sample from the graph only."""
    fig = go.Figure()
    for prefix, period, symbol in (
        ("", "כל 15 העונות", "circle"),
        ("last5_", "חמש העונות האחרונות", "diamond"),
    ):
        for pop, label, color in (
            ("clean", "נקיים", mv.BLUE),
            ("all_games", "כל המשחקים", mv.GRAY),
        ):
            p = frame.loc[frame.population.eq(pop) & frame.stage.ne("middle")]
            fig.add_trace(
                go.Scatter(
                    x=p.stage.map(mv.VALUES),
                    y=p[f"{prefix}goals_residual"],
                    **two_se(p[f"{prefix}goals_se"]),
                    mode="markers",
                    name=f"{period}: {label}",
                    marker=dict(color=color, size=10, symbol=symbol),
                    customdata=p[[f"{prefix}n_games"]],
                    hovertemplate="%{x}<br>שארית %{y:.3f}"
                    "<br>משחקים %{customdata[0]}<extra></extra>",
                )
            )
    fig.add_hline(y=0, line_dash="dash", line_color=mv.GRAY)
    fig.update_yaxes(title="שערים בפועל פחות הצפי")
    return mv._finish(fig, "בחמש האחרונות אין עודף ברור בפלייאוף הנקי; בתחתון מדגם קטן", height=850)
