"""Read-only additions for the shared notebook walkthrough."""

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.viz.goal_training_curves import frozen_full_report, full_validation_table
from ligat.viz.goals_notebook import COLORS, finish, season_counts, walk_forward_figure


def daily_decay_figure():  # type: ignore[no-untyped-def]
    from ligat.viz.goals_notebook import decay_figure

    figure = decay_figure(daily=True)
    for trace in figure.data:
        import base64

        values = trace.x
        if isinstance(values, dict):
            values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        trace.x = np.asarray(values) * 365.25
    figure.update_xaxes(title="גיל המשחק בימים")
    figure.add_trace(
        go.Scatter(
            x=[0, 8 * 365.25], y=[None, None], xaxis="x2", showlegend=False, hoverinfo="skip"
        )
    )
    figure.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            range=[0, 8 * 365.25],
            tickvals=[n * 365.25 for n in range(9)],
            ticktext=[str(n) for n in range(9)],
            title="גיל בעונות",
        )
    )
    return figure


def chain_html() -> str:
    boxes = [
        ("M0", "משחקי עבר ו-Elo נכנסים; פיצ׳רים מוקדמים יוצאים"),
        ("M1", "מצב הטבלה נכנס; אותות מוטיבציה יוצאים"),
        ("M2", "פיצ׳רים נכנסים; שני ממוצעי שערים λ יוצאים"),
        ("M3", "שני λ נכנסים; הסתברויות לתוצאות יוצאות"),
        ("סימולציה", "הסתברויות נכנסות; טבלאות עונה יוצאות"),
    ]
    return (
        '<div dir="rtl" class="goal-chain"'
        ' style="display:flex;gap:8px;align-items:center;font:14px Arial">'
        + '<span aria-hidden="true">←</span>'.join(
            '<div style="flex:1 1 0;min-width:0;box-sizing:border-box;'
            "padding:10px;overflow-wrap:anywhere;border:2px solid "
            + ("#2563EB;background:#DBEAFE" if name == "M2" else "#93C5FD;background:white")
            + ';border-radius:8px"><b>'
            + name
            + "</b><br>"
            + text
            + "</div>"
            for name, text in boxes
        )
        + "</div>"
    )


def input_summary(data: GoalData) -> str:
    context = data.context
    return (
        f'<div dir="rtl">עונות מחקר {context.season_id.min()}–{context.season_id.max()}; '
        f"{context.game_id.nunique():,} משחקים; {len(context):,} שורות קבוצה-משחק; "
        f"{context.season_id.between(26, 28).sum()} שורות מעונות 26–28.</div>"
    )


def provisional_summary(data: GoalData) -> str:
    """Display the dated decision over the immutable report's historical status."""
    from ligat.viz.goals_decision_display import opening

    return opening(frozen_full_report(load_goal_experiment()))


def season_games_figure(data: GoalData):  # type: ignore[no-untyped-def]
    counts = season_counts(data)
    games = counts.games.to_numpy()
    # Contiguous structural periods come from changes between count regimes.
    distinct = np.unique(games)
    cuts = sorted(np.argsort(np.diff(distinct))[-2:])
    lower, upper = [(distinct[i] + distinct[i + 1]) / 2 for i in cuts]
    regimes = np.where(
        games < lower, "ליגה מצומצמת", np.where(games > upper, "ליגה מורחבת", "מבנה נוכחי")
    )
    figure = go.Figure()
    labels = [f"{s}<br>עונת מחקר {n}" for s, n in zip(counts.season, counts.season_id, strict=True)]
    roles = [
        "חימום" if n < 11 else "קיפול 1" if n == 11 else "5 האחרונות" if n >= 21 else "עמוק"
        for n in counts.season_id
    ]
    for i, regime in enumerate(dict.fromkeys(regimes)):
        mask = regimes == regime
        figure.add_bar(
            x=np.flatnonzero(mask),
            y=games[mask],
            name=regime,
            marker_color=COLORS[i],
            customdata=np.array(roles)[mask],
            hovertemplate="%{y} משחקים<br>%{customdata}<extra>%{fullData.name}</extra>",
        )
    figure.update_xaxes(
        tickvals=list(range(len(counts))), ticktext=labels, tickangle=-55, title="עונה ומספרה במחקר"
    )
    figure.update_yaxes(title="משחקים בעונה", rangemode="tozero")
    for i, role in enumerate(roles):
        figure.add_annotation(
            x=i,
            y=-0.32,
            yref="paper",
            text=role,
            showarrow=False,
            textangle=-55,
            font=dict(size=11),
        )
    result = finish(figure, f"שלושה מבני ליגה לאורך {len(counts)} עונות פיתוח", height=850)
    result.update_layout(margin=dict(b=240), legend=dict(y=-0.55))
    return result


def warmup_html(data: GoalData) -> str:
    rows = data.context.loc[data.context.season_id.lt(11)]
    return (
        '<div dir="rtl"><p>קיפול 1 מתאמן רק על שלוש עונות החימום: '
        f"{rows.game_id.nunique():,} משחקים ו-{len(rows):,} שורות, מליגה במבנה אחר. "
        "לכן הוא אינו נכלל בכלל 8.</p></div>"
    )


def folds_figure(data: GoalData):  # type: ignore[no-untyped-def]
    figure = walk_forward_figure()
    for season in sorted(data.context.season_id.unique())[3:]:
        rows = int(data.context.season_id.lt(season).sum())
        figure.add_annotation(
            x=1.01,
            xref="paper",
            y=figure.data[0].y[season - 11],
            text=f"{rows:,} שורות",
            showarrow=False,
            xanchor="left",
        )
    figure.data[0].colorbar.x = 1.3
    figure.update_layout(margin=dict(r=250, b=150))
    return figure


def decay_result_html() -> str:
    validation = full_validation_table(load_goal_experiment())
    part = validation.loc[validation.stage_role.eq("S2-AB")]
    means = []
    for _, group in part.groupby("half_life", dropna=False):
        means.append(float(np.average(group.poisson_nll, weights=group.games)))
    return (
        '<div dir="rtl"><p>בתוצאות שלב B, טווח הממוצעים לפי זמן מחצית הוא '
        f"{min(means):.6f}–{max(means):.6f} NLL (פער {max(means) - min(means):.6f}); "
        "זה תיאור של הרשת. כמה תצורות קרובות לפי כלל 4; "
        "ההכרעות בפרק 6 נסגרו בה-ש31–ה-ש34.</p></div>"
    )
