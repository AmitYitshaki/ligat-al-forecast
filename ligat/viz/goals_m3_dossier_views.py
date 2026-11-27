"""Neutral tables and figures for the M3 candidate dossier."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.viz import goals_m3_dossier as core
from ligat.viz import goals_notebook as views
from ligat.viz import goals_walk_review as review

NAMES = dict(
    candidate="מועמד",
    scope="טווח",
    post_hoc="בדיקה בדיעבד?",
    seasons="עונות",
    games="משחקים",
    rows="שורות קבוצה-משחק",
    group="קבוצת איזון",
    predicted="חזוי",
    observed="נצפה",
    ratio="חזוי / נצפה",
    score="תוצאה",
    goals="סך שערים",
    outcome="תוצאת משחק",
    bin="קבוצת הסתברות",
    ece="ECE ברוחב 0.1 (מקורי, E3g)",
    segment="קבוצה",
    metric="מדד",
    mean="ממוצע",
    d="שיפור מול F5 — חיובי טוב",
    se="SE בין עונות",
    lower="תחתון CI95",
    upper="עליון CI95",
    includes_zero="כולל 0?",
    pearson="פיזור פירסון",
    nll="NLL עמוקים",
    nll_last5="NLL חמש אחרונות",
    draw_gap="פער תיקו",
    zero_gap="פער 0-0",
    one_gap="פער 1-1",
    ece_draw="ECE ברוחב 0.1 (מקורי, E3g)",
    ece_draw_deciles="ECE לפי עשירונים",
    fixed_noise_mean="רעש ממוצע, רוחב 0.1",
    fixed_noise_p95="אחוזון 95 לרעש, רוחב 0.1",
    deciles_noise_mean="רעש ממוצע, עשירונים",
    deciles_noise_p95="אחוזון 95 לרעש, עשירונים",
    maxcorr="מתאם λ הגבוה ביותר למועמד אחר",
)
VALUES = {
    **review.SCOPES,
    "regular": "עונה סדירה",
    "top_playoff": "פלייאוף עליון",
    "bottom_playoff": "פלייאוף תחתון",
    "middle_playoff": "פלייאוף אמצעי",
    "nll": "NLL",
    "rps": "RPS",
}


def table(frame: Any) -> str:
    p = frame.copy()
    if "group" in p:
        p["group"] = p["group"].map(
            {-1: "כל המשחקים", 0: "השליש המאוזן ביותר", 1: "השליש האמצעי", 2: "השליש הפחות מאוזן"}
        )
    for col in p:
        if p[col].dtype == object or isinstance(p[col].dtype, pd.StringDtype):
            p[col] = p[col].map(lambda v: VALUES.get(v, v) if isinstance(v, str) else v)
    names = {c: NAMES.get(c, c) for c in p}
    if "bin" in p and "ece" in p and "outcome" not in p:
        names["ece"] = "ECE לפי עשירונים"
    return review.render(p, names)


def summary(result: dict[str, Any]) -> Any:
    p = result["summary"].query("scope == 'deep'").copy()
    recent = result["summary"].query("scope == 'last5'").set_index("candidate")
    p["nll_last5"] = p.candidate.map(recent.nll)
    columns = [
        "candidate",
        "nll",
        "nll_last5",
        "draw_gap",
        "zero_gap",
        "one_gap",
        "ece_draw",
        "maxcorr",
        "pearson",
        "seasons",
        "games",
    ]
    if "ece_draw_deciles" in p:
        columns.insert(columns.index("ece_draw") + 1, "ece_draw_deciles")
    for kind, metric in (("fixed", "ece_draw"), ("deciles", "ece_draw_deciles")):
        fields = [f"{kind}_noise_mean", f"{kind}_noise_p95"]
        if fields[0] in p:
            index = columns.index(metric) + 1
            columns[index:index] = fields
    return p[columns]


def calibration_table(result: dict[str, Any], kind: str) -> str:
    key = "calibration" if kind == "fixed" else "draw_deciles"
    p = result[key].copy()
    mask = p.scope.eq("deep")
    if kind == "fixed":
        mask &= p.outcome.eq(core.OUTCOMES[1])
    for statistic in ("mean", "p95"):
        column = f"{kind}_noise_{statistic}"
        p.loc[mask, column] = p.loc[mask, "candidate"].map(result["ece_noise"][column])
    return table(p)


def section_table(result: dict[str, Any], key: str) -> str:
    if key == "D2":
        return table(result["scores"].query("scope == 'deep'")) + (
            "<details><summary>סך השערים וכל שלושת הטווחים</summary>"
            + table(result["totals"])
            + table(result["scores"])
            + "</details>"
        )
    if key == "D5":
        p = result["segments"]
        return (
            review.paragraph(
                "תמצית: NLL בשלישי האיזון בעמוקים המשותפים. "
                "כל ההשוואות לפי שלב ודגל, כולל RPS וכל הטווחים, בבלוק המלא."
            )
            + table(
                p.loc[p.scope.eq("deep") & p.metric.eq("nll") & p.segment.str.startswith("איזון")]
            )
            + "<details><summary>כל הקבוצות, המדדים והטווחים — SE ו-CI95</summary>"
            + table(p)
            + "</details>"
        )
    if key == "D6":
        return table(result["dispersion"])
    return table(summary(result))


def draw_figure(result: dict[str, Any]) -> Any:
    if "draw_bootstrap" in result:
        from ligat.viz.goals_draw_calibration import gap_figure

        return gap_figure(result)
    p = result["draw"].query("scope == 'deep' and group >= 0")
    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.12,
        shared_xaxes=True,
        subplot_titles=["השליש המאוזן ביותר לפי R1", "השליש האמצעי", "השליש הפחות מאוזן"],
    )
    for group in range(3):
        q = p.loc[p.group.eq(group)]
        for field, color, label in [
            ("predicted", views.COLORS[0], "חזוי"),
            ("observed", "#94A3B8", "נצפה"),
        ]:
            fig.add_bar(
                x=q[field],
                y=q.candidate,
                orientation="h",
                name=label,
                marker_color=color,
                legendgroup=field,
                showlegend=group == 0,
                row=group + 1,
                col=1,
            )
    fig.update_xaxes(title="שיעור תיקו", tickformat=".0%")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_layout(barmode="group")
    fig = views.finish(fig, "D1 — תחזיות תיקו מול תוצאות, באותם משחקים", height=1500)
    fig.update_layout(margin=dict(l=310, b=150), legend=dict(y=-0.08))
    return fig


def calibration_figure(result: dict[str, Any]) -> Any:
    p = result["calibration"].query("scope == 'deep'")
    fig = go.Figure()
    names = list(p.candidate.unique())
    buttons: list[dict[str, Any]] = []
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="כיול מושלם", line=dict(color="#94A3B8", dash="dash")
    )
    for i, name in enumerate(names):
        q = p.loc[p.candidate.eq(name)]
        for j, outcome in enumerate(core.OUTCOMES):
            rows = q.loc[q.outcome.eq(outcome) & q.games.gt(0)]
            if j == 1 and "draw_deciles" in result:
                rows = result["draw_deciles"].loc[
                    lambda p, n=name: p.scope.eq("deep") & p.candidate.eq(n)
                ]
            fig.add_scatter(
                x=rows.predicted,
                y=rows.observed,
                mode="lines+markers",
                name=outcome,
                line=dict(color=views.COLORS[j]),
                marker=dict(
                    symbol="circle-open" if "בדיעבד" in name else "circle",
                    size=(6 + 7 * np.sqrt(rows.games / rows.games.max())).tolist() if j == 1 else 9,
                    line=dict(color="white", width=0.6),
                ),
                customdata=rows[["games"]],
                hovertemplate=(
                    "חזוי %{x:.4f}, נצפה %{y:.4f}<br>"
                    "%{customdata[0]} משחקים<extra>%{fullData.name}</extra>"
                ),
                visible=i == 0,
            )
        eces = q.groupby("outcome", sort=False).ece.first()
        extra = ""
        if "draw_deciles" in result:
            dec = result["draw_deciles"].loc[
                lambda p, n=name: p.scope.eq("deep") & p.candidate.eq(n)
            ]
            extra = (
                f"<br>ECE לפי עשירונים: {dec.ece.iloc[0]:.6f}<br>"
                f"{len(dec)} קבוצות; {int(dec.games.min())}–{int(dec.games.max())} "
                "משחקים בקבוצה.<br>"
                "ערכים זהים לא מפוצלים; הגדלים אינם בהכרח שווים."
            )
        annotation = dict(
            text=name
            + "<br>"
            + " · ".join(f"ECE מקורי {k}: {v:.4f}" for k, v in eces.items())
            + extra,
            x=1,
            y=1.12,
            yanchor="bottom",
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            bgcolor="#F1F5F9",
            font=dict(size=12),
        )
        if "ece_noise" in result:
            noise = result["ece_noise"].loc[name]
            annotation["text"] += (
                f"<br>רעש ברוחב 0.1: ממוצע {noise.fixed_noise_mean:.4f}, "
                f"אחוזון 95 {noise.fixed_noise_p95:.4f}"
                f"<br>רעש לפי עשירונים: ממוצע {noise.deciles_noise_mean:.4f}, "
                f"אחוזון 95 {noise.deciles_noise_p95:.4f}"
            )
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    dict(visible=[True] + [k // 3 == i for k in range(3 * len(names))]),
                    dict(annotations=[annotation]),
                ],
            )
        )
    fig.update_xaxes(title="הסתברות חזויה", range=[0, 1])
    fig.update_yaxes(title="שיעור נצפה", range=[0, 1])
    fig.update_layout(
        annotations=buttons[0]["args"][1]["annotations"],
        updatemenus=[dict(buttons=buttons, active=0, x=0, y=1.02, yanchor="bottom")],
    )
    fig = views.finish(fig, "D3 — כיול בית, תיקו וחוץ בעמוקים המשותפים", height=1050)
    fig.update_layout(
        margin=dict(t=320, b=150),
        title=dict(y=0.98),
    )
    return fig


def diversity_figure(result: dict[str, Any]) -> Any:
    fig = go.Figure()
    buttons: list[dict[str, Any]] = []
    modes = {
        "lambda_correlation": "מתאם λ",
        "lambda_difference": "ממוצע הפרש מוחלט ב-λ",
        "nll_correlation": "מתאם NLL של משחקים",
    }
    names = list(result["matrices"]["deep"]["lambda_correlation"].index)
    short = [
        "F5 המוצע",
        "GLM בבסיס",
        "GLM סרגל",
        "Maher",
        "CatBoost<br>רדוד, בדיעבד",
        "XGBoost<br>עומק 1, בדיעבד",
        "R1",
    ]
    choices = [(s, m) for s in ["deep", "all", "last5"] for m in modes]
    for i, (scope, metric) in enumerate(choices):
        p = result["matrices"][scope][metric]
        fig.add_heatmap(
            z=p.to_numpy(),
            x=short,
            y=short,
            visible=i == 0,
            colorscale="Cividis",
            text=np.round(p.to_numpy(), 3),
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="%{y} / %{x}: %{z:.6f}<extra></extra>",
        )
        counts = result["summary"].loc[result["summary"].scope.eq(scope)].iloc[0]
        annotation = dict(
            text=f"{modes[metric]} · {review.SCOPES[scope]} · "
            f"{int(counts.seasons)} עונות, {int(counts.games)} משחקים",
            x=1,
            y=1.06,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
        )
        buttons.append(
            dict(
                label=modes[metric] + " · " + review.SCOPES[scope],
                method="update",
                args=[
                    dict(visible=[k == i for k in range(len(choices))]),
                    dict(annotations=[annotation]),
                ],
            )
        )
    assert len(names) == len(short)
    fig = views.finish(fig, "D4 — דמיון בין תחזיות, בשלושה מדדים", height=1000)
    fig.update_layout(
        margin=dict(t=240, b=180, l=170),
        title=dict(y=0.98),
        annotations=buttons[0]["args"][1]["annotations"],
        updatemenus=[dict(buttons=buttons, active=0, x=0, y=1.12, yanchor="bottom")],
    )
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10), autorange="reversed")
    return fig


def figures(result: dict[str, Any]) -> dict[str, Any]:
    return dict(
        m3_draw=draw_figure(result),
        m3_calibration=calibration_figure(result),
        m3_diversity=diversity_figure(result),
    )


def explanation(name: str, result: dict[str, Any], *, after: bool) -> str:
    if "draw_deciles" in result and name in {"m3_draw", "m3_calibration"}:
        from ligat.viz.goals_draw_calibration import explanation as updated

        return updated(result, name, after=after)
    sample = result["summary"].query("scope == 'deep'").iloc[0]
    if name == "m3_draw":
        text = (
            "D1: בכל חלונית קבוצת איזון אחרת, לפי הפרש הסתברויות הבית והחוץ של R1. "
            "השליש הראשון הוא המאוזן ביותר. החלוקה מחושבת על המשחקים בעמוקים המשותפים "
            "וזהה לכל המועמדים; קשרים נשברים בסדר מזהה המשחק. כחול הוא ממוצע ההסתברות "
            "לתיקו ואפור שיעור התיקו שנצפה. כל זוג עמודות מתייחס בדיוק לאותם משחקים. "
            "השמות מציינים גם בדיקות בדיעבד; הנתונים לכל הטווחים נשמרים בטבלה המלאה."
        )
        if after:
            text = (
                f"במודל המוצע פער התיקו הכולל בעמוקים הוא {sample.draw_gap:+.6f}. "
                "הפער בגרף עשוי להשתנות בין משחקים מאוזנים למשחקים בעלי פייבוריט ברור. "
                "בכדורגל, מודל שמחסיר תיקו במשחקים שקולים משאיר לכיול של M3 תיקון "
                "מסוג אחר ממודל שמחסיר תיקו בכל המשחקים. זהו תיאור ממוצעים ולא בדיקת "
                "מובהקות או בחירת מועמד. הבדיקות בדיעבד נשארות מסומנות, והחלוקה אינה "
                "משתנה כדי להחמיא למודל מסוים."
            )
    elif name == "m3_calibration":
        text = (
            "D3: הבורר מחליף מועמד בלבד; המשחקים והטווח העמוק המשותף נשארים קבועים. "
            "בציר האופקי ממוצע ההסתברות החזויה, ובאנכי שיעור התוצאה שנצפתה, לכל אחת "
            "משלוש תוצאות המשחק בצבע אחר. ההסתברויות מחולקות לעשר קבוצות ברוחב 0.1; "
            "קבוצות ריקות אינן נקודות אפס. הקו המקווקו הוא שוויון חזוי ונצפה. בתיבה "
            "מופיעים שם המועמד ו-ECE המשוקלל במספר המשחקים, המתעדכנים עם הבורר."
        )
        if after:
            text = (
                f"ECE של תיקו במודל המוצע הוא {sample.ece_draw:.6f} בעמוקים. "
                "מרחק מהאלכסון מתאר חוסר התאמה בין ביטחון התחזית לבין שכיחות התוצאה. "
                "בכדורגל, תיקו יכול להיות מדורג היטב יחסית בין משחקים ובכל זאת לקבל "
                "הסתברות נמוכה מדי. M3 יוכל לבחון תיקון כזה על תחזיות מחוץ לאימון; "
                "הגרף כאן מתאר בלבד ואינו מכייל או בוחר מודל. ECE תלוי בקבוצות "
                "ההסתברות, ולכן משווים תחת אותה הגדרה ומציגים גם את הספירות המלאות."
            )
    else:
        text = (
            "D4: בכל תא במטריצה זוג מועמדים. הבורר מחליף טווח ומדד: מתאם בין ערכי λ "
            "בשורות קבוצה-משחק, ממוצע ההפרש המוחלט ביניהם, או מתאם בין NLL של משחקים. "
            "NLL למשחק הוא ממוצע שני ציוני הקבוצות, כפי שמדווח השופט החתום. האלכסון "
            "הוא 1 במתאמים ואפס בהפרש המוחלט. הצבע הוא סולם מספרי ניטרלי, אינו ציון "
            "טוב או רע ואינו מדרג מועמדים. מספר העונות והמשחקים מתעדכן עם הבחירה."
        )
        if after:
            text = (
                f"המתאם הגבוה של המודל המוצע למועמד אחר בעמוקים הוא {sample.maxcorr:.6f}. "
                "מתאם גבוה ב-λ מתאר מידע דומה על כוח ההתקפה וההגנה, ואילו הפרש "
                "מוחלט יכול לגלות שינוי בסקאלה גם כשהמתאם גבוה. מתאם NLL מתאר "
                "אם אותם משחקים מפתיעים את שני המודלים. בכדורגל, שני מודלים דומים "
                "עשויים להשאיר ל-M3 כמעט אותו מידע; שונות בתחזיות אינה מוכיחה "
                "שהשילוב ישפר חיזוי. אין במטריצות בחירה או המלצה על צירוף מודלים."
            )
    return '<div dir="rtl" class="graph-explanation"><p>' + text + "</p></div>"
