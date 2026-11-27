"""Post-decision draw diagnostics: whole-season uncertainty and tie-safe deciles."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.viz import goals_m3_dossier as core
from ligat.viz import goals_notebook as views
from ligat.viz import goals_walk_review as review


def season_bootstrap(
    seasons: Any, counts: Any, draws: Any, *, B: int = 2000, seed: int = 0
) -> tuple[Any, Any]:
    """Resample complete season blocks; rows within a block never move separately."""
    seasons = np.asarray(seasons)
    counts, draws = np.asarray(counts), np.asarray(draws)
    if len(seasons) < 2 or B < 1 or counts.shape != draws.shape or len(counts) != len(seasons):
        raise ValueError("Bootstrap requires aligned whole-season counts")
    indices = np.random.default_rng(seed).integers(0, len(seasons), size=(B, len(seasons)))
    denominator = counts[indices].sum(axis=1)
    if (denominator <= 0).any():
        raise ValueError("Every bootstrap balance group needs games")
    return draws[indices].sum(axis=1) / denominator, seasons[indices]


def draw_bootstrap(data: dict[str, Any]) -> tuple[Any, Any]:
    frames = core.scoped(data, "deep")
    reference = frames[review.LABELS["R1"]]
    seasons = np.array(sorted(reference.season_id.unique()))
    counts, draws = [], []
    for season in seasons:
        p = reference.loc[reference.season_id.eq(season)]
        counts.append([int(p.balance.eq(j).sum()) for j in range(3)])
        draws.append([int((p.balance.eq(j) & p.outcome.eq(1)).sum()) for j in range(3)])
    rates, samples = season_bootstrap(seasons, counts, draws)
    low, high = np.quantile(rates, [0.025, 0.975], axis=0)
    rows = []
    for name, p in frames.items():
        for j in range(3):
            q = p.loc[p.balance.eq(j)]
            predicted, observed = float(q.pd.mean()), float(q.outcome.eq(1).mean())
            rows.append(
                dict(
                    candidate=name,
                    group=j,
                    predicted=predicted,
                    observed=observed,
                    gap=predicted - observed,
                    lower=predicted - high[j],
                    upper=predicted - low[j],
                    games=len(q),
                    seasons=q.season_id.nunique(),
                )
            )
    return pd.DataFrame(rows), samples


def equal_count_bins(probabilities: Any) -> Any:
    """Quantile cutpoints never split identical predicted probabilities."""
    p = pd.Series(np.asarray(probabilities, dtype=float))
    if p.empty or not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Calibration requires finite probabilities")
    if p.nunique() == 1:
        return np.zeros(len(p), dtype=int)
    labels = pd.qcut(p, 10, labels=False, duplicates="drop")
    return pd.factorize(labels, sort=True)[0]


def draw_deciles(data: dict[str, Any]) -> Any:
    rows: list[dict[str, Any]] = []
    for scope in core.SCOPES:
        for name, p in core.scoped(data, scope).items():
            labels = equal_count_bins(p.pd)
            bins = []
            for label in np.unique(labels):
                q = p.iloc[np.flatnonzero(labels == label)]
                bins.append(
                    dict(
                        candidate=name,
                        scope=scope,
                        bin=int(label) + 1,
                        predicted=float(q.pd.mean()),
                        observed=float(q.outcome.eq(1).mean()),
                        games=len(q),
                        seasons=q.season_id.nunique(),
                    )
                )
            ece = sum(x["games"] * abs(x["predicted"] - x["observed"]) for x in bins) / len(p)
            rows.extend(dict(**row, ece=ece) for row in bins)
    return pd.DataFrame(rows)


def augment(result: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Add diagnostics; the original E3g tables and ECE values remain intact."""
    added = dict(result)
    added["draw_bootstrap"], added["bootstrap_samples"] = draw_bootstrap(data)
    added["draw_deciles"] = draw_deciles(data)
    ece = added["draw_deciles"].groupby(["scope", "candidate"]).ece.first()
    summary = result["summary"].copy()
    summary["ece_draw_deciles"] = [ece.loc[(r.scope, r.candidate)] for r in summary.itertuples()]
    added["summary"] = summary
    return added


def gap_figure(result: dict[str, Any]) -> Any:
    p = result["draw_bootstrap"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["השליש המאוזן ביותר", "השליש האמצעי", "השליש הפחות מאוזן"],
    )
    for j in range(3):
        q = p.loc[p.group.eq(j)]
        fig.add_scatter(
            x=100 * q.gap,
            y=q.candidate,
            mode="markers",
            showlegend=False,
            marker=dict(
                color=views.COLORS[0],
                size=10,
                symbol=["circle-open" if "בדיעבד" in n else "circle" for n in q.candidate],
            ),
            error_x=dict(
                array=100 * (q.upper - q.gap), arrayminus=100 * (q.gap - q.lower), symmetric=False
            ),
            customdata=q[["predicted", "observed", "games"]],
            hovertemplate=(
                "%{y}<br>פער %{x:.3f} נקודות אחוז<br>"
                "חזוי %{customdata[0]:.4f}<br>נצפה %{customdata[1]:.4f}<br>"
                "%{customdata[2]} משחקים<extra></extra>"
            ),
            row=j + 1,
            col=1,
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748B")
    fig.update_xaxes(
        title="חזוי פחות נצפה, בנקודות אחוז",
        range=[100 * p.lower.min() - 0.5, 100 * p.upper.max() + 0.5],
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.add_annotation(
        x=1,
        y=1.07,
        xref="paper",
        yref="paper",
        xanchor="right",
        showarrow=False,
        text=(
            "הטווח: אי-ודאות של השיעור הנצפה.<br>"
            "דוגמים עונות שלמות, כי משחקים מאותה עונה תלויים זה בזה.<br>"
            "13 עונות עמוקות משותפות; 2,000 דגימות; זרע 0."
        ),
        font=dict(size=12),
    )
    fig = views.finish(fig, "D1 — פער התיקו לפי איזון, וטווח לשיעור הנצפה", height=1500)
    fig.update_layout(margin=dict(t=240, b=160, l=310), title=dict(y=0.98))
    return fig


def explanation(result: dict[str, Any], name: str, *, after: bool) -> str:
    if name == "m3_draw":
        text = (
            "הציר מציג פער תיקו חזוי פחות נצפה בנקודות אחוז; קו האפס הוא התאמה. "
            "אותם שלישי איזון של R1 נשמרים מ-E3g. דגימת עונות שלמות, block bootstrap, "
            "מגרילה עם החזרה 13 עונות מתוך 13 העמוקות המשותפות, 2,000 פעמים בזרע 0. "
            "כל משחקי העונה שנדגמה נשמרים יחד. הטווח הוא אחוזוני 2.5 ו-97.5 של השיעור "
            "הנצפה, שמחוסרים מהממוצע החזוי הקבוע. היחידה היא נקודת אחוז, בטווח אפשרי "
            "של מינוס 100 עד 100 לפער. זה אינו טווח לאימון מחדש של המודל."
        )
        if after:
            q = result["draw_bootstrap"].loc[lambda p: p.candidate.eq(review.LABELS["F5"])]
            numbers = "; ".join(
                f'שליש {int(r.group) + 1}: <bdi dir="ltr">{100 * r.gap:+.2f}, '
                f"[{100 * r.lower:+.2f}, {100 * r.upper:+.2f}]</bdi>"
                for r in q.itertuples()
            )
            excluded = int(((q.lower > 0) | (q.upper < 0)).sum())
            text = (
                f"ב-F5, בנקודות אחוז: {numbers}. אפס מחוץ לטווח ב-{excluded} מתוך 3 שלישים. "
                "לעומת עמודות E3g רואים עכשיו את הפער ואת חוסר הוודאות של שיעור התיקו, "
                "בלי לשנות תחזית או ספירה. בכדורגל, חוסר בתיקו במשחק שקול עשוי להשאיר "
                "ל-M3 תיקון; טווח רחב מזכיר שעונות דומות אינן משחקים עצמאיים. אין כאן "
                "מבחן בחירה חדש, ואין הוכחה לסיבה של החוסר או להצלחת תיקון עתידי."
            )
    else:
        text = (
            "בית וחוץ נשארים בקבוצות ברוחב 0.1. לתיקו נוספו עשירונים לפי מספר משחקים: "
            "ממיינים הסתברויות וקובעים גבולות כמותיים, בלי לפצל ערכים זהים. ליד כל מועמד "
            "מופיעים מספר הקבוצות וטווח הספירות; גודל הנקודה לפי מספר המשחקים. ECE הוא "
            "סכום הפערים המוחלטים חזוי מול נצפה, במשקל מספר המשחקים בכל קבוצה, בטווח 0–1; "
            "אפס הוא התאמה בקבוצות האלה. ECE לפי עשירונים אינו אותו מדד מחולק כמו ECE "
            "ברוחב 0.1 המקורי של E3g, ולכן שניהם מוצגים בנפרד, ללא החלפה של נתון חתום."
        )
        if after:
            row = result["summary"].query("scope == 'deep'").iloc[0]
            text = (
                f"אצל F5 בעמוקים, ECE תיקו ברוחב 0.1 נשאר {row.ece_draw:.6f}; "
                f"ECE לפי עשירונים הוא {row.ece_draw_deciles:.6f}. לעומת E3g הקו מתאר יותר "
                "מקטעים בתוך תחום הסתברויות התיקו הצר, במקום להישען על שתיים או שלוש נקודות. "
                "המספר החדש תלוי בחלוקה, ולא מוכיח שהמודל השתנה. בכדורגל אפשר לראות "
                "היכן תחזית לתיקו חסרה גם בתוך משחקים דומים; נקודה קטנה נשענת על מעט "
                "משחקים. כיול הוא אבחון של שכיחות, והטווחים אינם סיכוי לנחש משחק מסוים."
            )
    return '<div dir="rtl"><p>' + text + "</p></div>"
