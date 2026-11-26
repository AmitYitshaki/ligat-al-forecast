"""Second-walk evidence computed exclusively from frozen fold results."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.evaluate.selection import paired_statistics
from ligat.viz import goals_notebook as views
from ligat.viz import goals_qa as qa
from ligat.viz import goals_qa_graphs as graphs
from ligat.viz import goals_walk_review as review
from ligat.viz.seasons import season_label

METRICS = {"poisson_nll": "NLL", "log_loss": "log loss בית/תיקו/חוץ", "rps": "RPS"}


def candidate_folds(b: dict[str, Any]) -> dict[str, Any]:
    result = {
        review.LABELS.get(name, name): review.folds(b, name)
        for name in ["glm", "EXP-002", "maher", "R1"]
    }
    checks: list[tuple[str, Callable[[dict[str, Any]], bool], str]] = [
        (
            "P4",
            lambda p: p.get("iterations") == 150 and p.get("l2_leaf_reg") == 10,
            "CatBoost רדוד — בדיקה בדיעבד",
        ),
        (
            "P3",
            lambda p: (
                p.get("max_depth") == 1
                and p.get("n_estimators") == 400
                and p.get("learning_rate") == 0.05
            ),
            "XGBoost עומק אחד — בדיקה בדיעבד",
        ),
    ]
    for check, predicate, label in checks:
        record = next(
            r for r in b["post"]["records"] if r["check"] == check and predicate(r["parameters"])
        )
        result[label] = pd.DataFrame(record["folds"])
    return result


def candidate_table(b: dict[str, Any]) -> Any:
    reference = review.folds(b, "F5")
    rows = []
    for name, candidate in candidate_folds(b).items():
        for scope in review.SCOPES:
            seasons = set(candidate.season_id) & set(reference.season_id)
            if scope == "deep":
                seasons &= set(candidate.loc[candidate.deep_fold, "season_id"])
                seasons &= set(reference.loc[reference.deep_fold, "season_id"])
            elif scope == "last5":
                seasons &= set(range(21, 26))
            a = candidate.loc[candidate.season_id.isin(seasons)].assign(deep_fold=True)
            r = reference.loc[reference.season_id.isin(seasons)].assign(deep_fold=True)
            for metric in METRICS:
                stats = paired_statistics(a, r, score=metric)
                assert stats.ci_lower is not None and stats.ci_upper is not None
                rows.append(
                    dict(
                        candidate=name,
                        scope=scope,
                        metric=metric,
                        candidate_mean=float(np.average(a[metric], weights=a.games)),
                        f5_mean=float(np.average(r[metric], weights=r.games)),
                        d=stats.mean_improvement,
                        se=stats.se,
                        lower=stats.ci_lower,
                        upper=stats.ci_upper,
                        includes_zero=stats.ci_lower <= 0 <= stats.ci_upper,
                        n=stats.n,
                        post_hoc="בדיעבד" in name,
                    )
                )
    return pd.DataFrame(rows)


def candidate_tables_html(b: dict[str, Any]) -> str:
    from ligat.viz.goals_configuration_names import configuration_name

    table = candidate_table(b)
    parts = []
    for check in ["P4", "P3"]:
        record = next(
            r
            for r in b["post"]["records"]
            if r["check"] == check
            and (
                (
                    check == "P4"
                    and r["parameters"].get("iterations") == 150
                    and r["parameters"].get("l2_leaf_reg") == 10
                )
                or (
                    check == "P3"
                    and r["parameters"].get("max_depth") == 1
                    and r["parameters"].get("n_estimators") == 400
                    and r["parameters"].get("learning_rate") == 0.05
                )
            )
        )
        label = configuration_name(
            record["parameters"], model="catboost" if check == "P4" else "xgb"
        )
        parts.append(review.paragraph(check + " — בדיקה בדיעבד: " + label))
    for name, p in table.groupby("candidate", sort=False):
        p = p.drop(columns=["candidate", "post_hoc"]).copy()
        p["scope"] = p.scope.map(review.SCOPES)
        p["metric"] = p.metric.map(METRICS)
        parts.append(
            "<h3>"
            + name
            + "</h3>"
            + review.render(
                p,
                dict(
                    scope="טווח",
                    metric="מדד",
                    candidate_mean="ממוצע המועמד",
                    f5_mean="ממוצע F5 באותן עונות",
                    d="שיפור מזווג מול F5 — חיובי טוב",
                    se="SE",
                    lower="תחתון CI95",
                    upper="עליון CI95",
                    includes_zero="כולל 0?",
                    n="עונות משותפות",
                ),
            )
        )
    return (
        '<div dir="rtl"><h2>המועמדים: כל מודל מול המודל המוצע F5</h2>' + "".join(parts) + "</div>"
    )


def candidate_figure(b: dict[str, Any]) -> Any:
    table = candidate_table(b)
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=["NLL", "log loss H/D/A", "RPS"],
    )
    for col, metric in enumerate(METRICS, 1):
        for scope, color in zip(review.SCOPES, views.COLORS[:3], strict=True):
            p = table.loc[table.metric.eq(metric) & table.scope.eq(scope)]
            fig.add_trace(
                go.Scatter(
                    x=p.d,
                    y=p.candidate,
                    mode="markers",
                    error_x=dict(
                        type="data", array=p.upper - p.d, arrayminus=p.d - p.lower, symmetric=False
                    ),
                    marker=dict(
                        color=color,
                        size=9,
                        symbol=["circle-open" if v else "circle" for v in p.post_hoc],
                    ),
                    name=review.SCOPES[scope],
                    legendgroup=scope,
                    showlegend=col == 1,
                    customdata=np.column_stack([p.n, p.se, p.includes_zero]),
                    hovertemplate=(
                        "שיפור %{x:+.6f}<br>n=%{customdata[0]}<br>"
                        "SE=%{customdata[1]}<br>כולל 0? %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748B")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="שיפור מול F5", tickformat=".3f", tickangle=0, nticks=3)
    for col, metric in enumerate(METRICS, 1):
        part = table.loc[table.metric.eq(metric)]
        low, high = float(part.lower.min()), float(part.upper.max())
        padding = (high - low) * 0.08
        fig.update_xaxes(
            tickvals=[low, 0, high],
            range=[low - padding, high + padding],
            tickfont=dict(size=10),
            row=1,
            col=col,
        )
    result = views.finish(
        fig,
        "בעמוקים רוב המתחרים קרובים ל-F5; בחמש האחרונות F5 עדיף על GLM"
        "<br>n=15 / 13–14 / 5; CI95 בין עונות",
        height=1000,
    )
    result.update_layout(
        margin=dict(l=280, b=200, t=170), title_font_size=16, legend=dict(y=-0.22), height=1050
    )
    counts = {
        scope: "/".join(str(int(n)) for n in sorted(part.n.unique()))
        for scope, part in table.groupby("scope")
    }
    result.update_layout(
        title_text="בעמוקים רוב המועמדים קרובים למודל המוצע"
        "<br>בחמש האחרונות המודל המוצע עדיף על שני מודלי הרגרסיה"
        f"<br>עונות: כל {counts['all']} · עמוקות {counts['deep']}"
        f" · אחרונות {counts['last5']}"
    )
    result.add_annotation(
        text="אפס = אין הבדל; הפסים הם CI95 בין עונות",
        x=0.5,
        y=-0.14,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return result


def candidate_explanation(b: dict[str, Any], *, after: bool) -> str:
    table = candidate_table(b)
    if not after:
        text = (
            "היער משווה כל מועמד למודל המוצע F5 על אותן עונות. בכל חלונית מדד אחר: "
            "NLL, log loss ו-RPS; בציר האופקי שיפור חיובי לטובת המועמד, ובאנכי שמות המודלים. "
            "הצבע מבחין בין כל העונות, העמוקות המשותפות לכל זוג וחמש האחרונות. הפס הוא CI95 "
            "בין עונות, ולא פיזור בין משחקים. אפס פירושו אין הבדל. סימון חלול הוא בדיקה "
            "בדיעבד, ולא מועמד שנבדק ברשת המקורית. ל-GLM בבסיס יש 13 עמוקות, ובאחרים 14; "
            "מספר העונות וה-SE מופיעים גם בטבלה ובריחוף."
        )
    else:
        p = table.loc[
            table.candidate.eq(review.LABELS["glm"]) & table.metric.eq("poisson_nll")
        ].set_index("scope")
        text = (
            f"מול GLM בבסיס, השיפור של המועמד לעומת F5 הוא {p.loc['deep', 'd']:+.6f} "
            f"בעמוקים עם SE {p.loc['deep', 'se']:.6f}, ובחמש האחרונות "
            f"{p.loc['last5', 'd']:+.6f}. בעמוקים כל המועמדים מלבד Maher בתוך הרעש "
            "מ-F5: רווחי הסמך כוללים אפס. בחמש האחרונות F5 עדיף בכיוון השיפור על שני "
            "מודלי ה-GLM ושלושת המדדים מסכימים בכיוון; אין לטעון שכל פס נפרד מוציא אפס. "
            "הדירוג על כל העונות מושפע מהעונה הראשונה, שבה לעצים רק שלוש עונות אימון. "
            "בכדורגל הפרדה זו מונעת לבלבל התאמה לתקופה עם חוזק חיזוי יציב; אין כאן בחירה או המלצה."
        )
    return '<div dir="rtl" class="graph-explanation"><p>' + text + "</p></div>"


def flagged_seasons(b: dict[str, Any]) -> Any:
    v = b["validation"]
    flagged = v.loc[v.log_loss.lt(0.98)].copy()
    flagged["kind"] = "עונה מסומנת"
    seasons = set(flagged.season_id)
    comparison = v.loc[v.stage_role.isin(["R0", "R1"]) & v.season_id.isin(seasons)].copy()
    comparison["kind"] = "השוואה באותה עונה"
    result = pd.concat([flagged, comparison], ignore_index=True)
    result["model"] = result.stage_role.map(lambda name: review.LABELS.get(name, name))
    result["half_life"] = result.half_life.map(lambda h: "בלי דעיכה" if pd.isna(h) else str(h))
    return result[["model", "season_id", "half_life", "log_loss", "kind"]]


def leakage_html(b: dict[str, Any]) -> str:
    v = b["validation"]
    f = flagged_seasons(b)
    flag = f.loc[f.kind.eq("עונה מסומנת")]
    diagnostics = qa.outcome_diagnostics(v)
    assert diagnostics.loss.min() >= 0.98, "Stop: configuration-average leakage threshold"
    r0 = f.loc[f.model.eq(review.LABELS["R0"]), "log_loss"].iloc[0]
    r1 = f.loc[f.model.eq(review.LABELS["R1"]), "log_loss"].iloc[0]
    return render_flagged(f) + review.paragraph(
        f"נמצאו {len(flag)} תוצאות עונה מתחת ל-0.98, באותה עונה, "
        f"ו-{int(v.log_loss.lt(1).sum())} מתחת ל-1.00. המינימום העונתי הוא "
        f"{flag.log_loss.min():.6f}; באותה עונה R1 נותן {r1:.6f} ו-R0 נותן {r0:.6f}. "
        "גם סרגל Elo ובית בלבד חזה היטב את אותה עונה; העדויות הללו מתאימות למידע "
        "לגיטימי על כוח קבוצות, ולא לממצא דליפה. בקרת ההצצה ההיסטורית מ-EXP-001 "
        "היא ממוצע על כל העונות, ואין להשוות אותה ישירות לעונה יחידה. "
        f"המינימום בממוצעי {len(diagnostics)} התצורות הוא {diagnostics.loss.min():.6f}; "
        "כלל העצירה לממוצע תצורה לא הופעל."
    )


def render_flagged(frame: Any) -> str:
    frame = frame.assign(season_id=frame.season_id.map(season_label))
    return review.render(
        frame,
        dict(
            model="מודל",
            season_id="עונה",
            half_life="זמן מחצית",
            log_loss="log loss של העונה",
            kind="תפקיד השורה",
        ),
    )


def family_ranking(b: dict[str, Any]) -> Any:
    entries, _ = qa.entry_tables(b["report"])
    p = entries.loc[entries.name.isin(["F1-R", "F1-A", "F2", "F3", "F4", "F5", "F6"])].copy()
    p = p.sort_values("d", ascending=False)
    meanings = {
        "F1-R": "כושר עשיר במקום מצומצם",
        "F1-A": "תוספת כושר עשיר",
        "F2": "כושר בנקודות",
        "F3": "מצב העונה והטבלה",
        "F4": "העונה הקודמת",
        "F5": "הדגל לקבוצה וליריבה",
        "F6": "זהות קבוצה ויריבה",
    }
    p["meaning"] = p.name.map(meanings)
    p["rank"] = range(1, len(p) + 1)
    p["elo"] = 100 * p.d / graphs._elo_denominator(b["report"])
    return p[["rank", "name", "meaning", "d", "se", "elo", "better", "passes"]]


def family_explanation(b: dict[str, Any]) -> str:
    p = family_ranking(b)
    stats = next(d for d in b["report"]["decisions"] if d["name"] == "F5" and d["kind"] == "entry")[
        "rule8"
    ]["stats"]
    positive = sum(d["difference"] > 0 for d in stats["differences"])
    return review.paragraph(
        "F1–F4 מסכמות תוצאות עבר. Elo כבר מסכם אותן ולכן נשאר להן מעט. "
        "F5 שונה: היא מוטיבציה ולא כוח; הפירוק מראה שהערך מגיע מהדגל של היריבה. "
        "F3 קרובה לעבור, וזהות הקבוצה ב-F6 מתאימה להסבר השינון."
    ) + review.paragraph(
        f"נבדקו {len(p)} משפחות מול אותו רף. כלל 8 הוא כלל החלטה ולא מבחן מובהקות, "
        "ולכן אין לו 'סיכוי מזל' מדויק. ובכל זאת, ככל שבודקים יותר משפחות, גדל הסיכוי "
        "שאחת מהן תעבור את הרף במקרה. לכן F5 לא נשען רק על הכלל. מה שמחזק אותו: "
        f"{positive} עונות חיוביות מתוך {stats['n']} בבדיקת הכניסה, הסבר כדורגלני "
        "(קבוצה שכבר אין לה על מה לשחק, והיריבה שלה), והפירוק שמראה שהערך מגיע מהדגל של היריבה."
    )
