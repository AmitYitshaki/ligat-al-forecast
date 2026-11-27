"""Final walk figures from signed folds; presentation only."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.evaluate.selection import paired_statistics
from ligat.viz import goals_notebook as views
from ligat.viz import goals_qa_graphs as graphs
from ligat.viz import goals_walk_evidence as evidence
from ligat.viz import goals_walk_review as review
from ligat.viz.seasons import season_label

NAMES = ["R0", "R1", "S1", "S2", "F5", "glm", "EXP-002", "EXP-003", "maher", "catboost-selected"]


def population(b: dict[str, Any], scope: str) -> set[int]:
    cache = b.setdefault("walk_populations", {})
    if scope in cache:
        return set(cache[scope])
    parts = [review.folds(b, name) for name in NAMES]
    sets = [set(p.loc[p.deep_fold, "season_id"] if scope == "deep" else p.season_id) for p in parts]
    seasons = set.intersection(*sets)
    result = seasons & set(range(21, 26)) if scope == "last5" else seasons
    cache[scope] = result
    return result


def selected(b: dict[str, Any], name: str, scope: str) -> Any:
    p = review.folds(b, name)
    return p.loc[p.season_id.isin(population(b, scope))].assign(deep_fold=True)


def paired(b: dict[str, Any], name: str, reference: str, scope: str) -> Any:
    return paired_statistics(
        selected(b, name, scope), selected(b, reference, scope), score="poisson_nll"
    )


def paired_forest(b: dict[str, Any]) -> Any:
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.3, subplot_titles=["המתחרים הקרובים", "רחוקים"]
    )
    for col, names in [
        (1, ["F5", "S1", "R1", "glm", "EXP-002"]),
        (2, ["R0", "catboost-selected", "maher"]),
    ]:
        for name in names:
            s = paired(b, name, "S2", "deep")
            assert s.ci_upper is not None and s.ci_lower is not None
            label = review.LABELS.get(name, name)
            fig.add_trace(
                go.Scatter(
                    x=[s.mean_improvement],
                    y=[label],
                    mode="markers",
                    error_x=dict(
                        array=[s.ci_upper - s.mean_improvement],
                        arrayminus=[s.mean_improvement - s.ci_lower],
                        symmetric=False,
                    ),
                    marker=dict(
                        size=16 if name == "F5" else 10,
                        color=graphs.GREEN if name == "F5" else views.COLORS[0],
                    ),
                    showlegend=False,
                    customdata=[s.n],
                    hovertemplate="שיפור %{x:+.6f}<br>n=%{customdata}<extra>CI95</extra>",
                ),
                row=1,
                col=col,
            )
    fig.add_vline(x=0, line_dash="dash", line_color=graphs.GRAY)
    fig.update_xaxes(title="שיפור מול S2; אפס = אין הבדל; CI95")
    return views.finish(
        fig,
        f"F5 קרוב לבסיס; המודלים הרחוקים נפרדים — {len(population(b, 'deep'))} עונות משותפות",
        height=850,
    )


def metric_by_season(b: dict[str, Any]) -> Any:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
        subplot_titles=["המודלים הקרובים לסרגל", "CatBoost מהרשת בנפרד"],
    )
    r1 = review.folds(b, "R1").set_index("season_id")
    for i, name in enumerate(
        ["S1", "S2", "F5", "EXP-002", "EXP-003", "maher", "glm", "catboost-selected"]
    ):
        p = review.folds(b, name).sort_values("season_id")
        fig.add_scatter(
            x=p.season_id.map(season_label),
            y=p.poisson_nll.to_numpy() - r1.loc[p.season_id, "poisson_nll"].to_numpy(),
            name=review.LABELS.get(name, name),
            mode="lines+markers",
            line=dict(
                width=4 if name == "F5" else 1.5, color=(*views.COLORS, "#0F766E", "#A16207")[i]
            ),
            row=2 if name == "catboost-selected" else 1,
            col=1,
        )
    fig.add_scatter(
        x=r1.index.map(season_label),
        y=np.zeros(len(r1)),
        name=review.LABELS["R1"],
        line=dict(color=graphs.GRAY, dash="dash"),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=graphs.GRAY)
    values = np.concatenate(
        [np.asarray(trace.y, dtype=float) for trace in fig.data if trace.yaxis != "y2"]
    )
    pad = (values.max() - values.min()) * 0.1
    fig.layout.yaxis.range = [float(values.min() - pad), float(values.max() + pad)]
    fig.add_annotation(
        text="2009/10: לעצים שלוש עונות אימון בלבד",
        xref="paper",
        yref="paper",
        x=0,
        y=1.1,
        showarrow=False,
        xanchor="left",
    )
    fig.update_yaxes(title="NLL פחות R1", title_standoff=15)
    fig.update_xaxes(title="עונת ולידציה", tickangle=-35)
    result = views.finish(fig, "F5 והמתחרים ביחס לסרגל Elo באותה עונה", height=950)
    result.update_layout(legend=dict(y=-0.24), margin=dict(b=230))
    return result


def budget_table(b: dict[str, Any]) -> Any:
    v = b["validation"]
    v = v.loc[v.stage_role.eq("S2-A") & v.preset.eq("grid")]
    cfg = b["report"]["winners"]["S2"]["parameters"]
    proposal = v.loc[
        v.depth.eq(cfg["max_depth"])
        & v.n_estimators.eq(cfg["n_estimators"])
        & v.learning_rate.eq(cfg["learning_rate"])
        & v.min_child_weight.eq(cfg["min_child_weight"])
    ].assign(deep_fold=True)
    r0 = review.folds(b, "R0").assign(deep_fold=True)
    scale = 100 / graphs._elo_denominator(b["report"], "all")
    rows = []
    groups = [
        p
        for _, p in v.loc[v.min_child_weight.eq(30)].groupby(
            ["depth", "n_estimators", "learning_rate"]
        )
    ]
    groups.append(proposal)
    for p in groups:
        s = paired_statistics(p.assign(deep_fold=True), r0, score="poisson_nll")
        other = paired_statistics(p.assign(deep_fold=True), proposal, score="poisson_nll")
        assert s.ci_lower is not None and s.ci_upper is not None
        assert s.mean_improvement is not None
        row = p.iloc[0]
        rows.append(
            dict(
                depth=row.depth,
                trees=row.n_estimators,
                rate=row.learning_rate,
                caution=row.min_child_weight,
                proposal=bool(row.min_child_weight == cfg["min_child_weight"]),
                elo=s.mean_improvement * scale,
                lower=s.ci_lower * scale,
                upper=s.ci_upper * scale,
                versus_proposal=other.mean_improvement,
                includes_zero=s.ci_lower <= 0 <= s.ci_upper,
                n=s.n,
            )
        )
    return pd.DataFrame(rows)


def elo_by_scope(b: dict[str, Any]) -> Any:
    fig = go.Figure()
    names = ["S1", "S2", "F5", "EXP-002", "EXP-003", "catboost-selected", "glm", "maher"]
    for i, scope in enumerate(review.SCOPES):
        denom = paired(b, "R1", "R0", scope).mean_improvement
        stats = [paired(b, name, "R0", scope) for name in names]
        means = [s.mean_improvement * 100 / denom for s in stats]
        fig.add_bar(
            y=[review.LABELS.get(n, n) for n in names],
            x=means,
            orientation="h",
            name=review.SCOPES[scope],
            marker_color=views.COLORS[i],
            error_x=dict(
                array=[(float(s.ci_upper) - s.mean_improvement) * 100 / denom for s in stats],
                arrayminus=[(s.mean_improvement - float(s.ci_lower)) * 100 / denom for s in stats],
                symmetric=False,
            ),
        )
        for position, (mean, s) in enumerate(zip(means, stats, strict=True)):
            fig.add_annotation(
                x=float(s.ci_upper) * 100 / denom + 3,
                y=position + (i - 1) * 0.8 / 3,
                xref="x",
                yref="y",
                text=f"{mean:.1f}%",
                showarrow=False,
                xanchor="left",
                bgcolor="white",
                font=dict(size=12, color=views.COLORS[i]),
            )
    fig.add_vline(x=0, line_dash="dash", line_color=graphs.GRAY)
    fig.add_vline(x=100, line_dash="dot", line_color=graphs.GRAY)
    fig.update_xaxes(title="אחוז מתרומת Elo; 0 = רק בית; 100 = Elo ובית; CI95 מזווג מול R0")
    fig.update_layout(barmode="group", margin=dict(l=260, r=150))
    fig.update_yaxes(type="category")
    return views.finish(
        fig, f"F5 והסרגלים באותן עונות; עמוקים n={len(population(b, 'deep'))}", height=1050
    )


def models_vs_baselines(b: dict[str, Any]) -> Any:
    parts = {name: selected(b, name, "deep") for name in NAMES if name not in ["R0", "R1"]}
    for name, p in evidence.candidate_folds(b).items():
        if "בדיעבד" in name:
            parts[name] = p.loc[p.season_id.isin(population(b, "deep"))]
    fig = go.Figure()
    for name, p in parts.items():
        s = graphs._summary(p)
        mean = 100 * np.exp(-s["mean"])
        fig.add_scatter(
            x=[mean],
            y=[review.LABELS.get(name, name)],
            mode="markers",
            marker=dict(
                size=15 if name == "F5" else 10,
                symbol="circle-open" if "בדיעבד" in name else "circle",
                color=graphs.GREEN if name == "F5" else views.COLORS[0],
            ),
            showlegend=False,
            error_x=dict(
                array=[100 * np.exp(-s["lower"]) - mean],
                arrayminus=[mean - 100 * np.exp(-s["upper"])],
                symmetric=False,
            ),
        )
    fig.update_xaxes(title="הסתברות גאומטרית לתוצאה שנצפתה; אחוזים ו-CI95")
    fig.update_layout(margin=dict(l=300))
    return views.finish(
        fig,
        f"המודלים קרובים — {len(population(b, 'deep'))} עמוקות משותפות; חלול = בדיעבד",
        height=1000,
    )


def review_plot(b: dict[str, Any]) -> Any:
    p = review.review_table(b)
    fig = go.Figure()
    for row in pd.DataFrame(p.attrs["deep_rows"]).itertuples():
        fig.add_scatter(
            x=[row.deep_gap],
            y=[row.question],
            mode="markers",
            showlegend=False,
            error_x=dict(array=[row.deep_se]),
            marker=dict(
                symbol="cross" if row.deep_gap == 0 else "circle",
                color=graphs.GREEN if row.deep_gap == 0 else graphs.RED,
                size=12,
            ),
        )
    fig.add_vline(x=0, line_dash="dash", line_color=graphs.GRAY)
    fig.update_xaxes(title="תוספת NLL מול המובילה בעמוקים; פס ±SE; פלוס ירוק ב-0 = ההצעה מובילה")
    fig.update_layout(margin=dict(l=280))
    return views.finish(fig, "הכרעות שונות בלבד: רוב הפערים קטנים מהאי-ודאות", height=950)


def learning_changes(diagnostic: dict[str, Any]) -> Any:
    outputs = [o for o in diagnostic["outputs"] if o["identity"]["config"]["n_estimators"] == 1000]
    rows = []
    for end in [400, 1000]:
        changes = np.array(
            [o["validation_nll"][149] - o["validation_nll"][end - 1] for o in outputs]
        )
        rows.append(
            dict(
                trees=end,
                median=float(np.median(changes)),
                better=int((changes > 0).sum()),
                n=len(changes),
            )
        )
    return pd.DataFrame(rows)


def apply(figures: dict[str, Any], b: dict[str, Any]) -> None:
    for key, fn in [
        ("paired_forest", paired_forest),
        ("metric_by_season", metric_by_season),
        ("models_vs_baselines", models_vs_baselines),
        ("elo_by_scope", elo_by_scope),
        ("open_reviews", review_plot),
    ]:
        figures[key] = fn(b)
    fig = figures["ladder_progress"]
    ranking = evidence.family_ranking(b)
    fig.update_yaxes(categoryorder="array", categoryarray=list(ranking.name), autorange="reversed")
    fig.update_xaxes(title="שיפור באחוז מתרומת Elo; 0 = אין הבדל בתצורת הכניסה; CI95")
    base = figures["base_family_forest"]
    seen = set()
    for trace in base.data:
        trace.y = [review.LABELS.get(n, n) for n in trace.y]
        clear = float(trace.x[0]) + float(trace.error_x.array[0]) < 0
        trace.marker.color = graphs.RED if clear else graphs.GRAY
        trace.name = "ההוצאה פוגעת: הרכיב נחוץ" if clear else "אין השפעה ברורה"
        trace.legendgroup = trace.name
        trace.showlegend = trace.name not in seen
        seen.add(trace.name)
    record = next(r for r in b["post"]["records"] if r["check"] == "P1")
    r = pd.DataFrame(record["folds"])
    s = paired_statistics(r, review.folds(b, "S2"), score="poisson_nll")
    assert s.ci_lower is not None and s.ci_upper is not None
    assert s.mean_improvement is not None
    scale = 100 / graphs._elo_denominator(b["report"])
    base.add_scatter(
        x=[s.mean_improvement * scale],
        y=["כל המשפחות בלי Elo (P1, בדיקה בדיעבד)"],
        mode="markers",
        marker=dict(symbol="circle-open", size=13, color=graphs.GRAY),
        name="בדיקה בדיעבד — ראו 6.2",
        error_x=dict(
            array=[(s.ci_upper - s.mean_improvement) * scale],
            arrayminus=[(s.mean_improvement - s.ci_lower) * scale],
            symmetric=False,
        ),
    )
    base.update_xaxes(title="השפעת ההוצאה באחוז מתרומת Elo; 0 = אין הבדל; CI95")
    base.update_layout(
        title_text="הוצאת Elo פוגעת בבסיס; כל המשפחות יחד מפצות חלקית (P1)", margin=dict(l=340)
    )
    # A neutral training scale does not mark memorization as predictive success.
    heat = figures["train_vs_validation_heatmap"]
    for trace in heat.data[::2]:
        trace.colorscale = [[0, "#F8FAFC"], [1, "#64748B"]]
    hist = figures["hda_histogram"]
    hist.layout.annotations = []
    for i, name in enumerate(["F5", "R1", "EXP-002"]):
        p = review.folds(b, name)
        value = float(np.average(p.log_loss, weights=p.games))
        hist.add_vline(x=value, line_color=views.COLORS[i], line_dash="dot")
        hist.add_annotation(
            x=0.98,
            xref="paper",
            y=0.98 - i * 0.09,
            yref="paper",
            text=f"{review.LABELS.get(name, name)}: {value:.6f}",
            showarrow=False,
            xanchor="right",
            bgcolor="white",
            font=dict(color=views.COLORS[i]),
        )
    hist.add_vline(x=0.968, line_color=graphs.RED, line_dash="dash")
    hist.add_annotation(
        x=0.98,
        xref="paper",
        y=0.68,
        yref="paper",
        text="בקרת הצצה (EXP-001): 0.968",
        showarrow=False,
        xanchor="right",
        bgcolor="white",
        font=dict(color=graphs.RED),
    )
    for level, text in [(0.59, "סף עצירה לממוצע תצורה: 0.98"), (0.50, "סף דגל לממוצע תצורה: 1.00")]:
        hist.add_annotation(
            x=0.98,
            y=level,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="right",
            bgcolor="white",
            font=dict(color=graphs.GRAY),
        )


def explanation(name: str, b: dict[str, Any], fig: Any, *, after: bool) -> str:
    if not after:
        changed = {
            "paired_forest",
            "metric_by_season",
            "models_vs_baselines",
            "elo_by_scope",
            "open_reviews",
            "base_family_forest",
        }
        if name in changed:
            axes = "; ".join(
                str(fig.layout[k].title.text or "")
                for k in fig.layout
                if k.startswith(("xaxis", "yaxis"))
            )
            return review.paragraph(
                "איך קוראים: " + axes + ". בכל שורה או קו מופיע מודל או מבחן, "
                "והמקרא מפריד בין מודלים וטווחים. F5 מסומן כמוצע; סימון חלול מציין "
                "בדיקה בדיעבד. פסים אופקיים הם רווחי סמך מזווגים של 95% בין עונות "
                "ביער המזווג ובאחוז מתרומת Elo, ורווחים מוחלטים בגרף ההסתברות; "
                "בהכרעות הפתוחות הפס הוא SE אחד. בגרף העונתי אין פס: כל נקודה היא עונה. "
                "שמות הצירים מגדירים את האפס והכיוון. יש לקרוא את המספר ואת אוכלוסיית "
                "העונות יחד, ולא להסיק על עוצמת חיזוי מסדר השורות בלבד."
            )
        return graphs.explain_graph(name, b, fig, after=False)
    if name == "metric_by_season":
        text = (
            "הציר מציג NLL פחות R1 באותה עונה, ולכן אפס הוא אותה תוצאה כמו סרג"
            "ל Elo ובית. F5 מודגש ובמקומו אין F6. השוואה זו מסירה את רמת הקושי"
            " המשותפת של העונה: קו שלילי טוב מ-R1 וקו חיובי גרוע ממנו. אין לפר"
            "ש עלייה של כל הקווים כעלייה מוחלטת בקושי. בכדורגל השאלה כאן היא ה"
            "אם מידע נוסף מעבר לכוח הקבוצות עוזר באותה תקופה, ולא האם עונה אחת"
            " קלה לחיזוי מאחרת."
        )
    elif name == "base_family_forest":
        text = (
            "זהו פירוק של השפעת ההוצאה, ולא תרומות הכניסה. אדום מסמן שההוצאה פ"
            "וגעת וברווח הסמך אין אפס; אפור מציין אי-ודאות. השורה החלולה של P1"
            " היא בדיקה בדיעבד של כל המשפחות בלי Elo, מול S2, ולא שינוי בהחלטת"
            " כלל 8. השוואתה לבסיס בלי Elo מראה שמידע על העבר מפצה חלקית על הו"
            "צאת מדד הכוח, אבל לא מבטל את ערכו. בבניית חיזוי כדורגל זו עדות לח"
            "פיפה בין תוצאות עבר למדד הכוח; המשך הפירוק מופיע ב-6.2."
        )
    elif name == "elo_by_scope":
        text = (
            "לצד כל עמודה מופיע ערכה עצמו באחוז מתרומת Elo. אפס הוא רמת רק בית,"
            " ו-100 הוא רמת Elo ובית, באותן עונות בכל טווח. הפס מזווג מול R0; "
            "זה אינו רווח על NLL מוחלט. בעמוקים כל המודלים משתמשים ב-13 עונות "
            "משותפות, כך ש-GLM לא נהנה מחוסר של עונה קשה. F5 מופיע כמוצע, ותוו"
            "יות המודלים אופקיות. ההפרדה לטווחים מציגה תלות בתקופה, ולא בחירה "
            "במודל; בכדורגל היא מראה אם מידע מעבר לכוח הקבוצות נשאר שימושי גם "
            "בתקופה האחרונה."
        )
    elif name == "open_reviews":
        text = (
            "כל שורה היא הכרעה שונה מטבלת ההכרעות, ולא רשימה כפולה של אותו זוג"
            " תצורות. פלוס ירוק בדיוק באפס פירושו שההצעה היא המובילה בעמוקים. "
            "שאר הנקודות מציגות את תוספת ה-NLL של ההצעה מול המובילה העמוקה, עם"
            " פס של SE אחד. פער קטן ביחס לפס אינו מספק הכרעה יציבה, והגרף אינו"
            " פותר את כלל 4 או כלל 6. בכדורגל שינוי קטן של תקציב או דעיכה עשוי"
            " להזיז דירוג בלי לשנות את יכולת החיזוי באופן ברור; ההכרעה נשארת א"
            "צל עמית."
        )
    elif name == "hda_histogram":
        text = (
            "ההתפלגות היא של ממוצעי תצורות על כל עונותיהן, ולא של עונות יחידות"
            ". F5, סרגל Elo ובית ו-GLM סרגל מסומנים עם ערכיהם המדויקים; בקרת ה"
            "הצצה ההיסטורית ב-0.968 מוצגת בנפרד. הדירוג כאן מושפע מהעונה הראשו"
            "נה, שבה לעצים יש רק שלוש עונות אימון. טבלת המועמדים משווה על אותן"
            " עונות ומפרידה עמוקות וחמש אחרונות. בכדורגל עונה אחת צפויה אינה ר"
            "איית דליפה; כלל העצירה מתייחס לממוצע תצורה. ראו גם טבלת העונות המ"
            "סומנות מעל הגרף."
        )
    elif name == "models_vs_baselines":
        text = (
            "כל הנקודות משתמשות באותן 13 עונות עמוקות, כולל שני מודלי GLM והבד"
            "יקות בדיעבד. הסימון החלול מבדיל את CatBoost הרדוד ואת XGBoost בעו"
            "מק אחד מהבדיקות שנקבעו מראש. פס רחב בהסתברות הגאומטרית משקף שונות"
            " בין עונות; הוא אינו תחליף להשוואה מזווגת. לכן טבלת המועמדים ליד "
            "הגרף מציגה גם שיפור, SE ושלושה מדדים לכל זוג מול F5. בכדורגל דירו"
            "ג קרוב יכול להשתנות בין תקופות בלי שיהיה הבדל יציב באיכות החיזוי,"
            " והגרף אינו בוחר מודל."
        )
    elif name == "paired_forest":
        s = paired(b, "F5", "S2", "deep")
        text = (
            f"השיפור של F5 מול S2 הוא {s.mean_improvement:+.6f}, עם SE {s.se:.6f}, "
            f"על {s.n} עונות עמוקות משותפות. אפס הוא אין הבדל; חיובי לטובת המועמד. "
            "שתי סקאלות נפרדות מונעות מהמודלים הרחוקים לדחוס את המתחרים הקרובים. "
            "F5 מודגש לצד S1, R1 ושני מודלי GLM. בכדורגל תוספת מידע קטנה עשויה "
            "לעזור בעקביות בלי ליצור פער גדול בעונה יחידה. יש לקרוא את רווח הסמך "
            "ואת האוכלוסייה המשותפת לפני הדירוג. הגרף מתאר את התוצאה ואינו בוחר מודל."
        )
    else:
        return graphs.explain_graph(name, b, fig, after=True)
    if name in {"models_vs_baselines", "elo_by_scope"}:
        text = text.replace("13 עונות", f"{len(population(b, 'deep'))} עונות")
    if name == "elo_by_scope":
        text += f" גם הטווח המלא כאן מוגבל ל-{len(population(b, 'all'))} עונות משותפות."
    return '<div dir="rtl" class="graph-explanation"><p>' + text + "</p></div>"
