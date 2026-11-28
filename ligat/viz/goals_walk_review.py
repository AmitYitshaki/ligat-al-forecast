"""Read-only tables for the second goals walkthrough; no model decisions."""

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.evaluate.selection import paired_statistics
from ligat.models.m2_glm import _COLUMNS
from ligat.viz import goals_qa as qa
from ligat.viz import goals_qa_graphs as graphs
from ligat.viz.presentation import display_table

LABELS = {
    "glm": "GLM בבסיס (D3)",
    "EXP-002": "GLM סרגל (EXP-002)",
    "EXP-003": "GLM סרגל עם הקשר אשתקד (EXP-003)",
    "maher": "Maher",
    "catboost-selected": "CatBoost מהרשת",
    "F1": "בסיס מכוונן (מסלול מקוצר)",
    "F5": "F5 (המוצע)",
    "R0": "R0: רק בית",
    "R1": "R1: Elo ובית",
    "base-without-elo": "בסיס בלי Elo",
    "base-without-home": "בסיס בלי ביתיות",
    "base-without-form": "בסיס בלי כושר",
    "base-without-value": "בסיס בלי שווי סגל",
    "base-without-stage": "בסיס בלי שלב",
    "base-without-remaining": "בסיס בלי מרחק לסיום",
    "base-without-promoted": "בסיס בלי דגל עולה",
}
SCOPES = {"all": "כל הקיפולים", "deep": "העמוקים המשותפים", "last5": "חמש האחרונות"}


def scope_description(scope: str) -> str:
    return {
        "all": "כל קיפולי הפיתוח, כולל העונות שבהן היסטוריית האימון קצרה.",
        "deep": "כל המודלים נמדדים על אותן 13 עונות עמוקות משותפות; "
        "ל-GLM בבסיס פחות היסטוריית אימון.",
        "last5": "חמש עונות הפיתוח האחרונות, באותה אוכלוסיית משחקים לכל המודלים.",
    }[scope]


TERM_LINKS = {
    term: f"walk-term-{index}"
    for index, term in enumerate(
        [
            "סולם",
            "קיפול עמוק",
            "חמש האחרונות",
            "בדיקה בדיעבד",
            "תצורת כניסה",
            "בסיס מכוונן",
            "תקציב למידה",
            "R0 / R1 / S1 / S2 / F1–F6",
            "prior לעולה",
            "בדיקות מרובות",
        ]
    )
}


def link_terms(value: str) -> str:
    pieces = re.split(r"(<[^>]*>)", value)
    anchor = False
    for i, piece in enumerate(pieces):
        if piece.startswith("<"):
            if piece.startswith("<a "):
                anchor = True
            elif piece.startswith("</a"):
                anchor = False
            continue
        if anchor:
            continue
        pattern = "|".join(re.escape(term) for term in sorted(TERM_LINKS, key=len, reverse=True))
        pieces[i] = re.sub(
            pattern,
            lambda match: '<a href="#' + TERM_LINKS[match[0]] + '">' + match[0] + "</a>",
            piece,
        )
    return "".join(pieces)


def paragraph(text: str) -> str:
    return '<div dir="rtl"><p>' + text + "</p></div>"


def model_names_figure(figure: Any) -> Any:
    """Translate visual model labels without changing categorical coordinates or IDs."""
    fig = go.Figure(figure)
    for trace in fig.data:
        if trace.name in LABELS:
            trace.name = LABELS[trace.name]
    for key in fig.layout:
        if not re.fullmatch(r"[xy]axis\d*", key):
            continue
        axis = fig.layout[key]
        coordinate = key[0]
        ref = key.replace("axis", "")
        categories = []
        for trace in fig.data:
            if (getattr(trace, coordinate + "axis", None) or coordinate) != ref:
                continue
            values = getattr(trace, coordinate, None)
            for value in [] if values is None else values:
                if isinstance(value, str) and value not in categories:
                    categories.append(value)
        if any(value in LABELS for value in categories):
            axis.tickmode = "array"
            axis.tickvals = categories
            axis.ticktext = [LABELS.get(value, value) for value in categories]
    return fig


def rules_html(b: dict[str, Any]) -> str:
    s = next(d for d in b["report"]["decisions"] if d["name"] == "F5" and d["kind"] == "entry")[
        "rule8"
    ]["stats"]
    rows = [
        dict(
            rule="4",
            checks="דיוק מול פשטות",
            threshold="בתוך SE אחת מהמובילה",
            example="הצעת S2",
            where="טבלת ההכרעות",
            pending="הצעות כלל 4 נשארות (ה-ש32)",
        ),
        dict(
            rule="6",
            checks="מובילות בעמוקים ובחמש האחרונות",
            threshold="המובילות שונות",
            example="חלופות הדעיכה",
            where="טבלת ההכרעות",
            pending="ההצעה של כלל 4 נשארת (ה-ש32)",
        ),
        dict(
            rule="8 גרסה 2",
            checks="כניסת מידע חדש",
            threshold="ממוצע מעל SE; שלוש מחמש האחרונות; ממוצע אחרונות חיובי",
            example=f"F5: {s['mean_improvement']:.6f} מול SE {s['se']:.6f}",
            where="כניסות המשפחות ופירוק F5",
            pending="ל-M3 עוברים F5, GLM בבסיס ו-Maher (ה-ש33)",
        ),
        dict(
            rule="10",
            checks="אי-הסכמה בין מדדים",
            threshold="NLL ו-RPS מכריעים הפוך",
            example="דגלי הדוח החתום",
            where="טבלת דגלי כלל 10",
            pending="נסגר לפי NLL (ה-ש34)",
        ),
    ]
    return (
        '<div dir="rtl"><h3>כללי ההכרעה</h3></div>'
        + paragraph(
            f"דוגמה מחושבת: בבדיקת הכניסה של F5 ממוצע ההפרש הוא {s['mean_improvement']:.6f}, "
            f"SE הוא {s['se']:.6f}, ועל פני {s['n']} עונות CI95 הוא "
            f"[{s['ci_lower']:+.6f}, {s['ci_upper']:+.6f}]."
        )
        + render(
            pd.DataFrame(rows),
            dict(
                rule="כלל",
                checks="מה בודק",
                threshold="הסף",
                example="דוגמה",
                where="איפה בפרק 6",
                pending="הכרעת עמית מ-27.11.2026",
            ),
        )
        + paragraph('<a href="#open-decisions">להכרעות הפתוחות ולתלויות שלהן</a>')
    )


def selector_gap(b: dict[str, Any]) -> str:
    cfg = b["report"]["winners"]["S2"]["parameters"]
    v = b["validation"]
    v = v.loc[v.stage_role.eq("S2-A") & v.preset.eq("grid")]
    for column, key in [
        ("depth", "max_depth"),
        ("n_estimators", "n_estimators"),
        ("learning_rate", "learning_rate"),
        ("min_child_weight", "min_child_weight"),
    ]:
        v = v.loc[v[column].eq(cfg[key])]
    logs = b["training"].loc[b["training"].unit_id.isin(v.unit_id)]
    last = logs.sort_values("iteration").groupby("unit_id").tail(1)
    joined = v.merge(last[["unit_id", "train_loss"]], on="unit_id", validate="one_to_one")
    train = float(np.average(joined.train_loss, weights=joined.games))
    validation = float(np.average(joined.poisson_nll, weights=joined.games))
    gap = validation - train
    return (
        f"בתצורת ברירת המחדל ממוצע NLL האימון המשוקלל במשחקים הוא {train:.6f}, "
        f"וממוצע הוולידציה המשוקלל באותם משחקים הוא {validation:.6f}. "
        f"הפער ולידציה פחות אימון הוא {gap:.6f}, "
        f"{100 * gap / graphs._elo_denominator(b['report'], 'all'):.2f}% "
        "מתרומת Elo. קו האימון בגרף מתאר חציון ו-IQR; ההפרש המספרי כאן משווה "
        "שני ממוצעים משוקללים בלבד. בכדורגל הפער מתאר התאמה לעבר לעומת עונה חדשה, "
        "והוא אינו מבחן מובהקות או הוראה לשנות את תקציב האימון."
    )


def learning_description(b: dict[str, Any], diagnostic: dict[str, Any]) -> str:
    outputs = diagnostic["outputs"]
    selected = diagnostic["selected_config"]
    same = [
        o
        for o in outputs
        if o["identity"]["config"]["max_depth"] == selected["max_depth"]
        and o["identity"]["config"]["learning_rate"] == selected["learning_rate"]
    ]
    longest = max(o["identity"]["config"]["n_estimators"] for o in same)
    long = [o for o in same if o["identity"]["config"]["n_estimators"] == longest]
    short = [o for o in same if o["identity"]["config"]["n_estimators"] == selected["n_estimators"]]
    end = selected["n_estimators"]
    v150 = float(np.median([o["validation_nll"][end - 1] for o in short]))
    v400 = float(np.median([o["validation_nll"][min(400, longest) - 1] for o in long]))
    vend = float(np.median([o["validation_nll"][-1] for o in long]))
    weighted = graphs._summary(folds(b, "S2"))["mean"]
    # The full-grid selector reports the stage-A version, before stage-B retuning.
    cfg = b["report"]["winners"]["S2"]["parameters"]
    p = b["validation"].loc[b["validation"].stage_role.eq("S2-A")]
    for col, key in [
        ("depth", "max_depth"),
        ("n_estimators", "n_estimators"),
        ("learning_rate", "learning_rate"),
        ("min_child_weight", "min_child_weight"),
    ]:
        p = p.loc[p[col].eq(cfg[key])]
    weighted = float(np.average(p.poisson_nll, weights=p.games))
    return (
        '<div dir="rtl"><h2>6.1 — הוולידציה מתייצבת לפני שהאימון מסתיים</h2></div>'
        + paragraph(
            "מה נבדק: הצעת S2 ההיסטורית ושכנותיה בעומק ובקצב, ותקציב הלמידה הארוך. "
            "יתר הפרמטרים קבועים; האבחון חתום ונפרד מהבחירה המקורית. "
            "הצעות כלל 4 נשארו בה-ש32; אבחון F5 שנבחר יתווסף ב-E4b."
        )
        + paragraph(
            f"איך קוראים: חציון לא משוקלל של {len(short)} קיפולים ופס IQR. "
            f"בסוף ההצעה חציון הוולידציה הוא {v150:.6f}; בבורר 4.1 מוצג "
            f"ממוצע משוקלל לפי משחקים {weighted:.6f}. אלה סוגי סיכום שונים. "
            "קו מלא הוא אימון ומקווקו ולידציה; הקו האנכי מציין את התקציב המקורי."
        )
        + paragraph(
            f"מה רואים: בתקציב {longest} עצים חציון הוולידציה הוא {v400:.6f} "
            f"אחרי {min(400, longest)} עצים ומגיע ל-{vend:.6f} בסוף, כשהאימון "
            f"ממשיך לרדת. בתקציב {end} העצים הוולידציה עוד יורדת מעט; זה "
            "קרוב לאזור שבו השיפור מתייצב, ללא עצירה מוקדמת או בחירה מחדש."
        )
    )


def render(frame: Any, names: dict[str, str]) -> str:
    """Render complete precision, including both CI endpoints, without an index."""
    from ligat.viz.goals_configuration_names import configuration_name

    frame = frame.copy()
    for field in frame.columns:
        frame[field] = frame[field].map(
            lambda value: configuration_name(value) if isinstance(value, str) else value
        )
    return display_table(
        frame,
        names,
        show_index=False,
        precision={
            c: (
                0
                if pd.api.types.is_integer_dtype(frame[c])
                or c in {"n", "seasons", "games", "rows", "positive", "better", "rank"}
                else 6
            )
            for c in frame.select_dtypes(include="number")
        },
    ).content


def folds(b: dict[str, Any], name: str, scope: str = "all") -> Any:
    p = graphs._selected_folds(b, name)
    if scope == "deep":
        p = p.loc[p.deep_fold]
    elif scope == "last5":
        p = p.loc[p.season_id.between(21, 25)]
    return p.copy()


def decay_table(b: dict[str, Any]) -> Any:
    rows = []
    for model, role in [("XGBoost", "S2-AB"), ("GLM בבסיס (D3)", "glm-base"), ("Maher", "maher")]:
        for scope in SCOPES:
            p = b["validation"].loc[b["validation"].stage_role.eq(role)]
            if scope == "deep":
                p = p.loc[p.deep_fold]
            elif scope == "last5":
                p = p.loc[p.season_id.between(21, 25)]
            for half, g in p.groupby("half_life", dropna=False):
                rows.append(
                    dict(
                        model=model,
                        scope=SCOPES[scope],
                        half="בלי דעיכה" if pd.isna(half) else str(half),
                        nll=np.average(g.poisson_nll, weights=g.games),
                        seasons=g.season_id.nunique(),
                    )
                )
    return pd.DataFrame(rows)


def decay_conclusion(b: dict[str, Any]) -> str:
    p = decay_table(b)
    denom = graphs._elo_denominator(b["report"], "all")
    values = p.loc[p.scope.eq(SCOPES["all"])].groupby("model").nll.agg(["min", "max"])
    return paragraph(
        "; ".join(
            f"{name}: הפער הגדול ביותר {row['max'] - row['min']:.6f} NLL, "
            f"שהם {100 * (row['max'] - row['min']) / denom:.1f}% מתרומת Elo"
            for name, row in values.iterrows()
        )
        + ". Maher ללא דעיכה גרוע בהרבה; ל-XGBoost ול-GLM יש Elo וכושר "
        "המתעדכנים אחרי משחקים, ולכן ללא דעיכה נמצא אצלם בין הטובים."
    )


def ruler_html(b: dict[str, Any]) -> str:
    rows = [
        dict(
            model=LABELS.get(name, name),
            nll=graphs._scope_row(b, name, "all")["poisson_nll"],
            probability=100 * np.exp(-graphs._scope_row(b, name, "all")["poisson_nll"]),
        )
        for name in ["R0", "R1", "F5"]
    ]
    denominator = graphs._elo_denominator(b["report"], "all")
    return (
        '<div dir="rtl" id="elo-reading"><h3>סרגל לקריאת הפרשי NLL</h3></div>'
        + render(
            pd.DataFrame(rows), dict(model="ייחוס", nll="NLL", probability="הסתברות גאומטרית %")
        )
        + paragraph(
            f"כל מה ש-Elo מוסיף על יתרון הבית הוא {denominator:.3f} NLL; "
            f"שיפור של 0.001 הוא {100 * 0.001 / denominator:.1f}% מתרומת Elo. "
            '<bdi dir="ltr">100 × (NLL(R0) − NLL(model)) / '
            "(NLL(R0) − NLL(R1))</bdi> הוא אחוז מתרומת Elo; "
            "מאה אחוז = R1, אפס = R0. זה אינו אחוז ניחושים נכונים."
        )
    )


def glm_comparison() -> str:
    rows = [
        dict(
            model="GLM סרגל (EXP-002)",
            columns=", ".join(_COLUMNS["ruler"]),
            count=len(_COLUMNS["ruler"]),
            start="2006/2007",
            decay="בלי דעיכה",
        ),
        dict(
            model="GLM בבסיס (D3)",
            columns=", ".join(_COLUMNS["base"]),
            count=len(_COLUMNS["base"]),
            start="2007/2008",
            decay="ארבעה ערכי דעיכה",
        ),
    ]
    return render(
        pd.DataFrame(rows),
        dict(
            model="מודל",
            columns="פיצ׳רים",
            count="מספר פיצ׳רים",
            start="עונת תחילת האימון",
            decay="דעיכה",
        ),
    ) + paragraph(
        "ל-GLM בבסיס דרושים ערכי אשתקד שאינם קיימים בעונה הראשונה. הוא אינו מטפל "
        "ב-NaN כמו שיטות העצים ולכן מתחיל בעונה הבאה. המקור: prereg, Q27; "
        "בקיפול 2010/2011 יש לו רק שלוש עונות אימון."
    )


def metrics_table(b: dict[str, Any], scope: str) -> Any:
    names = ["R0", "R1", "S1", "S2", "F1", "F5", "EXP-002", "glm", "maher", "catboost-selected"]
    frames = {name: folds(b, name, scope) for name in names}
    if scope == "deep":
        common = set.intersection(*(set(p.season_id) for p in frames.values()))
        frames = {name: p.loc[p.season_id.isin(common)] for name, p in frames.items()}
    rows = []
    r0 = frames["R0"]
    r1 = frames["R1"]
    base = np.average(r0.poisson_nll, weights=r0.games)
    denominator = base - np.average(r1.poisson_nll, weights=r1.games)
    for name, p in frames.items():
        mean = float(np.average(p.poisson_nll, weights=p.games))
        by = pd.DataFrame(b["report"]["readable_metrics"][name]["by_season"])
        by = by.loc[by.season_id.isin(p.season_id)]
        total = by.better_rows.sum() + by.worse_rows.sum()
        better = (
            "—" if name == "S2" else f"{100 * by.better_rows.sum() / total:.2f}" if total else "—"
        )
        residuals = []
        cache = b.setdefault("walk_residuals", {})
        for unit in p.unit_id:
            if unit not in cache:
                path = Path("data/derived/exp005/full") / (unit + ".json")
                predictions = pd.DataFrame(json.loads(path.read_bytes())["payload"]["per_team"])
                predictions = predictions.set_index(["game_id", "team_id"])
                cache[unit] = predictions["lambda"] - b["data"].y.reindex(predictions.index)
                assert cache[unit].notna().all()
            residuals.extend(cache[unit].tolist())
        errors = np.asarray(residuals, dtype=float)
        rows.append(
            dict(
                model=LABELS.get(name, name),
                nll=mean,
                probability=100 * np.exp(-mean),
                elo=100 * (base - mean) / denominator,
                mae=float(np.abs(errors).mean()),
                rmse=float(np.sqrt(np.mean(errors**2))),
                better=better,
                seasons=p.season_id.nunique(),
            )
        )
    return pd.DataFrame(rows)


def paired_models_table(b: dict[str, Any]) -> Any:
    names = ["R0", "R1", "S1", "S2", "F1", "F5", "EXP-002", "glm", "maher", "catboost-selected"]
    frames = {name: folds(b, name, "deep") for name in names}
    common = set.intersection(*(set(p.season_id) for p in frames.values()))
    reference = frames["S2"].loc[frames["S2"].season_id.isin(common)]
    rows = []
    for name, p in frames.items():
        s = paired_statistics(p.loc[p.season_id.isin(common)], reference, score="poisson_nll")
        rows.append(
            dict(
                name=LABELS.get(name, name),
                d=s.mean_improvement,
                se=s.se,
                lower=s.ci_lower,
                upper=s.ci_upper,
                n=s.n,
            )
        )
    return pd.DataFrame(rows)


def review_table(b: dict[str, Any]) -> Any:
    raw = qa.review_evidence(b["report"], b["validation"])
    rows = []
    for stage, group in raw.groupby("stage", sort=False):
        proposal = group.loc[group.kind.eq("הצעה זמנית")].iloc[0]
        choice = next(r for r in b["report"]["open_reviews"] if r["name"] == stage)
        decision = next(d for d in b["report"]["decisions"] if d["name"] == stage)
        deep_key = choice["deep_leaders"][0]["key"]
        proposal_key = choice["proposal"]
        other_key = choice["recent_leaders"][0]["key"] if proposal_key == deep_key else deep_key
        selected = qa.configuration_folds(b["validation"], decision, proposal_key)
        alternative = qa.configuration_folds(b["validation"], decision, other_key)
        comparison = paired_statistics(alternative, selected, score="poisson_nll")
        original = decision["configurations"][proposal_key]
        changed = decision["configurations"][other_key]
        terms = {
            "max_depth": "עומק",
            "n_estimators": "עצים",
            "learning_rate": "קצב",
            "min_child_weight": "זהירות",
            "reg_lambda": "קנס",
            "subsample": "דגימה",
        }
        changes = [
            f"{terms.get(key, key)}: {value} במקום {original['parameters'].get(key)}"
            for key, value in changed["parameters"].items()
            if value != original["parameters"].get(key)
        ]
        if changed["half_life"] != original["half_life"]:

            def decay(value: Any) -> str:
                return "בלי דעיכה" if value is None else f"{value:g} עונות"

            changes.append(
                f"דעיכה: {decay(changed['half_life'])} במקום {decay(original['half_life'])}"
            )
        dependencies = (
            "כל בדיקות הסולם והמדדים התלויים בבסיס"
            if stage == "S2-AB"
            else "המסלול המקוצר והמדדים של המשפחה"
            if str(stage).startswith("F")
            else "התצורה והמדדים של "
            + {
                "S1-AB": "סרגל Elo ובית המכוונן",
                "catboost-selected": "CatBoost",
                "glm-base": "GLM בבסיס",
                "maher": "Maher",
            }.get(str(stage), str(stage))
        )
        rows.append(
            dict(
                question=f"{stage}: איזו תצורה תמשיך?",
                stage=stage,
                proposal=proposal.label,
                alternative=decision["labels"][other_key],
                changes="; ".join(changes) or "אותה תצורה",
                gap=comparison.mean_improvement,
                se=comparison.se,
                deep_gap=proposal.gap,
                deep_se=proposal.se,
                default="ההצעה של כלל 4 נשארת (ה-ש32)",
                depends=dependencies,
            )
        )
    frame = pd.DataFrame(rows)
    frame["signature"] = frame.apply(
        lambda row: (
            row.proposal,
            row.alternative,
            row.changes,
            round(row.gap, 12),
            round(row.se, 12),
        ),
        axis=1,
    )
    merged = []
    for _, group in frame.groupby("signature", sort=False):
        row = group.iloc[0].drop(labels="signature").to_dict()
        stage = str(group.stage.iloc[0])
        questions = {
            "S1-AB": "איזו תצורת Elo ובית (S1)?",
            "S2-AB": "איזו תצורה לבסיס S2?",
            "F1-short": "איזו תצורה למסלול F1–F4?",
            "F5-short": "איזו תצורה אחרי דגלי המוטיבציה?",
            "catboost-selected": "איזו תצורת CatBoost?",
            "glm-base": "איזו דעיכה ל-GLM בבסיס?",
            "maher": "איזה זמן מחצית ל-Maher?",
        }
        row["question"] = questions.get(stage, stage + ": איזו תצורה תמשיך?")
        row["depends"] = "; ".join(dict.fromkeys(group.depends))
        merged.append(row)
    result = pd.DataFrame(merged).drop(columns="stage")
    result.attrs["deep_rows"] = result[["question", "deep_gap", "deep_se"]].to_dict("records")
    return result.drop(columns=["deep_gap", "deep_se"])


def post_summary(b: dict[str, Any], check: str) -> str:
    from ligat.viz.goals_configuration_names import configuration_name

    p, intro, conclusion = graphs.post_hoc_sections(b, check)
    p = p.assign(includes_zero=p.lower.le(0) & p.upper.ge(0))
    rows = []
    for cfg, g in p.groupby("configuration", sort=False):

        def pick(ref: str, scope: str, frame: Any = g) -> Any:
            return frame.loc[frame.reference.eq(ref) & frame.scope.eq(scope)].iloc[0]

        sd, sl, fd, fl = (
            pick(*args)
            for args in [("S2", "deep"), ("S2", "last5"), ("F5", "deep"), ("F5", "last5")]
        )
        rows.append(
            dict(
                configuration=cfg,
                s2_deep=sd.d,
                s2_last=sl.d,
                f5_deep=fd.d,
                f5_last=fl.d,
                passes=sd.d > sd.se and sd.better >= 3 and sl.d > 0,
            )
        )
    names = dict(
        configuration="תצורה",
        s2_deep="מול S2 בעמוקים",
        s2_last="מול S2 בחמש האחרונות",
        f5_deep="מול F5 בעמוקים",
        f5_last="מול F5 בחמש האחרונות",
        passes="עומד במספרי כלל 8 מול S2",
    )
    full = p.rename(
        columns={
            "configuration": "תצורה",
            "check": "בדיקה",
            "scope": "טווח",
            "reference": "ייחוס",
            "nll": "NLL מועמד",
            "reference_nll": "NLL ייחוס",
            "d": "שיפור",
            "se": "SE",
            "lower": "תחתון CI95",
            "upper": "עליון CI95",
            "includes_zero": "כולל 0?",
            "n": "עונות",
            "positive": "עונות חיוביות",
            "better": "עדיפות בחמש האחרונות",
            "probability": "הסתברות גאומטרית %",
            "elo": "אחוז מתרומת Elo",
            "hda": "log loss",
        }
    )
    return configuration_name(
        intro
        + conclusion
        + render(pd.DataFrame(rows), names)
        + "<details><summary>כל הטבלאות המלאות</summary>"
        + render(full, {c: c for c in full})
        + "</details>"
    )


def expanded_glossary(original: str, b: dict[str, Any]) -> str:
    """Merge both existing dictionaries, retaining anchor IDs and unique concepts."""
    extra = graphs.explanation_glossary(b)
    rows = re.findall(r"<tr(?:\s[^>]*)?>.*?</tr>", original + extra, flags=re.S)
    unique = {}
    ids = {}
    for row in rows:
        cells = re.findall(r"<td>(.*?)</td>", row, flags=re.S)
        if not cells:
            continue
        key = graphs.plain_explanation(cells[0])
        if key == "Γ ו-deviance":
            continue
        unique[key] = cells
        match = re.search(r'id="([^"]+)"', row)
        if match:
            ids[key] = match[1]
    terms = {
        "דגימת עונות חוזרת (block bootstrap)": (
            "דגימה עם החזרה של עונות שלמות; שומרת יחד משחקים מאותה עונה. "
            "טווחי אחוזונים מתארים אי-ודאות של המדד, ולא תוצאה מובטחת למשחק."
        ),
        "ECE לפי עשירונים": (
            "פער מוחלט חזוי מול נצפה, משוקלל בגודל קבוצות הסתברות לפי מספר משחקים. "
            "ערכים זהים לא מפוצלים; הטווח 0–1, נמוך פירושו התאמה טובה בקבוצות שנוצרו."
        ),
        "כיול": "התאמה בין ההסתברות החזויה לבין שכיחות התוצאה בפועל",
        "ECE": "ממוצע משוקלל בגודל קבוצות ההסתברות של הפער המוחלט בין חזוי לנצפה",
        "פיזור יתר": "שונות מותנית גדולה מזו של Poisson; שגיאת ממוצע יכולה להיראות דומה",
        "סטטיסטי הפיזור של פירסון": (
            'ממוצע <bdi dir="ltr">(Y−λ)²/λ</bdi>; משלב שונות ושגיאות בתחזית הממוצע'
        ),
        "Dixon-Coles": "תיקון לתלות בתוצאות נמוכות, כגון 0-0 ו-1-1",
        "מתאם": "מדד לתנועה לינארית משותפת; אינו מוכיח סיבתיות או ערך בשילוב מודלים",
        "סולם": "הוספת משפחת מידע בכל פעם מול אותו ייחוס",
        "קיפול עמוק": "קיפול עם די עונות אימון לפי הסף שנרשם מראש",
        "חמש האחרונות": "עונות 2019/2020–2023/2024",
        "בדיקה בדיעבד": "בדיקה שנוספה אחרי שנראו עונות הפיתוח",
        "תצורת כניסה": "הפרמטרים הקבועים שבהם נמדדת תרומת משפחה",
        "בסיס מכוונן": "אותו מידע עם קנס, דגימה ודעיכה של המסלול המקוצר",
        "תקציב למידה": "מספר העצים כפול קצב הלמידה",
        "R0 / R1 / S1 / S2 / F1–F6": "בית; Elo ובית; שלב ראשון; בסיס; משפחות מידע",
        "prior לעולה": "מידע מוקדם על כוח עולה ביחס לליגה לפני שיש לה משחקים",
        "בדיקות מרובות": (
            "בדיקת חלופות רבות מגדילה את הסיכוי לממצא מקרי. כלל 8 הוא כלל החלטה "
            "ולא מבחן מובהקות, ולכן אין לו סיכוי מזל מדויק."
        ),
    }
    for key, value in terms.items():
        unique[key] = [key, value, "—", "הגדרה לקריאת הניסוי; אינה הכרעה"]
    examples = {
        "דגימת עונות חוזרת (block bootstrap)": (
            "אם עונת 2023/24 נדגמה פעמיים, כל משחקיה נספרים פעמיים.",
            "כך בודקים עד כמה פער התיקו תלוי בהרכב העונות.",
        ),
        "ECE לפי עשירונים": (
            "קבוצה עם תחזית 25% ושכיחות 27% תורמת פער 0.02 לפי משקלה.",
            "רואים אם משחקים שקיבלו תחזיות תיקו דומות אכן מסתיימים בתיקו בשכיחות דומה.",
        ),
        "פיזור יתר": (
            "במשחקים עם אותו λ, ייתכנו יותר תוצאות קיצוניות מכפי ש-Poisson חוזה.",
            "קצב שערים תקין בממוצע עשוי להחמיץ משחקים עם מעט מאוד או הרבה שערים.",
        ),
        "prior לעולה": (
            "Maher מתחיל עם מידע מוקדם בעוצמה של שישה משחקים.",
            "בתחילת העונה, כוח העולה מקבל עוגן עד שמצטברות תוצאות מליגת העל.",
        ),
        "רעש בכיול": (
            "בתחזית תיקו של 25%, מדמים תיקו בהסתברות 25%.",
            "גם מודל מכויל יראה פער זמני בין תחזית לשכיחות בגלל מקריות התוצאות.",
        ),
    }
    unique["רעש בכיול"] = [
        "רעש בכיול",
        "התפלגות ECE במדגמים מדומים תחת כיול מושלם",
        *examples["רעש בכיול"],
    ]
    for key, example in examples.items():
        unique[key][2:] = example
    for key, definition, illustration, meaning in [
        (
            "SHAP",
            "פירוק תחזית לוג λ לערך בסיס ולתרומות הפיצ׳רים",
            "תרומה של 0.1 בלוג מכפילה את הקצב בערך ב-1.105",
            "עוזר להבין מה העלה את תחזית ההבקעה למשחק מסוים",
        ),
        (
            "gain",
            "שיפור במטרת האימון מפיצולים שמשתמשים בפיצ׳ר",
            "20% הוא חמישית מהסכום המנורמל בקיפול",
            "מראה מאיזה מידע הפיצולים בעצים הפיקו שיפור באימון",
        ),
        (
            "ממוצע גאומטרי של מכפיל",
            "ממוצע הלוגריתמים והמרה בחזרה למכפיל",
            "הממוצע הגאומטרי של 0.8 ו-1.25 הוא 1",
            "מסכם כמה גדל קצב ההבקעה היחסי של הקבוצות בבית",
        ),
        (
            "החלפה נגדית",
            "שינוי קלט אחד תוך שמירה על יתר הקלטים",
            "אותה שורת משחק עם דגל בית ואז עם דגל חוץ",
            "מבודד את תגובת המודל לדגל הבית בלבד",
        ),
    ]:
        unique[key] = [key, definition, illustration, meaning]
    groups: dict[str, list[str]] = {"מודלים": [], "מדדים": [], "כללי הכרעה": [], "ניסוי והנדסה": []}
    for key, cells in unique.items():
        group = (
            "מודלים"
            if key in ["Elo", "GLM", "XGBoost", "CatBoost", "Maher", "prior לעולה"]
            else "כללי הכרעה"
            if key.startswith("כלל") or key == "הכרעה פתוחה"
            else "מדדים"
            if key
            in [
                "NLL",
                "λ",
                "Γ",
                "deviance",
                "RPS",
                "log loss",
                "IQR",
                "SE",
                "d̄",
                "הסתברות גאומטרית",
                "אחוז מתרומת Elo",
                "Pearson r",
                "פיזור יתר",
                "גרף יער",
                "הפרש מזווג ו-SE עונתי",
            ]
            else "ניסוי והנדסה"
        )
        anchor = ids.get(key, TERM_LINKS.get(key, "term-" + str(list(unique).index(key))))
        groups[group].append(
            '<tr id="'
            + html.escape(anchor)
            + '">'
            + "".join("<td>" + c + "</td>" for c in cells)
            + "</tr>"
        )
    return (
        '<div dir="rtl" id="glossary"><h2>10 — מילון לפי נושא</h2>'
        + "".join(
            "<h3>"
            + name
            + "</h3><table><tr><th>מושג</th><th>מה זה</th><th>דוגמה</th><th>משמעות</th></tr>"
            + "".join(parts)
            + "</table>"
            for name, parts in groups.items()
        )
        + "</div>"
    )
