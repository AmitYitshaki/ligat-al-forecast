"""Amit's dated decisions displayed above immutable EXP-005 evidence."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.viz import goals_qa as qa
from ligat.viz import goals_walk_review as review


def opening(report: dict[str, Any], *, heading: str = "0 — התשובה: M2 שנבחר הוא F5") -> str:
    """Read the selected-model figures and matched feature-entry evidence."""
    rows = {r["scope"]: r for r in report["readable_metrics"]["F5"]["summary"]}
    d = next(d for d in report["decisions"] if d["name"] == "F5" and d["kind"] == "entry")["rule8"][
        "stats"
    ]
    removed = next(d for d in report["decisions"] if d["name"] == "F5-without-opp_nothing_to_play")[
        "rule8"
    ]["stats"]
    positive = sum(x["difference"] > 0 for x in d["differences"])
    return (
        f'<div dir="rtl"><h2>{heading}</h2>'
        "<p>השאלה: אילו פיצ׳רים ואיזה מודל נכנסים ל-M2? עמית הכריע ב-27.11.2026 "
        "(ה-ש30–ה-ש34): XGBoost על הבסיס ועוד דגל אין על מה לשחק לשני הצדדים.</p>"
        f"<p>הסתברות גאומטרית: {rows['deep']['observed_probability_percent']:.2f}% "
        "בעמוקים (14 עונות, 2010/11–2023/24) "
        f"ו-{rows['last5']['observed_probability_percent']:.2f}% בחמש האחרונות; "
        f"{rows['deep']['elo_share_percent']:.1f}% ו-{rows['last5']['elo_share_percent']:.1f}% "
        "מתרומת Elo, בהתאמה.</p>"
        f'<p>השיפור המזווג בכניסה מעל הבסיס: <bdi dir="ltr">d̄={d["mean_improvement"]:.6f}, '
        f"SE={d['se']:.6f}</bdi>; {positive} מתוך {d['n']} עונות עמוקות חיוביות "
        f"ו-{d['last5_folds_better']} מתוך {d['last5_folds']} האחרונות. "
        "שאר המשפחות לא עברו את כלל 8.</p>"
        "<p>ידיעה שלקבוצה, ובעיקר ליריבה, אין על מה לשחק משפרת את חיזוי השערים; "
        "הערך מגיע בעיקר מדגל היריבה: הוצאתו לבד פוגעת, "
        f'<bdi dir="ltr">CI95 [{removed["ci_lower"]:.5f}, {removed["ci_upper"]:.5f}]</bdi>.</p>'
        "<p>ל-M3 עוברים F5, GLM בבסיס ו-Maher; R1 הוא קו הבסיס. "
        "ממשיכים בשרשרת המלאה ללא גרסה 0. פרק 7 ושכבת ה-λ יושלמו ב-E4b וב-E4c.</p></div>"
    )


def rule10_table(b: dict[str, Any]) -> Any:
    """Game-weighted deep means, with a fixed RPS-winner minus NLL-winner sign."""
    rows = []
    for d in b["report"]["decisions"]:
        if d.get("rule10") is not True:
            continue
        if d["kind"] == "entry":
            a = qa.fold_group(b["validation"], d["config"], d["reference"])
            z = qa.fold_group(b["validation"], d["config"], d["candidate"])
            se = d["rule8"]["stats"]["se"]
        else:
            v = b["validation"]
            p = v.loc[v.stage_role.eq(d["name"]) & v.deep_fold]
            means = p.groupby("configuration_key")[["poisson_nll", "rps", "games"]].apply(
                lambda g: pd.Series(
                    {k: np.average(g[k], weights=g.games) for k in ["poisson_nll", "rps"]}
                )
            )
            a = p.loc[p.configuration_key.eq(means.poisson_nll.idxmin())]
            z = p.loc[p.configuration_key.eq(means.rps.idxmin())]
            se = np.nan
        a, z = a.loc[a.deep_fold], z.loc[z.deep_fold]
        nll_a, nll_z = (float(np.average(p.poisson_nll, weights=p.games)) for p in [a, z])
        rps_a, rps_z = (float(np.average(p.rps, weights=p.games)) for p in [a, z])
        rows.append(
            dict(
                stage=d["name"],
                nll_a=nll_a,
                nll_z=nll_z,
                rps_a=rps_a,
                rps_z=rps_z,
                nll_gap=nll_z - nll_a,
                rps_gap=rps_z - rps_a,
                se=se,
                seasons=a.fold.nunique(),
                decision="נסגר לפי NLL (ה-ש34)",
            )
        )
    return pd.DataFrame(rows)


def rule10_html(b: dict[str, Any]) -> str:
    return review.paragraph(
        "שלושת דגלי כלל 10 נסגרו לפי NLL (ה-ש34). A היא המנצחת לפי NLL, "
        "B לפי RPS; הפערים הם B פחות A. בשורת ההוצאה, A שומרת את עמודת "
        "המחזורים ו-B מוציאה אותה; שיפור ההוצאה הוא מינוס הפער ב-NLL. "
        "SE מוצג עבור בדיקת ההוצאה. אלה החלטות מעל הדוח ההיסטורי, שאינו משתנה."
    ) + review.render(
        rule10_table(b),
        dict(
            stage="בדיקה",
            nll_a="NLL של A",
            nll_z="NLL של B",
            rps_a="RPS של A",
            rps_z="RPS של B",
            nll_gap="פער NLL",
            rps_gap="פער RPS",
            se="SE להוצאה",
            seasons="עונות",
            decision="הכרעת עמית",
        ),
    )


def final_summary(b: dict[str, Any], diagnostic: dict[str, Any] | None = None) -> str:
    from ligat.viz import goals_walk_evidence as evidence

    frames = evidence.candidate_folds(b)
    frames[review.LABELS["F5"]] = review.folds(b, "F5")
    frames[review.LABELS["catboost-selected"]] = review.folds(b, "catboost-selected")
    common = set.intersection(*(set(p.loc[p.deep_fold].season_id) for p in frames.values()))
    means = {
        n: float(
            np.average(
                p.loc[p.season_id.isin(common)].poisson_nll,
                weights=p.loc[p.season_id.isin(common)].games,
            )
        )
        for n, p in frames.items()
    }
    r0 = review.folds(b, "R0")
    r0 = r0.loc[r0.season_id.isin(common)]
    base = float(np.average(r0.poisson_nll, weights=r0.games))
    ruler = means[review.LABELS["R1"]]
    ranking = pd.DataFrame(
        [
            dict(model=n, nll=m, elo=100 * (base - m) / (base - ruler), seasons=len(common))
            for n, m in sorted(means.items(), key=lambda x: x[1])
        ]
    )
    middle = "חוסר התיקו בשליש האמצעי של האיזון"
    if diagnostic is not None:
        p = diagnostic["draw_bootstrap"]
        row = p.loc[p.candidate.eq(review.LABELS["F5"]) & p.group.eq(1)].iloc[0]
        middle += (
            f' אצל F5, בנקודות אחוז <bdi dir="ltr">{100 * row.gap:+.2f} '
            f"[{100 * row.lower:+.2f}, {100 * row.upper:+.2f}]</bdi>"
        )
    ranking["nll"] = ranking.nll.map("{:.4f}".format)
    ranking["elo"] = ranking.elo.map("{:.1f}%".format)
    return (
        opening(b["report"], heading="10 — סיכום שנקרא לבד")
        + review.paragraph(
            "דירוג תיאורי לפי NLL בעמוקים המשותפים בלבד; שמות במילים ובדיקות בדיעבד מסומנות. "
            "הדירוג אינו מבטל את בחירת עמית בשלושה, שנעשתה גם לפי פשטות וגיוון."
        )
        + review.render(
            ranking, dict(model="אפשרות שנבדקה", nll="NLL", elo="אחוז מתרומת Elo", seasons="עונות")
        )
        + review.paragraph(
            "פ-15 וכללים 4, 6 ו-10 נסגרו בה-ש31, ה-ש32 וה-ש34. "
            f"ל-06 נשארו {middle}, דפוס התוצאות הנמוכות ש-Dixon-Coles מתקן, "
            "ועודף 0-0 בחמש האחרונות המחייב לבדוק את יציבות ρ. יעד חדש לפרויקט ייקבע בתכנון 06. "
            "אין סימן לפיזור יתר; המדד לא מפריד בין פיזור יתר לטעות ב-λ. Maher מדווח בנפרד: 1.027. "
            "פרק 7 ושכבת ה-λ יושלמו ב-E4b וב-E4c."
        )
    )


def clean(text: str) -> str:
    """Remove obsolete presentation wording, preserving every numeric value."""
    for old, new in {
        "טיוטה": "תוצאות EXP-005",
        "F5 (המוצע)": "F5 (נבחר)",
        "F5 המוצע": "F5 (נבחר)",
        "המודל המוצע": "M2 שנבחר",
        "למודל המוצע": "ל-M2 שנבחר",
        "במודל המוצע": "ב-M2 שנבחר",
        "המודל הסופי": "M2 שנבחר (F5, ה-ש31)",
        "הכרעות הפתוחות": "הכרעות 27.11.2026",
        "הכרעות פתוחות": "הכרעות 27.11.2026",
        "לפני הכרעות עמית": "היסטוריית הבדיקות; עמית הכריע בה-ש31–ה-ש34",
        "אין כאן בחירת מודל": "הבחירה התקבלה בה-ש31–ה-ש33",
        "ההכרעות בפרק 6 זמניות": "ההכרעות נסגרו בה-ש31–ה-ש34",
        "Poisson מספיק": "אין סימן לפיזור יתר; המדד לא מפריד בין פיזור יתר לטעות ב-λ",
    }.items():
        text = text.replace(old, new)
    return text
