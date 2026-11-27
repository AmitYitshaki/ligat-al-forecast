"""Human-readable configuration names and selector context, presentation only."""

import re
from typing import Any

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.viz import goals_qa_graphs as graphs


def configuration_name(value: Any, *, model: str = "xgb", stage_b: bool = False) -> str:
    """The one display formatter; native keys and frozen labels remain unchanged."""
    if isinstance(value, str):
        text = value
        for key, label in {
            "max_depth": "עומק",
            "depth": "עומק",
            "iterations": "איטרציות",
            "n_estimators": "עצים",
            "learning_rate": "קצב",
            "half_life": "זמן מחצית",
            "min_child_weight": "זהירות",
            "reg_lambda": "קנס",
            "subsample": "דגימה",
            "preset": "תצורת ייחוס",
            "library_default": "ברירת המחדל של הספרייה",
            "None": "בלי דעיכה",
        }.items():
            text = re.sub(r"\b" + re.escape(key) + r"\b", label, text)
        text = re.sub(r"(\b(?:2|3|4)(?:\.0)?) עונה\b", r"\1 עונות", text)
        patterns: list[tuple[str, Any]] = [
            (r"\bd(\d+(?:\.\d+)?)\b", r"עומק \1"),
            (r"\bT(\d+)", r"\1 עצים"),
            (r"\bn(\d+)(?= ·)", r"\1 עצים"),
            (r"\blr(\d+(?:\.\d+)?)", r"קצב \1"),
            (r"\b(?:mcw|cw)(\d+(?:\.\d+)?)", r"זהירות \1"),
            (r"L2=(\d+(?:\.\d+)?)", r"קנס \1"),
            (r"\bl2=(\d+(?:\.\d+)?)", r"קנס \1"),
            (r"\bit(\d+)", r"\1 איטרציות"),
            (r"\boh(\d+)", r"סף קידוד \1"),
            (r"\bcatboost\b", "CatBoost"),
            (r"\bglm-base\b", "GLM בבסיס (D3)"),
            (r"\brl(\d+(?:\.\d+)?)", r"קנס \1"),
            (r"\bss(\d+(?:\.\d+)?)", lambda m: f"דגימה {100 * float(m[1]):g}%"),
            (r"\bcs\d+(?:\.\d+)?\s*·?", ""),
            (r"\bseed\d+\s*·?", ""),
            (r"proposal", "★ ההצעה"),
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        if "CatBoost" in text:
            text = re.sub(r"(\d+) עצים", r"\1 איטרציות", text)
        return text.replace(" / ", " · ").replace("XGB ", "XGBoost · ")
    p = value.get("parameters", value)
    depth = p.get("max_depth", p.get("depth"))
    trees = p.get("n_estimators", p.get("iterations"))
    rate = p.get("learning_rate")
    if model == "catboost":
        return f"CatBoost · עומק {depth} · {trees} איטרציות · קנס {p.get('l2_leaf_reg')}"
    if model in ("glm", "maher"):
        decay = value.get("half_life")
        return ("GLM בבסיס (D3)" if model == "glm" else "Maher") + (
            " · בלי דעיכה" if decay is None else f" · דעיכה {decay:g} עונות"
        )
    text = f"עומק {depth:g} · {trees:g} עצים · קצב {rate:g}"
    if "min_child_weight" in p:
        text += f" · זהירות {p['min_child_weight']:g}"
    if stage_b:
        text += f" · קנס {p.get('reg_lambda', 0):g} · דגימה {100 * p.get('subsample', 1):g}%"
        decay = value.get("half_life")
        text += " · בלי דעיכה" if decay is None else f" · דעיכה {decay:g} עונות"
    return text


def figure_labels(figure: Any) -> Any:
    """Translate visible strings, including selector updates, without changing numbers."""

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return configuration_name(value)
        if isinstance(value, dict):
            styles = {"template", "colorscale", "marker", "line", "font", "bdata", "dtype"}
            return {key: item if key in styles else walk(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(item) for item in value]
        if isinstance(value, np.ndarray) and value.dtype.kind in "OU":
            return walk(value.tolist())
        return value

    return go.Figure(walk(figure.to_dict()))


def _dots(value: float, values: list[float]) -> str:
    position = values.index(value)
    return "●" * (position + 1) + "○" * (len(values) - position - 1)


def selectors(figures: dict[str, Any], b: dict[str, Any]) -> None:
    v = b["validation"]
    v = v.loc[v.stage_role.eq("S2-A") & v.preset.eq("grid")]
    cfg = b["report"]["winners"]["S2"]["parameters"]
    fields = ["depth", "n_estimators", "learning_rate", "min_child_weight"]
    proposal_budget = cfg["n_estimators"] * cfg["learning_rate"]
    train = b["training"].sort_values("iteration").groupby("unit_id").tail(1)
    joined = v.merge(train[["unit_id", "train_loss"]], on="unit_id", validate="one_to_one")
    for name in ["training_selector", "train_vs_validation_heatmap", "hyperparameter_heatmap"]:
        fig = figures[name]
        menu = fig.layout.updatemenus[0]
        buttons = list(menu.buttons)
        old_preferred = menu.active
        ordered = [old_preferred] + [i for i in range(len(buttons)) if i != old_preferred]
        fixed_annotations = [annotation.to_plotly_json() for annotation in fig.layout.annotations]
        new_buttons = []
        if name == "training_selector":
            groups = [p for _, p in v.groupby("group_key", sort=True)]
        else:
            groups = [p for _, p in v.groupby(["n_estimators", "learning_rate"])]
        ordered[1:] = sorted(ordered[1:], key=lambda i: tuple(groups[i].iloc[0][f] for f in fields))
        initial_annotations = []
        for index in ordered:
            button = buttons[index].to_plotly_json()
            part = groups[index]
            row = part.iloc[0]
            budget = float(row.n_estimators * row.learning_rate)
            if name == "training_selector":
                hover = (
                    f"<br>דגימת עמודות {100 * float(row.colsample_bytree):g}% · seed {cfg['seed']}"
                )
                for trace in fig.data[index * 4 : index * 4 + 4]:
                    if trace.hovertemplate:
                        trace.hovertemplate = trace.hovertemplate.replace(
                            "<extra>", hover + "<extra>"
                        )
                label = configuration_name(
                    dict(
                        max_depth=row.depth,
                        n_estimators=row.n_estimators,
                        learning_rate=row.learning_rate,
                        min_child_weight=row.min_child_weight,
                    )
                )
                lines = []
                meanings = ["שאלות בכל עץ", "תיקונים מצטברים", "גודל כל תיקון", "ראיות לפני חוק"]
                hebrew = ["עומק", "עצים", "קצב", "זהירות"]
                for field, meaning, title in zip(fields, meanings, hebrew, strict=True):
                    val = float(row[field])
                    values = sorted(float(x) for x in v[field].unique())
                    if field == "min_child_weight":
                        meaning = (
                            "כמעט בלי דרישת ראיות" if val == min(values) else "יותר ראיות לפני חוק"
                        )
                    lines.append(f"{title} {val:g} {_dots(val, values)} — {meaning}")
                j = joined.loc[joined.unit_id.isin(part.unit_id)]
                training = float(np.average(j.train_loss, weights=j.games))
                validation = float(np.average(j.poisson_nll, weights=j.games))
                gap = validation - training
                lines.append(f"אימון {training:.6f} · ולידציה {validation:.6f} · פער {gap:.6f}")
                elo_gap = 100 * gap / graphs._elo_denominator(b["report"], "all")
                lines.append(f"פער: {elo_gap:.1f}% מתרומת Elo")
            else:
                label = f"{int(row.n_estimators)} עצים · קצב {row.learning_rate:g}"
                tree_dots = _dots(float(row.n_estimators), sorted(v.n_estimators.unique()))
                rate_dots = _dots(float(row.learning_rate), sorted(v.learning_rate.unique()))
                lines = [
                    f"{int(row.n_estimators)} עצים {tree_dots}",
                    f"קצב {row.learning_rate:g} {rate_dots}",
                ]
            budget_text = f"תקציב {budget:g} · פי {budget / proposal_budget:.2f} מההצעה"
            if name == "training_selector":
                lines.append(budget_text + f" ({proposal_budget:g})")
            else:
                lines[-1] += " · " + budget_text
            if index == old_preferred:
                label = "★ ההצעה · " + label
            annotation = dict(
                text="התצורה שנבחרה<br>" + "<br>".join(lines),
                x=1,
                y=1.14,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                align="right",
                bgcolor="#F1F5F9",
                borderpad=8,
                font=dict(size=13),
            )
            annotations = fixed_annotations + [annotation]
            button.update(
                label=label, method="update", args=[button["args"][0], {"annotations": annotations}]
            )
            new_buttons.append(button)
            if index == old_preferred:
                initial_annotations = annotations
        menu.buttons = new_buttons
        menu.active = 0
        menu.y = 1.02
        menu.x = 0
        fig.update_layout(
            annotations=initial_annotations,
            margin=dict(t=330, b=150),
            height=1050,
            title=dict(y=0.98, yanchor="top"),
        )
        if name != "training_selector":
            fig.update_xaxes(title="זהירות: כמה ראיות לפני שבונים חוק")
            fig.layout.yaxis.title.text = "עומק: כמה שאלות בכל עץ"
