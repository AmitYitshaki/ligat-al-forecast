"Read-only QA walkthrough graphs, using signed full and post-hoc evidence."

import base64
import html
import re
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from scipy.stats import t  # type: ignore[import-untyped]

from ligat.evaluate.selection import paired_statistics
from ligat.viz import goal_training_curves as curves
from ligat.viz import goals_notebook as views
from ligat.viz import goals_qa as qa
from ligat.viz.seasons import season_label

GREEN, RED, GRAY = "#16A34A", "#DC2626", "#6B7280"


def bundle(spec: Any) -> dict[str, Any]:
    from ligat.evaluate.goals_post_hoc import load_post_hoc

    return dict(
        report=curves.frozen_full_report(spec),
        training=curves.training_curves(spec, stage="full"),
        validation=curves.full_validation_table(spec).assign(season_id=lambda x: x.fold + 10),
        post=load_post_hoc(spec),
        data=views.development_data(),
    )


def _array(value: Any) -> np.ndarray[Any, Any]:
    if isinstance(value, dict):
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
    return np.asarray(value, dtype=float)


def _summary(frame: Any, field: str = "poisson_nll") -> dict[str, float]:
    values = frame[field].to_numpy(float)
    n = len(values)
    mean = float(np.average(values, weights=frame.games))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    radius = float(t.ppf(0.975, n - 1)) * se if n > 1 else float("nan")
    return dict(mean=mean, se=se, lower=mean - radius, upper=mean + radius, n=n)


def _elo_denominator(report: dict[str, Any], scope: str = "deep") -> float:
    scores = {
        name: next(
            x["poisson_nll"]
            for x in report["readable_metrics"][name]["summary"]
            if x["scope"] == scope
        )
        for name in ["R0", "R1"]
    }
    return float(scores["R0"] - scores["R1"])


def training_selector(b: dict[str, Any]) -> Any:
    train, v, report = b["training"], b["validation"], b["report"]
    v = v.loc[v.stage_role.eq("S2-A") & v.preset.eq("grid")]
    fig = go.Figure()
    buttons = []
    preferred = 0
    cfg = report["winners"]["S2"]["parameters"]
    for index, (_, part) in enumerate(v.groupby("group_key", sort=True)):
        row = part.iloc[0]
        logs = train.loc[train.unit_id.isin(part.unit_id)]
        grouped = (
            logs.groupby("iteration")
            .train_loss.quantile([0.25, 0.5, 0.75])
            .unstack()
            .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
        )
        for name, fill in [("q1", None), ("q3", "tonexty")]:
            fig.add_trace(
                go.Scatter(
                    x=grouped.index,
                    y=grouped[name],
                    line=dict(width=0),
                    fill=fill,
                    fillcolor="rgba(37,99,235,.12)",
                    showlegend=False,
                    hoverinfo="skip",
                    visible=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=grouped.index,
                y=grouped["median"],
                line=dict(color=views.COLORS[0]),
                name="אימון: חציון ו-IQR",
                visible=False,
                hovertemplate="עץ %{x}<br>NLL אימון %{y:.5f}<extra>n=15</extra>",
            )
        )
        stats = _summary(part)
        end = int(row.n_estimators)
        fig.add_trace(
            go.Scatter(
                x=[end],
                y=[stats["mean"]],
                mode="markers",
                error_y=dict(type="data", array=[stats["upper"] - stats["mean"]]),
                marker=dict(size=10, color=views.COLORS[1]),
                name="ולידציה בסוף בלבד: נקודה ו-CI95",
                visible=False,
                hovertemplate="NLL ולידציה %{y:.5f}<extra>n=15</extra>",
            )
        )
        label = f"d{row.depth:g} / T{end} / lr{row.learning_rate:g} / mcw{row.min_child_weight:g}"
        if all(
            float(row[field]) == float(cfg[key])
            for field, key in [
                ("depth", "max_depth"),
                ("n_estimators", "n_estimators"),
                ("learning_rate", "learning_rate"),
                ("min_child_weight", "min_child_weight"),
            ]
        ):
            preferred = index
        buttons.append(dict(label=label, method="update", args=[{}]))
    for i, button in enumerate(buttons):
        button["args"] = [dict(visible=[j // 4 == i for j in range(len(fig.data))])]
    for j in range(preferred * 4, preferred * 4 + 4):
        fig.data[j].visible = True
    fig.update_layout(
        updatemenus=[
            dict(buttons=buttons, active=preferred, x=0, y=1.07, xanchor="left", yanchor="bottom")
        ]
    )
    fig.update_xaxes(title="מספר עצים")
    fig.update_yaxes(title="Poisson NLL")
    result = views.finish(
        fig, "אימון נמוך אינו מבטיח ולידציה טובה; n=15, IQR ו-CI95 — טיוטה", height=850
    )
    result.update_layout(margin=dict(t=190, b=160), legend=dict(y=-0.2))
    return result


def slice_maps(b: dict[str, Any]) -> dict[str, Any]:
    train, v, report = b["training"], b["validation"], b["report"]
    v = v.loc[v.stage_role.eq("S2-A") & v.preset.eq("grid")]
    final = (
        train.loc[train.unit_id.isin(v.unit_id)].sort_values("iteration").groupby("unit_id").tail(1)
    )
    joined = v.merge(final[["unit_id", "train_loss"]], on="unit_id", validate="one_to_one")
    compare = make_subplots(rows=1, cols=2, subplot_titles=["אימון: חציון", "ולידציה: ממוצע"])
    share = go.Figure()
    buttons_compare = []
    buttons_share = []
    preferred = 0
    r0 = next(
        x["poisson_nll"] for x in report["readable_metrics"]["R0"]["summary"] if x["scope"] == "all"
    )
    denom = _elo_denominator(report, "all")
    depths = sorted(joined.depth.unique())
    cautions = sorted(joined.min_child_weight.unique())
    for index, ((trees, rate), part) in enumerate(
        joined.groupby(["n_estimators", "learning_rate"])
    ):
        rows = []
        for (depth, caution), group in part.groupby(["depth", "min_child_weight"]):
            rows.append(
                dict(
                    depth=depth,
                    caution=caution,
                    train=float(group.train_loss.median()),
                    train_lower=float(group.train_loss.quantile(0.25)),
                    train_upper=float(group.train_loss.quantile(0.75)),
                    **_summary(group),
                )
            )
        cells = pd.DataFrame(rows)
        for col, field in [(1, "train"), (2, "mean")]:
            pivot = cells.pivot(index="depth", columns="caution", values=field).reindex(
                index=depths, columns=cautions
            )
            text = [[f"{value:.4f}" for value in row] for row in pivot.to_numpy()]
            lower_field, upper_field = (
                ("train_lower", "train_upper") if col == 1 else ("lower", "upper")
            )
            intervals = np.stack(
                [
                    cells.pivot(index="depth", columns="caution", values=key)
                    .reindex(index=depths, columns=cautions)
                    .to_numpy()
                    for key in [lower_field, upper_field]
                ],
                axis=-1,
            )
            compare.add_trace(
                go.Heatmap(
                    z=pivot.to_numpy(),
                    x=list(range(len(cautions))),
                    y=list(range(len(depths))),
                    text=text,
                    texttemplate="%{text}",
                    textfont=dict(color="#111827"),
                    customdata=intervals,
                    colorscale=["#DCFCE7", "#F9FAFB", "#FEE2E2"],
                    zmin=float(pivot.min().min()),
                    zmax=float(pivot.max().max()),
                    visible=False,
                    colorbar=dict(x=0.44 if col == 1 else 1.02, thickness=12),
                    hovertemplate=(
                        "NLL %{z:.5f}<br>פס [%{customdata[0]:.5f}, "
                        "%{customdata[1]:.5f}]<extra>n=15; אימון IQR, ולידציה CI95</extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        pivot = cells.pivot(index="depth", columns="caution", values="mean").reindex(
            index=depths, columns=cautions
        )
        elo = 100 * (r0 - pivot) / denom
        elo_intervals = np.stack(
            [
                100
                * (
                    r0
                    - cells.pivot(index="depth", columns="caution", values=key)
                    .reindex(index=depths, columns=cautions)
                    .to_numpy()
                )
                / denom
                for key in ["upper", "lower"]
            ],
            axis=-1,
        )
        # Each slice uses its own range. Zero still separates negative/red from
        # positive/green, while positive values retain visible contrast.
        low, high = float(elo.min().min()), float(elo.max().max())
        if low == high:
            low, high = low - 1, high + 1
        if low < 0 < high:
            zero = -low / (high - low)
            scale = [[0, "#FCA5A5"], [zero, "#FFFFFF"], [1, "#16A34A"]]
        elif high <= 0:
            scale = [[0, "#FCA5A5"], [1, "#FFF1F2"]]
        else:
            scale = [[0, "#F0FDF4"], [1, "#16A34A"]]
        share.add_trace(
            go.Heatmap(
                z=elo.to_numpy(),
                x=list(range(len(cautions))),
                y=list(range(len(depths))),
                text=[[f"{x:+.1f}%".replace("-", "−") for x in row] for row in elo.to_numpy()],
                texttemplate="%{text}",
                textfont=dict(color="#111827"),
                customdata=elo_intervals,
                colorscale=scale,
                zmin=low,
                zmax=high,
                visible=False,
                hovertemplate=(
                    "תרומת Elo %{z:+.1f}%<br>CI95 [%{customdata[0]:.1f}, "
                    "%{customdata[1]:.1f}]<extra>n=15</extra>"
                ),
            )
        )
        label = f"T{int(trees)} / lr{rate:g}"
        if (
            trees == report["winners"]["S2"]["parameters"]["n_estimators"]
            and rate == report["winners"]["S2"]["parameters"]["learning_rate"]
        ):
            preferred = index
            label += " (proposal)"
        buttons_compare.append(dict(label=label, method="update", args=[{}]))
        buttons_share.append(dict(label=label, method="update", args=[{}]))
    for i, button in enumerate(buttons_compare):
        button["args"] = [dict(visible=[j // 2 == i for j in range(len(compare.data))])]
    for i, button in enumerate(buttons_share):
        button["args"] = [dict(visible=[j == i for j in range(len(share.data))])]
    for j in [2 * preferred, 2 * preferred + 1]:
        compare.data[j].visible = True
    share.data[preferred].visible = True
    for fig, buttons in [(compare, buttons_compare), (share, buttons_share)]:
        fig.update_layout(updatemenus=[dict(buttons=buttons, active=preferred, x=0, y=1.08)])
        fig.update_xaxes(
            title="זהירות (min_child_weight)",
            tickvals=list(range(len(cautions))),
            ticktext=[str(int(x)) for x in cautions],
        )
        fig.update_yaxes(
            title="עומק", tickvals=list(range(len(depths))), ticktext=[str(int(x)) for x in depths]
        )
    compare.update_yaxes(title=None, row=1, col=2)
    a = views.finish(
        compare, "תקציב וזהירות משנים את הפער בין אימון לוולידציה; n=15 — טיוטה", height=850
    )
    c = views.finish(share, "עצים רדודים מחזירים יותר מתרומת Elo; n=15 — טיוטה", height=850)
    for fig in [a, c]:
        fig.update_layout(margin=dict(t=190, b=110))
    return dict(train_vs_validation_heatmap=a, hyperparameter_heatmap=c)


def budget_table(b: dict[str, Any]) -> pd.DataFrame:
    v = b["validation"]
    r = b["report"]
    v = v.loc[v.stage_role.eq("S2-A") & v.preset.eq("grid")]
    r0 = next(
        x["poisson_nll"] for x in r["readable_metrics"]["R0"]["summary"] if x["scope"] == "all"
    )
    denom = _elo_denominator(r, "all")
    rows = []
    for (depth, trees, rate), p in v.loc[v.min_child_weight.eq(v.min_child_weight.max())].groupby(
        ["depth", "n_estimators", "learning_rate"]
    ):
        s = _summary(p)
        rows.append(
            dict(
                depth=depth,
                trees=trees,
                rate=rate,
                elo=100 * (r0 - s["mean"]) / denom,
                n=s["n"],
                lower=100 * (r0 - s["upper"]) / denom,
                upper=100 * (r0 - s["lower"]) / denom,
            )
        )
    return pd.DataFrame(rows)


def learning_curves(b: dict[str, Any], summary: dict[str, Any]) -> Any:
    # Plotly fills left to right. Put the frozen proposal second so it is the
    # first panel for a Hebrew reader (top right), without changing evidence.
    keys = list(dict.fromkeys(_learning_key(o) for o in summary["outputs"]))
    selected = summary["selected_config"]
    proposal = (
        selected.get("max_depth", 2),
        selected.get("learning_rate", 0.05),
        selected["n_estimators"],
    )
    if proposal in keys and len(keys) > 1:
        keys.remove(proposal)
        keys.insert(1, proposal)
    ordered = dict(
        summary, outputs=sorted(summary["outputs"], key=lambda o: keys.index(_learning_key(o)))
    )
    fig = curves.learning_curve_figure(ordered)
    values = [float(y) for trace in fig.data for y in _array(trace.y)]
    low, high = min(values), max(values)
    padding = 0.03 * (high - low)
    for name in fig.layout:
        if name.startswith("yaxis"):
            fig.layout[name].range = [low - padding, high + padding]
            if name != "yaxis":
                fig.layout[name].matches = "y"
    fig.layout.yaxis2.title.text = ""
    fig.layout.yaxis4.title.text = ""
    return fig


def _learning_key(output: dict[str, Any]) -> tuple[Any, Any, Any]:
    cfg = output["identity"]["config"]
    return cfg["max_depth"], cfg["learning_rate"], cfg["n_estimators"]


def paired_folds(b: dict[str, Any]) -> Any:
    decision = next(
        d for d in b["report"]["decisions"] if d["name"] == "F5" and d["kind"] == "entry"
    )
    s = decision["rule8"]["stats"]
    rows = s["differences"]
    fig = go.Figure()
    fig.add_bar(
        x=[season_label(x["season_id"]) for x in rows],
        y=[x["difference"] for x in rows],
        marker=dict(
            color=[GREEN if x["difference"] >= 0 else RED for x in rows],
            pattern=dict(shape=["/" if x["season_id"] >= 21 else "" for x in rows]),
        ),
        text=[f"{x['difference']:+.4f}".replace("-", "−") for x in rows],
        textposition="outside",
        customdata=[x["games"] for x in rows],
        hovertemplate="d %{y:+.6f}<br>משחקים %{customdata}<extra>חיובי טוב</extra>",
    )
    fig.add_hrect(
        y0=s["mean_improvement"] - s["se"],
        y1=s["mean_improvement"] + s["se"],
        fillcolor="rgba(107,114,128,.12)",
        line_width=0,
    )
    fig.add_hline(y=s["mean_improvement"], line_color=GRAY, line_dash="dash")
    fig.add_hline(y=0, line_color=GRAY)
    fig.update_xaxes(title="עונה; הצלבה = חמש האחרונות", tickangle=-45)
    fig.update_yaxes(title="תרומת F5: NLL ללא המשפחה פחות איתה")
    return views.finish(
        fig,
        f"F5 עקבית ב-{s['last5_folds_better']} מחמש האחרונות; n={s['n']}, פס ±SE — טיוטה",
        height=850,
    )


def family_ladder(b: dict[str, Any], *, base: bool = False) -> Any:
    report = b["report"]
    entries, _ = qa.entry_tables(report)
    part = entries.loc[
        entries.name.str.startswith("base-without-")
        if base
        else entries.name.isin(["F1-R", "F1-A", "F2", "F3", "F4", "F5", "F6"])
    ]
    scale = 100 / _elo_denominator(report)
    fig = go.Figure()
    for row in part.itertuples():
        mean = row.d * scale
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[row.name],
                mode="markers+text",
                text=[f"{mean:+.2f}%".replace("-", "−")],
                textposition="top center",
                error_x=dict(
                    type="data",
                    array=[(row.upper - row.d) * scale],
                    arrayminus=[(row.d - row.lower) * scale],
                    symmetric=False,
                ),
                marker=dict(
                    color=GREEN if row.passes else RED,
                    symbol="cross" if row.passes else "x",
                    size=11,
                ),
                name="עברה את כלל 8" if row.passes else "לא עברה את כלל 8",
                legendgroup=str(bool(row.passes)),
                showlegend=not any(
                    trace.legendgroup == str(bool(row.passes)) for trace in fig.data
                ),
                customdata=[[row.n, row.se * scale]],
                hovertemplate=(
                    "שיפור %{x:+.3f}%<br>n=%{customdata[0]}<br>SE "
                    "%{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_color=GRAY, line_dash="dash")
    fig.update_xaxes(title="תרומת מידע באחוז מתרומת Elo; CI95")
    fig.update_yaxes(title="בדיקה", autorange="reversed")
    return views.finish(
        fig,
        "רק F5 עברה את כלל 8; תרומת מידע בלי כוונון מחדש; n=14, CI95 — טיוטה"
        if not base
        else "הוצאת רכיבי הבסיס כמעט אינה משפרת; n=14, CI95 — טיוטה",
        height=850,
    )


def paired_forest(b: dict[str, Any]) -> Any:
    fig = go.Figure()
    for name, m in b["report"]["readable_metrics"].items():
        if name.startswith("F") or name == "S2":
            continue
        s = m["paired"]
        mean = s["mean_improvement"]
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[name],
                mode="markers",
                error_x=dict(
                    type="data",
                    array=[s["ci_upper"] - mean],
                    arrayminus=[mean - s["ci_lower"]],
                    symmetric=False,
                ),
                marker=dict(
                    color=GREEN if mean >= 0 else RED, symbol="cross" if mean >= 0 else "x", size=11
                ),
                showlegend=False,
                customdata=[s["n"]],
                hovertemplate="שיפור %{x:+.6f}<br>n=%{customdata}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_color=GRAY, line_dash="dash")
    fig.update_xaxes(title="שיפור מזווג מול S2 — חיובי טוב, CI95")
    fig.update_yaxes(title="מודל")
    return views.finish(
        fig, "אי-הוודאות בין עונות חשובה מדירוג זעיר; n משותף בריחוף — טיוטה", height=850
    )


def metric_by_season(b: dict[str, Any]) -> Any:
    fig = go.Figure()
    v = b["validation"]
    report = b["report"]
    for i, name in enumerate(
        ["S1", "S2", "F6", "EXP-002", "EXP-003", "maher", "glm", "catboost-selected"]
    ):
        if name in report["winners"]:
            p = qa.fold_group(v, report["winners"][name], report["stages"][name])
        elif name in report["comparators"]:
            p = qa.fold_group(
                v, report["comparators"][name]["config"], report["comparators"][name]["features"]
            )
        elif name in ["glm", "maher"]:
            cfg = report["comparators"].get("glm-base" if name == "glm" else name)
            if cfg is None:
                continue
            p = qa.fold_group(v, cfg["config"], cfg["features"])
        else:
            p = v.loc[v.stage_role.eq(name)].assign(season_id=lambda x: x.fold + 10)
        if p.empty:
            continue
        p = p.sort_values("fold")
        fig.add_trace(
            go.Scatter(
                x=p.season_id.map(season_label),
                y=p.poisson_nll,
                mode="lines+markers",
                name=name,
                line=dict(color=(*views.COLORS, "#0F766E", "#A16207")[i]),
                customdata=p.games,
                hovertemplate=(
                    "NLL %{y:.5f}<br>משחקים %{customdata}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_xaxes(title="עונת ולידציה", tickangle=-45)
    fig.update_yaxes(title="Poisson NLL")
    return views.finish(
        fig, ("פערי העונות גדולים מהפערים הזעירים בין חלק מהמודלים; 15 קיפולים — טיוטה"), height=850
    )


def win_rate_by_season(b: dict[str, Any]) -> Any:
    rows = b["report"]["readable_metrics"]["F5"]["by_season"]
    p = pd.DataFrame(rows)
    values = p.better_non_tie_percent.to_numpy(float)
    n = len(values)
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    radius = float(t.ppf(0.975, n - 1)) * se
    mean = float(np.mean(values))
    fig = go.Figure(
        go.Bar(
            x=p.season_id.map(season_label),
            y=values.tolist(),
            marker_color=[GREEN if x >= 50 else RED for x in values],
            text=[f"{x - 50:+.1f}".replace("-", "−") for x in values],
            customdata=p.better_rows + p.worse_rows,
            hovertemplate="%{y:.1f}%<br>שורות לא שוות %{customdata}<extra>תיאור עונה</extra>",
        )
    )
    fig.add_hline(y=50, line_color=GRAY, line_dash="dash")
    fig.add_hrect(
        y0=mean - radius, y1=mean + radius, fillcolor="rgba(107,114,128,.12)", line_width=0
    )
    fig.update_xaxes(title="עונה", tickangle=-45)
    fig.update_yaxes(title="אחוז שורות עדיפות מול S2; הטקסט = פער מ-50%")
    return views.finish(
        fig, f"יתרון השורות משתנה בין עונות; n={n}, הפס CI95 לממוצע העונות — טיוטה", height=850
    )


def models_vs_baselines(b: dict[str, Any]) -> Any:
    v = b["validation"]
    report = b["report"]
    fig = go.Figure()
    labels = {
        "F5": "F5: המודל המוצע (S2+F5)",
        "glm": "GLM בבסיס (D3)",
        "EXP-002": "GLM סרגל (EXP-002)",
        "catboost-selected": "CatBoost מהרשת",
    }
    for name in ["S1", "S2", "F5", "F6", "EXP-002", "EXP-003", "catboost-selected", "glm", "maher"]:
        if name in report["winners"]:
            p = qa.fold_group(v, report["winners"][name], report["stages"][name])
        elif name in report["comparators"]:
            p = qa.fold_group(
                v, report["comparators"][name]["config"], report["comparators"][name]["features"]
            )
        else:
            p = v.loc[v.stage_role.eq(name)].copy()
        p = p.loc[p.deep_fold]
        if p.empty:
            continue
        s = _summary(p)
        mean = 100 * np.exp(-s["mean"])
        lo = 100 * np.exp(-s["upper"])
        hi = 100 * np.exp(-s["lower"])
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[labels.get(name, name)],
                mode="markers",
                error_x=dict(
                    type="data", array=[hi - mean], arrayminus=[mean - lo], symmetric=False
                ),
                name=name,
                showlegend=False,
                marker=dict(
                    color=views.COLORS[1] if name == "F5" else views.COLORS[0],
                    size=15 if name == "F5" else 10,
                    symbol="diamond" if name == "F5" else "circle",
                ),
                customdata=[s["n"]],
                hovertemplate=(
                    "הסתברות גאומטרית %{x:.2f}%<br>n=%{customdata}<extra>CI95 בין עונות</extra>"
                ),
            )
        )
    fig.update_xaxes(title="הסתברות גאומטרית לתוצאה שנצפתה, %, CI95")
    fig.update_yaxes(title="מודל")
    return views.finish(
        fig, "חלק מהמודלים כמעט חופפים בהסתברות לתוצאה; קיפולים עמוקים — טיוטה", height=850
    )


def elo_by_scope(b: dict[str, Any]) -> Any:
    fig = go.Figure()
    report = b["report"]
    names = ["S1", "S2", "F6", "EXP-002", "EXP-003", "catboost-selected", "glm", "maher"]
    for index, scope in enumerate(["all", "deep", "last5"]):
        values = [
            next(
                x["elo_share_percent"]
                for x in report["readable_metrics"][name]["summary"]
                if x["scope"] == scope
            )
            for name in names
        ]
        errors = []
        for name in names:
            if name in report["winners"]:
                part = qa.fold_group(
                    b["validation"], report["winners"][name], report["stages"][name]
                )
            elif name in report["comparators"]:
                c = report["comparators"][name]
                part = qa.fold_group(b["validation"], c["config"], c["features"])
            else:
                part = b["validation"].loc[b["validation"].stage_role.eq(name)]
            if scope == "deep":
                part = part.loc[part.deep_fold]
            elif scope == "last5":
                part = part.loc[part.season_id.ge(21)]
            s = _summary(part)
            errors.append(100 * (s["upper"] - s["mean"]) / _elo_denominator(report, scope))
        fig.add_bar(
            x=names,
            y=values,
            name={"all": "כל העונות", "deep": "עמוקות", "last5": "חמש האחרונות"}[scope],
            marker_color=views.COLORS[index],
            error_y=dict(type="data", array=errors),
            text=[f"{x - 100:+.1f}".replace("-", "−") for x in values],
            hovertemplate="תרומת Elo %{y:.2f}%<extra>%{fullData.name}</extra>",
        )
    for line in [100, 102]:
        fig.add_hline(y=line, line_color=GRAY, line_dash="dash")
    fig.update_layout(barmode="group")
    fig.update_xaxes(
        title="מודל",
        tickvals=names,
        ticktext=["CatBoost" if name == "catboost-selected" else name for name in names],
    )
    fig.update_yaxes(title="אחוז מתרומת Elo; הטקסט = פער מ-100%")
    return views.finish(
        fig, "טווח העונות משנה את ההשוואה; n=15/13–14/5, CI95 בין עונות — טיוטה", height=850
    )


def review_plot(b: dict[str, Any]) -> Any:
    p = qa.review_evidence(b["report"], b["validation"])
    fig = go.Figure()
    for row in p.itertuples():
        fig.add_trace(
            go.Scatter(
                x=[row.gap],
                y=[_review_label(row.stage) + " · " + row.kind],
                mode="markers",
                error_x=dict(type="data", array=[row.se]),
                showlegend=False,
                marker=dict(
                    color=GREEN if row.gap <= 0 else RED,
                    symbol="cross" if row.gap <= 0 else "x",
                    size=9,
                ),
                customdata=[[row.n, row.stage]],
                hovertemplate="תוספת NLL %{x:+.6f}<br>n=%{customdata[0]}"
                "<br>ID: %{customdata[1]}<extra>±SE</extra>",
            )
        )
    fig.add_vline(x=0, line_color=GRAY, line_dash="dash")
    fig.update_xaxes(title="תוספת NLL מול המובילה העמוקה; פס ±SE")
    fig.update_yaxes(title="הכרעה ומועמדת", autorange="reversed")
    return views.finish(
        fig, "רוב פערי ההכרעות קטנים מהאי-ודאות שלהם; n משותף בריחוף — טיוטה", height=1150
    )


def _review_label(stage: str) -> str:
    labels = {
        "S1-AB": "S1: Elo ובית",
        "S2-AB": "S2: בסיס",
        "catboost-selected": "CatBoost מהרשת",
        "glm-base": "GLM בבסיס (D3)",
        "maher": "Maher",
    }
    return labels.get(stage, stage.replace("-short", " (מסלול מקוצר)"))


def post_hoc_forest(b: dict[str, Any]) -> Any:
    """Compare every diagnostic with S2 and F5 on the same deep seasons."""
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.07,
        subplot_titles=["מול המודל המוצע F5", "מול הבסיס S2"],
    )
    tables = pd.concat(
        [
            scoped_post_hoc_table(b, check, references=("S2", "F5"))
            for check in ["P1", "P2", "P3", "P4"]
        ],
        ignore_index=True,
    )
    labels = []
    for record in b["post"]["records"]:
        p = record["parameters"]
        detail = _post_hoc_label(record)
        prominent = (record["check"] == "P4" and p["iterations"] == 150) or (
            record["check"] == "P3"
            and p["max_depth"] == 1
            and p["n_estimators"] == 400
            and p["learning_rate"] == 0.05
        )
        label = (
            record["check"] + " " + ("CatBoost " if record["check"] == "P4" else "XGB ") + detail
        )
        label = f"<b>{label} ★</b>" if prominent else label
        labels.append(label)
        for col, ref in [(1, "F5"), (2, "S2")]:
            s = tables.loc[
                tables.check.eq(record["check"])
                & tables.configuration.eq(detail)
                & tables.reference.eq(ref)
                & tables.scope.eq("deep")
            ].iloc[0]
            value = s.d
            fig.add_trace(
                go.Scatter(
                    x=[value],
                    y=[label],
                    mode="markers",
                    error_x=dict(
                        type="data",
                        array=[s.upper - value],
                        arrayminus=[value - s.lower],
                        symmetric=False,
                    ),
                    marker=dict(
                        color=GREEN if value >= 0 else RED,
                        symbol="cross" if value >= 0 else "x",
                        size=11 if prominent else 8,
                    ),
                    showlegend=False,
                    customdata=[[s.n, ref, record["check"]]],
                    hovertemplate="שיפור %{x:+.6f}<br>n=%{customdata[0]}"
                    "<br>מול %{customdata[1]}<extra>CI95</extra>",
                ),
                row=1,
                col=col,
            )
    fig.add_vline(x=0, line_color=GRAY, line_dash="dash")
    fig.update_xaxes(title="שיפור NLL מזווג מול הייחוס, CI95")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, autorange="reversed")
    deep = tables.loc[tables.scope.eq("deep")]
    low, high = float(deep.lower.min()), float(deep.upper.max())
    padding = (high - low) * 0.05
    fig.update_xaxes(range=[low - padding, high + padding])
    fig.layout.xaxis2.matches = "x"
    result = views.finish(
        fig,
        "CatBoost רדוד ועומק 1 משפרים מול S2; בלי Elo גרוע יותר"
        "<br>n=14, CI95 — טיוטה, לפני הכרעות עמית",
        height=1150,
    )
    result.update_layout(margin=dict(l=330, t=185, b=130), title_font_size=17)
    return result


def decay_figures() -> dict[str, Any]:
    result = {}
    for daily in [False, True]:
        fig = views.decay_figure(daily=daily)
        halves = [None, 0.5, 1, 2] if daily else [None, 1, 2, 4]
        for trace, half in zip(fig.data, halves, strict=True):
            trace.name = (
                "בלי דעיכה"
                if half is None
                else "זמן מחצית: עונה אחת"
                if half == 1
                else f"זמן מחצית: {half:g} עונות"
            )
            if not daily:
                trace.line.shape = "hv"
        fig.add_hline(y=0.5, line_color=GRAY, line_dash="dash")
        if daily:
            for trace in fig.data:
                trace.x = _array(trace.x) * 365.25
            fig.update_xaxes(title="כמה ימים לפני יום החיזוי")
        result["daily_decay" if daily else "season_decay"] = views.finish(
            fig,
            "ל-Maher דעיכה חיונית; ללא דעיכה מאבד דיוק"
            if daily
            else "ל-XGBoost ול-GLM ללא דעיכה נמצא בין הטובים",
        )
    return result


def correlation_scatter(b: dict[str, Any]) -> Any:
    p = b["data"].X[["prev_ppm_diff", "prev_gd_diff"]].dropna()
    r = float(p.corr().iloc[0, 1])
    fig = go.Figure(
        go.Scatter(
            x=p.prev_ppm_diff,
            y=p.prev_gd_diff,
            mode="markers",
            marker=dict(color=views.COLORS[0], size=3, opacity=0.2),
        )
    )
    fig.update_xaxes(title="פער נקודות למשחק בעונה הקודמת")
    fig.update_yaxes(title="פער הפרש שערים למשחק בעונה הקודמת")
    return views.finish(
        fig, f"נקודות ושערים קודמים חופפים במידע; r={r:.3f}, n={len(p):,}", height=850
    )


def hda_histogram(b: dict[str, Any]) -> Any:
    p = qa.outcome_diagnostics(b["validation"])
    fig = go.Figure(go.Histogram(x=p.loss, marker_color=views.COLORS[0], nbinsx=30))
    reference = b["validation"].loc[b["validation"].stage_role.eq("EXP-002")]
    loss = float(np.average(reference.log_loss, weights=reference.games))
    for i, (value, name) in enumerate([(0.98, "עצירה"), (1, "דגל"), (loss, "EXP-002")]):
        fig.add_vline(x=value, line_dash="dash", line_color=GRAY)
        fig.add_annotation(
            x=value, y=1 - i * 0.09, yref="paper", text=name, showarrow=False, xanchor="left"
        )
    fig.update_xaxes(title="log loss ממוצע בית/תיקו/חוץ", range=[0.965, float(p.loss.max()) + 0.01])
    fig.update_yaxes(title="מספר תצורות")
    return views.finish(
        fig, f"אין תצורה מתחת לסף הדגל; n={len(p)}, מינימום {p.loss.min():.4f}", height=850
    )


def worker_bars() -> Any:
    p = views.engineering_tables()["bench"]
    fig = go.Figure(
        go.Bar(
            x=p.n_jobs.astype(str),
            y=p.units_per_hour,
            marker_color=[views.COLORS[0] if n == 10 else "#C5C5C5" for n in p.n_jobs],
            text=p.units_per_hour.round(0),
            textposition="outside",
        )
    )
    fig.update_xaxes(title="מספר תהליכים")
    fig.update_yaxes(title="יחידות לשעה במדידת הפיילוט")
    return views.finish(fig, "10 תהליכים היו מהירים מ-15 במדידה הקבועה; n=60 לכל מדידה")


def hyperparameter_effects(b: dict[str, Any]) -> Any:
    v = b["validation"].loc[b["validation"].preset.eq("grid")]
    figures = curves.full_figures(b["training"], v, b["report"])
    fig = figures["hyperparameter_effects"]
    for name in ["yaxis2", "yaxis4", "yaxis6", "yaxis8"]:
        fig.layout[name].title.text = ""
    fig.data[7].visible = False
    fig.layout.xaxis8.visible = False
    fig.layout.yaxis8.visible = False
    fig.add_annotation(
        x=0.78,
        y=0.1,
        xref="paper",
        yref="paper",
        showarrow=False,
        text="דגימת פיצ׳רים קבועה ברשת: 0.8<br>אין כאן שינוי שמאפשר לאמוד השפעה",
    )
    return views.finish(
        fig, "הדפוס תלוי בתכנון A ו-B; פס ±SE בין תצורות, n בריחוף — טיוטה", height=1150
    )


def all_figures(b: dict[str, Any], learning: dict[str, Any]) -> dict[str, Any]:
    return dict(
        training_selector=training_selector(b),
        **slice_maps(b),
        learning_curves_train_vs_validation=learning_curves(b, learning),
        paired_folds=paired_folds(b),
        ladder_progress=family_ladder(b),
        base_family_forest=family_ladder(b, base=True),
        paired_forest=paired_forest(b),
        metric_by_season=metric_by_season(b),
        win_rate_by_season=win_rate_by_season(b),
        models_vs_baselines=models_vs_baselines(b),
        elo_by_scope=elo_by_scope(b),
        open_reviews=review_plot(b),
        post_hoc_forest=post_hoc_forest(b),
        correlation_scatter=correlation_scatter(b),
        hda_histogram=hda_histogram(b),
        worker_bars=worker_bars(),
        hyperparameter_effects=hyperparameter_effects(b),
        **decay_figures(),
    )


def post_hoc_table(b: dict[str, Any], check: str) -> tuple[pd.DataFrame, str]:
    """Compatibility entry point with correctly scoped descriptive evidence."""
    table, introduction, conclusion = post_hoc_sections(b, check)
    return table, introduction + conclusion


def graph_explanation(name: str, fig: Any, *, after: bool) -> str:
    "Explain the encodings, then report numeric evidence from visible traces."
    notes = {
        "training_selector": (
            (
                "ציר x הוא מספר העצים וציר y הוא NLL. כחול: חציון אימון ופס IQR; "
                "נקודה סגולה: ממוצע ולידציה ו-CI95 בסוף בלבד, 15 עונות. הבורר משנה את "
                "אותה תצורה בשני המדדים."
            ),
            (
                "ירידה באימון מתארת התאמה לעבר; נקודת הוולידציה מודדת עונה שלא "
                "נראתה באימון. ההפרש בין הערך האחרון בכחול לנקודה הסגולה הוא פער"
                " האימון והוולידציה של התצורה הפעילה."
            ),
        ),
        "train_vs_validation_heatmap": (
            (
                "x: זהירות, y: עומק. שמאל חציון NLL אימון, ימין ממוצע NLL ולידציה."
                " ירוק נמוך וטוב, אדום גבוה ורע, עם סולם נפרד לכל צד. בריחוף: IQR "
                "לאימון, CI95 לוולידציה, n=15."
            ),
            (
                "מספר קטן באימון אינו מבטיח מספר קטן בוולידציה. משווים את המספרים "
                "בין הצדדים; שני גוונים זהים אינם בהכרח אותו NLL."
            ),
        ),
        "hyperparameter_heatmap": (
            (
                "x: זהירות, y: עומק. הצבע והמספר הם אחוז מתרומת Elo; ירוק חיובי "
                "ואדום שלילי, בסולם נפרד לכל פרוסה לפי הטווח שלה. בבורר כל ששת "
                "צירופי עצים וקצב. בריחוף CI95 על 15 "
                "קיפולים, כשהסרגלים מוחזקים קבועים."
            ),
            (
                "הפרוסה הפעילה מציגה כמה מהשיפור של Elo נשמר בכל עומק וזהירות. ערך"
                " מתחת לאפס פירושו גרוע גם מיתרון בית בלבד; הוא מסומן במינוס, ולא "
                "רק בצבע."
            ),
        ),
        "hyperparameter_effects": (
            (
                "כל חלונית מציגה פרמטר בציר x ו-NLL ולידציה בציר y. כחול שלב A, "
                "סגול שלב B. הפס הוא ±SE בין תצורות, לא בין משחקים; n בריחוף. "
                "ברירת המחדל מחוץ לרשת הוסרה."
            ),
            (
                "אלה ממוצעים בתכנון רשת משולב: שינוי הקו אינו מבודד סיבה. פיזור "
                "בין תצורות אינו CI להשפעת פרמטר בכדורגל."
            ),
        ),
        "learning_curves_train_vs_validation": (
            (
                "x: מספר עצים, y: NLL לא משוקלל ובאותה סקאלה בכל ארבע החלוניות. קו"
                " מלא אימון, מקווקו ולידציה; קו אנכי בתקציב שנבחר. כל חלונית תצורה "
                "אחת, והקווים והפסים הם חציון ו-IQR על 15 עונות. ההצעה מוצגת "
                "ראשונה בקריאה מימין לשמאל, למעלה מימין."
            ),
            (
                "ציר y המשותף מאפשר להשוות את אותו גובה בין כל ארבע התצורות. ירידה "
                "באימון במקביל לעלייה בוולידציה מתארת התאמת יתר. פרק 6.2 בודק גם "
                "עומק 1."
            ),
        ),
        "paired_folds": (
            (
                "x: עונות עם שנים, y: NLL ללא F5 פחות איתה. ירוק חיובי, אדום "
                "שלילי; הצלבה בחמש האחרונות. הקו הוא הממוצע והפס ±SE בין העונות "
                "העמוקות; מספר המשחקים בריחוף."
            ),
            (
                "כל עמודה משווה את אותם משחקים באותה עונה. השיפור אינו מופיע בכל "
                "עונה, ולכן קוראים גם את הפיזור ואת חמש האחרונות; Q60 נשארת פתוחה "
                "מול מספר ה-QA."
            ),
        ),
        "ladder_progress": (
            (
                "y: משפחה שנבדקה בכניסה, x: תרומתה באחוז מתרומת Elo על העונות "
                "העמוקות. פס CI95, n ו-SE בריחוף. פלוס ירוק עבר כלל 8; איקס אדום "
                "לא עבר. F6 נבדקה אחרי F5, ללא כוונון מחדש."
            ),
            (
                "רק F5 עברה. הנקודות הן תרומת המידע עצמה, כך שמשפחות שלא נכנסו "
                "אינן מקבלות את השיפור של כוונון הבסיס. זהו הבדל בין תרומת פיצ׳ר "
                "לבין איכות מודל מכוונן."
            ),
        ),
        "base_family_forest": (
            (
                "y: רכיב שהוצא מהבסיס, x: השיפור לאחר הוצאתו באחוז מתרומת Elo. פס "
                "CI95 על 14 עונות. פלוס ירוק עבר כלל 8, איקס אדום לא עבר; הסימן "
                "המספרי מראה את כיוון הפער."
            ),
            (
                "שיפור חיובי בהוצאה היה מעיד שהרכיב מזיק. פס שחוצה אפס משאיר "
                "אי-ודאות, ואינו הוכחה שאין לפיצ׳ר שום מידע."
            ),
        ),
        "paired_forest": (
            (
                "y: מודל, x: שיפור NLL מול S2 על העונות העמוקות המשותפות. ירוק "
                "ופלוס חיובי, אדום ואיקס שלילי, פס CI95; n בריחוף. GLM משתמש ב-13 "
                "עונות משותפות בגלל עומק האימון שלו."
            ),
            (
                "פסים שחוצים אפס מראים מדוע הבדלים קטנים בדירוג דורשים זהירות. "
                "ההשוואה מזווגת באותה עונה ולא בין שתי קבוצות בלתי תלויות של "
                "משחקים."
            ),
        ),
        "metric_by_season": (
            (
                "x: עונת ולידציה עם שנים, y: NLL; כל צבע הוא מודל והנקודות הן "
                "ציוני עונה. 15 קיפולים; מספר המשחקים בריחוף. אין פס לעונה בודדת; "
                "השונות בין עונות גלויה בקווים."
            ),
            (
                "הקווים מתארים עונות קשות וקלות, ולא רק שיפור בין מודלים. שימוש "
                "באותן עונות בהשוואה המזווגת מסיר חלק מהתנודות המשותפות האלה."
            ),
        ),
        "win_rate_by_season": (
            (
                "x: עונה, y: אחוז שורות שבהן M2 שנבחר (F5, ה-ש31) מדויק יותר מ-S2 מתוך "
                "שורות לא שוות. ירוק מעל 50%, אדום מתחת; הטקסט הוא ההפרש מ-50. הפס"
                " CI95 לממוצע 15 שיעורי העונות, לא לכל עונה בנפרד."
            ),
            (
                "יתרון קרוב ל-50% אינו סותר שיפור NLL: גודל השיפור בכל שורה חשוב. "
                "שורות שתי הקבוצות באותו משחק תלויות, ולכן הפס חושב בין עונות ולא "
                "כאילו כל שורה היא הטלת מטבע עצמאית."
            ),
        ),
        "models_vs_baselines": (
            (
                "y: מודל, x: 100×exp(−NLL), ההסתברות הגאומטרית למה שקרה. פס CI95 "
                "מהמרת ממוצע עונתי, n בריחוף. F5, המודל המוצע, מודגש ביהלום סגול; מיקום ימינה טוב "
                "יותר, ללא קידוד דירוג בצבע."
            ),
            (
                "רווחי הסמך כאן הם על הרמה המוחלטת, והם חופפים בגלל הבדלי הקושי "
                "בין עונות. החפיפה אינה אומרת שאין הבדל בין מודלים; ההשוואה "
                "הנכונה היא המזווגת לפי עונה, בגרף היער."
            ),
        ),
        "elo_by_scope": (
            (
                "x: מודלים, y: אחוז מתרומת Elo. הצבעים הם טווחי העונות: כל 15, "
                "13–14 עמוקות, וחמש אחרונות. הקווים ב-100 וב-102; הטקסט הוא פער מ-100. "
                "פס CI95 בין עונות, עם סרגלים קבועים."
            ),
            (
                "הבחירה בטווח משנה את התמונה. החלק של סרגל Elo מכוון ל-100%; ערכים"
                " מעליו משיגים יותר מהשיפור שלו מעל בית בלבד. Q61 מתעדת את הפער "
                "בין הטווח האחרון לכתוב ב-QA."
            ),
        ),
        "open_reviews": (
            (
                "y: הכרעה ומועמדת, x: תוספת NLL ביחס למובילה העמוקה. כאן חיובי רע "
                "ולכן איקס אדום; שלילי טוב ופלוס ירוק. הפס ±SE, n משותף בריחוף. "
                "התוויות מתארות את השלב במילים; ה-ID נשמר בריחוף."
            ),
            (
                "רוב הפערים קטנים מרוחב אי-הוודאות. הגרף אינו מאשר אף הצעה: כללים "
                "4 ו-6 עדיין מחכים להכרעת עמית."
            ),
        ),
        "post_hoc_forest": (
            (
                "y: בדיקה ותצורה; x: שיפור NLL מזווג. מימין מול S2, משמאל מול F5, המודל המוצע. "
                "חיובי טוב ופלוס ירוק, שלילי רע ואיקס אדום. הפסים CI95 על 14 עונות"
                " עמוקות משותפות; n בריחוף. כוכבית ושם מודגש מציינים את תצורות "
                "עומק 1 ו-CatBoost הרדוד הבולטות."
            ),
            (
                "ההפרדה בין החלוניות מראה אם תצורה משתפרת מעל בסיס S2, וגם מה "
                "נשאר מהיתרון מול S2+F5. ההשוואה לסרגל GLM נשארת בטבלאות, "
                "כדי לשמור על קריאות הגרף."
            ),
        ),
        "correlation_scatter": (
            (
                "x: פער נקודות למשחק בעונה הקודמת, y: פער הפרש שערים. כל נקודה "
                "שורת קבוצה-משחק מלאה. כחול שקוף מציג צפיפות; אין כאן אימון או פס "
                "CI של מודל."
            ),
            (
                "המתאם הגבוה מצביע על מידע דומה. המסנן השאיר הפרש שערים, מדד עם "
                "יותר פירוט מהתוצאה הקטגורית ניצחון/תיקו/הפסד. המידע נלקח רק "
                "מהעבר."
            ),
        ),
        "hda_histogram": (
            (
                "x: log loss ממוצע של בית/תיקו/חוץ, y: מספר תצורות. כחול מציג את "
                "התפלגות 283 התצורות; קווים ב-0.98 לעצירה, 1.00 לדגל ובסרגל "
                "EXP-002. זהו מפקד התצורות ולא אומדן עם CI."
            ),
            (
                "אין תצורה מתחת לסף הדגל בניסוי הקפוא. הסף הוא בדיקת דליפה, ולא "
                "יעד שיפור; ציון טוב מאוד מחייב קודם לבדוק שלא נעשה שימוש בתוצאה."
            ),
        ),
        "worker_bars": (
            (
                "x: מספר תהליכים, y: יחידות אימון לשעה. 10 מודגש בכחול והשאר "
                "באפור; כל מדידה על אותו תקציב של 60 יחידות. זו מדידת זמן יחידה "
                "לכל תקציב, ללא חזרות שמאפשרות CI."
            ),
            (
                "מקביליות נוספת אינה מבטיחה קיצור זמן, כי תהליכים מתחרים על משאבי "
                "המחשב. זהו ממצא הנדסי של המחשב והתקציב שנמדדו, ללא משמעות "
                "כדורגלנית."
            ),
        ),
        "season_decay": (
            (
                "x: כמה עונות לפני החיזוי, y: משקל משחק באימון. כל צבע זמן מחצית "
                "אחר; מדרגות משום שהמשקל קבוע לעונה. הקו ב-0.5. אלה פונקציות משקל "
                "מדויקות, ללא n או אי-ודאות מדגמית."
            ),
            (
                "אחרי זמן מחצית אחד המשקל הוא 0.5. למשל, זמן מחצית של עונה אחת "
                "נותן לעונה הקודמת חצי ממשקל העונה האחרונה."
            ),
        ),
        "daily_decay": (
            (
                "x: כמה ימים לפני יום החיזוי, y: משקל משחק. כל צבע זמן מחצית"
                " אחר, הקו ב-0.5. Maher מעדכן מדי יום ולכן הקווים רציפים. אלה "
                "פונקציות משקל, ללא אי-ודאות מדגמית."
            ),
            (
                "זמן מחצית קצר מעדיף תוצאות קרובות. הדעיכה משנה את חשיבות העבר, "
                "ואינה מתירה שימוש במשחק שטרם התקיים."
            ),
        ),
    }
    before, conclusion = notes[name]
    if not after:
        return '<div dir="rtl"><p>' + before + "</p></div>"
    if name == "training_selector":
        active = [trace for trace in fig.data if trace.visible is True]
        train_end = float(_array(active[2].y)[-1])
        validation = float(_array(active[3].y)[-1])
        conclusion += (
            f" בתצורה הפעילה: חציון האימון בסיום {train_end:.6f}, "
            f"ולידציה {validation:.6f}, פער {validation - train_end:.6f}."
        )
    elif name == "paired_folds":
        values = _array(fig.data[0].y)
        mean = float(np.average(values, weights=_array(fig.data[0].customdata)))
        se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        conclusion += f" d̄ מחושב {mean:.9f}, SE {se:.9f}, n={len(values)}."
    elif name == "ladder_progress":
        trace = next(trace for trace in fig.data if trace.y[0] == "F5")
        value = float(_array(trace.x)[0])
        conclusion += f" תרומת הכניסה של F5: {value:+.3f}% מתרומת Elo."
    elif name == "hda_histogram":
        values = _array(fig.data[0].x)
        conclusion += f" מינימום {values.min():.9f}, מתחת ל-1: {int((values < 1).sum())}."
    elif name == "win_rate_by_season":
        values = _array(fig.data[0].y)
        conclusion += (
            f" ממוצע שיעורי העונות {values.mean():.3f}%; "
            f"מעל 50% ב-{int((values > 50).sum())} מתוך {len(values)} עונות."
        )
    elif name == "elo_by_scope":
        value = float(_array(fig.data[2].y)[2])
        conclusion += f" בנתון החתום: M2 שנבחר (F5, ה-ש31) בחמש האחרונות {value:.6f}% מתרומת Elo."
    field = (
        "x"
        if name
        in {
            "ladder_progress",
            "base_family_forest",
            "paired_forest",
            "models_vs_baselines",
            "open_reviews",
            "post_hoc_forest",
        }
        else "z"
        if "heatmap" in name
        else "y"
    )
    plot_values: list[float] = []
    for trace in fig.data:
        if trace.visible is False or getattr(trace, field, None) is None:
            continue
        try:
            plot_values.extend(_array(getattr(trace, field)).reshape(-1).tolist())
        except (ValueError, TypeError):
            continue
    finite = [v for v in plot_values if np.isfinite(v)]
    numeric = (
        f" טווח הערכים המוצגים בברירת המחדל: {min(finite):.6f} עד {max(finite):.6f}."
        if len(finite)
        else ""
    )
    return '<div dir="rtl"><p>' + conclusion + numeric + "</p></div>"


"""Descriptive notebook evidence; all calculations read the verified bundle."""


def _scope_row(b: dict[str, Any], name: str, scope: str) -> dict[str, Any]:
    return next(
        row for row in b["report"]["readable_metrics"][name]["summary"] if row["scope"] == scope
    )


def _selected_folds(b: dict[str, Any], name: str) -> Any:
    report, v = b["report"], b["validation"]
    if name in report["winners"]:
        return qa.fold_group(v, report["winners"][name], report["stages"][name])
    if name in report["comparators"]:
        c = report["comparators"][name]
        return qa.fold_group(v, c["config"], c["features"])
    return v.loc[v.stage_role.eq(name)].copy()


def plain_explanation(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]*>", "", value)).strip()


def _post_hoc_label(record: dict[str, Any]) -> str:
    p = record["parameters"]
    return (
        f"d{p['depth']} T{p['iterations']} L2={p['l2_leaf_reg']}"
        if record["check"] == "P4"
        else f"d{p['max_depth']} T{p['n_estimators']} lr{p['learning_rate']} "
        f"cw{p['min_child_weight']}"
    )


def scoped_post_hoc_table(
    b: dict[str, Any], check: str, *, references: tuple[str, ...] = ("S2", "EXP-002")
) -> Any:
    """Descriptive comparisons on explicit populations, never used by selection.

    The frozen rule 8 still uses deep folds. For the requested all/last5 display,
    copies of already selected rows declare the reporting population to the pure
    paired-statistics core. Original depth flags and all signed files stay intact.
    """
    records = [r for r in b["post"]["records"] if r["check"] == check]
    baselines = {name: _selected_folds(b, name) for name in references}
    rows = []
    for record in records:
        candidate = pd.DataFrame(record["folds"])
        label = _post_hoc_label(record)
        for summary in record["summary"]:
            scope = summary["scope"]
            for reference, baseline in baselines.items():
                seasons = set(candidate.season_id)
                if scope == "deep":
                    seasons &= set(candidate.loc[candidate.deep_fold, "season_id"])
                    seasons &= set(baseline.loc[baseline.deep_fold, "season_id"])
                elif scope == "last5":
                    seasons &= set(range(21, 26))
                a = candidate.loc[candidate.season_id.isin(seasons)].assign(deep_fold=True)
                r = baseline.loc[baseline.season_id.isin(seasons)].assign(deep_fold=True)
                stats = paired_statistics(a, r, score="poisson_nll")
                rows.append(
                    dict(
                        check=check,
                        configuration=label,
                        scope=scope,
                        nll=summary["poisson_nll"],
                        probability=summary["observed_probability_percent"],
                        elo=summary["elo_share_percent"],
                        hda=summary["log_loss"],
                        reference=reference,
                        reference_nll=float(np.average(r.poisson_nll, weights=r.games)),
                        d=stats.mean_improvement,
                        se=stats.se,
                        lower=stats.ci_lower,
                        upper=stats.ci_upper,
                        positive=int(stats.differences.difference.gt(0).sum()),
                        better=stats.last5_folds_better,
                        n=stats.n,
                    )
                )
    return pd.DataFrame(rows)


def post_hoc_sections(b: dict[str, Any], check: str) -> tuple[Any, str, str]:
    """Question, complete scoped evidence and football interpretation for one P."""
    records = [r for r in b["post"]["records"] if r["check"] == check]
    table = scoped_post_hoc_table(b, check, references=("S2", "EXP-002", "F5"))
    columns = ", ".join(records[0]["features"])
    configurations = "; ".join(table.configuration.drop_duplicates())
    questions = {
        "P1": "האם אפשר לוותר על Elo כאשר מצב הטבלה, הכושר והעונה הקודמת זמינים "
        "יחד? זה אותו בסיס אחרי מסנן המתאם, בצירוף F1-A, F2, F3, F4 ו-F5, "
        "ובלי שני משתני Elo. אין מזהי קבוצה או יריבה ואין F6. הפרמטרים "
        "וזמן המחצית נשארו אלה של S2 החתומה, כדי שהשינוי יהיה במידע הזמין "
        "ולא בכוונון חדש.",
        "P2": "האם הכנסת כל המשפחות יחד עם Elo מוסיפה משהו שלא התגלה בסולם? "
        "רשימת העמודות זהה ל-P1 בתוספת Elo, והפרמטרים הקפואים של S2 נשמרו."
        " משפחות שלא עברו בנפרד עשויות לעבוד יחד, אך הן עלולות גם לספק "
        "מידע חוזר ורעש. לכן ההשוואה היא בדיקה תיאורית של כל המידע יחד, "
        "ולא רשות לעקוף את כללי הכניסה.",
        "P3": "האם אזור פשוט יותר מזה שנבדק ברשת המקורית מתאים יותר לחיזוי עונה "
        "חדשה? על בסיס S2 נבדקו עצי עומק אחד עם מספרי עצים וקצבים שונים, "
        "וגם עומק שניים עם דרישה מחמירה יותר למספר דוגמאות בעלה. כל שאר "
        "הפרמטרים וזמן המחצית נגזרו מ-S2 החתומה. עץ רדוד יכול לתקן הבדל "
        "רחב בכוח בלי ללמוד צירוף מקרי של תנאים.",
        "P4": "האם CatBoost חלש כשלעצמו, או שנבדק רק בתקציב גדול שמאפשר שינון? "
        "נבדקו עצים רדודים, שני תקציבי איטרציות ושני קנסות על העלים, על "
        "אותם פיצ׳רי בסיס. הקצב, הזרע ו-one_hot_max_size נשארו כפי שנרשמו "
        "מראש לאבחון. זו השוואה של תקציב ואילוץ למידה, בלי להוסיף מזהי "
        "קבוצות ובלי לשנות את השופט.",
    }
    intro = (
        f'<div dir="rtl"><h3>{check}</h3><p>{questions[check]} '
        f"מספר התצורות שנשמרו: {len(records)}. התצורות הן "
        f"<code>{html.escape(configurations)}</code>. "
        f"רשימת הפיצ׳רים בפועל אחרי המסנן: <code>{html.escape(columns)}</code>. "
        "הטבלאות מופרדות לפי אוכלוסיית ההשוואה: כל הקיפולים, העמוקים וחמש העונות האחרונות. "
        "בכל טבלה NLL המועמד והייחוס, שיפור חיובי לטובת המועמד, SE ו-CI95,"
        " וכן מספר העונות החיוביות בטווח ובחמש האחרונות. "
        "CI95 אינו IQR: הוא מתאר אי-ודאות בממוצע ההפרש בין עונות, ולא את "
        "החצי האמצעי של ציוני האימון.</p></div>"
    )
    conclusion = _post_hoc_interpretation(table, check)
    return table, intro, conclusion


def _post_hoc_interpretation(table: Any, check: str) -> str:
    deep = table.loc[table.scope.eq("deep") & table.reference.eq("S2")]
    if check == "P1":
        r = deep.iloc[0]
        text = (
            "כל המשפחות בלי Elo מחזירות חלק גדול ממידע החוזק, אך אינן תחליף מלא לו. "
            f"בעמוקים מול S2 השיפור הוא {r.d:+.6f}, עם SE של {r.se:.6f} "
            f"ו-CI95=[{r.lower:+.6f}, {r.upper:+.6f}], כולו לרעת המודל בלי Elo. "
            "הפער נשאר גם כשהטבלה, הכושר והעבר נגישים יחד ולא רק כל משפחה בנפרד. "
            "בכדורגל, סיכום רצף המפגשים ואיכות היריבות ב-Elo עדיין מכיל מידע "
            "שהמדדים האחרים אינם משחזרים באותה יעילות."
        )
    elif check == "P2":
        r = deep.iloc[0]
        text = (
            "הכנסת כל המשפחות עם Elo כמעט אינה מוסיפה דיוק לבסיס S2. "
            f"בעמוקים השיפור הוא {r.d:+.6f}, לעומת SE של {r.se:.6f}; "
            f"CI95=[{r.lower:+.6f}, {r.upper:+.6f}] כולל אפס. "
            "התוספת קטנה מאוד ביחס לאי-הוודאות, ולכן אין כאן רווח מבוסס מכל המידע יחד. "
            "בכדורגל, חלק ממדדי הטבלה, העבר והכושר מתארים איכות קבוצה ש-Elo "
            "והבסיס כבר מסכמים, והעמסתם יכולה להוסיף גם מידע חוזר ורעש."
        )
    elif check == "P3":
        good = deep.loc[deep.configuration.str.startswith("d1 T400 lr0.05 ")].iloc[0]
        weak = deep.loc[deep.configuration.str.startswith("d1 T150 lr0.05 ")].iloc[0]
        text = (
            "עומק 1 מצליח כשהתקציב מספיק: XGBoost עם 400 עצים וקצב 0.05 "
            f"נותן מול S2 בעמוקים שיפור של {good.d:+.6f}, עם SE של {good.se:.6f} "
            f"וחיובי ב-{int(good.better)} מתוך חמש האחרונות. "
            "התצורה עומדת במספרים של כלל 8 גרסה 2 מול S2, כולל ממוצע חיובי "
            "בחמש האחרונות, אילו הייתה ברשת מראש. "
            f"לעומתה, עומק 1 עם 150 עצים וקצב 0.05 גרוע בבירור: {weak.d:+.6f}. "
            "בכדורגל, תיקוני כוח פשוטים יכולים להיות יציבים, אך בתקציב קצר מדי "
            "המודל אינו מספיק ללמוד את ההבדלים בין מצבי המשחק."
        )
    else:
        short = deep.loc[deep.configuration.str.startswith("d2 T150 ")]
        long = deep.loc[deep.configuration.str.startswith("d2 T400 ")]
        r3 = short.loc[short.configuration.str.endswith("L2=3.0")].iloc[0]
        r10 = short.loc[short.configuration.str.endswith("L2=10.0")].iloc[0]
        text = (
            "CatBoost רדוד אינו חלש כשלעצמו: בעומק 2 ו-150 איטרציות שני הקנסות "
            f"נותנים מול S2 בעמוקים שיפור של {short.d.min():+.5f} עד {short.d.max():+.5f}, "
            f"עם SE של {short.se.min():.5f}–{short.se.max():.5f} "
            "ו-CI95 שאינו כולל אפס. "
            f"בקנס 3 השיפור חיובי ב-{int(r3.better)} מתוך חמש האחרונות ובקנס 10 "
            f"ב-{int(r10.better)} מתוכן; שתיהן עומדות במספרים של כלל 8 גרסה 2 "
            "מול S2, אילו היו ברשת מראש. "
            "שתי התצורות כמעט זהות בתוצאותיהן ולכן אינן שתי ראיות נפרדות. "
            f"ב-400 איטרציות השיפור כבר שלילי בשני הקנסות "
            f"({long.d.min():+.6f} עד {long.d.max():+.6f}), עקבי עם שינון. "
            "בכדורגל, תקציב קצר מאפשר ללמוד קשרי כוח רחבים בלי להתאים כל פרט לעבר."
        )
    return '<div dir="rtl"><p>' + text + "</p></div>"


def post_hoc_chapter_summary(b: dict[str, Any]) -> str:
    """Four direct answers for the human model decision, without selecting."""
    p1 = next(r for r in b["post"]["records"] if r["check"] == "P1")
    p1_all = next(s for s in p1["summary"] if s["scope"] == "all")
    base = b["validation"].loc[b["validation"].stage_role.eq("base-without-elo")]
    base_loss = float(np.average(base.poisson_nll, weights=base.games))
    r0 = _scope_row(b, "R0", "all")["poisson_nll"]
    base_share = 100 * (r0 - base_loss) / _elo_denominator(b["report"], "all")
    return (
        '<div dir="rtl"><h3>ארבע תשובות לקראת הכרעת המודל ל-M3</h3>'
        f"<p>(1) בלי Elo: כל המשפחות מחזירות {p1_all['elo_share_percent']:.1f}% "
        f"מתרומת Elo בכל הקיפולים, לעומת {base_share:.1f}% בבסיס בלי Elo; "
        "עדיין גרועות מ-S2 בעמוקים.</p>"
        "<p>(2) כל המשפחות עם Elo: כמעט זהה ל-S2; אין רווח מבוסס.</p>"
        "<p>(3) רדוד מנצח עמוק בחלק מהתצורות: עומק 1 ו-CatBoost רדוד "
        "עומדים במספרים של כלל 8 מול S2. CatBoost רדוד משתווה למודל המוצע "
        "בעמוקים; המודל המוצע עדיף בחמש האחרונות.</p>"
        "<p>(4) מה זה אומר: עמית הכריע: ל-M3 עוברים F5, GLM בבסיס ו-Maher (ה-ש33). "
        "הבדיקה בדיעבד אינה בוחרת. </p></div>"
    )


def elo_example(b: dict[str, Any]) -> str:
    r0 = _scope_row(b, "R0", "last5")["poisson_nll"]
    r1 = _scope_row(b, "R1", "last5")["poisson_nll"]
    final = _scope_row(b, "F5", "last5")
    share = 100 * (r0 - final["poisson_nll"]) / (r0 - r1)
    return (
        "אחוז מתרומת Elo אינו אחוז ניחושים נכונים, והוא יכול להיות שלילי או לעלות על מאה: "
        f"בחמש האחרונות, "
        f"100×({r0:.6f}−{final['poisson_nll']:.6f})/({r0:.6f}−{r1:.6f})={share:.1f}%. "
        f"באותו טווח סרגל GLM של EXP-002 מקבל "
        f"{_scope_row(b, 'EXP-002', 'last5')['elo_share_percent']:.1f}%. "
        "המשמעות היא כמה מההפחתה בהפתעת החיזוי שנותן Elo מעל יתרון בית "
        "בלבד השיג המודל, ולא כמה משחקים הוא ניצח בניחוש."
    )


def _specific_graph_numbers(name: str, b: dict[str, Any], fig: Any) -> str:
    if name == "chain":
        c = b["data"].context
        return (
            f"השרשרת נשענת כאן על {c.game_id.nunique():,} משחקי עבר, {len(c):,} "
            f"שורות קבוצה-משחק ו-{c.season_id.nunique()} עונות פיתוח."
        )
    if name == "season_games":
        p = b["data"].context.groupby("season_id").game_id.nunique()
        return (
            f"מוצגות {len(p)} עונות; בעונה הקצרה {int(p.min())} משחקים ובעונה "
            f"הארוכה {int(p.max())}. כל משחק מספק שתי שורות יעד, ולכן מספר שורות "
            f"האימון כפול ממספר המשחקים."
        )
    if name == "walk_forward":
        c = b["data"].context
        return (
            f"בקיפול הראשון {int(c.season_id.lt(11).sum()):,} שורות אימון, "
            f"ובאחרון {int(c.season_id.lt(25).sum()):,}; מספר עונות הוולידציה "
            f"הוא {c.loc[c.season_id.ge(11), 'season_id'].nunique()}. הגידול הוא "
            f"בהיסטוריה הזמינה, ולא בהיתר להשתמש בעונה שנבדקת."
        )
    if name == "poisson_example":
        from scipy.stats import poisson

        _, example = views.poisson_example(b["data"])
        rate = float(example.lambda_example.iloc[0])
        return (
            f"בקצב ההמחשה λ={rate:g}, הסתברות שער אחד היא "
            f"{poisson.pmf(1, rate):.6f}, וה-NLL שלו "
            f"{float(-poisson.logpmf(1, rate)):.6f}; לאפס שערים "
            f"NLL={float(-poisson.logpmf(0, rate)):.6f}. זו המחשת Poisson ליד "
            f"שורה אמיתית, ולא תחזית שנבחרה לניסוי."
        )
    if name in ["season_decay", "daily_decay"]:
        trace = fig.data[1 if name == "season_decay" else 2]
        ages, weights = _array(trace.x), _array(trace.y)
        if name == "daily_decay":
            ages = ages / 365.25
        values = [
            float(weights[int(np.argmin(np.abs(ages - age), axis=None))]) for age in [1, 2, 3]
        ]
        return (
            f"בזמן מחצית של עונה אחת, משקלי המשחקים בגילים של עונה, שתי עונות "
            f"ושלוש עונות הם בהתאמה {values[0]:.3f}, {values[1]:.3f}, "
            f"{values[2]:.3f}. המספרים מחושבים מפונקציית המשקל, ואינם ציון של "
            f"איכות חיזוי."
        )
    if name == "training_selector":
        traces = [trace for trace in fig.data if trace.visible is True]
        train, val = float(_array(traces[2].y)[-1]), float(_array(traces[3].y)[-1])
        q1, q3 = float(_array(traces[0].y)[-1]), float(_array(traces[1].y)[-1])
        radius = float(_array(traces[3].error_y.array)[-1])
        return (
            f"בברירת המחדל חציון NLL האימון בסיום הוא {train:.6f}, ו-NLL "
            f"הוולידציה {val:.6f}, פער {val - train:.6f}. ה-IQR בסיום הוא "
            f"[{q1:.6f}, {q3:.6f}]: זה הפיזור המרכזי בין העונות, ואינו פס אמון "
            f"בממוצע. CI95 לממוצע הוולידציה הוא "
            f"[{val - radius:.6f}, {val + radius:.6f}]."
        )
    if name == "train_vs_validation_heatmap":
        traces = [trace for trace in fig.data if trace.visible is True]
        a, v = _array(traces[0].z), _array(traces[1].z)
        return (
            f"בפרוסת ברירת המחדל חציון loss האימון נע בין {a.min():.6f} "
            f"ל-{a.max():.6f}, ואילו NLL הוולידציה נע בין {v.min():.6f} "
            f"ל-{v.max():.6f}. ההשוואה היא בין אותם שילובים וקיפולים; אין לפרש "
            f"שני גוונים שווים כשני ציונים שווים, כי סולמות הצבע נפרדים."
        )
    if name == "hyperparameter_heatmap":
        p = budget_table(b)
        q = p.loc[
            p.depth.eq(p.depth.min()) & p.trees.eq(p.trees.min()) & p.rate.eq(p.rate.min())
        ].iloc[0]
        aggressive = p.loc[
            p.depth.eq(p.depth.max()) & p.trees.eq(p.trees.max()) & p.rate.eq(p.rate.max())
        ].iloc[0]
        return (
            f"בזהירות הגבוהה, עומק {q.depth:g} עם {q.trees:g} עצים וקצב "
            f"{q.rate:g} מחזיר {q.elo:.2f}% מתרומת Elo. בעומק "
            f"{aggressive.depth:g}, {aggressive.trees:g} עצים וקצב "
            f"{aggressive.rate:g}, התרומה היא {aggressive.elo:+.2f}%. הערך "
            f"השלילי אומר גרוע מיתרון בית בלבד."
        )
    if name == "hyperparameter_effects":
        y = _array(fig.data[0].y)
        se = float(_array(fig.data[0].error_y.array)[0])
        return (
            f"בממוצע תצורות שלב A, NLL בעומק הקטן הוא {y[0]:.6f} ובעומק הגדול "
            f"{y[-1]:.6f}; ההפרש {y[-1] - y[0]:.6f}, ו-SE בעומק הקטן {se:.6f}. "
            f"דגימת הפיצ׳רים קבועה בכל "
            f"הרשת, ולכן החלונית שלה מסומנת ככזו שאין בה שינוי לאמוד. זו השפעה "
            f"בתכנון רשת משולב, ולא ניסוי סיבתי מבודד."
        )
    if name == "learning_curves_train_vs_validation":
        train, val = float(_array(fig.data[-4].y)[-1]), float(_array(fig.data[-1].y)[-1])
        trees = int(_array(fig.data[-1].x)[-1])
        return (
            f"בחלונית התקציב הארוך, אחרי {trees} עצים, חציון האימון {train:.6f} "
            f"וחציון הוולידציה {val:.6f}, פער {val - train:.6f}. כל ארבע "
            f"החלוניות משתמשות באותו טווח y, כך שגובה זהה הוא אותו NLL גם "
            f"כשהתקציב בציר x שונה."
        )
    if name == "paired_folds":
        d = next(d for d in b["report"]["decisions"] if d["name"] == "F5" and d["kind"] == "entry")[
            "rule8"
        ]["stats"]
        return (
            f"בבדיקת כניסת F5, d̄={d['mean_improvement']:.6f} מול "
            f"SE={d['se']:.6f}; בחמש האחרונות השיפור הממוצע "
            f"{d['last5_improvement']:.6f}, וחיובי ב-{d['last5_folds_better']} "
            f"מתוך {d['last5_folds']}. לכן תנאי כלל 8 שנבדקו מתקיימים; אין כאן "
            f"שינוי או חישוב מחדש של ההכרעה."
        )
    if name in ["ladder_progress", "base_family_forest"]:
        p, _ = qa.entry_tables(b["report"])
        wanted = (
            ["base-without-elo", "base-without-home", "base-without-value"]
            if name == "base_family_forest"
            else ["F3", "F5", "F6"]
        )
        scale = 100 / _elo_denominator(b["report"])
        return (
            "תרומות הכניסה, ללא כוונון מחדש: "
            + "; ".join(
                f"{n}: {float(p.loc[p.name.eq(n), 'd'].iloc[0]) * scale:+.3f}% מתרומת Elo"
                for n in wanted
            )
            + ". הנקודה היא ממוצע הפרש על אותן עונות, הפס CI95 וקו האפס מייצג "
            "היעדר שינוי; צבע הסימון קובע אם כלל הכניסה עבר, ולא רק אם הממוצע "
            "חיובי."
        )
    if name == "paired_forest":
        m = b["report"]["readable_metrics"]
        return (
            "השיפור מול S2 הוא "
            + "; ".join(
                f"{n}: {m[n]['paired']['mean_improvement']:+.6f}, "
                f"SE={m[n]['paired']['se']:.6f}, n={m[n]['paired']['n']}"
                for n in ["EXP-002", "maher", "catboost-selected"]
            )
            + ". הפסים הם השוואה מזווגת של אותם משחקים בכל עונה, ולכן אינם פסים "
            "של שני ממוצעים בלתי תלויים."
        )
    if name == "metric_by_season":
        a, r = (
            _selected_folds(b, "S2").sort_values("season_id"),
            _selected_folds(b, "EXP-002").sort_values("season_id"),
        )
        c = _selected_folds(b, "catboost-selected").sort_values("season_id")
        return (
            f"בעונת הוולידציה הראשונה S2 מקבלת NLL "
            f"{float(a.poisson_nll.iloc[0]):.6f}, הסרגל "
            f"{float(r.poisson_nll.iloc[0]):.6f} ו-CatBoost "
            f"{float(c.poisson_nll.iloc[0]):.6f}. העונה הראשונה נשענת על פחות "
            f"עבר, ולכן חשוב להפריד אותה מההשוואה בעונות עמוקות ולא להסיק "
            f"מהעקומה על כיוון קבוע בכל תקופה."
        )
    if name == "win_rate_by_season":
        y = _array(fig.data[0].y)
        return (
            f"ממוצע שיעורי העונות הוא {float(np.mean(y)):.3f}%; הטווח בין "
            f"{float(y.min()):.3f}% ל-{float(y.max()):.3f}%, ומעל חמישים אחוז "
            f"ב-{int((y > 50).sum())} מתוך {len(y)} עונות. זו עדיפות בשורת "
            f"שערים, לא הסתברות לניצחון של קבוצה במשחק ולא אחוז הצלחה בטבלת "
            f"הליגה."
        )
    if name == "models_vs_baselines":
        probabilities = {trace.name: float(_array(trace.x)[0]) for trace in fig.data}
        return (
            f"ההסתברות הגאומטרית בעמוקות היא {probabilities['F5']:.3f}% למודל המוצע F5, "
            f"{probabilities['EXP-002']:.3f}% לסרגל "
            f"ו-{probabilities['catboost-selected']:.3f}% ל-CatBoost. חפיפה בפסים מתארת "
            f"את שונות ציוני העונות; ההבדל הקטן בין שני המובילים מחייב גם את "
            f"ההשוואה המזווגת, ולא הכרעה מהנקודות לבדן."
        )
    if name == "elo_by_scope":
        return (
            elo_example(b) + f" S2 בחמש האחרונות נותנת "
            f"{_scope_row(b, 'S2', 'last5')['elo_share_percent']:.1f}%, כך שטווח "
            f"העונות משנה את גודל התוספת המוצגת."
        )
    if name == "open_reviews":
        p = qa.review_evidence(b["report"], b["validation"])
        maher = p.loc[p.stage.eq("maher")].iloc[0]
        s2 = p.loc[p.stage.eq("S2-AB")].iloc[1]
        return (
            f"מוצגות {len(p)} מועמדות ב-{p.stage.nunique()} הכרעות. בהצעה הזמנית "
            f"של S2 תוספת NLL {s2.gap:.6f} מול SE {s2.se:.6f}; ב-Maher המובילה "
            f"האחרונה גרועה מהעמוקה ב-{maher.gap:.6f}, עם SE {maher.se:.6f}. לכן "
            f"הגרף ממחיש מה דורש דיון, ואינו מקבל אוטומטית את אחת ההצעות."
        )
    if name == "post_hoc_forest":
        p1 = next(r for r in b["post"]["records"] if r["check"] == "P1")["paired"]["S2"]
        p2 = next(r for r in b["post"]["records"] if r["check"] == "P2")["paired"]["S2"]
        return (
            f"בעמוקות, השיפור של כל הפיצ׳רים בלי Elo מול S2 הוא "
            f"{p1['mean_improvement']:+.6f}, CI95=[{p1['ci_lower']:+.6f}, "
            f"{p1['ci_upper']:+.6f}]. עם Elo השיפור "
            f"{p2['mean_improvement']:+.6f}, CI95=[{p2['ci_lower']:+.6f}, "
            f"{p2['ci_upper']:+.6f}]. אלה נקודות מוצא לקריאת P1–P4, ולא הכרעה "
            f"בין המודלים."
        )
    if name == "correlation_scatter":
        p = b["data"].X[["prev_ppm_diff", "prev_gd_diff"]].dropna()
        return (
            f"Pearson r={float(p.corr().iloc[0, 1]):.6f} על {len(p):,} שורות "
            f"מלאות, מתוך {len(b['data'].X):,}; ל-{len(b['data'].X) - len(p):,} "
            f"שורות חסרה היסטוריה מלאה. r נע בין מינוס אחת לפלוס אחת, וקרבה "
            f"לפלוס אחת פירושה ששני המשתנים נוטים לעלות יחד, לא שקשר אחד גורם "
            f"לאחר."
        )
    if name == "hda_histogram":
        p = qa.outcome_diagnostics(b["validation"])
        return (
            f"נבדקו {len(p)} תצורות מקוריות; המינימום {float(p.loss.min()):.6f}, "
            f"ומספר הממוצעים מתחת לסף הדגל הוא {int(p.loss.lt(1).sum())}. קווי "
            f"העצירה והדגל אינם יעד לשיפור. בקרת ההצצה ההיסטורית מ-EXP-001, "
            f"שמתועדת ב-EXPERIMENTS ובה-ת18, נפרדת מהתוצאות החדשות המוצגות כאן."
        )
    if name == "worker_bars":
        p = views.engineering_tables()["bench"].set_index("n_jobs")
        rate10, rate15 = float(p.loc[10, "units_per_hour"]), float(p.loc[15, "units_per_hour"])
        return (
            f"נמדדו {float(p.loc[6, 'units_per_hour']):.1f} יחידות בשעה בשישה "
            f"תהליכים, {rate10:.1f} בעשרה ו-{rate15:.1f} בחמישה עשר. בעשרה "
            f"תהליכים התפוקה גבוהה ב-{100 * (rate10 / rate15 - 1):.2f}% מהמדידה "
            f"של חמישה עשר. זו השוואת מחשב בתנאי המדידה, ולא ציון תחזית; אין "
            f"חזרות שמאפשרות לאמוד כאן CI95 של זמן הריצה."
        )
    raise KeyError(name)


def explain_graph(name: str, b: dict[str, Any], fig: Any, *, after: bool) -> str:
    """A graph-specific reading guide and a computed football conclusion."""
    early = {
        "chain": (
            "התרשים עונה על השאלה מה נכנס לשכבת השערים ומה יוצא ממנה. קוראים "
            "את התיבות מימין לשמאל: מידע מוקדם, מוטיבציה, שני ממוצעי שערים, "
            "הסתברויות לתוצאות וסימולציה של טבלה. הכחול מדגיש את M2, שבו עוסקת"
            " המחברת; החצים הם סדר עיבוד, לא קשר סטטיסטי שנאמד. אין ציר מספרי,"
            " אין פס אי-ודאות ואין בחירת מודל בתוך התרשים.",
            "כל שלב משתמש רק במה שזמין בבוקר המשחק. שני צדדי המשחק מקבלים "
            "תחזיות שערים נפרדות, והמרה לניצחון או לתיקו שייכת לשלב הבא. "
            "התרשים מסביר חוזה בין רכיבים ולא מודד איכות חיזוי; אין להסיק ממנו"
            " שהוספת רכיב משפרת את התוצאה. בכדורגל הוא מבהיר מדוע חיזוי שערים "
            "אינו עדיין תחזית של טבלת עונה.",
        ),
        "season_games": (
            "העמודות עונות על השאלה כמה משחקים זמינים בכל עונת פיתוח. ציר x "
            "הוא עונה עם השנים שלה וציר y הוא מספר המשחקים; כחול מזהה את כמות "
            "הדאטה, ולא טוב או רע במודל. כל משחק תורם שתי שורות קבוצה-משחק, "
            "שאינן שתי תוצאות משחק עצמאיות. זה מפקד של משחקים שנשמרו ולכן אין "
            "פסי אי-ודאות מדגמית: לא נאמדה כאן אוכלוסייה שלא נצפתה.",
            "מבנה הליגה ומספר המשחקים משתנים לאורך הזמן. עונה עם יותר משחקים "
            "מספקת יותר דוגמאות לאימון ולהערכה, אך אינה מבטיחה שהכדורגל בה קל "
            "יותר לחיזוי. ההבחנה בין משחקים לשורות חשובה כאשר מפרשים את גודל "
            "המדגם ואת אי-הוודאות בין עונות; אין להכפיל באופן מלאכותי את הראיות כי "
            "לכל משחק יש שני צדדים.",
        ),
        "walk_forward": (
            "המפה עונה על השאלה מתי כל משחק זמין למאמן. ציר x הוא עונות "
            "ההיסטוריה, וציר y הוא קיפול ולידציה. התאים והמקרא מבחינים בין "
            "אימון על העבר, עונה שנבחנת ועתיד שאסור למודל לראות. מספר השורות "
            "ליד כל שורה הוא תקציב האימון. זו תוכנית זמן קבועה ולא ציון מודל, "
            "ולכן אין בה פס אי-ודאות; התאים מראים סדר סיבתי, לא אי-ודאות "
            "סטטיסטית.",
            "עם מעבר לעונה הבאה המאמן מקבל יותר עבר, אבל לעולם לא את התוצאות "
            "של אותה עונת מבחן מראש. בכדורגל זה מחקה הכנה לעונה חדשה מתוך מה "
            "שכבר ידוע. קיפול מוקדם עם מעט עונות עלול להתנהג אחרת מהעמוקים; "
            "לכן מפרידים את ההשוואה המלאה מהאוכלוסייה שבה יש מספיק היסטוריה, "
            "בלי לשנות את פיצול העונות בדיעבד.",
        ),
        "poisson_example": (
            "הגרף עונה על השאלה איך ממוצע שערים הופך להתפלגות של תוצאות "
            "אפשריות. ציר x הוא מספר שערים, וציר y ההסתברות שקיבל מספר זה; "
            "הכחול מציג הסתברות ולא שיפור מול מודל אחר. מוצג קצב המחשה קבוע "
            "ליד משחק אמיתי, לא תחזית שנבחרה ל-M3. אין פס אי-ודאות משום שלא נאמד "
            "כאן פרמטר מהדוגמה: ההתפלגות מחושבת בדיוק עבור הקצב הנתון.",
            "תוצאה אפשרית יכולה לקבל הסתברות נמוכה גם אם הממוצע סביר, ולכן "
            "מודדים את ההפתעה במה שקרה ולא רק מרחק מהממוצע. NLL הוא מינוס "
            "הלוגריתם של ההסתברות שניתנה לתוצאה שנצפתה; ערך קטן טוב יותר, אך "
            "אין לו יחידות של שערים. בכדורגל, תחזית ממוצע טובה אינה הבטחה "
            "לתוצאה מסוימת ואינה עדיין הסתברות לניצחון.",
        ),
    }
    if name in early:
        base = early[name][int(after)]
    else:
        base = plain_explanation(graph_explanation(name, fig, after=after))
        if after:
            base = base.split(" טווח הערכים המוצגים בברירת המחדל:", 1)[0]
        base = base.replace("Q60 נשארת פתוחה מול מספר ה-QA.", "המספר החתום אושר בבדיקת Claude.")
        base = base.replace(
            "Q61 מתעדת את הפער בין הטווח האחרון לכתוב ב-QA.", "המספר החתום אושר בבדיקת Claude."
        )
    details = {
        "training_selector": "הבורר מחליף עומק, מספר עצים, קצב וזהירות יחד, וברירת המחדל היא "
        'פרמטרי הצעת S2 הקפואה. <a href="#g-iqr-ci">IQR ו-CI95</a> מודדים '
        "דברים שונים: הראשון מכיל את החצי האמצעי של ציוני האימון, והשני את"
        " אי-הוודאות בממוצע עונות הוולידציה. NLL יכול לנוע מאפס כלפי מעלה;"
        ' נמוך יותר פירושו פחות הפתעה. זו <a href="#g-training-gap">עקומת '
        "אימון מול ולידציה</a>, ולא ולידציה בכל איטרציה.",
        "train_vs_validation_heatmap": "הבורר מחליף את מספר העצים והקצב, ופותח בפרוסה"
        " של הצעת S2 החתומה. "
        "בכל תא אותם קיפולים בשני הצדדים; ריחוף מציג IQR לאימון ו-CI95 "
        "לוולידציה. מאחר שהסקאלות נפרדות, הבדל צבע לבדו אינו הפרש NLL. "
        "השאלה היא האם תוספת יכולת להתאים את משחקי העבר נשמרת גם במשחקי "
        "העונה הבאה.",
        "hyperparameter_heatmap": "הבורר מחליף את התקציב ואינו בוחר מודל חדש; הפרוסה הקפואה מסומנת "
        'proposal. <a href="#g-elo-share">אחוז מתרומת Elo</a> נמדד מול '
        "אותם שני סרגלים, ובו ירוק גבוה טוב ואדום שלילי רע. בריחוף מופיע "
        "CI95 בין עונות כשהסרגלים מוחזקים קבועים. " + elo_example(b),
        "hyperparameter_effects": "כל נקודה מסכמת תצורות רשת ששיתפו ערך של "
        "פרמטר, ולא משנה פרמטר אחד"
        " במודל זהה. הפסים הם SE בין תצורות, ומספרן מופיע בריחוף; אין "
        "לפרשם כ-CI95 על משחקים או כ-IQR עונתי. דגימת הפיצ׳רים נשארה קבועה"
        " ולכן לא מצויר עבורה אפקט. ברירת המחדל שמחוץ לרשת אינה נכללת "
        'בגרף. <a href="#g-paired">SE, שגיאת תקן</a>, הוא סטיית התקן בין '
        "התצורות חלקי שורש מספרן: אי-ודאות בממוצע באותן יחידות NLL, מאפס "
        "ומעלה. ערך לדוגמה מחושב אחרי הגרף; זה אינו ה-SE העונתי המזווג.",
        "learning_curves_train_vs_validation": "כל חלונית היא תצורת היפר-פרמטרים אחרת, ובכולן"
        " אותו טווח y הכולל "
        "את כל הקווים ופסי ה-IQR. ההשוואה מראה אם תקציב נוסף מוריד רק את "
        "ההפתעה באימון או גם בעונה החדשה. אין כאן early stopping ואין "
        "שינוי במספר העצים שנבחר בסולם; זהו אבחון מאוחר ומופרד של הבחירה "
        "הזמנית.",
        "paired_folds": '<a href="#g-paired">הפרש מזווג</a> מחסיר את ציוני שני המודלים על '
        "אותה עונה, כך שעונה קלה אינה נזקפת בטעות לזכות אחד מהם. SE מתאר "
        "את אי-הוודאות בממוצע ההפרשים בין עונות, לא את פיזור השערים במשחק."
        " דוגמת F5 בהמשך מחושבת מקובץ ההכרעה; העמודות האחרונות מסומנות "
        "בהצלבה כדי לבדוק עקביות לאורך זמן.",
        "ladder_progress": 'זה <a href="#g-forest">גרף יער</a>: כל נקודה היא ממוצע, הפס CI95 '
        "וקו האפס הוא היעדר הבדל. ציר x יכול להיות שלילי או חיובי; פלוס "
        "ירוק מציין כניסה לפי כלל 8 ואיקס אדום אי-כניסה. כל משפחה נבדקת "
        "בתצורת הכניסה הקבועה שלה, והפער מנורמל לתרומת Elo. כוונון מחדש של"
        " הבסיס אינו נחשב תרומה של משפחה. ירוק = עברה את כלל 8; אדום = לא עברה. "
        "הצבע והסמל מציינים את הכלל, ואילו סימן המספר מציין את כיוון התרומה.",
        "base_family_forest": "השאלה כאן הפוכה מהוספת משפחה: האם הוצאת רכיב מהבסיס מועילה. נקודה"
        " שלילית אומרת שההוצאה פגעה בחיזוי, ופס שחוצה אפס משאיר אי-ודאות "
        "לגבי הממוצע. כל רכיב מושווה על אותם משחקים בעונות העמוקות, וההפרש"
        " מוצג בסולם אחוז מתרומת Elo. אין בגרף הכרעה חדשה על רשימת "
        "הפיצ׳רים.",
        "paired_forest": "כל שורה משווה מודל שלם ל-S2 על העונות העמוקות המשותפות שלו. מספר "
        "העונות עשוי להשתנות עם עומק האימון של המודל ולכן n מופיע בריחוף. "
        "הפסים הם CI95 של הפרש מזווג, לא IQR של שתי אוכלוסיות נפרדות; "
        "נקודה חיובית ופלוס ירוק פירושם פחות NLL. עדיין צריך לבדוק גם את "
        "העונות האחרונות ואת כללי ההכרעה.",
        "metric_by_season": "הקו מקשר עונות לצורך קריאה בלבד; הוא אינו מניח שהדיוק משתנה באופן"
        " חלק בזמן. מספר המשחקים מופיע בריחוף, והצבעים הם זהות המודל, לא "
        "קידוד טוב ורע. אין פס שגיאה לעונה בודדת בגרף זה; אי-הוודאות "
        "הרלוונטית להשוואה מוצגת בגרפים המזווגים. כך אפשר לראות אם פער "
        "ממוצע נשען על תקופה מסוימת.",
        "win_rate_by_season": "העמודות סופרות שורות יעד שהשתפרו, בלי שורות שבהן הציונים שווים. "
        "המספר על העמודה הוא הפרש מנקודת האמצע, ולא אחוז השיפור ב-NLL. הפס"
        " מתאר את ממוצע שיעורי העונות; הוא אינו נותן CI95 נפרד לכל עמודה. "
        "משחק אחד יוצר שתי שורות תלויות, ולכן אין להתייחס לכל שורה כהטלת "
        "מטבע עצמאית.",
        "models_vs_baselines": "הציר מומר באמצעות exp של מינוס NLL, ולכן הוא תחום בין אפס למאה "
        "אחוז. זה ממוצע גאומטרי של ההסתברות למספר השערים שנצפה, לא "
        "ההסתברות לתוצאה הסבירה ביותר ולא אחוז ניצחונות שחזינו נכון. הכחול"
        " מסמן מיקום של מודל, והפס מתאר CI95 בין עונות. השוואת שני מודלים "
        "קרובים דורשת גם את יער ההפרשים המזווגים.",
        "elo_by_scope": "כל צבע הוא אוכלוסיית עונות אחרת, והמספרים על העמודות הם הפרש מקו "
        "הייחוס של מאה אחוז. הפסים מחושבים בין עונות כאשר סרגלי הנרמול "
        "מוחזקים קבועים; הם אינם פסי שיפור מזווג. ברוב המודלים יש אותה "
        "אוכלוסייה עמוקה, ול-GLM עונה משותפת אחת פחות. אין בגרף רשות לבחור"
        " רק את הטווח שמחמיא למודל.",
        "open_reviews": "הסימן כאן מכוון לרעת המועמדת: תוספת NLL חיובית היא הפסד מול "
        "המובילה העמוקה. לכן איקס אדום מתאים לנקודה חיובית, ופלוס ירוק "
        "לנקודה שאינה מפסידה. כל פס הוא SE אחד, ולא CI95. שתי שורות של "
        "אותה הכרעה אינן שתי בדיקות בלתי תלויות; הן שתי חלופות שמחכות "
        "להכרעת עמית.",
        "post_hoc_forest": "כל תצורה מופיעה מול שני ייחוסים, ולכן אין לספור שתי שורות כראיות "
        "עצמאיות. היער מציג את העונות העמוקות בלבד; הטבלאות והפסקאות בהמשך"
        " מפרידות גם את כל הקיפולים ואת חמש האחרונות. שני הצירים חולקים "
        "את אותו טווח, כך שאותו אורך פס ואותו מרחק מקו האפס נקראים "
        "באותן יחידות בשתי החלוניות. ההשוואה ל-EXP-002 נשמרת בטבלאות.",
        "correlation_scatter": "הפיזור כולל רק שורות עם מידע מלא על שני המשתנים; חוסר עבר של עולה"
        " או קבוצה אינו מוחלף כאן באפס. הצבע השקוף מאפשר לראות צפיפות של "
        "נקודות חופפות ולא מייצג יתרון של אחת הקבוצות. מתאם מודד תנועה "
        "משותפת של משתנים, אך אינו אומר איזה מהם סיבתי או איזה יוסיף NLL "
        "לאחר שכבר הכנסנו Elo.",
        "hda_histogram": "כל תצורה מקורית תורמת ממוצע אחד להיסטוגרמה, ולא כל קיפול עונתי "
        "בנפרד. סף העצירה חל על ממוצע של תצורה; עונה בודדת טובה אינה לבדה "
        "הוכחת דליפה. ההמרה לבית/תיקו/חוץ והשופט זהים לניסוי הקפוא. זהו "
        "מפקד תצורות, ולכן אין פס CI95 על גובה העמודות ואין כאן רשת "
        "פרמטרים חדשה.",
        "worker_bars": "העמודה הכחולה היא מספר התהליכים שנבחר להפעלה, ושתי האפורות הן "
        "מדידות ההשוואה. הגובה הוא קצב עבודה בזמן קיר, לא סכום זמני המעבד "
        "של כל התהליכים. אין כאן פס CI95 כי לא נעשו חזרות עצמאיות של אותו "
        "תנאי. הוספת תהליכים יכולה להוסיף תחרות על מעבד וזיכרון, ולכן יותר"
        " תהליכים אינם בהכרח יותר יחידות לשעה.",
        "season_decay": "הבורר אינו משתנה בגרף זה: כל קו הוא זמן מחצית אחר לפי המקרא. "
        "המדרגות מדגישות שכל המשחקים באותה עונת אימון מקבלים אותו משקל, גם"
        " אם התקיימו בתאריכים שונים. כחול ללא דעיכה שומר את המשקל, ואילו "
        "הקו הקצר נותן לעבר הרחוק השפעה קטנה יותר. לא נלמדת כאן תחזית "
        "שערים ולא נמדדת איכות המודל.",
        "daily_decay": "המשקל של Maher מתעדכן לפי גיל המשחק ביום החיזוי, ולכן הקווים "
        "רציפים. ציר הזמן מוצג בימים, ואילו זמני המחצית במקרא ניתנים בעונות; "
        "עונה מומרת ל-365.25 ימים. כל צבע זמן מחצית אחר לפי המקרא, והקו האופקי מדגים את "
        "נקודת מחצית המשקל. אין פס אי-ודאות כי זו פונקציה קבועה; אין "
        "להשתמש בה כדי להציץ במשחק עתידי.",
    }
    football = {
        "training_selector": "בכדורגל, פער בין התאמה לעבר לחיזוי עונה חדשה יכול להעיד שהמודל "
        "למד צירופים מקומיים של קבוצות ותקופות. הפער לבדו אינו מבחן "
        "מובהקות, ולכן לא בוחרים תצורה מהמרחק בין עקומת האימון לנקודת הוולידציה בלבד.",
        "train_vs_validation_heatmap": "המשמעות הכדורגלנית היא שזיכרון מפורט של "
        "תוצאות עבר יכול לפגוע "
        "בהכללה לקבוצות ולסגלים של העונה הבאה. המפה מסבירה את מחיר הגמישות"
        " ולא את סיבת השינוי בכוח של קבוצה מסוימת.",
        "hyperparameter_heatmap": "המשמעות הכדורגלנית היא שכוח כללי ויציב של הקבוצות מועיל יותר "
        "מצירופים מפורטים מדי של מאפייני משחק. גם מודל מורכב יכול לשנן "
        "רעש, ולכן תרומה שלילית היא אזהרה אמפירית ולא הוכחה ש-Elo הוא כל "
        "הסיפור.",
        "hyperparameter_effects": "אין לפרמטר טכני משמעות כדורגלנית בפני עצמו. הפרשנות בכדורגל "
        "עקיפה: כמה מורכבות נדרשת כדי לתאר כוח ויתרון בית, בלי לשנן תוצאות"
        " מקריות של מפגשים מסוימים.",
        "learning_curves_train_vs_validation": "בכדורגל, למידה נוספת של העבר אינה בהכרח שיפור"
        " כשסגלים ומצבים "
        "משתנים. ההתרחקות בין האימון לוולידציה מצביעה על שינון אפשרי; היא "
        "עוזרת להבין את התקציב ואינה משנה את הבחירה המקורית.",
        "paired_folds": "המשמעות הכדורגלנית האפשרית היא שמצב המטרות של היריבה משפיע על "
        "עוצמת המשחק שלה, אך זהו קשר תיאורי. הגרף מראה מתי אות המוטיבציה "
        "תרם לחיזוי שערים; הוא אינו מוכיח התנהגות סיבתית של שחקנים.",
        "ladder_progress": "בכדורגל, הערך של מידע חדש נמדד במה שנוסף מעבר לכוח ולכושר שכבר "
        "ידועים, ולא בכך שהוא נשמע רלוונטי. כישלון כלל הכניסה משאיר את "
        "המשפחה מחוץ לבסיס הזמני ואינו קובע שאין בה שום מידע בעולם.",
        "base_family_forest": "בכדורגל, פגיעה אחרי הוצאת Elo מצביעה על ערך החוזק המצטבר, ואילו "
        "חפיפה בפסי רכיבים אחרים מתאימה למידע שמשותף לכושר, שווי ומצב "
        "העונה. אין להסיק מהפסים שכל קשר כדורגלני נמדד ללא רעש.",
        "paired_forest": "המשמעות בכדורגל היא שצריך להשוות איכות חיזוי על אותן עונות, "
        "שחולקות סביבה של קבוצות, שערים ויתרון בית. פער קטן ליד פס רחב "
        "אינו סיבה מספקת לבחור ספרייה או מבנה מודל.",
        "metric_by_season": "בכדורגל, עונה חריגה בכמות השערים או באיזון הליגה יכולה להזיז את "
        "כל המודלים יחד. פער משותף כזה אינו תרומה של פיצ׳ר מסוים; הוא "
        "הסיבה להשתמש גם בהשוואה מזווגת לפי עונה.",
        "win_rate_by_season": "בכדורגל, מודל יכול לשפר מעט שורות רבות או לשפר מאוד שורות מעטות. "
        "לכן שיעור קרוב למטבע אינו סותר שיפור בממוצע NLL, והוא אינו מספר "
        "לנו מי תזכה באליפות או תנצח במשחק יחיד.",
        "models_vs_baselines": "בכדורגל, פיזור של מספר שערים מתאר הפתעות אפשריות גם כשהקבוצה "
        "חזקה. המדד מתגמל הסתברות סבירה למה שקרה בפועל; הוא אינו אומר שבכל"
        " משחק רבע מן התחזית נכונה.",
        "elo_by_scope": "בכדורגל, דפוסי יתרון בית, כושר ומוטיבציה משתנים בין תקופות, ולכן "
        "יתרון בטווח האחרון יכול להיות שונה מהתמונה הארוכה. יש לשמור את "
        "שלושת הטווחים גלויים ולא לבחור בדיעבד את המחמיא ביותר.",
        "open_reviews": "אין לגרף הכרעה כדורגלנית עצמאית: הוא מציג את מחיר הדיוק של חלופות"
        " טכניות ותקופתיות. ב-Maher יש סיבה לדיון נפרד, ובשאר המקרים יש "
        "לקרוא את גודל הפער מול הרעש בין עונות.",
        "post_hoc_forest": "בכדורגל, אפשר לייצג איכות קבוצה גם בלי Elo דרך תוצאות, כושר "
        "ומיקום, אך ייתכן שיאבד סיכום יציב של איכות היריבות. היער מניע את "
        "השאלה הזאת; ארבע הפסקאות שאחריו עונות עליה עם כל התצורות "
        "והטווחים.",
        "correlation_scatter": "בכדורגל, נקודות והפרש שערים בעונה הקודמת משקפים איכות דומה. הפרש "
        "השערים שומר פירוט על גודל התוצאה, ולכן נשמר במסנן; זו הכרעת ייצוג"
        " מוקדמת ולא בחירה לפי ציוני המבחן.",
        "hda_histogram": "אין לסף משמעות של יעד כדורגלני: תוצאה טובה מאוד עשויה להיות שיפור"
        " אמיתי או הצצה אסורה. קודם בודקים את סדר הזמנים ואת מקור Elo, ורק"
        " אחר כך מפרשים שיפור בחיזוי.",
        "worker_bars": "אין לתפוקה זו משמעות כדורגלנית. היא מסבירה מדוע הריצה תוכננה "
        "למחשב הנוכחי, ולא מדוע מודל אחד חוזה טוב יותר. המדידה אינה מבטיחה"
        " שאותו מספר תהליכים יהיה מהיר ביותר במחשב אחר או בתקציב אימון "
        "אחר.",
        "season_decay": "בכדורגל, זמן מחצית קצר מעדיף מצבה עדכנית של קבוצה על עונות עם "
        "סגלים אחרים. המחיר הוא פחות ראיות היסטוריות. הגרף מדגים את הפשרה "
        "במשקל, וההשוואה האמפירית בטבלה הסמוכה קובעת מה קרה בניסוי.",
        "daily_decay": "בכדורגל, Maher יכול לעדכן חוזק התקפה והגנה אחרי משחקים שכבר "
        "הסתיימו בעונה הנוכחית. הדעיכה היומית מצמצמת השפעת תוצאות ישנות, "
        "אך אינה מתירה אימון על המשחק שחוזים או על תוצאות שטרם נודעו.",
    }
    if after:
        result = base + " " + _specific_graph_numbers(name, b, fig) + " " + football.get(name, "")
        if name not in early and name not in [
            "season_decay",
            "daily_decay",
            "correlation_scatter",
            "worker_bars",
            "hda_histogram",
        ]:
            result += ""
    else:
        result = base + " " + details.get(name, "")
    if after and name == "training_selector":
        from ligat.viz.goals_walk_review import selector_gap

        result = selector_gap(b)
    assert len(plain_explanation(result)) >= 250, (name, after, len(plain_explanation(result)))
    return '<div dir="rtl" class="graph-explanation"><p>' + result + "</p></div>"


def explanation_glossary(b: dict[str, Any]) -> str:
    p = next(d for d in b["report"]["decisions"] if d["name"] == "F5" and d["kind"] == "entry")[
        "rule8"
    ]["stats"]
    rows = [
        (
            "g-iqr-ci",
            "IQR מול CI95",
            "IQR הוא הקטע מאחוזון 25 לאחוזון 75 של התוצאות; CI95 הוא פס "
            "אי-ודאות באומדן ממוצע לפי העונות. שניהם באותן יחידות של המדד, אך "
            "אינם אותה שאלה.",
            f"בכניסת F5, CI95 לשיפור הוא [{p['ci_lower']:+.6f}, "
            f"{p['ci_upper']:+.6f}]; IQR עקומת האימון מתאר את הפיזור המרכזי בין "
            f"הקיפולים ולא את הפס הזה.",
            "IQR מראה הבדלים בין עונות אימון, ו-CI95 מתאר כמה בטוחים בתוספת דיוק ממוצעת.",
        ),
        (
            "g-forest",
            "גרף יער",
            "כל שורה בדיקה: נקודה לממוצע, פס לאי-ודאות וקו אפס להיעדר שינוי. "
            "ציר שיפור יכול להיות חיובי או שלילי; חציית אפס משאירה אי-ודאות "
            "בכיוון.",
            f"נקודת F5 בכניסה היא d̄={p['mean_improvement']:.6f}, והפס שלה "
            f"CI95=[{p['ci_lower']:+.6f}, {p['ci_upper']:+.6f}].",
            "מאפשר להשוות תרומות מידע בלי לבלבל בינן לבין כוונון מחדש של מודל.",
        ),
        (
            "g-paired",
            "הפרש מזווג ו-SE עונתי",
            "מחשבים לכל עונה ציון ייחוס פחות ציון מועמד; חיובי הוא שיפור. "
            "הממוצע משוקלל במשחקים וה-SE לפי פיזור ההפרשים בין עונות, לא לפי "
            "כל שורת משחק כאילו היתה עצמאית.",
            f"ל-F5: d̄={p['mean_improvement']:.6f}, SE={p['se']:.6f}, n={p['n']} עונות עמוקות.",
            "עונות קשות משפיעות על שני המודלים, והזיווג מקטין בלבול בין איכות עונה לאיכות חיזוי.",
        ),
        (
            "g-elo-share",
            "אחוז מתרומת Elo",
            "נרמול שיפור NLL בין בית בלבד לבין Elo. מאה אחוז הוא שיפור זהה "
            "ל-Elo; שלילי גרוע מבית בלבד, ומעל מאה משיג יותר מתרומת Elo.",
            elo_example(b),
            "מדד יחסי של מידע על כוח קבוצות, ולא אחוז ניחושים נכונים.",
        ),
        (
            "g-training-gap",
            "עקומת אימון מול ולידציה",
            "אימון מודד את ההתאמה לעבר ששימש ללמידה; ולידציה מודדת עונה שלא "
            "שימשה לאימון. ירידה באימון בלי ירידה בוולידציה יכולה לתאר שינון. "
            "שניהם NLL, שערכו קטן טוב יותר.",
            "המספרים של התצורה הפעילה מופיעים בפסקה שאחרי הבורר, כולל NLL "
            "האימון, NLL הוולידציה וההפרש ביניהם.",
            "תקציב גדול יכול לזכור צירופי קבוצות ותוצאות שאינם חוזרים בעונה הבאה.",
        ),
    ]
    return (
        '<div dir="rtl"><h3>מושגים לגרפים ולבדיקות '
        "בדיעבד</h3><table><thead><tr><th>מושג</th><th>מה זה "
        "והטווח</th><th>דוגמה מחושבת</th><th>משמעות "
        "בכדורגל</th></tr></thead><tbody>"
        + "".join(
            '<tr id="'
            + name
            + '">'
            + "".join("<td>" + value + "</td>" for value in fields)
            + "</tr>"
            for name, *fields in rows
        )
        + "</tbody></table></div>"
    )


def approved_findings(b: dict[str, Any]) -> str:
    from ligat.viz.goals_decision_display import opening

    return opening(b["report"])


def historical_leakage_reference() -> str:
    from pathlib import Path

    source = Path("docs/EXPERIMENTS.md").read_text(encoding="utf-8")
    row = next(line for line in source.splitlines() if "בקרת דליפה" in line and "אחרי" in line)
    match = re.search(r"נותן\s+(\d+\.\d+)", row)
    assert match is not None
    value = float(match.group(1))
    return (
        '<div dir="rtl"><p>נקודת ייחוס היסטורית: בקרת הצצה מ-EXP-001, שבה '
        "משתמשים ב-Elo של אחרי המשחק, "
        f"קיבלה log loss={value:.3f}. הנתון מתועד ב-"
        '<a href="../../docs/EXPERIMENTS.md">EXPERIMENTS, EXP-001, בקרת דליפה</a> '
        'וב-<a href="../../docs/PROJECT_PRINCIPLES.md">יומן ההחלטות, ה-ת18</a>. '
        "זו דוגמה מתועדת מניסוי קודם, ולא תוצאה חדשה של EXP-005 או יעד להגיע אליו.</p></div>"
    )
