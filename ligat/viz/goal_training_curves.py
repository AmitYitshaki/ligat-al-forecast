"""Signed, derived training views; full validation is gated by the frozen D3 report."""

import hashlib
import inspect
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.evaluate.goals_experiment import GoalExperimentSpec, GoalUnit, pilot_units
from ligat.evaluate.goals_runner import (
    _payload_digest,
    _read_journal,
    goal_provenance,
    goal_units_status,
    load_goal_inputs,
    load_goal_unit,
)
from ligat.models.training_log import atomic_bytes
from ligat.viz.goals_notebook import BLUE_SCALE, COLORS, finish

PARAMETERS = (
    "depth",
    "n_estimators",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "half_life",
    "l2",
)
_VERIFIED_STAGES: dict[tuple[str, str], str] = {}


def _sources(
    spec: GoalExperimentSpec, stage: str
) -> tuple[Any, Any, list[GoalUnit], dict[str, Any]]:
    if stage not in ("pilot", "full"):
        raise ValueError("Training views require pilot or full")
    if stage == "full":
        spec.guard_full()
    data = load_goal_inputs(spec)
    provenance = goal_provenance(data, prematch_version=spec.raw["inputs"]["prematch_version"])
    if stage == "pilot":
        units = list(pilot_units(spec))
    else:
        from ligat.evaluate.goals_ladder import _read_full_plan

        units = _read_full_plan(spec.output / "full" / "unit_plan.json", provenance)
    directory = spec.output / stage
    sources = {}
    for u in units:
        path = directory / (u.unit_id + ".json")
        journal = path.with_suffix(".trainlog.json.gz")
        try:
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            sources[journal.name] = hashlib.sha256(journal.read_bytes()).hexdigest()
        except FileNotFoundError as exc:
            status = goal_units_status([u], data, provenance, spec.output, stage=stage)[0]
            raise ValueError(
                f"Training table unavailable: {status['status']}: {status['reason']}"
            ) from exc
    binding = dict(
        index_sha256=hashlib.sha256((directory / "index.csv").read_bytes()).hexdigest(),
        sources=sources,
        code_sha256=_transform_signature(),
        provenance=provenance.identity(),
    )
    # C5 verification is reusable only for exactly the same bytes and inputs.
    # Hash every unit/journal/index on every call; a missing or changed source
    # must pass C5 again, even when the derived table itself is already cached.
    slot = (str(directory.resolve()), stage)
    verification = _payload_digest({"slot": slot, **binding})
    if _VERIFIED_STAGES.get(slot) != verification:
        statuses = goal_units_status(units, data, provenance, spec.output, stage=stage)
        invalid = [r for r in statuses if r["status"] != "complete"]
        if invalid:
            raise ValueError(
                f"Training table unavailable: {invalid[0]['status']}: {invalid[0]['reason']}"
            )
        if (
            any(
                hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
                for name, digest in sources.items()
            )
            or hashlib.sha256((directory / "index.csv").read_bytes()).hexdigest()
            != binding["index_sha256"]
        ):
            raise ValueError(
                "Training sources changed during C5 verification; retry after completion"
            )
        _VERIFIED_STAGES[slot] = verification
    return data, provenance, units, binding


def _transform_signature() -> str:
    """Bind table transformations without invalidating data for a plot or diagnostic edit."""
    functions = (
        _sources,
        _transform_signature,
        _cached,
        _publish_manifest,
        _flat,
        training_curves,
        frozen_full_report,
        full_validation_table,
    )
    return _payload_digest(
        {
            "parameters": PARAMETERS,
            "functions": {function.__name__: inspect.getsource(function) for function in functions},
        }
    )


def _cached(path: Path, binding: dict[str, Any]) -> bool:
    manifest = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest.exists():
        return False
    try:
        saved = json.loads(manifest.read_bytes())
        payload = saved["payload"]
        return bool(
            saved["sha256"] == _payload_digest(payload)
            and payload["binding"] == binding
            and payload["output_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    except (ValueError, KeyError):
        return False


def _publish_manifest(path: Path, binding: dict[str, Any], rows: int, wall: float) -> None:
    payload = dict(
        binding=binding,
        rows=rows,
        bytes=path.stat().st_size,
        build_seconds=wall,
        output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    atomic_bytes(
        path.with_suffix(".manifest.json"),
        json.dumps(dict(payload=payload, sha256=_payload_digest(payload))).encode(),
    )


def _flat(unit: GoalUnit, actual: dict[str, Any] | None = None) -> dict[str, Any]:
    p = unit.parameters
    actual = actual or {}

    def resolved(name: str) -> Any:
        value = p.get(name)
        return actual.get(name) if value is None else value

    return dict(
        unit_id=unit.unit_id,
        backend=unit.backend,
        label=unit.label,
        fold=unit.season - 10,
        stage_role=unit.role,
        feature_name=unit.role[:2]
        if unit.role.startswith(("S1-", "S2-"))
        else unit.role.removesuffix("-short"),
        feature_list=json.dumps(unit.features),
        preset=p.get("preset", "fixed"),
        depth=resolved("max_depth") if "max_depth" in p else resolved("depth"),
        n_estimators=resolved("n_estimators") if "n_estimators" in p else resolved("iterations"),
        learning_rate=resolved("learning_rate"),
        min_child_weight=resolved("min_child_weight"),
        subsample=resolved("subsample"),
        colsample_bytree=resolved("colsample_bytree"),
        reg_lambda=resolved("reg_lambda"),
        half_life=p.get("half_life") if unit.backend == "maher" else unit.half_life,
        l2=resolved("l2_leaf_reg"),
    )


def training_curves(spec: GoalExperimentSpec, *, stage: str = "pilot") -> pd.DataFrame:
    """Stream native train-only journals, compressing repeated metadata in Parquet."""
    _, _, units, binding = _sources(spec, stage)
    path = spec.output / stage / "training_curves.parquet"
    if _cached(path, binding):
        return pd.read_parquet(path)
    started = perf_counter()
    schema = pa.schema(
        [
            (n, pa.string())
            for n in (
                "unit_id",
                "backend",
                "label",
                "stage_role",
                "feature_list",
                "as_of",
                "preset",
            )
        ]
        + [
            ("feature_name", pa.string()),
            ("fold", pa.int32()),
            ("iteration", pa.int32()),
            ("iterations", pa.int32()),
            ("converged", pa.bool_()),
            ("train_loss", pa.float32()),
        ]
        + [(n, pa.float64()) for n in PARAMETERS]
    )
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    rows = 0
    buffer = []
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for unit in units:
                source = spec.output / stage / (unit.unit_id + ".json")
                payload = json.loads(source.read_bytes())["payload"]
                logs = _read_journal(source, payload)
                for log in logs:
                    metadata = _flat(unit, log.get("actual_params", {}))
                    losses = log["train_loss"]
                    if unit.backend == "maher":
                        points = [(log["iterations"], log["final_loss"])]
                    else:
                        points = list(enumerate(losses, 1)) if losses else [(0, None)]
                    for iteration, loss in points:
                        buffer.append(
                            dict(
                                **metadata,
                                as_of=log.get("as_of"),
                                iteration=iteration,
                                iterations=log["iterations"],
                                train_loss=loss,
                                converged=log["converged"],
                            )
                        )
                    if len(buffer) >= 100000:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                        rows += len(buffer)
                        buffer = []
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                rows += len(buffer)
        os.replace(temporary, path)
        _publish_manifest(path, binding, rows, perf_counter() - started)
    finally:
        temporary.unlink(missing_ok=True)
    return pd.read_parquet(path)


def frozen_full_report(spec: GoalExperimentSpec) -> dict[str, Any]:
    """A completed signed ladder is mandatory before any new validation view."""
    spec.guard_full()
    saved = json.loads((spec.output / "full" / "ladder_report.json").read_bytes())
    if saved["sha256"] != _payload_digest(saved["payload"]):
        raise ValueError("D3 report signature mismatch")
    data = load_goal_inputs(spec)
    provenance = goal_provenance(data, prematch_version=spec.raw["inputs"]["prematch_version"])
    if (
        saved["payload"]["provenance"]["code_sha256"] != provenance.code_sha256
        or saved["payload"]["provenance"]["input_sha256"] != provenance.input_sha256
    ):
        raise ValueError("D3 report is outdated")
    return dict(saved["payload"])


def full_validation_table(spec: GoalExperimentSpec, *, stage: str = "full") -> pd.DataFrame:
    """Full-stage audit from saved lambdas, never fit or alter the frozen choices."""
    if stage != "full":
        raise ValueError("Validation belongs only to full")
    frozen_full_report(spec)
    from ligat.evaluate.goals_ladder import LadderConfiguration, _score_unit_results
    from ligat.evaluate.selection_rules import leakage_check

    data, provenance, units, binding = _sources(spec, "full")
    binding["report_sha256"] = hashlib.sha256(
        (spec.output / "full" / "ladder_report.json").read_bytes()
    ).hexdigest()
    path = spec.output / "full" / "validation_views.parquet"
    if _cached(path, binding):
        return pd.read_parquet(path)
    started = perf_counter()
    groups: dict[str, list[GoalUnit]] = {}
    for u in units:
        key = _payload_digest({k: v for k, v in u.identity().items() if k != "season"})
        groups.setdefault(key, []).append(u)
    rows = []
    for key, group in groups.items():
        results = [
            load_goal_unit(spec.output / "full" / (u.unit_id + ".json"), u, provenance, data)
            for u in group
        ]
        scored = _score_unit_results(results, data)
        leakage_check(scored.per_fold, score="log_loss")
        configs = {u.season: u for u in group}
        actual = {r.unit.season: r.metadata.get("actual_params", {}) for r in results}
        for record in scored.per_fold.to_dict("records"):
            u = configs[record["season_id"]]
            rows.append(
                dict(
                    **_flat(u, actual[u.season]),
                    configuration_key=LadderConfiguration(u.backend, u.parameters, u.half_life).key,
                    group_key=key,
                    poisson_nll=record["poisson_nll"],
                    log_loss=record["log_loss"],
                    rps=record["rps"],
                    games=record["games"],
                    deep_fold=record["deep_fold"],
                )
            )
    table = pd.DataFrame(rows)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    _publish_manifest(path, binding, len(table), perf_counter() - started)
    return table


def _grid_labels(
    figure: Any,
    pivot: pd.DataFrame,
    lower: float,
    upper: float,
    *,
    col: int | None = None,
    precision: int = 4,
    suffix: str = "",
) -> None:
    placement = {} if col is None else dict(row=1, col=col)
    for iy, y in enumerate(pivot.index):
        for ix, x in enumerate(pivot.columns):
            value = float(pivot.loc[y, x])
            if not np.isfinite(value):
                continue
            fraction = (value - lower) / (upper - lower) if upper > lower else 0.5
            figure.add_annotation(
                x=ix,
                y=iy,
                text=(f"{value:.{precision}f}" + suffix).replace("-", "−"),
                showarrow=False,
                font=dict(color="white" if fraction >= 0.5 else "#111827", size=14),
                **placement,
            )
    figure.update_xaxes(
        type="linear",
        range=[-0.5, len(pivot.columns) - 0.5],
        tickvals=list(range(len(pivot.columns))),
        ticktext=[f"{x:g}" for x in pivot.columns],
        **placement,
    )
    figure.update_yaxes(
        type="linear",
        range=[-0.5, len(pivot.index) - 0.5],
        tickvals=list(range(len(pivot.index))),
        ticktext=[f"{y:g}" for y in pivot.index],
        **placement,
    )


def _band(
    figure: Any, frame: pd.DataFrame, name: str, color: str, *, row: int = 1, col: int = 1
) -> None:
    grouped = frame.groupby("iteration").train_loss.agg(
        median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), n="size"
    )
    for y, fill in [("q1", None), ("q3", "tonexty")]:
        figure.add_trace(
            go.Scatter(
                x=grouped.index,
                y=grouped[y],
                line=dict(width=0),
                fill=fill,
                fillcolor="rgba("
                + ",".join(str(int(color[i : i + 2], 16)) for i in (1, 3, 5))
                + ",0.12)",
                showlegend=False,
                legendgroup=name,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    figure.add_trace(
        go.Scatter(
            x=grouped.index,
            y=grouped["median"],
            name=name,
            legendgroup=name,
            line=dict(color=color),
            customdata=grouped.n,
            hovertemplate=(
                "עץ %{x}<br>חציון %{y:.4f}<br>קיפולים %{customdata}<extra>%{fullData.name}</extra>"
            ),
        ),
        row=row,
        col=col,
    )


def pilot_figures(table: pd.DataFrame) -> dict[str, Any]:
    xgb = table.loc[
        table.backend.eq("xgb") & table.stage_role.eq("S2-A") & table.n_estimators.eq(1000)
    ]
    curves = make_subplots(rows=1, cols=2, subplot_titles=["קצב 0.05", "קצב 0.1"])
    for col, lr in enumerate((0.05, 0.1), 1):
        for i, depth in enumerate((2, 3, 4, 6, 8)):
            part = xgb.loc[
                xgb.learning_rate.eq(lr) & xgb.depth.eq(depth) & xgb.min_child_weight.eq(10)
            ]
            _band(curves, part, f"עומק {depth}, קצב {lr}", COLORS[i], col=col)
    curves.update_xaxes(
        type="log",
        title="מספר עצים",
        tickvals=[1, 10, 100, 1000],
        ticktext=["1", "10", "100", "1,000"],
    )
    plotted = xgb.loc[xgb.min_child_weight.eq(10)].train_loss
    curves.update_yaxes(
        title="loss אימון מקורי של XGBoost",
        range=[float(plotted.min()) - 0.02, float(plotted.max()) + 0.02],
    )
    final = xgb.sort_values("iteration").groupby("unit_id", sort=False).tail(1)
    heat = make_subplots(rows=1, cols=2, subplot_titles=["קצב 0.05", "קצב 0.1"])
    for col, lr in enumerate((0.05, 0.1), 1):
        grouped = (
            final.loc[final.learning_rate.eq(lr)].groupby(["depth", "min_child_weight"]).train_loss
        )
        part = grouped.median().unstack()
        q1 = grouped.quantile(0.25).unstack()
        q3 = grouped.quantile(0.75).unstack()
        heat.add_trace(
            go.Heatmap(
                z=part.to_numpy(),
                x=list(range(len(part.columns))),
                y=list(range(len(part.index))),
                text=[[f"עומק {d:g}, זהירות {m:g}" for m in part.columns] for d in part.index],
                colorscale=BLUE_SCALE,
                customdata=np.stack([q1.to_numpy(), q3.to_numpy()], axis=-1),
                zmin=float(final.train_loss.min()),
                zmax=float(final.train_loss.max()),
                showscale=col == 2,
                hovertemplate=(
                    "%{text}<br>חציון loss %{z:.4f}"
                    "<br>IQR %{customdata[0]:.4f}–%{customdata[1]:.4f}"
                    "<extra>n=3 קיפולים</extra>"
                ),
            ),
            row=1,
            col=col,
        )
        _grid_labels(
            heat, part, float(final.train_loss.min()), float(final.train_loss.max()), col=col
        )
    heat.update_xaxes(title="זהירות (min_child_weight)")
    heat.update_yaxes(title="עומק")
    cats = make_subplots(rows=1, cols=1)
    cat = table.loc[table.backend.eq("catboost")]
    default = cat.loc[cat.preset.eq("library_default")]
    _band(cats, default, "ברירת מחדל של הספרייה", COLORS[0])
    for i, depth in enumerate((4, 6, 8), 1):
        _band(
            cats,
            cat.loc[cat.preset.eq("grid") & cat.depth.eq(depth) & cat.l2.eq(3)],
            f"רשת: עומק {depth}, קנס 3",
            COLORS[i],
        )
    cats.update_xaxes(title="מספר עצים")
    cats.update_yaxes(title="loss אימון מקורי של CatBoost")
    maher = go.Figure()
    daily = table.loc[table.backend.eq("maher")]
    for i, half in enumerate((None, 0.5, 1, 2)):
        part = daily.loc[daily.half_life.isna() if half is None else daily.half_life.eq(half)]
        maher.add_trace(
            go.Scatter(
                x=part.as_of,
                y=part.iterations,
                mode="markers",
                name="בלי דעיכה" if half is None else f"מחצית חיים {half:g}",
                marker=dict(color=COLORS[i], size=5),
                hovertemplate="%{x}<br>איטרציות %{y}<extra>%{fullData.name}</extra>",
            )
        )
    maher.update_xaxes(title="יום האימון")
    maher.update_yaxes(title="איטרציות עד התכנסות", rangemode="tozero")
    return dict(
        train_loss_curves_by_depth=finish(
            curves, "עומק וקצב משנים את מסלול האימון; n=3, חציון ו-IQR", height=850
        ),
        final_train_loss_heatmap=finish(
            heat, "מורכבות משנה את loss האימון; חציון של 3 קיפולים", height=850
        ),
        catboost_curves=finish(cats, "ברירת המחדל והרשת לומדות במסלולים שונים; n=3 ו-IQR"),
        maher_convergence=finish(maher, f"Maher התכנס ב-{len(daily):,} אימונים יומיים"),
    )


def _fold_summary(frame: pd.DataFrame) -> dict[str, float]:
    from scipy.stats import t  # type: ignore[import-untyped]

    n = len(frame)
    mean = float(np.average(frame.poisson_nll, weights=frame.games))
    se = float(frame.poisson_nll.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    distance = float(t.ppf(0.975, n - 1)) * se if n > 1 else float("nan")
    return dict(mean=mean, se=se, lower=mean - distance, upper=mean + distance, n=float(n))


def full_figures(
    training: pd.DataFrame, validation: pd.DataFrame, report: dict[str, Any]
) -> dict[str, Any]:
    """Draft full-stage views; no decisions or fit counts can be changed here."""
    v = validation.loc[validation.backend.eq("xgb") & validation.stage_role.eq("S2-A")]
    train = (
        training.loc[training.unit_id.isin(v.unit_id)]
        .sort_values("iteration")
        .groupby("unit_id")
        .tail(1)
    )
    final = train[["unit_id", "train_loss"]].merge(v, on="unit_id", validate="one_to_one")
    selected = final.loc[final.n_estimators.eq(1000) & final.learning_rate.eq(0.1)]
    compare = make_subplots(
        rows=1, cols=2, subplot_titles=["אימון: חציון ו-IQR", "ולידציה: ממוצע ו-CI95"]
    )
    val_cells = []
    for (depth, mcw), part in selected.groupby(["depth", "min_child_weight"]):
        stats = _fold_summary(part)
        val_cells.append(
            dict(
                depth=depth,
                mcw=mcw,
                train=float(part.train_loss.median()),
                train_lower=float(part.train_loss.quantile(0.25)),
                train_upper=float(part.train_loss.quantile(0.75)),
                **stats,
            )
        )
    cells = pd.DataFrame(val_cells)
    heat_min = float(min(cells["train"].min(), cells["mean"].min()))
    heat_max = float(max(cells["train"].max(), cells["mean"].max()))
    for col, field, lower, upper in [
        (1, "train", "train_lower", "train_upper"),
        (2, "mean", "lower", "upper"),
    ]:
        pivot = cells.pivot(index="depth", columns="mcw", values=field)
        lo = cells.pivot(index="depth", columns="mcw", values=lower)
        hi = cells.pivot(index="depth", columns="mcw", values=upper)
        n = cells.pivot(index="depth", columns="mcw", values="n")
        custom = np.stack([lo.to_numpy(), hi.to_numpy(), n.to_numpy()], axis=-1)
        compare.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=list(range(len(pivot.columns))),
                y=list(range(len(pivot.index))),
                text=[[f"עומק {d:g}, זהירות {m:g}" for m in pivot.columns] for d in pivot.index],
                colorscale=BLUE_SCALE,
                zmin=heat_min,
                zmax=heat_max,
                customdata=custom,
                showscale=col == 2,
                hovertemplate=(
                    "%{text}<br>הפסד %{z:.4f}<br>טווח %{customdata[0]:.4f}–%{customdata[1]:.4f}"
                    "<br>n=%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        _grid_labels(compare, pivot, heat_min, heat_max, col=col)
    compare.update_xaxes(title="זהירות (min_child_weight)")
    compare.update_yaxes(title="עומק")
    r0 = next(r for r in report["readable_metrics"]["R0"]["summary"] if r["scope"] == "all")[
        "poisson_nll"
    ]
    r1 = next(r for r in report["readable_metrics"]["R1"]["summary"] if r["scope"] == "all")[
        "poisson_nll"
    ]
    if r0 <= r1:
        raise ValueError("Elo contribution denominator must be positive")
    normalized = cells.copy()
    for c in ("mean", "lower", "upper"):
        normalized[c] = 100 * (r0 - normalized[c]) / (r0 - r1)
    # A decreasing affine transform reverses confidence interval endpoints.
    normalized[["lower", "upper"]] = normalized[["upper", "lower"]].to_numpy()
    pivot = normalized.pivot(index="depth", columns="mcw", values="mean")
    lo = normalized.pivot(index="depth", columns="mcw", values="lower")
    hi = normalized.pivot(index="depth", columns="mcw", values="upper")
    share = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=list(range(len(pivot.columns))),
            y=list(range(len(pivot.index))),
            text=[[f"עומק {d:g}, זהירות {m:g}" for m in pivot.columns] for d in pivot.index],
            colorscale=BLUE_SCALE,
            customdata=np.stack([lo.to_numpy(), hi.to_numpy()], axis=-1),
            hovertemplate=(
                "%{text}<br>תרומת Elo: %{z:.1f}%<br>CI95: "
                "%{customdata[0]:.1f}–%{customdata[1]:.1f}%"
                "<extra>n=15 קיפולים</extra>"
            ),
        )
    )
    _grid_labels(
        share, pivot, float(pivot.min().min()), float(pivot.max().max()), precision=1, suffix="%"
    )
    share.update_xaxes(title="זהירות (min_child_weight)")
    share.update_yaxes(title="עומק")
    configs = []
    effect_source = validation.loc[
        validation.backend.eq("xgb") & validation.stage_role.isin(["S2-A", "S2-AB"])
    ]
    for key, part in effect_source.groupby("group_key"):
        configs.append(
            dict(
                group_key=key,
                stage_role=part.iloc[0].stage_role,
                **{p: part.iloc[0][p] for p in PARAMETERS if p != "l2"},
                **_fold_summary(part),
            )
        )
    config_table = pd.DataFrame(configs)
    settings = [
        ("depth", "S2-A", "A: עומק"),
        ("min_child_weight", "S2-A", "A: זהירות"),
        ("n_estimators", "S2-A", "A: עצים"),
        ("learning_rate", "S2-A", "A: קצב למידה"),
        ("subsample", "S2-AB", "B: דגימת משחקים"),
        ("reg_lambda", "S2-AB", "B: קנס על העלים"),
        ("half_life", "S2-AB", "B: זמן מחצית בעונות"),
        ("colsample_bytree", "S2-A", "A: דגימת פיצ׳רים"),
    ]
    effects = make_subplots(
        rows=4, cols=2, vertical_spacing=0.1, subplot_titles=[s[2] for s in settings]
    )
    for index, (parameter, role, title) in enumerate(settings):
        subset = config_table.loc[config_table.stage_role.eq(role)].copy()
        if parameter == "half_life":
            subset[parameter] = subset[parameter].map(
                lambda h: "ללא דעיכה" if pd.isna(h) else str(int(h))
            )
        grouped = subset.groupby(parameter, dropna=False)["mean"].agg(["mean", "std", "size"])
        se = grouped["std"] / np.sqrt(grouped["size"])
        effects.add_trace(
            go.Scatter(
                x=grouped.index,
                y=grouped["mean"],
                mode="lines+markers",
                error_y=dict(type="data", array=se),
                name=parameter,
                line=dict(color=COLORS[0] if role == "S2-A" else COLORS[1]),
                showlegend=False,
                customdata=grouped["size"],
                hovertemplate=(
                    "ערך %{x}<br>NLL %{y:.4f}<br>n=%{customdata} תצורות"
                    "<extra>±SE בין תצורות</extra>"
                ),
            ),
            row=index // 2 + 1,
            col=index % 2 + 1,
        )
        effects.update_xaxes(title=title, row=index // 2 + 1, col=index % 2 + 1)
    effects.update_yaxes(title="NLL ולידציה")
    ladder = go.Figure()
    for name in ("S1", "S2", "F1", "F2", "F3", "F4", "F5", "F6"):
        stats = report["readable_metrics"][name]["paired"]
        mean = stats["mean_improvement"]
        se = stats["se"]
        ladder.add_trace(
            go.Scatter(
                x=[name],
                y=[mean],
                mode="markers",
                name=name,
                marker=dict(
                    color="#16A34A" if mean >= 0 else "#DC2626",
                    symbol="cross" if mean >= 0 else "x",
                    size=12,
                ),
                error_y=dict(type="data", array=[se]),
                customdata=[[stats["n"], stats["ci_lower"], stats["ci_upper"]]],
                hovertemplate=(
                    "שיפור NLL: %{y:+.5f}<br>n=%{customdata[0]} עונות"
                    "<br>CI95 %{customdata[1]:.5f}–%{customdata[2]:.5f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    ladder.add_hline(y=0, line_dash="dash", line_color="#6B7280")
    ladder.update_xaxes(title="שלב בסולם")
    ladder.update_yaxes(title="שיפור מול S2: חיובי = פחות NLL; ±SE")
    return dict(
        train_vs_validation_heatmap=finish(
            compare, "אימון נמוך יותר אינו מבטיח ולידציה טובה יותר — טיוטה", height=850
        ),
        hyperparameter_heatmap=finish(
            share,
            f"ברשת המוצגת, עומק {cells.loc[cells['mean'].idxmin(), 'depth']:g} "
            f"וזהירות {cells.loc[cells['mean'].idxmin(), 'mcw']:g} מובילים; n=15 — טיוטה",
        ),
        hyperparameter_effects=finish(
            effects, "השפעות הפרמטרים תלויות גם בתכנון A ו-B — טיוטה", height=1150
        ),
        ladder_progress=finish(
            ladder,
            "נכנסו לפי כלל 8: "
            + ", ".join(report["entered"])
            + "; n=14, שיפור מול S2 ו-±SE — טיוטה",
        ),
    )
