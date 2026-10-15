"""Hebrew notebook presentation: bidi-safe Plotly labels and compact, immutable HTML tables."""

import html
import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

HEBREW = re.compile(r"[\u0590-\u05ff]")
# Keep a numeric/Latin island in reading order, including signs, percentages and parentheses.
LTR_ISLAND = re.compile(
    r"\([^()\u0590-\u05ff]+\)|"
    r"\[[^\[\]\u0590-\u05ff]+\]|"
    r"(?:[A-Za-z\u0370-\u03ff]+(?:[=≈<>]+)?)?"
    r"[+−-]?\d+(?:[.,:/–−+×=]\d+)*(?:%|\+)?|"
    r"[A-Za-z\u0370-\u03ff]+(?:[−-]\d+)?"
)


def rtl_text(text: str, *, force: bool = False) -> str:
    """Embed Hebrew as RTL, protecting numbers/symbols as LTR; preserve Plotly HTML tags.

    Direction controls affect display only. Repeated finalization is idempotent;
    English-only labels stay intact unless force requests a dynamic Hebrew hover.
    HTML tags and Plotly placeholder syntax are preserved.
    """
    if (not force and not HEBREW.search(text)) or "\u202b" in text:
        return text
    pieces = re.split(r"(<[^>]+>)", text)
    for i, part in enumerate(pieces):
        if not part or part.startswith("<"):
            continue
        # Keep placeholders inside the RTL sentence. Formatted numeric placeholders
        # (including a literal percent suffix) form one LTR island; text values inherit RTL.
        tokens = re.compile(r"%\{[^}]+\}%?|" + LTR_ISLAND.pattern)

        def protect(match: re.Match[str]) -> str:
            value = match.group()
            if value.startswith("%{") and ":" not in value:
                return value
            return "\u202a" + value + "\u202c"

        numeric = tokens.sub(protect, part)
        pieces[i] = "\u200f\u202b" + numeric + "\u202c\u200f"
    return "".join(pieces)


def rtl_figure(figure: Any, *, labels: Mapping[str, str] | None = None) -> Any:
    """Apply one text policy to Plotly titles, axes, annotations, legends and hover labels.

    Category values remain untouched: explicit tick labels provide RTL presentation
    without changing the data coordinates, category matching or hover event values.
    """

    def hover_label(template: str) -> str:
        # Translate literal labels only; never replace Plotly variable names.
        pieces = re.split(r"(%\{[^}]+\}|<[^>]+>)", template)
        for i, part in enumerate(pieces):
            if not part.startswith(("%{", "<")):
                pieces[i] = re.sub(
                    r"[A-Za-z_][A-Za-z_0-9]*",
                    lambda match: (labels or {}).get(match.group(), match.group()),
                    part,
                )
        return rtl_text("".join(pieces), force=True)

    for key in figure.layout:
        axis = figure.layout[key]
        if key.startswith("coloraxis") and axis.colorbar.title.text:
            axis.colorbar.title.text = rtl_text(axis.colorbar.title.text)
        if not re.fullmatch(r"[xy]axis\d*", key):
            continue
        if axis.title.text:
            axis.title.text = rtl_text(axis.title.text)
        if axis.ticktext is not None:
            axis.ticktext = [rtl_text(str(t)) for t in axis.ticktext]
        else:
            coordinate = "x" if key.startswith("x") else "y"
            ref = key.replace("axis", "")
            values: list[str] = []
            for trace in figure.data:
                if (getattr(trace, coordinate + "axis", None) or coordinate) != ref:
                    continue
                coordinates = getattr(trace, coordinate, None)
                for value in () if coordinates is None else coordinates:
                    if isinstance(value, str) and value not in values:
                        values.append(value)
            if any(HEBREW.search(value) for value in values):
                axis.tickmode = "array"
                axis.tickvals = values
                axis.ticktext = [rtl_text(t) for t in values]
    if figure.layout.title.text:
        figure.layout.title.text = rtl_text(figure.layout.title.text)
    if figure.layout.legend.title.text:
        figure.layout.legend.title.text = rtl_text(figure.layout.legend.title.text)
    for annotation in figure.layout.annotations:
        annotation.text = rtl_text(annotation.text)
    for trace in figure.data:
        if trace.name:
            trace.name = rtl_text(trace.name)
        if getattr(trace, "hovertemplate", None):
            value = trace.hovertemplate
            trace.hovertemplate = (
                hover_label(value) if isinstance(value, str) else [hover_label(v) for v in value]
            )
        for field in ("text", "hovertext"):
            value = getattr(trace, field, None)
            if value is not None:
                setattr(
                    trace,
                    field,
                    rtl_text(value)
                    if isinstance(value, str)
                    else [rtl_text(str(v)) for v in value],
                )
        if trace.legendgrouptitle.text:
            trace.legendgrouptitle.text = rtl_text(trace.legendgrouptitle.text)
        colorbar = getattr(trace, "colorbar", None)
        if colorbar is not None and colorbar.title.text:
            colorbar.title.text = rtl_text(colorbar.title.text)
    return figure


class TableDisplay:
    """Small HTML display value, with no dependency on a running notebook or browser."""

    def __init__(self, content: str) -> None:
        self.content = content

    def _repr_html_(self) -> str:
        return self.content


def display_table(
    frame: pd.DataFrame,
    names: Mapping[str, str],
    *,
    columns: Iterable[str] | None = None,
    percent: Iterable[str] = (),
    values: Mapping[str, str] | None = None,
    intervals: Mapping[str, tuple[str, str]] | None = None,
) -> TableDisplay:
    """Return a Hebrew RTL table without modifying data, selecting or aggregating rows.

    Column selection is explicit. Known CI pairs become one column; percentages and
    their intervals share a scale (one decimal). Missing numbers stay NA, not zero.
    Index levels are retained and labelled, including nullable/multi-level indices.
    """
    chosen = list(frame.columns if columns is None else columns)
    rates = set(percent)
    translations = values or {}

    def label(name: Any) -> str:
        key = str(name)
        if key not in names and re.search(r"[A-Za-z]", key):
            raise ValueError(f"Missing Hebrew table label: {key}")
        return names.get(key, key)

    def number(value: Any, column: str) -> str:
        if value is pd.NA or (pd.api.types.is_scalar(value) and pd.isna(value)):
            return "NA"
        if f"{column}:{value}" in translations:
            return translations[f"{column}:{value}"]
        if isinstance(value, bool):
            return "כן" if value else "לא"
        if isinstance(value, (int, float)) or pd.api.types.is_number(value):
            if column in rates:
                return f"{float(value) * 100:.1f}%"
            if float(value).is_integer():
                return f"{int(value):,}"
            return f"{float(value):.2f}"
        return translations.get(str(value), str(value))

    def escaped(value: str) -> str:
        # bdi isolates signed numbers/percentages from the surrounding RTL table.
        direction = "auto" if HEBREW.search(value) else "ltr"
        return f'<bdi dir="{direction}">{html.escape(value)}</bdi>'

    fields: list[tuple[str, str, str | None]] = []
    consumed: set[str] = set()
    for column in chosen:
        if column in consumed:
            continue
        if column.endswith(("_ci_lower", "_ci_upper")) or column in ("ci_lower", "ci_upper"):
            continue
        lower, upper = f"{column}_ci_lower", f"{column}_ci_upper"
        if intervals and column in intervals:
            lower, upper = intervals[column]
        elif column.endswith("_mean") and lower not in frame:
            lower, upper = column[:-5] + "_ci_lower", column[:-5] + "_ci_upper"
        if column == "rate" or column == "per_match":
            if "ci_lower" in frame and "ci_upper" in frame:
                lower, upper = "ci_lower", "ci_upper"
        # Some established schemas name the interval after the event, without '_rate'.
        if column.endswith("_rate") and lower not in frame:
            lower, upper = column[:-5] + "_ci_lower", column[:-5] + "_ci_upper"
        fields.append((label(column), column, None))
        if lower in frame and upper in frame:
            fields.append((f"טווח 95% — {label(column)}", lower, upper))
            consumed.update((lower, upper))
    # A generic CI can describe a named fraction rather than a column called rate.
    if (
        "ci_lower" in chosen
        and "ci_upper" in chosen
        and not any(f[2] == "ci_upper" for f in fields)
    ):
        fields.append(("טווח 95%", "ci_lower", "ci_upper"))
    index_names = [label(n) if n is not None else "" for n in frame.index.names]
    headers = index_names + [f[0] for f in fields]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    rows = []
    for index, row in frame.iterrows():
        index_values = index if isinstance(index, tuple) else (index,)
        cells = [
            escaped(number(v, str(n))) for v, n in zip(index_values, frame.index.names, strict=True)
        ]
        for _, column, upper_bound in fields:
            if upper_bound is None:
                value = number(row[column], column)
            else:
                base = column.removesuffix("_ci_lower")
                if base not in rates and base + "_rate" in rates:
                    base += "_rate"
                if column == "ci_lower" and ("rate" in rates or "fraction_below" in rates):
                    base = "rate" if "rate" in rates else "fraction_below"
                lo, hi = number(row[column], base), number(row[upper_bound], base)
                value = "NA" if "NA" in (lo, hi) else f"{lo.rstrip('%')}–{hi.rstrip('%')}"
            cells.append(escaped(value))
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return TableDisplay(
        '<div dir="rtl" style="text-align:right;overflow-x:auto">'
        "<style>.ligat-table th,.ligat-table td{padding:6px 10px;text-align:right;"
        "border-bottom:1px solid #d1d5db;font-variant-numeric:tabular-nums}"
        ".ligat-table th{background:#eef2ff}.ligat-table tbody tr:nth-child(even)"
        "{background:#f8fafc}</style>"
        '<table class="ligat-table" style="border-collapse:collapse;font-size:14px;'
        'color:#111827;background:white">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
