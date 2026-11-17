"""Explicit numeric display contracts shared by notebook-04 figures."""

import re
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np


def verify_rtl_labels(figure: Mapping[str, Any]) -> None:
    """Reject Hebrew presentation labels that skipped the shared RTL embedding.

    Coordinates/customdata remain data. Their displayed labels are ticktext and
    hover text, so those are checked without modifying category matching.
    """
    fields = {"text", "name", "hovertemplate", "hovertext", "ticktext"}

    def walk(value: Any, path: tuple[str, ...] = (), field: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, (*path, key), key)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)), field)
        elif isinstance(value, str) and field in fields and re.search(r"[\u0590-\u05ff]", value):
            if not value.startswith("\u202b"):
                raise ValueError(f"Hebrew label skipped rtl_figure: {'.'.join(path)}")

    walk(figure)


def normalize_minus(text: str) -> str:
    """Preserve Hebrew prefixes and identifiers while displaying negative numbers correctly."""
    text = re.sub(r"(EXP|T)−(?=\d)", r"\1-", text)
    text = re.sub(r"(?<=[בוהכלמש])−(?=\d)", "-", text)
    return re.sub(r"(?<![\w\u0590-\u05ff])-(?=\d)|(?<=\d[eE])-(?=\d)", "−", text)


def two_se(values: Any, *, axis: Literal["x", "y"] = "y") -> dict[str, Any]:
    """Build both an explicit ±2 SE error bar and the trace marker checked by finalization."""
    return {
        f"error_{axis}": dict(type="data", array=2 * np.asarray(values, dtype=float), visible=True),
        "meta": {"two_se": [axis]},
    }
