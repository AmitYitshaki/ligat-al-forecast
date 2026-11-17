"""Explicit numeric display contracts shared by notebook-04 figures."""

import re
from typing import Any, Literal

import numpy as np


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
