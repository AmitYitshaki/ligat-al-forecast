"""One season-label policy for notebook text, tables, axes and hover displays."""

import re
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

SHORT_SEASON = re.compile(r"(?<!\d)(\d{4})/(\d{2})(?!\d)")


def season_label(season_id: int) -> str:
    """Format an internal IFA season key using both full years; never alter the key."""
    if isinstance(season_id, bool) or not isinstance(season_id, Integral):
        raise ValueError("A season label requires an integer season_id")
    first = int(season_id) + 1998
    return f"{first}/{first + 1}"


def expand_season_text(text: str) -> str:
    """Expand season abbreviations mechanically; leave other prose unchanged."""

    def replace(match: re.Match[str]) -> str:
        first, last = int(match[1]), int(match[2])
        if (first + 1) % 100 != last:
            raise ValueError(f"Nonconsecutive season years: {match.group()}")
        return season_label(first - 1998)

    return SHORT_SEASON.sub(replace, text)


def with_season_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a view and attach full-year text while retaining numeric season_id keys."""
    data = frame.copy()
    if "season_id" in data.index.names:
        data["season"] = [season_label(int(s)) for s in data.index.get_level_values("season_id")]
    elif "season_id" in data:
        data["season"] = [season_label(int(s)) for s in data.season_id]
    elif "season" in data:
        data["season"] = [expand_season_text(str(s)) for s in data.season]
    return data


def format_figure_seasons(figure: Any) -> None:
    """Format season hover fields without changing point coordinates or color values."""
    for trace in figure.data:
        template = getattr(trace, "hovertemplate", None)
        custom = getattr(trace, "customdata", None)
        if custom is not None:
            array = np.asarray(custom, dtype=object).copy()
            for index in np.ndindex(array.shape):
                if isinstance(array[index], str):
                    array[index] = expand_season_text(str(array[index]))
            if isinstance(template, str) and array.ndim == 2:
                columns = re.findall(
                    r"season_id[^<%]*%\{customdata\[(\d+)\](?::[^}]*)?\}", template
                )
                for column in map(int, columns):
                    array[:, column] = [
                        s if isinstance(s, str) and "/" in s else season_label(int(s))
                        for s in array[:, column]
                    ]
            trace.customdata = array
        if isinstance(template, str) and re.search(r"season_id[^<%]*%\{marker.color", template):
            labels = [season_label(int(s)) for s in trace.marker.color]
            current = getattr(trace, "customdata", None)
            array = (
                np.empty((len(labels), 0), dtype=object)
                if current is None
                else np.asarray(current, dtype=object)
            )
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            column = array.shape[1]
            trace.customdata = np.column_stack([array, labels])
            trace.hovertemplate = template.replace("%{marker.color}", f"%{{customdata[{column}]}}")
