"""Pure descriptive estimates and visible-text acceptance checks for notebook 04."""

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from bs4 import BeautifulSoup

from ligat.analysis import motivation_review as mr

PERIODS = {"B1": "2009/2010–2013/2014", "B2": "2014/2015–2018/2019", "B3": "2019/2020–2023/2024"}
FORBIDDEN = (
    r"\bA4b\b",
    r"\bA5[a-z]?\b",
    r"\bS3\b",
    r"S4_comparisons",
    r"alive_only_importance",
    r"near_zero_vs_dead",
    r"alive_importance_difference",
    r"survival_direct",
    r"transparent:",
    r"\bnan\b",
    r"בלוק [123]",
    r"אינדקס מתחיל באפס",
    r"\bclean\b",
    r"SE אחד",
)


def signed_survival_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR of survival leverage by signed safety gap and the fixed remaining bins."""
    part = frame.loc[
        ~frame.nothing_to_play & ~frame.relegation_decided & frame.gap_to_safety.between(-12, 15)
    ].copy()
    part["remaining_bin"] = pd.cut(
        part.games_remaining_stage, [-np.inf, 1, 3, 6, np.inf], labels=["≤1", "2–3", "4–6", "7+"]
    )
    return (
        part.groupby(["gap_to_safety", "remaining_bin"], observed=True)
        .survival_lev.agg(
            n="size",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .sort_values(["remaining_bin", "gap_to_safety"])
    )


def alive_staircase(frame: pd.DataFrame, *, seed: int, repeats: int = 2000) -> pd.DataFrame:
    """Each living L5 level minus ALL living rows, including that level, within strata."""
    alive = mr.validation_rows(frame).loc[lambda x: ~x.nothing_to_play]
    rows = []
    for level in (0.25, 0.5, 0.75, 1.0):
        for metric in ("res_pts", "res_ga"):
            rows.append(
                dict(
                    level=level,
                    metric=metric,
                    **mr.within_strata_contrast(
                        alive,
                        alive.L5.eq(level),
                        metric,
                        seed=seed,
                        repeats=repeats,
                        years=mr.VALIDATION_SEASONS,
                        include_selected_in_reference=True,
                    ),
                )
            )
    return pd.DataFrame(rows)


def typical_effects(slopes: pd.DataFrame) -> pd.DataFrame:
    """Scale each coefficient and its season-bootstrap draws by the same observed median gap."""
    result = slopes.copy()
    result["estimate"] = result.median_effect
    result["se"] = result.median_effect_se
    return result


def interval_rows(
    frame: pd.DataFrame, *, label: str, estimate: str = "estimate", se: str = "se"
) -> pd.DataFrame:
    """Two-SE intervals and explicit zero-crossing status; a crossing does not prove no effect."""
    result = frame.copy()
    result["reading_label"] = result[label].astype(str)
    result["lower"] = result[estimate] - 2 * result[se]
    result["upper"] = result[estimate] + 2 * result[se]
    result["excludes_zero"] = (result.lower.gt(0) | result.upper.lt(0)) & result[se].notna()
    return result


def visible_blocks(notebook: Mapping[str, Any]) -> list[tuple[int, str]]:
    """Extract what the reader sees; ignore Python source, CSS and MIME fallbacks hidden by HTML."""
    blocks = []
    checking_claims = False
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            heading = BeautifulSoup(source, "html.parser").find("h2")
            if heading is not None:
                checking_claims = heading.get_text().startswith("6.10")
            blocks.append((i, source))
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "text/html" in data:
                html = "".join(data["text/html"])
                if checking_claims:
                    soup = BeautifulSoup(html, "html.parser")
                    for table in soup.find_all("table"):
                        headers = [h.get_text(strip=True) for h in table.find_all("th")]
                        if "שם טכני" in headers and "מה הטענה" in headers:
                            column = headers.index("שם טכני")
                            for row in table.find_all("tr"):
                                values = row.find_all("td", recursive=False)
                                if len(values) > column:
                                    value = values[column].get_text()
                                    # The exception allows internal names, never direction marks.
                                    values[column].string = "".join(
                                        re.findall(r"[\u200e\u200f]|[בוהכלמש]−\d+", value)
                                    )
                    html = str(soup)
                blocks.append((i, html))
            if "application/vnd.plotly.v1+json" in data:

                def walk(value: Any, key: str = "", cell_index: int = i) -> None:
                    if isinstance(value, dict):
                        for k, v in value.items():
                            walk(v, k)
                    elif isinstance(value, (list, tuple)):
                        for v in value:
                            walk(v, key)
                    elif isinstance(value, str) and key in {
                        "text",
                        "name",
                        "hovertemplate",
                        "hovertext",
                        "ticktext",
                    }:
                        blocks.append((cell_index, value))

                walk(data["application/vnd.plotly.v1+json"])
            if not data and output.get("output_type") == "stream":
                blocks.append((i, "".join(output["text"])))
    cleaned = []
    for i, block in blocks:
        soup = BeautifulSoup(block, "html.parser")
        for tag in soup(["style", "script"]):
            tag.decompose()
        cleaned.append((i, soup.get_text(" ", strip=True)))
    return cleaned


def visible_text_violations(notebook: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return forbidden terms/direction marks and mathematical minuses used as Hebrew prefixes."""
    violations = []
    for i, text in visible_blocks(notebook):
        for pattern in (*FORBIDDEN, r"[\u200e\u200f]", r"[בוהכלמש]−(?=\d)"):
            for match in re.finditer(pattern, text):
                violations.append(dict(cell=i, term=match.group(), pattern=pattern))
    return violations


def verify_bottom_line(fragment: str, calculated: Mapping[str, tuple[float, int]]) -> pd.DataFrame:
    """Require every data number to be inside a registered claim; exempt navigation/model IDs."""
    soup = BeautifulSoup(fragment, "html.parser")
    claims = {str(span["data-claim"]): span.get_text() for span in soup.select("span[data-claim]")}
    for span in soup.select("span[data-claim]"):
        span.decompose()
    for tag in soup.select(".reference"):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"(?:EXP-\d+|T-\d+|E\d+|M[0-6]|ה-מ\d+)", "", text)
    # Section/notebook references and explicitly marked season ranges are navigation, not estimates.
    text = re.sub(r"מחברת\s+0[0-9]", "", text)
    if re.search(r"\d", text):
        raise AssertionError("Unregistered number in bottom line")
    return mr.verify_rounded_claims(claims, calculated)
