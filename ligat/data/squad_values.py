"""Offline Transfermarkt season roster aggregates; these are not dated valuations."""

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml
from bs4 import BeautifulSoup

from ligat.data.repository import Repository
from ligat.data.schemas import make_frame


@dataclass(frozen=True)
class SquadPage:
    """Source rows and displayed league totals retained for validation."""

    rows: list[dict[str, Any]]
    total_eur: int | None
    known_sum_eur: int
    rounding_tolerance_eur: float


def money(value: str) -> tuple[int | None, float]:
    """Parse EUR amounts and half the displayed rounding precision; '-' stays null."""
    if value.strip() == "-":
        return None, 0
    match = re.fullmatch(r"\s*€\s*([\d,]+(?:\.\d+)?)\s*([mk]?)\s*", value)
    if not match:
        raise ValueError(f"Unexpected EUR amount: {value!r}")
    multiplier = {"": 1, "k": 1000, "m": 1_000_000}[match[2]]
    number = match[1].replace(",", "")
    precision = len(number.split(".")[1]) if "." in number else 0
    return int(Decimal(number) * multiplier), multiplier * 10**-precision / 2


def parse_squad_page(content: bytes, year: int) -> SquadPage:
    """Extract clubs by numeric ID, using only the requested season's roster table."""
    soup = BeautifulSoup(content, "lxml", from_encoding="utf-8")
    if str(year) not in [str(o.get("value")) for o in soup.select("option[selected]")]:
        raise ValueError(f"Wrong or missing selected season: {year}")
    candidates = []
    for table in soup.select("table.items"):
        headers = [h.get_text(" ", strip=True) for h in table.select("thead th")]
        if all(h in headers for h in ("Squad", "ø market value", "Total market value")):
            candidates.append((table, headers))
    if len(candidates) != 1:
        raise ValueError("Expected one Transfermarkt season squad-value table")
    table, headers = candidates[0]
    rows: list[dict[str, Any]] = []
    tolerance = 0.0
    ids = set()
    for row in table.select("tbody > tr"):
        links = row.select("a[href*='/startseite/verein/']")
        if not links:
            continue
        match = re.search(r"/startseite/verein/(\d+)/saison_id/(\d+)", str(links[0]["href"]))
        if match is None or int(match[2]) != year:
            raise ValueError("Club link does not identify the requested season")
        club_id = int(match[1])
        if club_id in ids:
            raise ValueError(f"Duplicate Transfermarkt club: {club_id}")
        ids.add(club_id)
        cells = [c.get_text(" ", strip=True) for c in row.find_all("td", recursive=False)]
        if len(cells) != len(headers):
            raise ValueError(f"Unexpected squad table row width for club {club_id}")
        squad = cells[headers.index("Squad")]
        if squad != "-" and not squad.isdigit():
            raise ValueError(f"Invalid squad size: {squad}")
        average, _ = money(cells[headers.index("ø market value")])
        total, rounding = money(cells[headers.index("Total market value")])
        tolerance += rounding
        name = next((a.get_text(" ", strip=True) for a in links if a.get_text(strip=True)), "")
        rows.append(
            {
                "tm_club_id": club_id,
                "club_name": name,
                "squad_size": None if squad == "-" else int(squad),
                "avg_value_eur": average,
                "total_value_eur": total,
            }
        )
    if not rows:
        raise ValueError("No clubs in season squad table")
    footer = table.select_one("tfoot tr")
    if footer is None:
        raise ValueError("Missing league total in squad table")
    footer_cells = footer.find_all("td", recursive=False)
    total, rounding = money(footer_cells[-1].get_text(" ", strip=True))
    tolerance += rounding
    known_sum = sum(r["total_value_eur"] for r in rows if r["total_value_eur"] is not None)
    missing = any(r["total_value_eur"] is None for r in rows)
    if total is not None and (
        known_sum > total + tolerance or (not missing and abs(known_sum - total) > tolerance)
    ):
        raise ValueError(
            f"League total mismatch: sum={known_sum}, source={total}, tolerance={tolerance}"
        )
    return SquadPage(rows, total, known_sum, tolerance)


def read_squad_page(root: Path, year: int, metadata: dict[str, Any]) -> bytes:
    """Verify gzip bytes and original response against their manifest before parsing."""
    compressed = (root / f"{year}.html.gz").read_bytes()
    content = gzip.decompress(compressed)
    if (
        len(compressed) != metadata["compressed_bytes"]
        or hashlib.sha256(compressed).hexdigest() != metadata["compressed_sha256"]
        or len(content) != metadata["bytes"]
        or hashlib.sha256(content).hexdigest() != metadata["sha256"]
    ):
        raise ValueError(f"Transfermarkt hash/size mismatch for season {year}")
    return content


def build_squad_values(
    raw_root: Path = Path("data/raw/transfermarkt/ISR1"),
    mapping_path: Path = Path("config/transfermarkt_teams.yaml"),
    repository: Repository | None = None,
) -> pd.DataFrame:
    """Build all 21 seasons only after validating IDs, totals and IFA season coverage."""
    repo = repository or Repository()
    entries = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))["clubs"]
    mapping = {int(row["tm_club_id"]): int(row["team_id"]) for row in entries}
    if len(mapping) != len(entries) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Transfermarkt mapping must be one-to-one")
    manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Transfermarkt collection is not completed")
    expected = repo.read("round_tables")[["season_id", "team_id"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    totals = []
    for year in range(2006, 2027):
        page = parse_squad_page(read_squad_page(raw_root, year, manifest["files"][str(year)]), year)
        unmapped = sorted({row["tm_club_id"] for row in page.rows} - set(mapping))
        if unmapped:
            raise ValueError(f"Missing Transfermarkt club mappings in {year}: {unmapped}")
        source_ids = {mapping[row["tm_club_id"]] for row in page.rows}
        expected_ids = set(expected.loc[expected.season_id == year - 1998, "team_id"])
        if not expected_ids or source_ids != expected_ids:
            raise ValueError(
                f"IFA coverage mismatch in {year}: "
                f"missing team_id={sorted(expected_ids - source_ids)}, "
                f"unexpected team_id={sorted(source_ids - expected_ids)}"
            )
        rows.extend(
            {
                "season_id": year - 1998,
                "team_id": mapping[row["tm_club_id"]],
                **{k: v for k, v in row.items() if k != "club_name"},
            }
            for row in page.rows
        )
        totals.append(
            {
                "season_id": year - 1998,
                "saison_id": year,
                "clubs": len(page.rows),
                "source_total_eur": page.total_eur,
                "known_sum_eur": page.known_sum_eur,
                "rounding_tolerance_eur": page.rounding_tolerance_eur,
            }
        )
    frame = make_frame("squad_values", rows)
    repo.write("squad_values", frame)
    report = repo.root.parent / "reports/t005b_build.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"rows": len(frame), "seasons": totals}, indent=2), encoding="utf-8"
    )
    return frame
