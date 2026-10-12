"""One-off T-005b probe: exactly three authorized requests, then offline analysis."""

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from typing import Any

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/transfermarkt/probe"
REPORT = ROOT / "data/reports/t005b_probe.json"
USER_AGENT = "LigatAlForecast/0.1 (research; https://github.com/AmitYitshaki/ligat-al-forecast)"
PAGES = [
    (
        "1_season_2024",
        "https://www.transfermarkt.com/ligat-haal/startseite/wettbewerb/ISR1/plus/?saison_id=2024",
    ),
    (
        "2_value_2025_05_15",
        "https://www.transfermarkt.com/ligat-haal/marktwerteverein/wettbewerb/ISR1/stichtag/2025-05-15",
    ),
    (
        "3_value_2025_09_01",
        "https://www.transfermarkt.com/ligat-haal/marktwerteverein/wettbewerb/ISR1/stichtag/2025-09-01",
    ),
]


def publish(path: Path, content: bytes) -> None:
    """Publish a new immutable raw artifact atomically without overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        staging = Path(temporary.name)
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.link(staging, path)
    finally:
        staging.unlink()


def fetch() -> None:
    """Fetch sequentially with no redirects, retries or cookie reuse."""
    if RAW.exists() and any(RAW.iterdir()):
        raise RuntimeError("Probe artifacts already exist; use offline analysis, not another fetch")
    manifest: dict[str, Any] = {"status": "running", "user_agent": USER_AGENT, "pages": []}
    for index, (name, url) in enumerate(PAGES):
        if index:
            time.sleep(3)
        started = datetime.now(UTC).isoformat()
        with requests.Session() as session:
            session.headers["User-Agent"] = USER_AGENT
            response = session.get(url, timeout=(15, 60), allow_redirects=False)
        content = response.content
        compressed = gzip.compress(content, mtime=0)
        publish(RAW / f"{name}.html.gz", compressed)
        soup = BeautifulSoup(content, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        blocked = response.status_code in (403, 429) or bool(
            re.search(r"captcha|just a moment|access denied|verify.*human", title, re.I)
            or soup.select_one("#challenge-form, .g-recaptcha, #cf-challenge-running")
        )
        manifest["pages"].append(
            {
                "url": url,
                "time": started,
                "http_status": response.status_code,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "compressed_size": len(compressed),
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "file": f"{name}.html.gz",
                "title": title,
            }
        )
        manifest["status"] = (
            "blocked" if blocked else ("failed" if response.status_code != 200 else "running")
        )
        if index == len(PAGES) - 1 and manifest["status"] == "running":
            manifest["status"] = "completed"
        RAW.mkdir(parents=True, exist_ok=True)
        temporary_manifest = RAW / "manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_manifest.replace(RAW / "manifest.json")
        print(json.dumps(manifest["pages"][-1], ensure_ascii=True), flush=True)
        if blocked or response.status_code != 200:
            raise RuntimeError(f"STOP: HTTP {response.status_code}, title={title}; no retry")


def inspect() -> None:
    """Show table headers and sample rows from saved pages without network access."""
    for name, _ in PAGES:
        path = RAW / f"{name}.html.gz"
        if not path.exists():
            continue
        soup = BeautifulSoup(gzip.decompress(path.read_bytes()), "lxml")
        print(name)
        for table in soup.select("table.items"):
            print("HEADERS", [h.get_text(" ", strip=True) for h in table.select("thead th")])
            for row in table.select("tbody > tr")[:2]:
                print(
                    "ROW",
                    [
                        cell.get_text(" ", strip=True)
                        for cell in row.find_all("td", recursive=False)
                    ],
                )
                print(
                    "CLUB_LINKS",
                    [a.get("href") for a in row.select("a[href*='/startseite/verein/']")],
                )
        print(
            "DATES",
            [
                (option.get("value"), option.get_text(" ", strip=True))
                for option in soup.select("select option[selected]")
            ],
        )


def euros(value: str) -> int | None:
    """Convert the displayed euro amount without rounding binary floats."""
    if value.strip() == "-":
        return None
    match = re.fullmatch(r"\s*€\s*([\d.,]+)\s*([mk]?)\s*", value)
    if not match:
        raise ValueError(f"Unexpected market value: {value!r}")
    multiplier = {"m": 1_000_000, "k": 1_000, "": 1}[match[2]]
    return int(Decimal(match[1].replace(",", "")) * multiplier)


def analyze() -> None:
    """Verify saved artifacts, extract dated columns and compare common club IDs."""
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or len(manifest["pages"]) != 3:
        raise RuntimeError("The three authorized pages have not all completed")
    datasets: list[dict[int, dict[str, Any]]] = []
    metadata = []
    for index, (name, url) in enumerate(PAGES):
        artifact = manifest["pages"][index]
        compressed = (RAW / f"{name}.html.gz").read_bytes()
        content = gzip.decompress(compressed)
        assert artifact["url"] == url
        assert len(compressed) == artifact["compressed_size"]
        assert hashlib.sha256(compressed).hexdigest() == artifact["compressed_sha256"]
        assert len(content) == artifact["size"]
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        soup = BeautifulSoup(content, "lxml", from_encoding="utf-8")
        expected_header = ["Total market value", "Value 15/05/2025", "Value 01/09/2025"][index]
        candidates = []
        for table in soup.select("table.items"):
            headers = [cell.get_text(" ", strip=True) for cell in table.select("thead th")]
            if expected_header in headers:
                candidates.append((table, headers.index(expected_header)))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one {expected_header!r} table")
        table, value_index = candidates[0]
        selected = [option.get("value") for option in soup.select("select option[selected]")]
        expected_selection = ["2024", "2025-05-15", "2025-09-01"][index]
        if expected_selection not in selected:
            raise RuntimeError(f"Requested date/season not selected: {selected}")
        clubs = {}
        for row in table.select("tbody > tr"):
            link = row.select_one("a[href*='/startseite/verein/']")
            if link is None:
                continue
            club_match = re.search(r"/startseite/verein/(\d+)/", str(link["href"]))
            if club_match is None:
                raise ValueError("Club link lacks a numeric Transfermarkt ID")
            club_id = int(club_match[1])
            if club_id in clubs:
                raise ValueError(f"Duplicate club {club_id}")
            cells = row.find_all("td", recursive=False)
            text = cells[value_index].get_text(" ", strip=True)
            text_link = next(
                (a for a in row.select("a[href*='/startseite/verein/']") if a.get_text(strip=True)),
                link,
            )
            clubs[club_id] = {
                "name": text_link.get_text(" ", strip=True),
                "value": euros(text),
                "source_text": text,
            }
        if not clubs:
            raise RuntimeError("No club values extracted")
        datasets.append(clubs)
        metadata.append(
            {
                "url": url,
                "selected_date_or_season": expected_selection,
                "column": expected_header,
                "clubs": len(clubs),
            }
        )
    common = sorted(set(datasets[0]) & set(datasets[1]) & set(datasets[2]))
    dates = ["2025-05-15", "2025-09-01"]
    rows: list[dict[str, Any]] = []
    for club_id in common:
        values = [dataset[club_id]["value"] for dataset in datasets]
        differences = {
            date: abs(values[0] - value) / value
            if values[0] is not None and value not in (None, 0)
            else None
            for date, value in zip(dates, values[1:], strict=True)
        }
        available = {
            date: difference for date, difference in differences.items() if difference is not None
        }
        closest = (
            [
                date
                for date, difference in available.items()
                if difference == min(available.values())
            ]
            if available
            else []
        )
        rows.append(
            {
                "tm_club_id": club_id,
                "name": datasets[0][club_id]["name"],
                "season_page_value": values[0],
                "v_2025_05_15": values[1],
                "v_2025_09_01": values[2],
                "relative_differences": differences,
                "closest_dates": closest,
            }
        )
    summary: dict[str, dict[str, Any]] = {}
    for date in dates:
        date_differences = [
            row["relative_differences"][date]
            for row in rows
            if row["relative_differences"][date] is not None
        ]
        summary[date] = {
            "median_relative_difference": median(date_differences) if date_differences else None,
            "comparable_clubs": len(date_differences),
            "uniquely_closest_clubs": sum(row["closest_dates"] == [date] for row in rows),
            "tied_closest_clubs": sum(
                date in row["closest_dates"] and len(row["closest_dates"]) > 1 for row in rows
            ),
        }
    valid_medians = {
        date: item["median_relative_difference"]
        for date, item in summary.items()
        if item["median_relative_difference"] is not None
    }
    best = min(valid_medians, key=lambda date: valid_medians[date]) if valid_medians else None
    recommendation = (
        "inconclusive_no_comparable_values"
        if best is None
        else "different_method_or_date_no_go_without_further_investigation"
        if valid_medians[best] > 0.15
        else "september_closer_possible_leakage_consider_t_minus_2"
        if best == "2025-09-01"
        else "may_closer_candidate_go_subject_to_date_confirmation"
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "http_requests": 3,
        "sources": metadata,
        "common_clubs": len(common),
        "excluded_clubs_by_page": [sorted(set(dataset) - set(common)) for dataset in datasets],
        "clubs": rows,
        "summary": summary,
        "closest_by_median": best,
        "recommendation": recommendation,
        "limitation": (
            "Value similarity is not proof of a measurement date or a causal transfer-window "
            "effect. Historical values may have been revised. "
            "Only the two authorized dates were compared."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="Perform the three approved HTTP requests once"
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Inspect saved table headers without network access"
    )
    args = parser.parse_args()
    if args.fetch:
        fetch()
    if args.inspect:
        inspect()
    else:
        analyze()
