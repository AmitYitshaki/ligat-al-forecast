"""Offline season-page parsing, blocking, integrity and completeness checks."""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
import yaml

from ligat.data.repository import Repository
from ligat.data.squad_values import build_squad_values, money, parse_squad_page
from ligat.scrape.ifa import AccessBlockedError
from ligat.scrape.transfermarkt import USER_AGENT, TransfermarktSource

FIXTURE = Path(__file__).parent / "fixtures/transfermarkt/2024.html.gz"


def page(year: int = 2024) -> bytes:
    return gzip.decompress(FIXTURE.read_bytes()).replace(b"2024", str(year).encode())


class Transport:
    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        self.calls += 1
        assert kwargs["headers"] == {"User-Agent": USER_AGENT}
        assert kwargs["allow_redirects"] is False
        status, content = next(self.responses)
        response = requests.Response()
        response.status_code, response._content, response.url = status, content, url
        return response


@pytest.mark.parametrize("value,expected", [("€14.25m", 14250000), ("€850k", 850000), ("-", None)])
def test_money(value: str, expected: int | None) -> None:
    assert money(value)[0] == expected


def test_real_season() -> None:
    parsed = parse_squad_page(page(), 2024)
    assert len(parsed.rows) == 14
    assert parsed.rows[0] == {
        "tm_club_id": 119,
        "club_name": "Maccabi Tel Aviv",
        "squad_size": 43,
        "avg_value_eur": 812000,
        "total_value_eur": 34930000,
    }
    assert parsed.total_eur == 179990000
    assert parsed.known_sum_eur == 180030000
    assert parsed.rounding_tolerance_eur == 75000


def test_missing_values_stay_null() -> None:
    content = page().replace("€812k".encode(), b"-").replace("€34.93m".encode(), b"-")
    parsed = parse_squad_page(content, 2024)
    assert parsed.rows[0]["total_value_eur"] is None
    assert parsed.rows[0]["avg_value_eur"] is None


@pytest.mark.parametrize(
    "content,year",
    [
        (b"<html>temporary error</html>", 2024),
        (page(), 2023),
        (page().replace("€179.99m".encode(), "€1.00m".encode()), 2024),
    ],
    ids=["error-page", "wrong-season", "wrong-total"],
)
def test_invalid_page(content: bytes, year: int) -> None:
    with pytest.raises(ValueError):
        parse_squad_page(content, year)


@pytest.mark.parametrize(
    "status,content",
    [
        (403, b"forbidden"),
        (429, b"slow down"),
        (200, b"<html><title>CAPTCHA</title></html>"),
        (200, b"<html><form id='challenge-form'></form></html>"),
    ],
)
def test_stop_without_retry(tmp_path: Path, status: int, content: bytes) -> None:
    transport = Transport([(status, content)])
    source = TransfermarktSource(tmp_path, session=transport)
    with pytest.raises(AccessBlockedError):
        source.collect([2024, 2025])
    with pytest.raises(AccessBlockedError):
        source.collect([2024])
    assert transport.calls == 1
    assert not (tmp_path / "2024.html.gz").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "blocked"
    assert len(list((tmp_path / "rejected").glob("*.gz"))) == 1


def test_invalid_response_then_resume(tmp_path: Path) -> None:
    first = Transport([(200, page(2023)), (200, b"wrong page")])
    delays = []
    with pytest.raises(ValueError):
        TransfermarktSource(tmp_path, session=first, sleep=delays.append).collect([2023, 2024])
    before = (tmp_path / "2023.html.gz").read_bytes()
    second = Transport([(200, page())])
    manifest = TransfermarktSource(tmp_path, session=second).collect([2023, 2024])
    assert second.calls == 1 and manifest["status"] == "completed"
    assert (tmp_path / "2023.html.gz").read_bytes() == before
    assert delays == [3]


def test_integrity_and_cached_resume(tmp_path: Path) -> None:
    transport = Transport([(200, page())])
    source = TransfermarktSource(tmp_path, session=transport)
    source.collect([2024])
    source.collect([2024])
    assert transport.calls == 1
    target = tmp_path / "2024.html.gz"
    target.write_bytes(gzip.compress(page() + b" "))
    with pytest.raises(ValueError, match="hash/size"):
        source.collect([2024])
    assert transport.calls == 1


def test_recover_unmanifested_page(tmp_path: Path) -> None:
    (tmp_path / "2024.html.gz").write_bytes(FIXTURE.read_bytes())
    transport = Transport([])
    result = TransfermarktSource(tmp_path, session=transport).collect([2024])
    assert transport.calls == 0
    assert result["files"]["2024"]["recovered_from_cache"]


@pytest.mark.parametrize("delay", [0, 2.99, float("nan"), float("inf")])
def test_delay_floor(delay: float) -> None:
    with pytest.raises(ValueError):
        TransfermarktSource(delay_seconds=delay)


def test_full_build_and_mapping_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "raw"
    transport = Transport([(200, page(year)) for year in range(2006, 2027)])
    TransfermarktSource(root, session=transport, sleep=lambda _: None).collect()
    club_ids = [row["tm_club_id"] for row in parse_squad_page(page(), 2024).rows]
    entries = [{"tm_club_id": club, "team_id": index + 1} for index, club in enumerate(club_ids)]
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump({"clubs": entries}))
    expected = pd.DataFrame(
        [{"season_id": season, "team_id": team} for season in range(8, 29) for team in range(1, 15)]
    )
    repo = Repository(tmp_path / "clean")
    monkeypatch.setattr(repo, "read", lambda _: expected)
    frame = build_squad_values(root, mapping, repo)
    assert len(frame) == 294
    assert str(frame.avg_value_eur.dtype) == "Int64"
    artifact = repo.root / "squad_values.parquet"
    before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    mapping.write_text(yaml.safe_dump({"clubs": entries[1:]}))
    with pytest.raises(ValueError, match="Missing Transfermarkt club mappings"):
        build_squad_values(root, mapping, repo)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
    mapping.write_text(yaml.safe_dump({"clubs": entries}))
    monkeypatch.setattr(repo, "read", lambda _: expected[expected.team_id != 1])
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_squad_values(root, mapping, repo)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
