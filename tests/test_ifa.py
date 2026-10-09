"""Offline scraper behavior tests: no request may reach the real network."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import requests
from typer.testing import CliRunner

from ligat.cli import app, parse_seasons
from ligat.scrape.ifa import (
    AccessBlockedError,
    IFASource,
    ScrapeConfig,
    SourceFormatError,
    load_scrape_config,
    parse_stages,
    validate_round,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ifa"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Tests must not access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


class Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.sleeps: list[float] = []
        self.wall = datetime(2026, 10, 9, 12, tzinfo=UTC)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.elapsed += seconds

    def monotonic(self) -> float:
        return self.elapsed

    def now(self) -> datetime:
        return self.wall


def response(body: bytes, status: int = 200) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = body
    result.url = "https://www.football.org.il/test"
    return result


class Session:
    def __init__(self, clock: Clock, replies: list[requests.Response | Exception]) -> None:
        self.clock = clock
        self.replies = list(replies)
        self.calls: list[tuple[str, dict[str, int], float]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, int],
        headers: dict[str, str],
        timeout: float,
        allow_redirects: bool,
    ) -> requests.Response:
        assert "LigatAlForecast" in headers["User-Agent"]
        assert timeout > 0
        assert allow_redirects is False
        self.calls.append((url, params.copy(), self.clock.elapsed))
        self.clock.elapsed += 0.25  # Fake request duration: delay must follow completion.
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_source(
    tmp_path: Path, replies: list[requests.Response | Exception]
) -> tuple[IFASource, Session, Clock]:
    config = load_scrape_config(Path("config/scrape.yaml"))
    config = config.model_copy(update={"raw_root": tmp_path / "raw"})
    clock = Clock()
    session = Session(clock, replies)
    source = IFASource(
        config, session=session, sleep=clock.sleep, monotonic=clock.monotonic, now=clock.now
    )
    return source, session, clock


def season_replies() -> list[requests.Response | Exception]:
    return [response((FIXTURES / "league.html").read_bytes())] + [
        response((FIXTURES / f"{kind}.xml").read_bytes())
        for kind in ["played", "future", "playoff", "promotion"]
    ]


def test_history_collects_every_range_once_preserving_raw_bytes(tmp_path: Path) -> None:
    source, session, _ = make_source(tmp_path, season_replies())
    result = source.scrape_history([11])[0]
    assert result.downloaded == 5  # One league page and four unique rounds.
    assert result.cached == 0
    assert [params["round_id"] for _, params, _ in session.calls[1:]] == [1, 2, 3, 4]
    assert all(params["box"] == 0 for _, params, _ in session.calls[1:])
    for index, kind in enumerate(["played", "future", "playoff", "promotion"], start=1):
        assert (result.directory / f"round_{index:02d}.xml").read_bytes() == (
            FIXTURES / f"{kind}.xml"
        ).read_bytes()
    manifest = json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["expected_rounds"] == [1, 2, 3, 4]
    assert manifest["files"]["round_04.xml"]["stages"][0]["name"] == "מבחן עלייה"
    assert len(manifest["files"]["round_03.xml"]["stages"]) == 2
    assert manifest["files"]["round_01.xml"]["http_status"] == 200


def test_delay_applies_after_request_completion(tmp_path: Path) -> None:
    source, session, clock = make_source(tmp_path, season_replies())
    source.scrape_history([8])
    starts = [start for _, _, start in session.calls]
    assert clock.sleeps == [2.0] * 4
    assert all(b - a >= 2.25 for a, b in zip(starts, starts[1:], strict=False))


def test_completed_history_never_refetches_or_rewrites_raw_files(tmp_path: Path) -> None:
    source, session, _ = make_source(tmp_path, season_replies())
    first = source.scrape_history([8])[0]
    raw_paths = list(first.directory.glob("*.xml")) + [first.directory / "league.html"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in raw_paths}
    second = source.scrape_history([8])[0]
    assert second.downloaded == 0
    assert second.cached == 5
    assert len(session.calls) == 5
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in raw_paths}


def test_interrupted_history_resumes_missing_rounds_only(tmp_path: Path) -> None:
    initial = season_replies()[:2] + [requests.Timeout("interrupted")] * 3
    source, _, _ = make_source(tmp_path, initial)
    with pytest.raises(requests.Timeout):
        source.scrape_history([8])
    source, session, _ = make_source(tmp_path, season_replies()[2:])
    result = source.scrape_history([8])[0]
    assert result.cached == 2
    assert result.downloaded == 3
    assert [params["round_id"] for _, params, _ in session.calls] == [2, 3, 4]
    assert (
        json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))["status"]
        == "completed"
    )


def test_live_creates_separate_snapshots_with_future_rounds(tmp_path: Path) -> None:
    source, session, clock = make_source(tmp_path, season_replies() + season_replies())
    first = source.scrape_live()
    clock.wall += timedelta(seconds=1)
    second = source.scrape_live()
    assert first.directory != second.directory
    assert first.directory.parent.name == "snapshot"
    assert first.directory.parent.parent.name == "28"
    assert len(session.calls) == 10
    for result in (first, second):
        assert (result.directory / "round_02.xml").read_bytes() == (
            FIXTURES / "future.xml"
        ).read_bytes()


def test_snapshot_collision_cannot_overwrite_files(tmp_path: Path) -> None:
    source, session, _ = make_source(tmp_path, season_replies())
    source.scrape_live()
    with pytest.raises(FileExistsError):
        source.scrape_live()
    assert len(session.calls) == 5


@pytest.mark.parametrize("failure", [response(b"busy", 503), requests.ConnectionError("offline")])
def test_retry_backoff_then_success(tmp_path: Path, failure: requests.Response | Exception) -> None:
    source, session, clock = make_source(tmp_path, [failure, failure] + season_replies())
    source.scrape_history([8])
    assert clock.sleeps[:2] == [2.0, 4.0]
    assert len(session.calls) == 7


@pytest.mark.parametrize("status", [403, 404, 429, 302])
def test_nonretryable_status_stops_immediately(tmp_path: Path, status: int) -> None:
    source, session, _ = make_source(tmp_path, [response(b"error", status)] + season_replies())
    with pytest.raises((AccessBlockedError, requests.HTTPError, SourceFormatError)):
        source.scrape_history([8])
    assert len(session.calls) == 1
    manifest = json.loads((tmp_path / "raw/8/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    if status == 403:
        with pytest.raises(AccessBlockedError):
            source.scrape_history([9])
        assert len(session.calls) == 1


def test_server_errors_exhaust_three_attempts(tmp_path: Path) -> None:
    source, session, _ = make_source(tmp_path, [response(b"busy", 500)] * 3)
    with pytest.raises(requests.HTTPError):
        source.scrape_history([8])
    assert len(session.calls) == 3


def test_corrupt_cache_is_not_refetched_or_overwritten(tmp_path: Path) -> None:
    source, session, _ = make_source(tmp_path, season_replies())
    result = source.scrape_history([8])[0]
    path = result.directory / "round_01.xml"
    path.write_bytes(b"corrupted by external process")
    with pytest.raises(SourceFormatError, match="hash mismatch"):
        source.scrape_history([8])
    assert path.read_bytes() == b"corrupted by external process"
    assert len(session.calls) == 5


def test_recovers_cached_file_without_manifest_without_refetch(tmp_path: Path) -> None:
    directory = tmp_path / "raw/8"
    directory.mkdir(parents=True)
    (directory / "league.html").write_bytes((FIXTURES / "league.html").read_bytes())
    source, session, _ = make_source(tmp_path, season_replies()[1:])
    result = source.scrape_history([8])[0]
    assert result.cached == 1
    assert len(session.calls) == 4
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["league.html"]["http_status"] is None
    assert manifest["files"]["league.html"]["recovered_from_cache"] is True


@pytest.mark.parametrize("body", [b"<html>error</html>", b"<select id='ddlBoxes'></select>"])
def test_missing_layout_fails(tmp_path: Path, body: bytes) -> None:
    source, session, _ = make_source(tmp_path, [response(body)])
    with pytest.raises(SourceFormatError):
        source.scrape_history([8])
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"<select id='ddlBoxes'><option data-min-round='x' data-max-round='2'>a</option></select>",
        b"<select id='ddlBoxes'><option data-min-round='3' data-max-round='2'>a</option></select>",
    ],
)
def test_invalid_layout_bounds(body: bytes) -> None:
    with pytest.raises(SourceFormatError):
        parse_stages(body)


@pytest.mark.parametrize(
    "body", [b"not XML", b"<ResponseData/>", b"<ResponseData><HtmlData/></ResponseData>"]
)
def test_invalid_round_is_preserved_and_run_fails(tmp_path: Path, body: bytes) -> None:
    source, _, _ = make_source(tmp_path, season_replies()[:1] + [response(body)])
    with pytest.raises(SourceFormatError):
        source.scrape_history([8])
    assert (tmp_path / "raw/8/round_01.xml").read_bytes() == body
    manifest = json.loads((tmp_path / "raw/8/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


@pytest.mark.parametrize("kind", ["played", "future", "playoff", "promotion"])
def test_four_round_document_types(kind: str) -> None:
    validate_round((FIXTURES / f"{kind}.xml").read_bytes())


@pytest.mark.parametrize("seasons", [[7], [28], [8, 29]])
def test_invalid_history_seasons_rejected_before_io(tmp_path: Path, seasons: list[int]) -> None:
    source, session, _ = make_source(tmp_path, [])
    with pytest.raises(ValueError):
        source.scrape_history(seasons)
    assert not session.calls
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(
    "field,value", [("delay_seconds", 1), ("max_attempts", 4), ("timeout_seconds", 0)]
)
def test_invalid_configuration(field: str, value: int) -> None:
    data = load_scrape_config(Path("config/scrape.yaml")).model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        ScrapeConfig.model_validate(data)


def test_cli_help_is_offline() -> None:
    runner = CliRunner()
    for command in [["--help"], ["scrape", "history", "--help"], ["scrape", "live", "--help"]]:
        assert runner.invoke(app, command).exit_code == 0


def test_cli_dispatch_uses_selected_config_and_seasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, session, _ = make_source(tmp_path, season_replies())
    monkeypatch.setattr("ligat.cli.IFASource", lambda config: source)
    result = CliRunner().invoke(app, ["scrape", "history", "--seasons", "8"])
    assert result.exit_code == 0, result.output
    assert len(session.calls) == 5


@pytest.mark.parametrize("value", ["27-8", "8-9-10", "not-a-season", "8-999999999999", "28"])
def test_invalid_cli_season_ranges(value: str) -> None:
    with pytest.raises(Exception, match="ascending range"):
        parse_seasons(value)


@pytest.mark.parametrize(
    "season,filename,expected_rounds",
    [(season, "league_2006.html", 33) for season in range(8, 11)]
    + [(11, "league_2009.html", 36), (12, "league_2010.html", 37), (13, "league_2011.html", 37)]
    + [(season, "league_2012.html", 36) for season in range(14, 28)]
    + [(28, "league_2026.html", 26)],
)
def test_reported_season_ranges(
    season: int, filename: str, expected_rounds: int, tmp_path: Path
) -> None:
    # These are reconstructed source layouts, not a live verification of the 21 seasons.
    replies = [response((FIXTURES / filename).read_bytes())] + [
        response((FIXTURES / "played.xml").read_bytes()) for _ in range(expected_rounds)
    ]
    source, session, _ = make_source(tmp_path, replies)
    result = source.scrape_live() if season == 28 else source.scrape_history([season])[0]
    assert result.downloaded == expected_rounds + 1
    assert len(list(result.directory.glob("round_*.xml"))) == expected_rounds
    assert [params["round_id"] for _, params, _ in session.calls[1:]] == list(
        range(1, expected_rounds + 1)
    )
    manifest = json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))
    names = [stage["name"] for stage in manifest["stages"]]
    if season in (11, 12):
        assert "מבחן עלייה" in names
    if season == 13:
        assert "פלייאוף אמצעי" not in names


def test_cli_reports_block_without_followup_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, session, _ = make_source(tmp_path, [response(b"blocked", 403)])
    monkeypatch.setattr("ligat.cli.IFASource", lambda config: source)
    result = CliRunner().invoke(app, ["scrape", "live"])
    assert result.exit_code == 1
    assert len(session.calls) == 1


def test_deduplicates_requested_seasons(tmp_path: Path) -> None:
    source, session, _ = make_source(tmp_path, season_replies())
    assert len(source.scrape_history([8, 8])) == 1
    assert len(session.calls) == 5


def test_xml_cdata_response_is_supported() -> None:
    validate_round(
        b"<ResponseData><HtmlData><![CDATA[<div>future round</div>]]></HtmlData></ResponseData>"
    )
