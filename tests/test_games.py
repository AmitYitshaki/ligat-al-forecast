"""Offline checks for stage-A limits, page identity, blocking and immutable publishing."""

import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from ligat.scrape.games import (
    GamePageSource,
    GameRef,
    collection_report,
    inspect_game_page,
    read_game_refs,
    run_games,
)
from ligat.scrape.ifa import AccessBlockedError, IFASource, SourceFormatError, load_scrape_config

PAGE = b"""<html><head><link rel="canonical" href="https://www.football.org.il/leagues/games/game/?game_id=113478"></head>
<body><section class="game-details" id="gfd_113478">
<span>&#1508;&#1512;&#1496;&#1497; &#1492;&#1502;&#1513;&#1495;&#1511;</span>
<div class="teams-names">
<div class="team-home"><a href="/team-details/?team_id=2266&amp;season_id=8">Home</a></div>
<div class="team-guest"><a href="/team-details/?team_id=2173&amp;season_id=8">Away</a></div></div>
<div class="game-time"><span class="date">26/08/2006 | 17:30</span>
<div class="total">4 : 1</div><div class="result-half">1 : 0</div></div>
<div class="timeline">Events</div><div class="players-cont">
<div class="home Active"><a href="/?player_id=1">Player</a></div>
<div class="guest Active"><a href="/?player_id=2">Player</a></div>
<div aria-labelledby="GAME_COACH_HOME"><a href="/?coach_id=3">Coach</a></div>
<div aria-labelledby="GAME_COACH_GUEST"><a href="/?coach_id=4">Coach</a></div>
</div></section></body></html>"""
GAME = GameRef(113478, 8, 2266, 2173)


def _response(content: bytes = PAGE, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = content
    response.url = "https://www.football.org.il/leagues/games/game/?game_id=113478"
    return response


def _source(tmp_path: Path, responses: list[requests.Response]) -> tuple[GamePageSource, Mock]:
    session = Mock()
    session.get.side_effect = responses
    http = IFASource(
        load_scrape_config(Path("config/scrape.yaml")), session=session, sleep=lambda _: None
    )
    return GamePageSource(http, tmp_path / "games"), session


def test_sections_and_identity() -> None:
    assert inspect_game_page(PAGE, GAME) == {
        "has_halftime": True,
        "has_timeline": True,
        "has_lineups": True,
        "has_coaches": True,
        "game_id_checked": True,
        "teams_and_season_checked": True,
        "canonical_game_id_checked": True,
    }
    with pytest.raises(SourceFormatError, match="identity mismatch"):
        inspect_game_page(PAGE, GameRef(999, 8, 2266, 2173))
    with pytest.raises(SourceFormatError, match="team links"):
        inspect_game_page(PAGE, GameRef(113478, 8, 999, 2173))
    with pytest.raises(SourceFormatError):
        inspect_game_page(b"<html><body>Temporary error</body></html>", GAME)


def test_cached_page_is_immutable_and_reuses_request(tmp_path: Path) -> None:
    source, session = _source(tmp_path, [_response()])
    first = source._collect(GAME)
    path = Path(first["path"])
    original = path.read_bytes()
    second = source._collect(GAME)
    assert second["cached"] is True
    assert path.read_bytes() == original
    assert session.get.call_count == 1
    path.write_bytes(gzip.compress(gzip.decompress(original) + b"corrupted"))
    with pytest.raises(SourceFormatError, match="hash mismatch"):
        source._collect(GAME)
    assert session.get.call_count == 1


def test_bad_200_is_archived_and_next_run_recovers(tmp_path: Path) -> None:
    source, session = _source(tmp_path, [_response(b"<html>Error</html>"), _response()])
    with pytest.raises(SourceFormatError):
        source._collect(GAME)
    assert not (source.raw_root / "8" / "113478.html.gz").exists()
    assert len(list((source.raw_root / "8" / "rejected").glob("*.html.gz"))) == 1
    assert source._collect(GAME)["cached"] is False
    assert session.get.call_count == 2


@pytest.mark.parametrize("status", [403, 429])
def test_blocking_stops_without_retry(tmp_path: Path, status: int) -> None:
    source, session = _source(tmp_path, [_response(status=status)])
    with pytest.raises(AccessBlockedError):
        source._collect(GAME)
    with pytest.raises(AccessBlockedError):
        source._collect(GAME)
    assert session.get.call_count == 1


def test_sample_rejects_full_collection_before_requests(tmp_path: Path) -> None:
    source, session = _source(tmp_path, [])
    with pytest.raises(ValueError, match="exactly ten"):
        source.collect_sample((GAME,))
    assert session.get.call_count == 0


@pytest.mark.parametrize(
    "sid,gid,home,away,lineups,coaches",
    [
        (8, 113478, 2266, 2173, False, False),
        (12, 240575, 2213, 2173, True, True),
        (16, 388606, 2176, 1068, True, True),
    ],
)
def test_real_saved_pages(
    sid: int, gid: int, home: int, away: int, lineups: bool, coaches: bool
) -> None:
    path = Path(__file__).parent / "fixtures" / "games" / f"{gid}.html"
    sections = inspect_game_page(path.read_bytes(), GameRef(gid, sid, home, away))
    assert sections["game_id_checked"]
    assert sections["has_halftime"]
    assert sections["has_timeline"]
    assert sections["has_lineups"] == lineups
    assert sections["has_coaches"] == coaches


def test_header_without_result_is_not_a_valid_game() -> None:
    with pytest.raises(SourceFormatError, match="result/date"):
        inspect_game_page(PAGE.replace(b"4 : 1", b"Temporary error"), GAME)


def _round_collection(directory: Path, sid: int, fixture: str = "mixed.xml") -> None:
    directory.mkdir(parents=True)
    content = (Path(__file__).parent / "fixtures" / "parse" / fixture).read_bytes()
    (directory / "round_01.xml").write_bytes(content)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "season_id": sid,
                "status": "completed",
                "completed_at": "2026-10-11T00:00:00+00:00",
                "expected_rounds": [1],
                "files": {
                    "round_01.xml": {
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "stages": [{"name": "Regular"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_raw_ids_latest_completed_snapshot_and_promotion_exclusion(tmp_path: Path) -> None:
    _round_collection(tmp_path / "28" / "snapshot" / "old", 28, "future.xml")
    _round_collection(tmp_path / "28" / "snapshot" / "current", 28)
    current_path = tmp_path / "28" / "snapshot" / "current" / "manifest.json"
    current = json.loads(current_path.read_text())
    current["completed_at"] = "2026-10-11T01:00:00+00:00"
    current_path.write_text(json.dumps(current))
    failed = tmp_path / "28" / "snapshot" / "failed"
    failed.mkdir()
    (failed / "manifest.json").write_text('{"status":"failed"}')
    games = read_game_refs(tmp_path, [28])
    assert len(games) == 5  # Two future rows in this partially played round have no IDs.
    manifest_path = tmp_path / "28" / "snapshot" / "current" / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["files"]["round_01.xml"]["stages"] = [{"name": "מבחן עלייה"}]
    manifest_path.write_text(json.dumps(document))
    assert read_game_refs(tmp_path, [28]) == ()


def test_raw_integrity_failure_happens_before_http(tmp_path: Path) -> None:
    _round_collection(tmp_path / "8", 8)
    (tmp_path / "8" / "round_01.xml").write_bytes(b"truncated")
    with pytest.raises(SourceFormatError, match="integrity"):
        read_game_refs(tmp_path, [8])


def test_interruption_resumes_and_reports_all_pages(tmp_path: Path) -> None:
    other = GameRef(113479, 8, 2266, 2173)
    other_page = PAGE.replace(b"113478", b"113479")
    source, session = _source(tmp_path, [_response(), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        source.collect((GAME, other), mode="history")
    manifest_path = source.raw_root / "8" / "manifest.json"
    assert json.loads(manifest_path.read_text())["status"] == "failed"
    original = (source.raw_root / "8" / "113478.html.gz").read_bytes()
    session.get.side_effect = [_response(other_page)]
    result = source.collect((GAME, other), mode="history")
    assert [row["cached"] for row in result] == [True, False]
    assert session.get.call_count == 3
    assert (source.raw_root / "8" / "113478.html.gz").read_bytes() == original
    assert gzip.decompress(original) == PAGE
    report = collection_report(source.raw_root, (GAME, other))[0]
    assert report == {
        "season_id": 8,
        "expected": 2,
        "valid": 2,
        "missing": [],
        "invalid": [],
        "rejected": 0,
        "status": "completed",
    }
    assert json.loads(manifest_path.read_text())["files"]["113478.html.gz"]["http_status"] == 200


def test_publish_collision_preserves_existing_file(tmp_path: Path) -> None:
    source, _ = _source(tmp_path, [])
    path = source.raw_root / "8" / "113478.html.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing immutable bytes")
    with pytest.raises(FileExistsError):
        source._publish(path, PAGE, GAME, {"files": {}}, 200)
    assert path.read_bytes() == b"existing immutable bytes"
    assert not list(path.parent.glob("*.tmp"))


def test_interrupted_atomic_publication_leaves_no_final_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, session = _source(tmp_path, [_response(), _response()])
    import os

    real_link = os.link
    monkeypatch.setattr("ligat.scrape.games.os.link", Mock(side_effect=OSError("disk interrupted")))
    with pytest.raises(OSError):
        source.collect((GAME,), mode="history")
    assert not (source.raw_root / "8" / "113478.html.gz").exists()
    assert not list((source.raw_root / "8").glob("*.tmp"))
    monkeypatch.setattr("ligat.scrape.games.os.link", real_link)
    source.collect((GAME,), mode="history")
    assert session.get.call_count == 2


def test_raw_technical_game_id_is_not_filtered(tmp_path: Path) -> None:
    _round_collection(tmp_path / "8", 8)
    manifest_path = tmp_path / "8" / "manifest.json"
    document = json.loads(manifest_path.read_text())
    path = tmp_path / "8" / "round_01.xml"
    from ligat.data.parse import parse_round

    original = parse_round(path.read_bytes()).games[0].game_id
    assert original is not None
    content = path.read_bytes().replace(str(original).encode(), b"732312")
    path.write_bytes(content)
    document["files"][path.name].update(
        bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )
    manifest_path.write_text(json.dumps(document))
    assert 732312 in {game.game_id for game in read_game_refs(tmp_path, [8])}


def test_legacy_sample_is_compressed_without_network_or_changing_original(tmp_path: Path) -> None:
    source, session = _source(tmp_path, [])
    path = source.raw_root / "8" / "113478.html"
    path.parent.mkdir(parents=True)
    path.write_bytes(PAGE)
    result = source.collect((GAME,), mode="history")
    assert result[0]["cached"]
    assert path.read_bytes() == PAGE
    assert gzip.decompress(Path(result[0]["path"]).read_bytes()) == PAGE
    assert session.get.call_count == 0


def test_technical_unplayed_page_without_events_is_valid() -> None:
    path = Path(__file__).parent / "fixtures" / "games" / "732312.html"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "906df127ca43a955a208df8f07e7bfc65f0e9a2e605786ee77483940b5bd9300"
    )
    result = inspect_game_page(path.read_bytes(), GameRef(732312, 23, 4823, 1005))
    assert result["game_id_checked"]
    assert result["has_halftime"]
    assert not result["has_timeline"]
    assert not result["has_lineups"]


def test_2012_page_with_lineups() -> None:
    path = Path(__file__).parent / "fixtures" / "games" / "268183.html"
    from urllib.parse import parse_qs, urlparse

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "lxml")
    ids = [
        int(
            parse_qs(urlparse(str(soup.select_one(f".teams-names .{side} a")["href"])).query)[
                "team_id"
            ][0]
        )
        for side in ("team-home", "team-guest")
    ]
    result = inspect_game_page(path.read_bytes(), GameRef(268183, 13, *ids))
    assert result["has_lineups"] and result["has_coaches"]


def test_invalid_cli_mode_does_not_construct_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ligat.scrape.games.IFASource", Mock(side_effect=AssertionError("Network")))
    with pytest.raises(ValueError, match="exactly one"):
        run_games(load_scrape_config(Path("config/scrape.yaml")), Path("config/game_pages.yaml"))


@pytest.mark.parametrize(
    "failure", [requests.ConnectionError(), requests.exceptions.ChunkedEncodingError(), 503]
)
def test_retry_backoff_and_throttling(tmp_path: Path, failure: object) -> None:
    session = Mock()
    session.get.side_effect = [
        _response(status=failure) if isinstance(failure, int) else failure,
        _response(),
    ]
    pauses = []
    http = IFASource(
        load_scrape_config(Path("config/scrape.yaml")),
        session=session,
        sleep=pauses.append,
        monotonic=lambda: 0.0,
    )
    GamePageSource(http, tmp_path).collect((GAME,), mode="history")
    assert session.get.call_count == 2
    assert pauses == [2.0, 2.0]  # Backoff, then the minimum inter-request delay.


def test_new_live_game_only_is_downloaded(tmp_path: Path) -> None:
    source, session = _source(tmp_path, [_response()])
    source.collect((GAME,), mode="live")
    other = GameRef(113479, 8, 2266, 2173)
    session.get.side_effect = [_response(PAGE.replace(b"113478", b"113479"))]
    result = source.collect((GAME, other), mode="live")
    assert [row["cached"] for row in result] == [True, False]
    assert session.get.call_count == 2


def test_report_only_never_constructs_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _round_collection(tmp_path / "rounds" / "8", 8)
    pages = tmp_path / "pages.yaml"
    pages.write_text(f"raw_root: {tmp_path.as_posix()}/games\n")
    config = load_scrape_config(Path("config/scrape.yaml")).model_copy(
        update={"raw_root": tmp_path / "rounds"}
    )
    monkeypatch.setattr("ligat.scrape.games.IFASource", Mock(side_effect=AssertionError("Network")))
    result = run_games(config, pages, seasons=(8,), report_only=True)
    assert result[0]["expected"] == 5
    assert len(result[0]["missing"]) == 5


def test_cli_rejects_ambiguous_modes_without_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from ligat.cli import app

    transport = Mock(side_effect=AssertionError("Network"))
    monkeypatch.setattr("ligat.scrape.games.IFASource", transport)
    result = CliRunner().invoke(app, ["scrape", "games", "--live", "--seasons", "8-27"])
    assert result.exit_code == 2
    transport.assert_not_called()
