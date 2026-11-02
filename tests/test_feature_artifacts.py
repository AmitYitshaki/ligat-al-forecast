"""Derived artifacts retain keys and metadata, and cannot overwrite clean inputs."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.features.artifacts import save_table_state


def artifact_frame() -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            game_id=[1, 1],
            team_id=[2, 3],
            season_id=8,
            as_of=pd.Timestamp("2006-08-01", tz="Asia/Jerusalem"),
            points=[0, 0],
        )
    ).set_index(["game_id", "team_id"])


def test_round_trip_and_checksummed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    frame = artifact_frame()
    directory = Path("data/derived/prematch")
    directory.mkdir(parents=True)
    previous = directory / "table_state_v1.parquet"
    previous_metadata = previous.with_suffix(".metadata.json")
    frame.to_parquet(previous)
    previous_metadata.write_text('{"version": "1"}\n', encoding="utf-8")
    preserved = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (previous, previous_metadata)
    }
    target, sidecar = save_table_state(frame, input_sha256="a" * 64, commit="test-commit")
    assert target.name == "table_state_v2.parquet"
    for path, snapshot in preserved.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == snapshot
    assert_frame_equal(frame, pd.read_parquet(target))
    metadata = json.loads(sidecar.read_text(encoding="utf8"))
    assert metadata["input_sha256"] == "a" * 64
    assert metadata["commit"] == "test-commit"
    assert metadata["index_columns"] == ["game_id", "team_id"]
    assert metadata["version"] == "2" and metadata["training_end"] is None
    assert len(metadata["changes_from_version_1"]) == 2
    assert "NA" in metadata["changes_from_version_1"][0]
    assert "places_decided" in metadata["changes_from_version_1"][1]
    assert "ה-ת10" in metadata["changes_from_version_1"][1]
    assert metadata["artifact_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize("directory", ["data/clean", "data/raw", "outside"])
def test_storage_boundary(directory: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="inside data/derived"):
        save_table_state(
            artifact_frame(), input_sha256="a" * 64, commit="test", directory=Path(directory)
        )
    assert not Path(directory).exists()


def test_unknown_feature_is_rejected_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unregistered"):
        save_table_state(
            artifact_frame().assign(referee="blocked"), input_sha256="a" * 64, commit="test"
        )
    assert not Path("data/derived").exists()
