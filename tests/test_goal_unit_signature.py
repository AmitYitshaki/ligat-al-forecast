"""Explicit unit dependency closure, per-file diagnostics and format migration."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ligat.evaluate.goals_runner import (
    UNIT_CODE_MODULES,
    UNIT_FORMAT_VERSION,
    _payload_digest,
    goal_provenance,
    goal_units_status,
    run_goal_unit,
)
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_runner import fake_factory, units


@pytest.mark.slow
def test_clean_process_covers_all_four_native_backend_dependencies():
    script = "exec(open('tests/goal_dependency_probe.py', encoding='utf-8').read())"
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["backends"] == 4 and set(result["loaded"]) <= set(UNIT_CODE_MODULES)


@pytest.mark.parametrize(
    "name,relevant",
    [
        ("evaluate/lambda_layer.py", False),
        ("evaluate/goals_ladder.py", False),
        ("cli.py", False),
        ("models/m2_xgb.py", True),
        ("evaluate/team_goal_backtest.py", True),
    ],
)
def test_status_depends_only_on_explicit_unit_sources(
    goal_sources, tmp_path, monkeypatch, name, relevant
):
    data = prepare(goal_sources, ("is_home",))
    unit, version = units()[0], goal_provenance(data)
    run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    original = Path.read_bytes
    target = (Path("ligat") / name).resolve()

    def changed(path):
        raw = original(path)
        return raw + b"\n# simulated change\n" if path.resolve() == target else raw

    monkeypatch.setattr(Path, "read_bytes", changed)
    current = goal_provenance(data)
    status = goal_units_status([unit], data, current, tmp_path)[0]
    assert status["status"] == ("outdated" if relevant else "complete")
    if relevant:
        assert name in status["reason"]
    saved = json.loads((tmp_path / "pilot" / (unit.unit_id + ".json")).read_text())["payload"]
    assert saved["format_version"] == UNIT_FORMAT_VERSION
    assert saved["provenance"]["code_files"] == version.code_files
    assert saved["provenance"]["libraries"] == version.libraries


def test_old_format_and_library_changes_are_outdated_without_refit(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    unit, version = units()[0], goal_provenance(data)
    run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    current = replace(version, libraries=version.libraries | {"numpy": "changed"})
    assert goal_units_status([unit], data, current, tmp_path)[0]["status"] == "outdated"
    path = tmp_path / "pilot" / (unit.unit_id + ".json")
    saved = json.loads(path.read_text())
    saved["payload"]["format_version"] = 2
    saved["sha256"] = _payload_digest(saved["payload"])
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    status = goal_units_status([unit], data, version, tmp_path)[0]
    assert status["status"] == "outdated" and "format" in status["reason"]
    assert path.read_bytes() == before
