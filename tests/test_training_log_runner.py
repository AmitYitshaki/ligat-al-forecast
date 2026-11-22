"""Journal integrity, rebuilt index and read-only status checks."""

import json
from dataclasses import asdict, replace

import pytest

from ligat.evaluate.goals_experiment import GoalUnit, load_goal_experiment
from ligat.evaluate.goals_runner import (
    PilotInterrupted,
    exp005_status,
    rebuild_goal_index,
    run_goal_unit,
    run_goal_units,
)
from ligat.models.m2_xgb import XGBGoalConfig
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_runner import fake_factory, provenance, units


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_journal_damage_rejects_resume_without_refitting(goal_sources, tmp_path, damage):
    data = prepare(goal_sources, ("is_home",))
    unit, version = units()[0], provenance(data)
    run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    journal = tmp_path / "pilot" / (unit.unit_id + ".trainlog.json.gz")
    if damage == "missing":
        journal.unlink()
    else:
        journal.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="training journal"):
        run_goal_unit(
            unit,
            data,
            version,
            tmp_path,
            model_factory=lambda _: pytest.fail("Refitted damaged cache"),
        )


def test_index_rebuild_and_status_do_not_train(goal_sources, tmp_path, monkeypatch):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    planned = units()[:2]
    run_goal_unit(planned[0], data, version, tmp_path, model_factory=fake_factory)
    path = rebuild_goal_index(tmp_path)
    original = path.read_bytes()
    path.unlink()
    assert rebuild_goal_index(tmp_path).read_bytes() == original
    spec = load_goal_experiment()
    spec = replace(spec, raw=spec.raw | {"output": str(tmp_path)})
    monkeypatch.setattr("ligat.evaluate.goals_runner.load_goal_inputs", lambda _: data)
    monkeypatch.setattr("ligat.evaluate.goals_runner.goal_provenance", lambda *a, **k: version)
    monkeypatch.setattr("ligat.evaluate.goals_runner.pilot_units", lambda _: planned)
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner._model", lambda *a: pytest.fail("Status trained")
    )
    assert [row["status"] for row in exp005_status(spec)] == ["complete", "missing"]
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner.goal_provenance",
        lambda *a, **k: replace(version, code_sha256="changed"),
    )
    result = exp005_status(spec)
    assert [row["status"] for row in result] == ["outdated", "missing"]
    assert "signature" in result[0]["reason"]
    assert "score" not in json.dumps(result)


def test_status_cli_dispatches_without_training(monkeypatch):
    from typer.testing import CliRunner

    from ligat.cli import app

    expected = [{"unit_id": "example", "status": "missing", "reason": "No saved unit"}]
    monkeypatch.setattr("ligat.evaluate.goals_runner.exp005_status", lambda _: expected)
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner.run_goal_pilot",
        lambda *a, **k: pytest.fail("Status CLI trained"),
    )
    result = CliRunner().invoke(app, ["run", "goals", "--stage", "pilot", "--status"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected


@pytest.mark.slow
def test_parallel_intentional_interruption_finishes_each_journal_unit_pair(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    with pytest.raises(PilotInterrupted):
        run_goal_units(
            units(),
            data,
            version,
            tmp_path,
            n_jobs=4,
            model_factory=fake_factory,
            stop_after_units=2,
        )
    saved = list((tmp_path / "pilot").glob("unit-*.json"))
    assert len(saved) == 2
    assert len(list((tmp_path / "pilot").glob("*.trainlog.json.gz"))) == 2
    resumed = run_goal_units(units(), data, version, tmp_path, n_jobs=4, model_factory=fake_factory)
    assert len(resumed) == 4 and sum(r.cached for r in resumed) == 2


@pytest.mark.slow
def test_native_training_curves_match_with_one_and_four_workers(goal_sources, tmp_path):
    import numpy as np

    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    configs = asdict(XGBGoalConfig(2, 1, 8, 0.1, 0.8, 0.8, 1))
    planned = [GoalUnit("xgb", configs, ("is_home",), s) for s in (11, 12, 15, 25)]
    serial = run_goal_units(planned, data, version, tmp_path / "one", n_jobs=1)
    parallel = run_goal_units(planned, data, version, tmp_path / "four", n_jobs=4)
    for a, b in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(
            a.evaluation.per_team["lambda"], b.evaluation.per_team["lambda"]
        )
        # Timings are measurements; deterministic model and journal fields match exactly.
        assert [{k: v for k, v in r.items() if k != "fit_seconds"} for r in a.training_log] == [
            {k: v for k, v in r.items() if k != "fit_seconds"} for r in b.training_log
        ]
