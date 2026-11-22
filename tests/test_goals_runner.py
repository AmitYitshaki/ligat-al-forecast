"""Fixed EXP-005 grids, blind signed artifacts, deterministic workers and guarded CLI."""

import hashlib
import json
from dataclasses import asdict, replace

import pandas as pd
import pytest
from typer.testing import CliRunner

from ligat.cli import app
from ligat.evaluate.goals_experiment import (
    GoalUnit,
    load_goal_experiment,
    pilot_units,
    xgb_stage_a,
    xgb_stage_b,
)
from ligat.evaluate.goals_runner import (
    GoalProvenance,
    PilotInterrupted,
    goal_data_fingerprint,
    run_goal_pilot,
    run_goal_unit,
    run_goal_units,
)
from ligat.features import registry
from ligat.features.prematch_v2 import PREMATCH_V2_COLUMNS
from ligat.models.m2_glm import GLMGoalConfig
from ligat.models.m2_xgb import XGBGoalConfig
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_team_goal_backtest import MeanGoalSpy


class LoggedMean(MeanGoalSpy):
    @property
    def training_log(self):
        return {
            "dataset": "training",
            "model": "fake",
            "train_loss": [1.0],
            "iterations": 1,
            "converged": True,
        }


def fake_factory(unit):
    return LoggedMean()


def units():
    return tuple(
        GoalUnit("glm", asdict(GLMGoalConfig("r0")), ("is_home",), season, role="synthetic")
        for season in (11, 12, 15, 25)
    )


def provenance(data):
    return GoalProvenance(goal_data_fingerprint(data), "0" * 64, "test")


def test_fixed_grids_and_score_free_pilot_plan():
    spec = load_goal_experiment()
    a = xgb_stage_a(spec)
    b = xgb_stage_b(spec, a[0])
    planned = pilot_units(spec)
    assert len(a) == 91 and len({c.config_id for c in a}) == 91
    assert len(b) == 24 and len({(c.config_id, hl) for c, hl in b}) == 24
    assert len(planned) == 408 and len({u.unit_id for u in planned}) == 408
    assert set(u.season for u in planned) == {12, 18, 25}
    assert spec.n_jobs == 15 and spec.raw["full_run_approved"] is False
    assert spec.raw["fast_path"] is None and spec.raw["correlation_drops"] == "pending"
    assert not any("team_id" in u.features for u in planned if u.backend == "catboost")
    assert sum(u.backend == "maher" for u in planned) == 12
    assert registry.M2_S1_COLUMNS == ("elo_pre", "elo_diff", "is_home")
    for columns in registry.M2_FAMILY_COLUMNS.values():
        assert set(columns) <= {*PREMATCH_V2_COLUMNS, "team_id"}


def test_unit_is_atomic_signed_blind_and_resumes_without_fitting(
    goal_sources, tmp_path, monkeypatch
):
    data = prepare(goal_sources, ("is_home",))
    unit = units()[0]
    version = provenance(data)

    def forbidden(*args, **kwargs):
        pytest.fail("Blind unit called scoring")

    monkeypatch.setattr("ligat.evaluate.team_goal_backtest.poisson_nll", forbidden)
    first = run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    assert not first.cached

    def do_not_refit(unit):
        pytest.fail("Completed unit was fitted again")

    second = run_goal_unit(unit, data, version, tmp_path, model_factory=do_not_refit)
    assert second.cached
    pd.testing.assert_frame_equal(first.evaluation.per_team, second.evaluation.per_team)
    pd.testing.assert_frame_equal(first.evaluation.per_fold, second.evaluation.per_fold)
    pd.testing.assert_series_equal(first.evaluation.summary, second.evaluation.summary)
    path = tmp_path / "pilot" / (unit.unit_id + ".json")
    saved = json.loads(path.read_text())
    assert not {"poisson_nll", "rps", "goals_scored", "log_loss"} & set(
        saved["payload"]["per_team"][0]
    )
    assert list((tmp_path / "pilot").glob("*.tmp")) == []


def test_interrupted_run_resumes_to_identical_predictions(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    with pytest.raises(PilotInterrupted):
        run_goal_units(
            units(),
            data,
            version,
            tmp_path / "resume",
            n_jobs=1,
            model_factory=fake_factory,
            stop_after_units=1,
        )
    assert len(list((tmp_path / "resume" / "pilot").glob("unit-*.json"))) == 1
    resumed = run_goal_units(
        units(), data, version, tmp_path / "resume", n_jobs=1, model_factory=fake_factory
    )
    fresh = run_goal_units(
        units(), data, version, tmp_path / "fresh", n_jobs=1, model_factory=fake_factory
    )
    assert sum(r.cached for r in resumed) == 1
    for a, b in zip(resumed, fresh, strict=True):
        pd.testing.assert_frame_equal(a.evaluation.per_team, b.evaluation.per_team)


@pytest.mark.slow
def test_one_and_four_loky_workers_are_bitwise_identical(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    serial = run_goal_units(
        units(), data, version, tmp_path / "one", n_jobs=1, model_factory=fake_factory
    )
    parallel = run_goal_units(
        units(), data, version, tmp_path / "four", n_jobs=4, model_factory=fake_factory
    )
    for a, b in zip(serial, parallel, strict=True):
        pd.testing.assert_frame_equal(a.evaluation.per_team, b.evaluation.per_team)
        assert a.unit.unit_id == b.unit.unit_id
        assert a.training_log == b.training_log


@pytest.mark.parametrize("damage", ["code", "input", "content"])
def test_mismatched_or_corrupt_unit_fails_without_overwriting(goal_sources, tmp_path, damage):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    unit = units()[0]
    run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    path = tmp_path / "pilot" / (unit.unit_id + ".json")
    if damage == "code":
        version = replace(version, code_sha256="1" * 64)
    elif damage == "input":
        version = replace(version, input_sha256="2" * 64)
    else:
        saved = json.loads(path.read_text())
        saved["payload"]["per_team"][0]["lambda"] += 1
        path.write_text(json.dumps(saved))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="signature"):
        run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    assert path.read_bytes() == before


def test_full_cli_blocks_before_loading_inputs(monkeypatch):
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner.load_goal_inputs",
        lambda *a, **k: pytest.fail("Full loaded inputs"),
    )
    result = CliRunner().invoke(
        app, ["run", "goals", "--config", "config/exp005.yaml", "--stage", "full"]
    )
    assert result.exit_code == 1 and "not approved" in result.output


def test_native_xgb_unit_on_synthetic_data_remains_blind(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    version = GoalProvenance(goal_data_fingerprint(data), "0" * 64, "engineering")
    cfg = XGBGoalConfig(2, 1, 8, 0.1, 0.8, 0.8, 1)
    unit = GoalUnit(
        "xgb",
        asdict(cfg),
        ("is_home",),
        15,
    )
    result = run_goal_unit(unit, data, version, tmp_path)
    expected_rows = int(data.context.season_id.eq(15).sum())
    assert len(result.evaluation.per_team) == expected_rows
    assert result.evaluation.per_team["lambda"].gt(0).all()
    path = tmp_path / "pilot" / (unit.unit_id + ".json")
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = run_goal_unit(unit, data, version, tmp_path)
    assert loaded.cached and hashlib.sha256(path.read_bytes()).hexdigest() == original
    print(
        json.dumps(
            {
                "synthetic_unit_rows": expected_rows,
                "fit_seconds": result.fit_seconds,
                "peak_rss_mb": result.peak_rss_mb,
                "cache_identical": True,
            }
        )
    )


def test_failed_atomic_publish_is_not_treated_as_a_completed_unit(
    goal_sources, tmp_path, monkeypatch
):
    data = prepare(goal_sources, ("is_home",))
    unit, version = units()[0], provenance(data)
    path = tmp_path / "pilot" / (unit.unit_id + ".json")
    with monkeypatch.context() as patch:

        def interrupted_replace(source, destination):
            raise OSError("simulated failure before publish")

        patch.setattr("ligat.evaluate.goals_runner.os.replace", interrupted_replace)
        with pytest.raises(OSError, match="before publish"):
            run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    assert not path.exists()
    assert list(path.parent.glob("*.tmp"))
    resumed = run_goal_unit(unit, data, version, tmp_path, model_factory=fake_factory)
    assert not resumed.cached and path.exists()


def test_pilot_dispatcher_interrupts_and_resumes_only_synthetic_units(
    goal_sources, tmp_path, monkeypatch
):
    data = prepare(goal_sources, ("is_home",))
    spec = load_goal_experiment()
    raw = dict(spec.raw)
    raw.update(n_jobs=1, output=str(tmp_path))
    spec = replace(spec, raw=raw)
    monkeypatch.setattr("ligat.evaluate.goals_runner.load_goal_inputs", lambda _: data)
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner.goal_provenance", lambda *a, **k: provenance(data)
    )
    monkeypatch.setattr("ligat.evaluate.goals_runner.pilot_units", lambda _: units())
    monkeypatch.setattr("ligat.evaluate.goals_runner._model", lambda *a: LoggedMean())

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic pilot dispatcher computed a score")

    monkeypatch.setattr("ligat.evaluate.team_goal_backtest.poisson_nll", forbidden)
    with pytest.raises(PilotInterrupted):
        run_goal_pilot(spec, stop_after_units=1)
    partial = json.loads((tmp_path / "pilot" / "interruption_report.json").read_text())
    assert partial["interrupted"] is True and partial["completed_files"] == 1
    result = run_goal_pilot(spec)
    assert result["units"] == 4 and result["cached_units"] == 1
    assert result["peak_total_rss_mb"] > 0
    saved = json.loads((tmp_path / "pilot" / "engineering_report.json").read_text())
    assert saved == result
    assert not list((tmp_path / "pilot").glob("engineering_report*.tmp"))
