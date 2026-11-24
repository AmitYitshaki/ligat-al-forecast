"""Cost ordering is fixed, and dispatch order cannot alter saved model evidence."""

import gzip
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

from ligat.evaluate.goals_experiment import GoalUnit
from ligat.evaluate.goals_runner import goal_unit_priority, run_goal_units
from ligat.models.m2_catboost import CatBoostGoalConfig
from ligat.models.m2_maher import MaherGoalConfig
from ligat.models.m2_xgb import XGBGoalConfig
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_runner import fake_factory, provenance, units


def test_fixed_cost_order_within_each_backend():
    base = XGBGoalConfig.library_default()
    light = GoalUnit("xgb", asdict(replace(base, preset="grid", max_depth=2)), ("is_home",), 18)
    heavy = replace(light, parameters=asdict(replace(base, preset="grid", n_estimators=1000)))
    cat = GoalUnit("catboost", asdict(CatBoostGoalConfig.library_default()), ("is_home",), 18)
    cat_light = replace(cat, parameters=asdict(CatBoostGoalConfig(2, 3, 150, 0.1)))
    cat_heavy = replace(cat, parameters=asdict(CatBoostGoalConfig(6, 3, 1000, 0.1)))
    mahers = [
        GoalUnit("maher", asdict(MaherGoalConfig(half_life=h)), ("is_home",), 18)
        for h in (None, 2, 1, 0.5)
    ]
    ordered = sorted([light, heavy, cat_light, cat_heavy, cat, *mahers], key=goal_unit_priority)
    assert ordered == [cat, cat_heavy, cat_light, heavy, light, *reversed(mahers)]


def test_reversing_dispatch_preserves_artifacts_except_times(goal_sources, tmp_path, monkeypatch):
    import ligat.evaluate.goals_runner as runner

    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    # RSS is an observation of process history, not a model result. Hold it fixed
    # while checking every non-timing field, rather than silently ignoring it.
    monkeypatch.setattr(
        runner.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1)),
    )
    run_goal_units(
        units(), data, version, tmp_path / "forward", n_jobs=1, model_factory=fake_factory
    )
    original = goal_unit_priority
    monkeypatch.setattr(
        runner, "goal_unit_priority", lambda unit: tuple(-ord(c) for c in unit.unit_id)
    )
    run_goal_units(
        tuple(reversed(units())),
        data,
        version,
        tmp_path / "reverse",
        n_jobs=1,
        model_factory=fake_factory,
    )
    monkeypatch.setattr(runner, "goal_unit_priority", original)
    for unit in units():
        name = unit.unit_id + ".json"
        payloads, journals = [], []
        for directory in ("forward", "reverse"):
            path = tmp_path / directory / "pilot" / name
            payload = json.loads(path.read_bytes())["payload"]
            for key in ("elapsed_seconds", "fit_seconds", "completed_at", "training_log_sha256"):
                payload.pop(key)
            payloads.append(payload)
            logs = json.loads(gzip.decompress(path.with_suffix(".trainlog.json.gz").read_bytes()))
            for record in logs:
                record.pop("fit_seconds", None)
            journals.append(logs)
        assert payloads[0] == payloads[1]
        assert journals[0] == journals[1]
