"""Synthetic-only M3 export: outside-season fits, signed resume and paired rates."""

import gzip
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goals_experiment import GoalUnit
from ligat.evaluate.lambda_layer import (
    LAMBDA_COLUMNS,
    LambdaLayerConfig,
    join_lambda_games,
    lambda_layer_status,
    load_lambda_layer,
    validate_lambda_layer,
    write_lambda_layer,
)
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.models.m2_catboost import CatBoostGoalConfig
from ligat.models.m2_glm import GLMGoalConfig
from ligat.models.m2_maher import MaherGoalConfig
from ligat.models.m2_xgb import XGBGoalConfig
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_runner import LoggedMean, fake_factory, provenance


class PriorSeasonDaily(LoggedMean):
    requires_as_of = True

    def __init__(self, season):
        super().__init__("per_matchday")
        self.season = season

    def fit(self, X, y, *, context, as_of, prediction_features):
        assert context.season_id.lt(self.season).all()
        assert context.date.lt(as_of).all()
        assert not X.index.isin(prediction_features.index).any()
        return super().fit(X, y, context=context)


def daily_factory(unit):
    return PriorSeasonDaily(unit.season)


def config():
    return LambdaLayerConfig("glm", asdict(GLMGoalConfig("r0")), ("is_home",), "prior_seasons_only")


@pytest.mark.parametrize("factory", [fake_factory, daily_factory])
def test_layer_covers_only_11_to_25_and_pairs_home_away(
    goal_sources, tmp_path, monkeypatch, factory
):
    data = prepare(goal_sources, ("is_home",))
    version, cfg = provenance(data), config()
    monkeypatch.setattr(
        "ligat.evaluate.team_goal_backtest.poisson_nll",
        lambda *a, **k: pytest.fail("Layer export computed validation scores"),
    )
    before = lambda_layer_status(cfg, data, tmp_path, provenance=version)
    assert before["status"] == "missing" and not list(tmp_path.iterdir())
    frame = write_lambda_layer(
        cfg, data, tmp_path, n_jobs=1, provenance=version, model_factory=factory
    )
    assert tuple(frame.columns) == LAMBDA_COLUMNS
    assert set(frame.season_id) == set(range(11, 26))
    assert not frame.season_id.isin(range(8, 11)).any()
    assert frame.train_end_season.lt(frame.season_id).all()
    assert frame.train_end_date.lt(frame.as_of).all()
    assert frame["lambda"].gt(0).all() and np.isfinite(frame["lambda"]).all()
    for season in range(11, 26):
        history = data.y.loc[data.context.season_id.lt(season)]
        assert frame.loc[frame.season_id.eq(season), "lambda"].eq(history.mean()).all()
    games = join_lambda_games(frame, data)
    assert len(games) * 2 == len(frame)
    indexed = frame.set_index(["game_id", "team_id"])
    for key, row in data.context.loc[data.context.season_id.ge(11)].iterrows():
        column = "lambda_home" if row.is_home else "lambda_away"
        assert games.loc[row.game_id, column] == indexed.loc[key, "lambda"]
    pd.testing.assert_frame_equal(frame, load_lambda_layer(cfg, data, tmp_path, provenance=version))
    saved = json.loads((tmp_path / "manifest.json").read_text())["payload"]
    assert saved["input_sha256"] == version.input_sha256
    assert (
        saved["output_sha256"]
        == hashlib.sha256((tmp_path / "lambda.parquet").read_bytes()).hexdigest()
    )
    assert len(saved["unit_ids"]) == 15
    assert saved["native_config_id"] == cfg.units()[0].configuration().config_id
    assert lambda_layer_status(cfg, data, tmp_path, provenance=version)["status"] == "complete"
    hashes = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def never_fit(unit):
        pytest.fail("Completed lambda layer retrained")

    repeated = write_lambda_layer(
        cfg, data, tmp_path, n_jobs=1, provenance=version, model_factory=never_fit
    )
    pd.testing.assert_frame_equal(repeated, frame)
    assert all(p.read_bytes() == raw for p, raw in hashes.items())


def test_current_season_targets_cannot_affect_even_daily_export(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    changed = data.y.copy()
    changed.loc[data.context.season_id.eq(12)] += 100
    other = GoalData(data.X, changed, data.context)
    a = write_lambda_layer(
        config(),
        data,
        tmp_path / "a",
        n_jobs=1,
        provenance=provenance(data),
        model_factory=daily_factory,
    )
    b = write_lambda_layer(
        config(),
        other,
        tmp_path / "b",
        n_jobs=1,
        provenance=provenance(other),
        model_factory=daily_factory,
    )
    np.testing.assert_array_equal(
        a.loc[a.season_id.le(12), "lambda"], b.loc[b.season_id.le(12), "lambda"]
    )
    assert (
        b.loc[b.season_id.eq(13), "lambda"].to_numpy()
        > a.loc[a.season_id.eq(13), "lambda"].to_numpy()
    ).all()


def test_daily_export_excludes_late_previous_season_fixture_until_as_of(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    context = data.context.copy()
    first = context.loc[context.season_id.eq(11), "as_of"].min()
    late = context.game_id.eq(1002)
    context.loc[late, "date"] = first + pd.Timedelta(days=1, hours=8)
    context.loc[late, "as_of"] = first + pd.Timedelta(days=1)
    data = GoalData(data.X, data.y, context)
    frame = write_lambda_layer(
        config(), data, tmp_path, n_jobs=1, provenance=provenance(data), model_factory=daily_factory
    )
    early = frame.loc[frame.season_id.eq(11) & frame.as_of.eq(first)]
    later = frame.loc[frame.season_id.eq(11) & frame.as_of.eq(first + pd.Timedelta(days=2))]
    assert early.train_end_date.lt(first).all()
    assert later.train_end_date.eq(first + pd.Timedelta(days=1, hours=8)).all()
    assert early["lambda"].eq(data.y.loc[data.context.season_id.lt(11) & ~late].mean()).all()
    assert later["lambda"].eq(data.y.loc[data.context.season_id.lt(11)].mean()).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend,parameters,features",
    [
        ("xgb", asdict(XGBGoalConfig(2, 1, 8, 0.1, 0.8, 0.8, 1)), ("is_home",)),
        ("catboost", asdict(CatBoostGoalConfig(4, 1, 8, 0.1)), ("is_home",)),
        ("glm", asdict(GLMGoalConfig("r0")), ("is_home",)),
        ("maher", asdict(MaherGoalConfig()), ("team_id", "opponent_id", "is_home")),
    ],
)
def test_native_backends_export_synthetic_data_only(
    goal_sources, tmp_path, backend, parameters, features
):
    prematch, games = goal_sources
    prematch = prematch.assign(promoted_in=False)
    columns = tuple(dict.fromkeys((*features, "promoted_in")))
    data = prepare((prematch, games), columns)
    cfg = LambdaLayerConfig(backend, parameters, features, "prior_seasons_only")
    frame = write_lambda_layer(cfg, data, tmp_path, n_jobs=1)
    validate_lambda_layer(frame, data)
    assert lambda_layer_status(cfg, data, tmp_path)["status"] == "complete"
    if backend == "glm":
        logs = [
            log
            for path in (tmp_path / "units" / "lambda").glob("*.gz")
            for log in json.loads(gzip.decompress(path.read_bytes()))
        ]
        assert any(log["iterations"] == 0 for log in logs)
        assert all(len(log["train_loss"]) == log["iterations"] for log in logs)
        index = pd.read_csv(tmp_path / "units" / "lambda" / "index.csv")
        zero = index.loc[index.iterations.eq(0)]
        assert zero.first_train_loss.isna().all() and zero.last_train_loss.isna().all()


@pytest.mark.parametrize(
    "damage", ["code", "input", "configuration", "version", "parquet", "journal", "manifest"]
)
def test_outdated_layer_is_never_silently_loaded_or_overwritten(goal_sources, tmp_path, damage):
    data = prepare(goal_sources, ("is_home",))
    cfg, version = config(), provenance(data)
    write_lambda_layer(
        cfg, data, tmp_path, n_jobs=1, provenance=version, model_factory=fake_factory
    )
    if damage == "code":
        version = replace(version, code_sha256="a" * 64)
    elif damage == "input":
        other = GoalData(data.X, data.y + 1, data.context)
        data, version = other, provenance(other)
    elif damage == "configuration":
        cfg = replace(cfg, half_life=1)
    elif damage == "version":
        version = replace(version, prematch_version="changed")
    elif damage == "parquet":
        (tmp_path / "lambda.parquet").write_bytes(b"corrupt")
    elif damage == "manifest":
        (tmp_path / "manifest.json").unlink()
    else:
        next((tmp_path / "units" / "lambda").glob("*.gz")).unlink()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert lambda_layer_status(cfg, data, tmp_path, provenance=version)["status"] == "outdated"
    with pytest.raises(ValueError, match="outdated"):
        load_lambda_layer(cfg, data, tmp_path, provenance=version)
    with pytest.raises(ValueError, match="outdated"):
        write_lambda_layer(
            cfg, data, tmp_path, n_jobs=1, provenance=version, model_factory=fake_factory
        )
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_failed_export_resumes_c5_units_without_refitting_completed_seasons(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    cfg, version = config(), provenance(data)
    calls = []

    def fail_at_15(unit):
        calls.append(unit.season)
        if unit.season == 15:
            raise RuntimeError("Synthetic failure")
        return LoggedMean()

    with pytest.raises(RuntimeError, match="Synthetic failure"):
        write_lambda_layer(
            cfg, data, tmp_path, n_jobs=1, provenance=version, model_factory=fail_at_15
        )
    assert calls[-1] == 15
    assert len(calls) == len(set(calls))
    completed = set(calls[:-1])
    preserved = {p: p.read_bytes() for p in (tmp_path / "units" / "lambda").glob("unit-*")}
    assert len(preserved) == 2 * len(completed)
    assert not (tmp_path / "manifest.json").exists()
    assert lambda_layer_status(cfg, data, tmp_path, provenance=version)["status"] == "missing"
    calls.clear()

    def resume(unit):
        calls.append(unit.season)
        return LoggedMean()

    frame = write_lambda_layer(
        cfg, data, tmp_path, n_jobs=1, provenance=version, model_factory=resume
    )
    assert set(calls) == set(range(11, 26)) - completed
    assert len(calls) == len(set(calls))
    assert all(p.read_bytes() == raw for p, raw in preserved.items())
    validate_lambda_layer(frame, data)


@pytest.mark.parametrize("damage", ["target", "season", "date", "missing", "infinite"])
def test_layer_schema_rejects_targets_bad_boundaries_and_coverage(goal_sources, tmp_path, damage):
    data = prepare(goal_sources, ("is_home",))
    frame = write_lambda_layer(
        config(), data, tmp_path, n_jobs=1, provenance=provenance(data), model_factory=fake_factory
    )
    if damage == "target":
        frame["goals_scored"] = 1
    elif damage == "season":
        frame.loc[0, "train_end_season"] = frame.loc[0, "season_id"]
    elif damage == "date":
        frame.loc[0, "train_end_date"] = frame.loc[0, "as_of"]
    elif damage == "missing":
        frame = frame.iloc[1:]
    else:
        frame.loc[0, "lambda"] = np.inf
    with pytest.raises(ValueError, match="schema|coverage"):
        validate_lambda_layer(frame, data)


@pytest.mark.parametrize("policy", ["per_fold", "per_matchday"])
def test_default_judge_and_unit_identity_are_unchanged_without_export_mode(goal_sources, policy):
    data = prepare(goal_sources, ("is_home",))
    unit = GoalUnit("glm", asdict(GLMGoalConfig("r0")), ("is_home",), 11)
    assert "prior_seasons_only" not in unit.identity()
    assert unit.unit_id == replace(unit, prior_seasons_only=False).unit_id
    assert unit.unit_id != replace(unit, prior_seasons_only=True).unit_id
    assert replace(unit, prior_seasons_only=True).identity()["prior_seasons_only"] is True
    with pytest.raises(ValueError, match="boolean"):
        replace(unit, prior_seasons_only=1)

    def factory():
        return LoggedMean(policy)

    a = evaluate_team_goals(factory, data, feature_columns=("is_home",), blind=True)
    b = evaluate_team_goals(
        factory, data, feature_columns=("is_home",), blind=True, prior_seasons_only=False
    )
    pd.testing.assert_frame_equal(a.per_team, b.per_team)
    with pytest.raises(ValueError, match="boolean"):
        evaluate_team_goals(LoggedMean, data, feature_columns=("is_home",), prior_seasons_only=1)


@pytest.mark.slow
def test_parallel_layer_matches_serial_layer_including_manifest(goal_sources, tmp_path):
    data = prepare(goal_sources, ("is_home",))
    version, cfg = provenance(data), config()
    serial = write_lambda_layer(
        cfg, data, tmp_path / "one", n_jobs=1, provenance=version, model_factory=daily_factory
    )
    parallel = write_lambda_layer(
        cfg, data, tmp_path / "four", n_jobs=4, provenance=version, model_factory=daily_factory
    )
    pd.testing.assert_frame_equal(serial, parallel)
    for name in ("lambda.parquet", "manifest.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "four" / name).read_bytes()
