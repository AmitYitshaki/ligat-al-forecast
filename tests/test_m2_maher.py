"""Identified Maher recovery, derivatives, chronology and blind daily real folds."""

import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.optimize import check_grad

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import prepare_goal_data
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.models.m2_maher import (
    MAHER_COLUMNS,
    MaherGoalConfig,
    MaherGoalModel,
    maher_objective,
    unpack_parameters,
)
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare


def synthetic_league():
    rng = np.random.default_rng(14401)
    a = rng.normal(0, 0.2, 10)
    a -= a.mean()
    b = rng.normal(0.1, 0.15, 10)
    gamma = np.log(1.25)
    rows = []
    game = 0
    for season in range(8, 12):
        for _repeat in range(2):
            for home in range(10):
                for away in range(10):
                    if home == away:
                        continue
                    game += 1
                    date = pd.Timestamp(1998 + season, 8, 1, tz="Asia/Jerusalem") + pd.Timedelta(
                        days=(game % 180) // 2
                    )
                    for own, opp, is_home in ((home, away, True), (away, home, False)):
                        rows.append(
                            dict(
                                game_id=game,
                                team_id=own + 1,
                                opponent_id=opp + 1,
                                is_home=is_home,
                                season_id=season,
                                date=date,
                                as_of=date.normalize(),
                                goals_scored=int(
                                    rng.poisson(np.exp(a[own] + b[opp] + gamma * is_home))
                                ),
                            )
                        )
    frame = pd.DataFrame(rows).set_index(["game_id", "team_id"], drop=False)
    return (
        frame.loc[:, MAHER_COLUMNS],
        frame.goals_scored,
        frame.drop(columns=["goals_scored"]),
        a,
        b,
        gamma,
    )


def test_identified_parameters_recover_known_ten_team_four_season_league(tmp_path):
    X, y, context, a, b, gamma = synthetic_league()
    as_of = context.as_of.max() + pd.Timedelta(days=1)
    model = MaherGoalModel(MaherGoalConfig()).fit(X, y, context=context, as_of=as_of)
    params = model.parameters
    np.testing.assert_allclose(params.log_alpha, a, rtol=0, atol=0.3)
    np.testing.assert_allclose(params.log_beta, b, rtol=0, atol=0.3)
    assert model.log_gamma == pytest.approx(gamma, abs=0.15)
    assert abs(params.log_alpha.sum()) < 1e-14
    own = X.team_id.to_numpy() - 1
    opp = X.opponent_id.to_numpy() - 1
    eta = a[own] + b[opp] + gamma * X.is_home.to_numpy()
    np.testing.assert_allclose(
        eta, (a + 0.7)[own] + (b - 0.7)[opp] + gamma * X.is_home.to_numpy(), atol=1e-15
    )
    rates = model.predict(X, context=context)
    path = tmp_path / "maher.json"
    model.save(path)
    loaded = MaherGoalModel.load(path)
    pd.testing.assert_series_equal(rates, loaded.predict(X, context=context))
    saved = json.loads(path.read_text())
    saved["payload"]["log_gamma"] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="signature"):
        MaherGoalModel.load(path)


@pytest.mark.parametrize("seed", range(4))
def test_analytic_gradient_and_unit_weights(seed):
    rng = np.random.default_rng(seed)
    n = 6
    m = 200
    team = rng.integers(0, n, m)
    opp = rng.integers(0, n, m)
    home = rng.integers(0, 2, m).astype(float)
    goals = rng.poisson(1.3, m).astype(float)
    weights = rng.uniform(0.2, 1, m)
    theta = rng.normal(0, 0.1, 2 * n)

    def objective(t, w=weights):
        return maher_objective(t, team, opp, home, goals, w, n)

    assert check_grad(lambda t: objective(t)[0], lambda t: objective(t)[1], theta) < 1e-5
    x, g = objective(theta, np.ones(m))
    u, h = objective(theta, np.full(m, 1.0))
    assert x == u
    np.testing.assert_array_equal(g, h)
    assert abs(unpack_parameters(theta, n)[0].sum()) < 1e-14


def test_unknown_team_gets_average_and_is_recorded_and_external_weights_rejected():
    X, y, context, *_ = synthetic_league()
    as_of = context.as_of.max() + pd.Timedelta(days=1)
    model = MaherGoalModel(MaherGoalConfig()).fit(X, y, context=context, as_of=as_of)
    unknown = X.iloc[:2].copy()
    unknown["team_id"] = 99
    unknown["opponent_id"] = 100
    rates = model.predict(unknown, context=context.iloc[:2])
    expected = np.exp(model.parameters.log_beta.mean() + model.log_gamma * unknown.is_home)
    np.testing.assert_array_equal(rates, expected)
    assert model.unseen_team_rows == 2 and model.unseen_teams == (99,)
    with pytest.raises(ValueError, match="external weights"):
        model.fit(X, y, pd.Series(1.0, index=X.index), context=context, as_of=as_of)


def test_failed_optimization_and_missing_reference_date_fail_closed(monkeypatch):
    X, y, context, *_ = synthetic_league()
    as_of = context.as_of.max() + pd.Timedelta(days=1)
    model = MaherGoalModel(MaherGoalConfig())
    with pytest.raises(ValueError, match="explicit"):
        model.fit(X, y, context=context)
    with pytest.raises(ValueError, match="strictly earlier"):
        model.fit(X, y, context=context, as_of=context.as_of.min())
    monkeypatch.setattr(
        "ligat.models.m2_maher.minimize",
        lambda *a, **k: SimpleNamespace(success=False, message="test failure"),
    )
    with pytest.raises(RuntimeError, match="as_of=.*test failure"):
        model.fit(X, y, context=context, as_of=as_of)
    with pytest.raises(RuntimeError):
        model.predict(X, context=context)


def test_daily_judge_passes_reference_date_and_never_same_day_results(goal_sources):
    data = prepare(goal_sources, MAHER_COLUMNS)
    models = []

    class Spy(MaherGoalModel):
        def fit(self, X, y, sample_weight=None, *, context, as_of=None, prediction_features=None):
            assert context.date.lt(as_of).all()
            assert set(prediction_features.index).isdisjoint(X.index)
            self.reference = as_of
            self.training_max = context.date.max()
            return super().fit(
                X,
                y,
                sample_weight,
                context=context,
                as_of=as_of,
                prediction_features=prediction_features,
            )

    def factory():
        model = Spy(MaherGoalConfig())
        models.append(model)
        return model

    result = evaluate_team_goals(
        factory,
        data,
        feature_columns=MAHER_COLUMNS,
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
        blind=True,
    )
    assert len(models) == result.per_fold.fits.sum()
    assert result.per_team["lambda"].gt(0).all()
    assert all(m.training_max < m.reference for m in models)
    assert "poisson_nll" not in result.per_team


@pytest.mark.real_data
def test_one_real_daily_season_reports_parameters_and_rates_without_scores(real_tables):
    path = Path("data/derived/prematch/prematch_v2.parquet")
    if not path.exists():
        pytest.skip("Frozen prematch is local only")
    seasons = tuple(range(8, 26))
    prematch = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    data = prepare_goal_data(
        prematch, real_tables.match_table, feature_columns=MAHER_COLUMNS, seasons=seasons
    )
    reports = []
    models = []

    class Timed(MaherGoalModel):
        def fit(self, X, y, sample_weight=None, *, context, as_of=None, prediction_features=None):
            start = perf_counter()
            super().fit(
                X,
                y,
                sample_weight,
                context=context,
                as_of=as_of,
                prediction_features=prediction_features,
            )
            reports.append((perf_counter() - start, float(y.mean())))
            return self

    def factory():
        model = Timed(MaherGoalConfig())
        models.append(model)
        return model

    started = perf_counter()
    result = evaluate_team_goals(
        factory,
        data,
        feature_columns=MAHER_COLUMNS,
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
        blind=True,
    )
    total = perf_counter() - started
    rates = result.per_team["lambda"]
    home = result.per_team.is_home
    assert rates.gt(0).all() and np.isfinite(rates).all()
    final = models[-1]
    params = final.parameters
    teams = yaml.safe_load(Path("config/teams.yaml").read_text(encoding="utf-8"))
    names = {row["team_id"]: row["name_he"] for row in teams["teams"]}
    print(
        json.dumps(
            {
                "maher_no_decay_engineering": {
                    "fits": len(models),
                    "total_seconds": total,
                    "fit_seconds_total": sum(t for t, _ in reports),
                    "fit_seconds_mean": float(np.mean([t for t, _ in reports])),
                    "lambda_min": float(rates.min()),
                    "lambda_max": float(rates.max()),
                    "lambda_mean": float(rates.mean()),
                    "training_goal_mean_average": float(np.mean([m for _, m in reports])),
                    "home_mean": float(rates.loc[home].mean()),
                    "away_mean": float(rates.loc[~home].mean()),
                    "last_home_multiplier": float(np.exp(final.log_gamma)),
                    "top_attack": [names[int(t)] for t in params.log_alpha.nlargest(3).index],
                    "best_defense": [names[int(t)] for t in params.log_beta.nsmallest(3).index],
                    "unseen_team_rows": sum(m.unseen_team_rows for m in models),
                    "team_config_type": type(teams).__name__,
                }
            }
        )
    )
