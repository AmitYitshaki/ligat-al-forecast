"""Season-relative entrant penalty, old-format rejection and blind convergence audit."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from ligat.models.m2_maher import MaherGoalConfig, MaherGoalModel
from ligat.models.m2_maher_prior import SeasonRoster
from tests.test_m2_maher import synthetic_league
from tests.test_m2_maher_prior import prior


def current_roster():
    return SeasonRoster(
        tuple((12, t, t == 11, "2010-08-01T00:00:00+03:00") for t in (*range(1, 7), 11))
    )


def with_weak_old_club():
    X, y, context, *_ = synthetic_league()
    rows = []
    for game in range(30):
        opponent = game % 6 + 1
        date = pd.Timestamp(2006, 8, 1, tz="Asia/Jerusalem") + pd.Timedelta(days=game, hours=12)
        for own, opp, home, goals in (
            (12, opponent, game % 2 == 0, int(game % 5 == 0)),
            (opponent, 12, game % 2 != 0, 3 + game % 2),
        ):
            rows.append(
                dict(
                    game_id=10000 + game,
                    team_id=own,
                    opponent_id=opp,
                    is_home=home,
                    date=date,
                    as_of=date.normalize(),
                    season_id=8,
                    goals_scored=goals,
                )
            )
    extra = pd.DataFrame(rows).set_index(["game_id", "team_id"], drop=False)
    return (X, y, context), (
        pd.concat([X, extra.loc[:, X.columns]]),
        pd.concat([y, extra.goals_scored]),
        pd.concat([context, extra.loc[:, context.columns]]),
    )


def test_unplayed_entrant_relative_to_season_is_independent_of_weak_retired_club():
    roster = current_roster()
    groups = with_weak_old_club()
    values = []
    for X, y, context in groups:
        model = MaherGoalModel(MaherGoalConfig(prior=prior()), roster=roster).fit(
            X, y, context=context, as_of=pd.Timestamp(2010, 8, 1, tz="Asia/Jerusalem")
        )
        p = model.parameters
        league = list(t for s, t, _, _ in roster.rows if s == 12)
        values.append(
            (
                float(p.loc[11, "log_alpha"] - p.loc[league, "log_alpha"].mean()),
                float(p.loc[11, "log_beta"] - p.loc[league, "log_beta"].mean()),
            )
        )
    print(json.dumps({"entrant_relative_before_after_old_club": values}))
    np.testing.assert_allclose(values[0], values[1], rtol=0, atol=2e-5)
    np.testing.assert_allclose(values[1], [prior().a_star, prior().b_star], rtol=0, atol=2e-5)


def test_no_prior_predictions_match_before_c4c_bitwise():
    X, y, context, *_ = synthetic_league()
    model = MaherGoalModel(MaherGoalConfig()).fit(
        X, y, context=context, as_of=context.as_of.max() + pd.Timedelta(days=1)
    )
    # Frozen C4b arithmetic, with its original cap, independent of the new penalty.
    teams = np.union1d(X.team_id, X.opponent_id)
    n = len(teams)
    ti = np.searchsorted(teams, X.team_id)
    oi = np.searchsorted(teams, X.opponent_id)
    home = X.is_home.to_numpy(float)
    goals = y.to_numpy(float)
    weights = np.ones(len(y))

    def legacy(t):
        a = np.r_[t[: n - 1], -t[: n - 1].sum()]
        b = t[n - 1 : 2 * n - 1]
        eta = a[ti] + b[oi] + t[-1] * home
        rates = np.exp(eta)
        residual = weights * (rates - goals)
        ga = np.bincount(ti, weights=residual, minlength=n)
        gb = np.bincount(oi, weights=residual, minlength=n)
        return float(weights @ (rates - goals * eta)) / weights.sum(), np.r_[
            ga[:-1] - ga[-1], gb, residual @ home
        ] / weights.sum()

    initial = np.zeros(2 * n)
    initial[n - 1 : 2 * n - 1] = np.log(np.average(goals, weights=weights))
    with threadpool_limits(limits=1):
        old = minimize(
            legacy,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-13},
        )
    assert old.success
    a = np.r_[old.x[: n - 1], -old.x[: n - 1].sum()]
    b = old.x[n - 1 : 2 * n - 1]
    np.testing.assert_array_equal(
        model.predict(X, context=context), np.exp(a[ti] + b[oi] + old.x[-1] * home)
    )


def test_two_promoted_clubs_observed_and_unplayed_satisfy_coupled_stationarity():
    X, y, context, *_ = synthetic_league()
    state = SeasonRoster(
        tuple((12, t, t in (1, 11), "2010-08-01T00:00:00+03:00") for t in range(1, 12))
    )
    model = MaherGoalModel(MaherGoalConfig(prior=prior()), roster=state).fit(
        X, y, context=context, as_of=pd.Timestamp(2010, 8, 1, tz="Asia/Jerusalem")
    )
    params = model.parameters
    league = params.loc[list(state.teams(12))]
    assert X.team_id.eq(1).any() and not X.team_id.eq(11).any()
    for column, target in (("log_alpha", prior().a_star), ("log_beta", prior().b_star)):
        residual = params.loc[list(state.promoted(12)), column] - league[column].mean() - target
        assert residual.loc[11] == pytest.approx(residual.sum() / len(league), abs=2e-5)
        assert residual.loc[11] == pytest.approx(residual.loc[1] / (len(league) - 1), abs=2e-5)


def test_old_maher_format_is_rejected_clearly(tmp_path):
    X, y, context, *_ = synthetic_league()
    model = MaherGoalModel(MaherGoalConfig()).fit(
        X, y, context=context, as_of=context.as_of.max() + pd.Timedelta(days=1)
    )
    path = tmp_path / "old.json"
    model.save(path)
    saved = json.loads(path.read_text())
    saved["payload"]["format_version"] = 2
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="format_version 2; expected 4"):
        MaherGoalModel.load(path)
