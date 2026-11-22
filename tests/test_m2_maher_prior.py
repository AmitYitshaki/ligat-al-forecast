"""Day decay, identified promoted penalty derivatives and frozen-prior reconstruction."""

import json
from pathlib import Path
from time import perf_counter

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
    day_weights,
    maher_objective,
)
from ligat.models.m2_maher_prior import (
    PromotedPrior,
    SeasonRoster,
    estimate_promoted_prior,
    load_promoted_prior,
    season_roster,
    verify_promoted_prior,
)
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_m2_maher import synthetic_league


def prior():
    return PromotedPrior(-0.2, 0.2, "0" * 64, 1)


def roster():
    return SeasonRoster(
        tuple(
            (season, team, team == 11, f"{1998 + season}-08-01T00:00:00+03:00")
            for season in (12, 13, 14)
            for team in range(1, 12)
            if season != 13 or team != 11
        )
    )


def test_day_decay_manual_half_life_and_configuration():
    as_of = pd.Timestamp("2010-01-01", tz="Asia/Jerusalem")
    dates = pd.Series([as_of - pd.Timedelta(days=365.25), as_of])
    np.testing.assert_array_equal(day_weights(dates, as_of, 1), [0.5, 1])
    np.testing.assert_array_equal(day_weights(dates, as_of, None), [1, 1])
    assert MaherGoalConfig(0.5).config_id != MaherGoalConfig(1).config_id
    assert "1" in MaherGoalConfig(1).label
    with pytest.raises(ValueError):
        MaherGoalConfig(True)


@pytest.mark.parametrize("seed", range(4))
def test_penalty_gradient_includes_league_beta_mean(seed):
    rng = np.random.default_rng(seed)
    n = 7
    m = 180
    team = rng.integers(0, n, m)
    opp = rng.integers(0, n, m)
    home = rng.integers(0, 2, m).astype(float)
    goals = rng.poisson(1.3, m).astype(float)
    weights = rng.uniform(0.1, 1, m)
    theta = rng.normal(0, 0.1, 2 * n)
    mask = np.array([False, True, False, False, True, False, False])

    def objective(t):
        reference = np.array([True, True, True, False, True, False, False])
        return maher_objective(
            t, team, opp, home, goals, weights, n, mask, -0.2, 0.3, 7.2, reference
        )

    assert check_grad(lambda t: objective(t)[0], lambda t: objective(t)[1], theta) < 1e-5


def test_rookie_without_matches_has_exact_relative_prior_and_only_current_season_penalty(tmp_path):
    X, y, context, *_ = synthetic_league()
    model = MaherGoalModel(MaherGoalConfig(prior=prior()), roster=roster())
    for season, year in ((12, 2010), (13, 2011), (14, 2012)):
        model.fit(X, y, context=context, as_of=pd.Timestamp(year, 8, 1, tz="Asia/Jerusalem"))
        assert model.penalty_teams == ((11,) if season != 13 else ())
        if season != 13:
            params = model.parameters
            league = list(roster().teams(season))
            assert params.loc[11, "log_alpha"] - params.loc[
                league, "log_alpha"
            ].mean() == pytest.approx(prior().a_star, abs=2e-5)
            assert params.loc[11, "log_beta"] - params.loc[
                league, "log_beta"
            ].mean() == pytest.approx(prior().b_star, abs=2e-5)
            assert model.penalty_strength == pytest.approx(6 * y.mean())
        else:
            assert 11 not in model.parameters.index
    path = tmp_path / "prior.json"
    model.save(path)
    loaded = MaherGoalModel.load(path)
    pd.testing.assert_series_equal(
        model.predict(X, context=context), loaded.predict(X, context=context)
    )
    assert loaded.roster == model.roster and loaded.config == model.config
    assert 6 / (30 + 6) == pytest.approx(1 / 6)  # Q45: approximately 17%, not zero.


def test_after_thirty_real_games_prior_parameter_effect_is_small():
    X, y, context, a, b, gamma = synthetic_league()
    rng = np.random.default_rng(14432)
    rows = []
    for game in range(30):
        other = game % 10 + 1
        home = game % 2 == 0
        date = pd.Timestamp(2010, 8, 1, tz="Asia/Jerusalem") + pd.Timedelta(days=game, hours=12)
        for own, opp, h in ((11, other, home), (other, 11, not home)):
            log_a = 0.1 if own == 11 else a[own - 1]
            log_b = 0.3 if opp == 11 else b[opp - 1]
            rows.append(
                dict(
                    game_id=10000 + game,
                    team_id=own,
                    opponent_id=opp,
                    is_home=h,
                    date=date,
                    as_of=date.normalize(),
                    season_id=12,
                    goals_scored=int(rng.poisson(np.exp(log_a + log_b + gamma * h))),
                )
            )
    new = pd.DataFrame(rows).set_index(["game_id", "team_id"], drop=False)
    all_X = pd.concat([X, new.loc[:, MAHER_COLUMNS]])
    all_y = pd.concat([y, new.goals_scored])
    all_context = pd.concat([context, new.drop(columns="goals_scored")])
    as_of = pd.Timestamp(2010, 10, 1, tz="Asia/Jerusalem")
    regular = MaherGoalModel(MaherGoalConfig()).fit(all_X, all_y, context=all_context, as_of=as_of)
    penalized = MaherGoalModel(MaherGoalConfig(prior=prior()), roster=roster()).fit(
        all_X, all_y, context=all_context, as_of=as_of
    )
    assert np.max(np.abs(regular.parameters.loc[11] - penalized.parameters.loc[11])) < 0.15


def test_known_synthetic_season_prior_averages(goal_sources, monkeypatch):
    prematch, games = goal_sources
    prematch = prematch.copy()
    prematch["promoted_in"] = (
        prematch.index.get_level_values("team_id") == 1
    ) & prematch.season_id.eq(9)
    data = prepare_goal_data(
        prematch,
        games,
        feature_columns=(*MAHER_COLUMNS, "promoted_in"),
        seasons=tuple(range(8, 26)),
    )

    class Known:
        def __init__(self, config):
            pass

        def fit(self, X, y, *, context, as_of):
            self.parameters = pd.DataFrame(
                {"log_alpha": [-0.3, 0.1, 0.1, 0.1], "log_beta": [0.7, 0.3, 0.5, 0.5]},
                index=[1, 2, 3, 4],
            )
            return self

    monkeypatch.setattr("ligat.models.m2_maher.MaherGoalModel", Known)
    saved = estimate_promoted_prior(data)
    assert saved["prior"]["a_star"] == pytest.approx(-0.3)
    assert saved["prior"]["b_star"] == pytest.approx(0.2)
    assert saved["prior"]["team_seasons"] == 1


@pytest.fixture(scope="module")
def real_prior_inputs(real_tables):
    path = Path("data/derived/prematch/prematch_v2.parquet")
    if not path.exists():
        pytest.skip("Frozen M0 is local only")
    seasons = tuple(range(8, 26))
    frame = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    data = prepare_goal_data(
        frame,
        real_tables.match_table,
        feature_columns=(*MAHER_COLUMNS, "promoted_in"),
        seasons=seasons,
    )
    return data, frame


@pytest.mark.real_data
def test_frozen_prior_recomputes_identically(real_prior_inputs):
    data, _ = real_prior_inputs
    saved = yaml.safe_load(Path("config/m2_maher_prior.yaml").read_text(encoding="utf-8"))
    actual = verify_promoted_prior(data)
    assert actual == saved
    assert actual["prior"]["team_seasons"] == 37
    print(json.dumps({"frozen_prior": actual["prior"], "season8_excluded": True}))


@pytest.mark.real_data
def test_real_daily_decay_and_promoted_first_last_game_without_scores(real_prior_inputs):
    data, frame = real_prior_inputs
    state = season_roster(data)
    frozen = load_promoted_prior()
    reports = []
    for hl in (None, 0.5, 1, 2):
        cfg = MaherGoalConfig(hl, prior=frozen)
        started = perf_counter()
        result = evaluate_team_goals(
            lambda c=cfg: MaherGoalModel(c, roster=state),
            data,
            feature_columns=data.X.columns,
            policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
            blind=True,
        )
        elapsed = perf_counter() - started
        entry = {"half_life": hl, "seconds": elapsed}
        if hl == 1:
            rates = result.per_team["lambda"]
            home = result.per_team.is_home
            entry.update(
                lambda_min=float(rates.min()),
                lambda_max=float(rates.max()),
                lambda_mean=float(rates.mean()),
                home_mean=float(rates.loc[home].mean()),
                away_mean=float(rates.loc[~home].mean()),
            )
            promoted_rows = result.per_team.loc[data.X.promoted_in.reindex(result.per_team.index)]
            matches = []
            for team, part in promoted_rows.groupby(level="team_id"):
                ordered = part.assign(round_id=frame["round"].reindex(part.index)).sort_values(
                    ["round_id", "as_of"]
                )
                matches.append(
                    {
                        "team_id": int(team),
                        "first_round": int(ordered.iloc[0].round_id),
                        "last_round": int(ordered.iloc[-1].round_id),
                        "first_lambda": float(ordered.iloc[0]["lambda"]),
                        "last_lambda": float(ordered.iloc[-1]["lambda"]),
                    }
                )
            entry["promoted_first_last_game"] = matches
            assert np.isfinite(rates).all() and rates.gt(0).all()
        reports.append(entry)
    print(json.dumps({"maher_prior_blind_timings": reports}))
