"""Complete, blind eight-configuration daily convergence audit for C4c."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from joblib import Parallel, delayed, parallel_config

from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.models.m2_maher import MaherGoalConfig, MaherGoalModel
from ligat.models.m2_maher_prior import load_promoted_prior, season_roster
from tests.test_m2_maher_prior import real_prior_inputs as real_prior_inputs


def audit_configuration(data, state, frozen, half_life, with_prior):
    records = []
    unknown = [0]

    class Audited(MaherGoalModel):
        def fit(self, *args, **kwargs):
            start = perf_counter()
            super().fit(*args, **kwargs)
            records.append({"iterations": self.iterations, "seconds": perf_counter() - start})
            return self

        def predict(self, X, *, context):
            result = super().predict(X, context=context)
            unknown[0] += self.unseen_team_rows
            return result

    cfg = MaherGoalConfig(half_life, frozen if with_prior else None)
    started = perf_counter()
    result = evaluate_team_goals(
        lambda: Audited(cfg, roster=state), data, feature_columns=data.X.columns, blind=True
    )
    rates = result.per_team["lambda"]
    iterations = np.array([r["iterations"] for r in records])
    assert np.isfinite(rates).all() and rates.gt(0).all()
    return {
        "half_life": half_life,
        "prior": with_prior,
        "fits": len(records),
        "failures": 0,
        "iterations_max": int(iterations.max()),
        "iterations_median": float(np.median(iterations)),
        "iterations_p99": float(np.quantile(iterations, 0.99)),
        "iterations": iterations.tolist(),
        "total_seconds": perf_counter() - started,
        "unseen_team_rows": unknown[0],
        "rows": len(rates),
    }


@pytest.mark.real_data
def test_all_daily_development_folds_converge_for_eight_configurations(real_prior_inputs):
    data, _ = real_prior_inputs
    state = season_roster(data)
    frozen = load_promoted_prior()
    expected = int(data.context.loc[data.context.season_id.ge(11), "as_of"].nunique())
    started = perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=8)(
            delayed(audit_configuration)(data, state, frozen, hl, p)
            for p in (False, True)
            for hl in (None, 0.5, 1, 2)
        )
    assert all(r["fits"] == expected and r["failures"] == 0 for r in results)
    all_iterations = np.concatenate([r.pop("iterations") for r in results])
    report = {
        "configurations": results,
        "fits": sum(r["fits"] for r in results),
        "failures": 0,
        "iterations_max": int(all_iterations.max()),
        "iterations_median": float(np.median(all_iterations)),
        "iterations_p99": float(np.quantile(all_iterations, 0.99)),
        "wall_seconds": perf_counter() - started,
        "all_lambda_positive_finite": True,
    }
    target = Path("data/reports/T-014/C4c_convergence.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


@pytest.mark.real_data
def test_first_day_entrants_use_their_season_league_reference(real_prior_inputs):
    data, _ = real_prior_inputs
    state = season_roster(data)
    frozen = load_promoted_prior()
    report = []
    for season in (15, 21, 25):
        as_of = state.start(season).tz_convert("Asia/Jerusalem")
        mask = data.context.date.lt(as_of)
        model = MaherGoalModel(MaherGoalConfig(1, frozen), roster=state).fit(
            data.X.loc[mask], data.y.loc[mask], context=data.context.loc[mask], as_of=as_of
        )
        params = model.parameters
        league = params.loc[list(state.teams(season))]
        promoted = params.loc[list(state.promoted(season))]
        d_alpha = promoted.log_alpha - league.log_alpha.mean() - frozen.a_star
        d_beta = promoted.log_beta - league.log_beta.mean() - frozen.b_star
        for team in state.promoted(season):
            count = int(data.X.loc[mask, "team_id"].eq(team).sum())
            a = float(params.loc[team, "log_alpha"] - league.log_alpha.mean())
            b = float(params.loc[team, "log_beta"] - league.log_beta.mean())
            report.append(
                {
                    "season": season,
                    "team_id": team,
                    "training_games": count,
                    "relative_log_alpha": a,
                    "a_star": frozen.a_star,
                    "relative_log_beta": b,
                    "b_star": frozen.b_star,
                    "reference_size": len(league),
                    "d_alpha": float(d_alpha.loc[team]),
                    "d_beta": float(d_beta.loc[team]),
                    "mean_promoted_d_alpha_over_R": float(d_alpha.sum() / len(league)),
                    "mean_promoted_d_beta_over_R": float(d_beta.sum() / len(league)),
                    "stationarity_applicable": count == 0,
                    "alpha_stationarity_residual": float(
                        d_alpha.loc[team] - d_alpha.sum() / len(league)
                    ),
                    "beta_stationarity_residual": float(
                        d_beta.loc[team] - d_beta.sum() / len(league)
                    ),
                    "other_promoted_alpha_over_R_minus_one": float(
                        d_alpha.drop(team).sum() / (len(league) - 1)
                    ),
                    "other_promoted_beta_over_R_minus_one": float(
                        d_beta.drop(team).sum() / (len(league) - 1)
                    ),
                }
            )
    target = Path("data/reports/T-014/C4c_reference.json")
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"season_reference": report}))
    for row in report:
        if row["training_games"] == 0:
            assert row["alpha_stationarity_residual"] == pytest.approx(0, abs=2e-5)
            assert row["beta_stationarity_residual"] == pytest.approx(0, abs=2e-5)
            assert row["d_alpha"] == pytest.approx(
                row["other_promoted_alpha_over_R_minus_one"], abs=2e-5
            )
            assert row["d_beta"] == pytest.approx(
                row["other_promoted_beta_over_R_minus_one"], abs=2e-5
            )
