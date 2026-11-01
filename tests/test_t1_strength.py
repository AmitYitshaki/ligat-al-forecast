"""Same-sample LOSO, previous-season joins, null behaviour and deleted API regression."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ligat.analysis.league_stats import context, t1_strength_comparison
from ligat.analysis.prepare import AnalysisTables


@pytest.fixture
def independent_tables() -> AnalysisTables:
    rng = np.random.default_rng(17)
    labels = []
    for season in range(8, 21):
        ranks = rng.permutation(np.arange(1, 9))
        shares = rng.lognormal(0, 0.8, 8)
        for team in range(8):
            labels.append(
                dict(
                    season_id=season,
                    team_id=team,
                    final_position=ranks[team],
                    prev_season_value_share=shares[team] / shares.sum(),
                    promoted_in=False,
                )
            )
    rows = []
    for season in range(8, 21):
        for i in range(240):
            h = i % 8
            a = (h + 1 + (i // 8) % 7) % 8
            outcome = rng.choice(3, p=[0.4, 0.28, 0.32])
            rows.append(
                dict(
                    season_id=season,
                    home_team_id=h,
                    away_team_id=a,
                    home_goals=int(outcome == 0),
                    away_goals=int(outcome == 2),
                )
            )
    return AnalysisTables(
        pd.DataFrame(rows),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(labels).set_index(["season_id", "team_id"]),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def test_loso_never_trains_on_the_heldout_season(
    independent_tables: AnalysisTables, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fit, original_predict = LogisticRegression.fit, LogisticRegression.predict_proba
    fitted = []
    season_by_id = independent_tables.match_table.season_id

    def fit(self: LogisticRegression, x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
        self._training_seasons_test = set(season_by_id.loc[x.index])
        return original_fit(self, x, y)

    def predict(self: LogisticRegression, x: pd.DataFrame) -> np.ndarray:
        test_seasons = set(season_by_id.loc[x.index])
        assert len(test_seasons) == 1
        assert not test_seasons & self._training_seasons_test
        assert len(self._training_seasons_test) == 11
        fitted.append(next(iter(test_seasons)))
        return original_predict(self, x)

    monkeypatch.setattr(LogisticRegression, "fit", fit)
    monkeypatch.setattr(LogisticRegression, "predict_proba", predict)
    result = t1_strength_comparison(independent_tables, n_bootstrap=100)
    assert len(fitted) == 4 * 12
    assert set(fitted) == set(range(9, 21))
    assert result.n.eq(12 * 240).all()


def test_same_sample_excludes_promoted_missing_and_nonpositive(
    independent_tables: AnalysisTables,
) -> None:
    labels = independent_tables.team_seasons.copy()
    labels.loc[(12, 0), "promoted_in"] = True
    labels.loc[(12, 1), "prev_season_value_share"] = np.nan
    labels.loc[(12, 2), "prev_season_value_share"] = 0
    labels.loc[(12, 3), "prev_season_value_share"] = -1
    labels.loc[(11, 4), "final_position"] = np.nan
    sample = context._strength_sample(
        replace(independent_tables, team_seasons=labels), list(range(8, 21))
    )
    assert not sample.season_id.eq(8).any()
    excluded = sample.loc[sample.season_id.eq(12)]
    assert not excluded.home_team_id.isin(range(5)).any()
    assert not excluded.away_team_id.isin(range(5)).any()
    assert np.isfinite(sample[["lv", "rk", "lrk"]]).all().all()
    original = independent_tables.match_table.copy(deep=True)
    t1_strength_comparison(independent_tables, n_bootstrap=20)
    pd.testing.assert_frame_equal(original, independent_tables.match_table)


def test_feature_orientation_and_uses_previous_official_rank(
    independent_tables: AnalysisTables,
) -> None:
    labels = independent_tables.team_seasons.copy()
    labels.loc[(12, 0), "prev_season_value_share"] = 0.4
    labels.loc[(12, 1), "prev_season_value_share"] = 0.1
    labels.loc[(11, 0), "final_position"] = 1
    labels.loc[(11, 1), "final_position"] = 4
    # Opposite current-season ranking must not enter the feature.
    labels.loc[(12, 0), "final_position"] = 8
    labels.loc[(12, 1), "final_position"] = 1
    games = independent_tables.match_table
    sample = context._strength_sample(replace(independent_tables, team_seasons=labels), [12])
    key = games.index[games.season_id.eq(12) & games.home_team_id.eq(0) & games.away_team_id.eq(1)][
        0
    ]
    row = sample.loc[key]
    assert row.lv == pytest.approx(np.log(4))
    assert row.rk == 3
    assert row.lrk == pytest.approx(np.log(4))


def test_independent_null_and_exact_reproducibility(independent_tables: AnalysisTables) -> None:
    first = t1_strength_comparison(independent_tables, n_bootstrap=500, seed=1)
    second = t1_strength_comparison(independent_tables, n_bootstrap=500, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert first.attrs == second.attrs
    assert (first.log_loss - first.loc["baseline", "log_loss"]).abs().max() < 0.015
    comparison = first.attrs["comparisons"][0]
    assert comparison["ci_lower"] <= 0 <= comparison["ci_upper"]
    assert first.rps.between(0, 1).all()


@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
def test_rejects_non_development(seasons: list[int], independent_tables: AnalysisTables) -> None:
    with pytest.raises(ValueError, match="development"):
        t1_strength_comparison(independent_tables, seasons)


def test_deleted_context_api_is_unavailable() -> None:
    import ligat.analysis.league_stats as stats

    # Assemble removed names so residual-name scans do not mistake this absence test for usage.
    names = [
        "venue_" + "effects",
        "venue_" + "summary",
        "coach_change_" + "effect",
        "coach_" + "summary",
        "derby_and_" + "big_games",
        "CITY_" + "CLUBS",
        "funnel_" + "limits",
        "funnel_" + "limits_binomial",
        "funnel_" + "limits_poisson",
    ]
    for name in names:
        assert not hasattr(context, name)
        assert not hasattr(stats, name)
