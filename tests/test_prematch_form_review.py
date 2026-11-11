"""Saved-only B2 diagnostics: weights, paired contrasts, joins and goal probabilities."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from ligat.analysis.prematch_b_review import (
    WORDS,
    EloReview,
    load_elo_review,
    top_configuration_contrasts,
)
from ligat.analysis.prematch_form_review import (
    EXP002_PREDICTIONS_DIRECTORY,
    SELECTED_P,
    FormReview,
    candidate_description,
    candidate_table,
    control_numbers,
    draw_diagnostics,
    ewma_illustrations,
    haifa_season,
    load_form_review,
    poisson_coefficients,
)
from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.goal_metrics import poisson_outcomes
from ligat.viz.prematch_b import calibration_chart, marginal_chart
from ligat.viz.prematch_form import haifa_chart


def empty_review(**overrides):
    defaults = dict(
        form=pd.DataFrame(),
        elo=pd.DataFrame(),
        metadata={},
        identities=pd.DataFrame(),
        p_rankings=pd.DataFrame(),
        o_rankings=pd.DataFrame(),
        folds=pd.DataFrame(),
        fits=pd.DataFrame(),
        predictions=pd.DataFrame(),
        team_predictions=pd.DataFrame(),
    )
    return FormReview(**(defaults | overrides))


def test_ewma_age_one_and_last_five_mass_are_separate_from_prior():
    weights, priors = ewma_illustrations()
    weights = weights.set_index("מה נמדד")
    priors = priors.set_index("שיטה")
    for half_life in (5, 10, 20):
        method = f"EWMA {half_life}"
        alpha = 1 - 0.5 ** (1 / half_life)
        assert weights.loc["משחק 1", method] == pytest.approx(alpha)
        assert weights.loc["משחק 5", method] == pytest.approx(alpha * 0.5 ** (4 / half_life))
        mass = weights.loc["חלק מחמשת האחרונים", method]
        assert mass == pytest.approx(sum(alpha * (1 - alpha) ** j for j in range(5)))
        assert mass + priors.loc[method, "אחרי 5 משחקים"] == pytest.approx(1)
    assert weights.loc["חלק מחמשת האחרונים", "EWMA 5"] == pytest.approx(0.5)
    assert weights.loc["משחק 10", "חלון 5"] == 0
    assert priors.loc["חלון 5", "אחרי 5 משחקים"] == 0
    assert WORDS in weights and WORDS in priors


@pytest.mark.parametrize(
    "candidate",
    [
        "w5",
        "w10",
        "w20",
        "ewma5",
        "ewma10",
        "ewma20",
        "std_m3",
        "std_m6",
        "std_m10",
        "prev",
        "hist",
    ],
)
def test_every_declared_candidate_explains_its_memory_and_venue(candidate):
    all_words, _ = candidate_description(candidate, "all")
    venue_words, _ = candidate_description(candidate, "venue")
    assert "כל המגרשים" in all_words
    assert "בנפרד" in venue_words
    assert len(all_words) > len(candidate)


def test_unknown_candidate_is_not_silently_labelled():
    with pytest.raises(ValueError):
        candidate_description("future_window", "all")


def test_paired_improvement_uses_weights_for_mean_but_unweighted_se():
    seasons = list(range(11, 26))
    differences = np.array([0.02, -0.01, 0.03] * 5)
    weights = np.arange(1, 16) * 2
    reference = pd.DataFrame(
        {"config_id": "P_reference_elo", "season_id": seasons, "rows": weights, "poisson_nll": 1.5}
    )
    candidate = reference.assign(config_id="P_ewma5_all_elo", poisson_nll=1.5 - differences)
    rankings = pd.DataFrame({"config_id": ["P_ewma5_all_elo"], "rank_all": [1]})
    identities = pd.DataFrame(
        {"config_id": ["P_ewma5_all_elo"], "candidate": ["ewma5"], "variant": ["all"]}
    )
    review = empty_review(
        folds=pd.concat([candidate, reference]), p_rankings=rankings, identities=identities
    )
    result = candidate_table(review, "P").iloc[0]
    expected_mean = np.average(differences, weights=weights)
    expected_se = differences.std(ddof=1) / np.sqrt(15)
    assert result.improvement == pytest.approx(expected_mean)
    assert result.improvement_se == pytest.approx(expected_se)
    assert result.z == pytest.approx(expected_mean / expected_se)
    assert result.seasons_better == 10
    bad = review.folds.loc[
        ~((review.folds.config_id == "P_ewma5_all_elo") & (review.folds.season_id == 11))
    ]
    with pytest.raises(ValueError, match="same folds"):
        candidate_table(replace(review, folds=bad), "P")


def test_haifa_join_uses_conceded_goals_and_only_each_venues_own_matches():
    index = pd.MultiIndex.from_arrays([[1, 2, 3, 4], [1005] * 4], names=["game_id", "team_id"])
    elo = pd.DataFrame({"as_of": pd.date_range("2024-01-01", periods=4)}, index=index)
    form = pd.DataFrame(
        {
            "gf_venue_ewma5": [1.5] * 4,
            "ga_venue_ewma5": [0.4] * 4,
            "gf_all_ewma20": [1.4] * 4,
            "ga_all_ewma20": [0.8] * 4,
        },
        index=index,
    )
    rows = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "team_id": [1005] * 4,
            "season_id": [25] * 4,
            "is_home": [True, False, True, False],
            "goals_scored": [2, 1, 1, 0],
        }
    )
    games = pd.DataFrame(
        {"game_id": [1, 2, 3, 4], "home_goals": [2, 3, 1, 2], "away_goals": [0, 1, 2, 0]}
    )
    joined = haifa_season(
        empty_review(form=form, elo=elo, predictions=games, team_predictions=rows)
    )
    assert joined.goals_against.tolist() == [0, 3, 2, 2]
    chart = haifa_chart(joined, "ga")
    assert [len(trace.x) for trace in chart.data] == [2, 2, 4, 2, 2]
    assert list(chart.data[3].y) == [0, 2]


def test_draw_tables_use_normalized_exact_scores_and_disjoint_tertiles():
    rates = np.column_stack([1 + np.arange(9) / 10, np.ones(9)])
    probabilities, _ = poisson_outcomes(rates)
    home = np.array([0, 1, 2, 0, 1, 2, 3, 1, 4])
    away = np.array([0, 0, 2, 1, 1, 0, 3, 2, 1])
    frame = pd.DataFrame(
        {
            "lambda_home": rates[:, 0],
            "lambda_away": rates[:, 1],
            "p_home": probabilities[:, 0],
            "p_draw": probabilities[:, 1],
            "p_away": probabilities[:, 2],
            "home_goals": home,
            "away_goals": away,
            "outcome": np.where(home > away, 0, np.where(home == away, 1, 2)),
        }
    )
    outcomes, strata, scores, totals = draw_diagnostics(frame)
    assert outcomes.predicted.sum() == pytest.approx(1)
    assert outcomes.observed.sum() == pytest.approx(1)
    assert strata.games.tolist() == [3, 3, 3]
    assert np.average(strata.predicted, weights=strata.games) == pytest.approx(
        probabilities[:, 1].mean()
    )
    retained = poisson.cdf(10, rates[:, 0]) * poisson.cdf(10, rates[:, 1])
    expected_zero = np.mean(np.exp(-rates.sum(axis=1)) / retained)
    assert scores.predicted.iloc[0] == pytest.approx(expected_zero)
    assert scores.ratio.iloc[0] == pytest.approx((1 / 9) / expected_zero)
    assert totals.predicted.iloc[0] == pytest.approx(rates.sum(axis=1).mean())


def test_calibration_encodes_sample_size_and_hollow_small_bins():
    bins = pd.DataFrame(
        {
            "outcome": ["בית", "בית", "תיקו", "חוץ"],
            "games": [10, 100, 20, 50],
            "predicted": [0.2, 0.5, 0.1, 0.4],
            "observed": [0.1, 0.5, 0, 0.4],
        }
    )
    figure = calibration_chart(bins)
    points = figure.data[1]
    assert list(points.marker.size) == [10, 100]
    assert list(points.marker.symbol) == ["circle-open", "circle"]
    assert list(points.customdata[:, 1]) == ["מדגם קטן", ""]


def test_integer_marginal_labels_do_not_change_means():
    frame = pd.DataFrame(
        {
            "parameter": ["k", "k"],
            "value": ["10.0", "20.0"],
            "mean_log_loss": [1.02, 1.01],
            "best_log_loss": [1.0, 1.0],
            "configs": [320, 320],
        }
    )
    chart = marginal_chart(frame)
    assert list(chart.data[0].x) == ["10", "20"]
    assert list(chart.data[0].y) == [1.02, 1.01]


@pytest.fixture
def relocated_review(tmp_path, monkeypatch):
    """Small real Parquets with original manifest names and derived predictions."""
    directory = tmp_path / "reports/experiments/EXP-002/full"
    directory.mkdir(parents=True)
    derived = tmp_path / EXP002_PREDICTIONS_DIRECTORY
    derived.mkdir(parents=True)
    index = pd.MultiIndex.from_tuples([(1, 1005), (1, 2000)], names=["game_id", "team_id"])
    form_path = tmp_path / "data/derived/form/form_v1.parquet"
    form_path.parent.mkdir(parents=True)
    pd.DataFrame({"gf_all_w5": [1.0, 2.0]}, index=index).to_parquet(form_path)
    elo_path = form_path.with_name("elo.parquet")
    pd.DataFrame({"season_id": [11, 11]}, index=index).to_parquet(elo_path)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    form_metadata = form_path.with_suffix(".metadata.json")
    form_metadata.write_text(
        json.dumps({"seasons": list(range(8, 26)), "artifact_sha256": digest(form_path)})
    )
    monkeypatch.setattr(
        "ligat.analysis.prematch_form_review.FORM_METADATA_SHA256", digest(form_metadata)
    )
    for name in ("fold_scores.parquet", "fits.parquet"):
        pd.DataFrame({"season_id": [11]}).to_parquet(directory / name)
    for name in ("comparison_P.csv", "comparison_O.csv"):
        pd.DataFrame({"config_id": [SELECTED_P]}).to_csv(directory / name, index=False)
    (directory / "configurations.json").write_text(
        json.dumps([{"config_id": SELECTED_P, "candidate": None}])
    )
    for name, count in (("predictions.parquet", 2), ("team_predictions.parquet", 4)):
        pd.DataFrame(
            {"season_id": [11] * (count + 1), "config_id": [SELECTED_P] * count + ["other"]}
        ).to_parquet(derived / name)
    files = {p.name: {"sha256": digest(p)} for p in directory.iterdir()}
    files.update({p.name: {"sha256": digest(p)} for p in derived.iterdir()})
    manifest = {
        "experiment_id": "EXP-002",
        "stage": "full",
        "completed": True,
        "inputs": {
            name: {"path": path.relative_to(tmp_path).as_posix(), "sha256": digest(path)}
            for name, path in (("form", form_path), ("elo", elo_path))
        },
        "files": files,
    }
    (directory / "metadata.json").write_text(json.dumps(manifest))
    return tmp_path


def test_load_form_review_reads_verified_predictions_from_derived(relocated_review):
    review = load_form_review(relocated_review)
    assert len(review.predictions) == 2
    assert len(review.team_predictions) == 4
    assert review.predictions.config_id.eq(SELECTED_P).all()
    assert review.form.index.equals(review.elo.index)
    directory = relocated_review / "reports/experiments/EXP-002/full"
    assert not (directory / "predictions.parquet").exists()
    assert not (directory / "team_predictions.parquet").exists()


@pytest.mark.parametrize("name", ["predictions.parquet", "team_predictions.parquet"])
@pytest.mark.parametrize("change", ["corrupt", "missing"])
def test_relocated_prediction_failure_stops_before_dataframe_read(
    relocated_review, monkeypatch, name, change
):
    path = relocated_review / EXP002_PREDICTIONS_DIRECTORY / name
    if change == "corrupt":
        path.write_bytes(b"modified")
    else:
        path.unlink()
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("read before checksum"))
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("read before checksum"))
    error = ValueError if change == "corrupt" else FileNotFoundError
    with pytest.raises(error):
        load_form_review(relocated_review)


def test_corrupt_experiment_stops_before_dataframe_read(relocated_review, monkeypatch):
    directory = relocated_review / "reports/experiments/EXP-002/full"
    (directory / "fold_scores.parquet").write_bytes(b"changed")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("read before checksum"))
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("read before checksum"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_form_review(relocated_review)


@pytest.mark.parametrize("name", ["fits.parquet", "comparison_P.csv", "../predictions.parquet"])
def test_non_prediction_manifest_member_cannot_use_alternate_location(relocated_review, name):
    directory = relocated_review / "reports/experiments/EXP-002/full"
    alternate = relocated_review / "alternate"
    alternate.mkdir()
    original = directory / name
    if original.is_file():
        original.rename(alternate / original.name)
    with pytest.raises(ValueError, match="Only explicit EXP-002 prediction filenames"):
        verify_calibration(
            directory / "metadata.json", file_locations={name: alternate / original.name}
        )


def test_relocation_cannot_add_an_unlisted_prediction_file(relocated_review):
    manifest = relocated_review / "reports/experiments/EXP-002/full/metadata.json"
    metadata = json.loads(manifest.read_text())
    del metadata["files"]["predictions.parquet"]
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="members of the saved manifest"):
        load_form_review(relocated_review)


def test_default_verification_does_not_search_for_relocated_files(relocated_review):
    manifest = relocated_review / "reports/experiments/EXP-002/full/metadata.json"
    with pytest.raises(FileNotFoundError):
        verify_calibration(manifest)


@pytest.mark.parametrize(("experiment", "stage"), [("EXP-001", "stage1"), ("EXP-002", "pilot")])
def test_alternate_locations_are_restricted_to_full_exp002(relocated_review, experiment, stage):
    manifest = relocated_review / "reports/experiments/EXP-002/full/metadata.json"
    metadata = json.loads(manifest.read_text())
    metadata.update(experiment_id=experiment, stage=stage)
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Only full EXP-002"):
        verify_calibration(
            manifest,
            file_locations={
                "predictions.parquet": relocated_review
                / EXP002_PREDICTIONS_DIRECTORY
                / "predictions.parquet"
            },
        )


def test_saved_controls_and_selected_coefficient_translation_without_refitting(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/derived/form/form_v1.parquet").exists():
        pytest.skip("Saved form/experiment artifacts are local only")
    monkeypatch.setattr(
        "ligat.models.form_models.fit_poisson", lambda *a, **k: pytest.fail("unexpected fit")
    )
    monkeypatch.setattr(
        "ligat.features.form.FormBlock.compute",
        lambda *a, **k: pytest.fail("unexpected form replay"),
    )
    review = load_form_review(root)
    controls = control_numbers(review).iloc[0]
    assert controls["P Elo בלבד"] == pytest.approx(1.410383, abs=5e-7)
    assert controls["P ללא Elo"] == pytest.approx(1.457037, abs=5e-7)
    assert controls["נבחר NLL"] == pytest.approx(1.409455, abs=5e-7)
    assert controls["נבחר z"] == pytest.approx(2.946413, abs=5e-6)
    assert controls["O Elo בלבד"] == pytest.approx(1.020423847325146, abs=1e-9)
    assert controls["מועמדי כושר משפרים"] == 0
    assert controls["ρ התקפה"] == pytest.approx(0.67696758687618)
    assert controls["ρ הגנה"] == pytest.approx(0.509820812102664)
    assert controls["ρ נקודות"] == pytest.approx(0.76198002092634)
    assert controls["תיקו חזוי (%)"] == pytest.approx(25.486218, abs=5e-7)
    coefficients = poisson_coefficients(review)
    assert coefficients.attack_change_percent.median() == pytest.approx(1.39408573)
    assert coefficients.home_win_change_pp.median() == pytest.approx(0.50229760)
    assert coefficients.defense_change_percent.median() == pytest.approx(0.874438, abs=5e-7)
    assert len(coefficients) == 15


def test_top_ten_contrasts_keep_common_settings_verified():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/derived/elo/exp001_predictions_stage1.parquet").exists():
        pytest.skip("Saved Elo artifacts are local only")
    review = load_elo_review(root)
    compact = top_configuration_contrasts(review)
    assert len(compact) == 10 and "מה מייחד אותה" in compact
    assert compact.log_loss.tolist() == review.tables["top20"].log_loss.head(10).tolist()
    bad_tables = {**review.tables, "top20": review.tables["top20"].assign(k=20)}
    with pytest.raises(ValueError, match="common settings"):
        top_configuration_contrasts(
            EloReview(bad_tables, review.predictions, review.prematch, review.execution)
        )
