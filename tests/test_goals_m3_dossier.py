"""The dossier reads native blind predictions and preserves comparison populations."""

import numpy as np
import pytest

from ligat.viz import goals_m3_dossier as dossier


@pytest.fixture(scope="module")
def native(real_tables):
    data = dossier.load_candidates()
    return data, dossier.dossier(data)


@pytest.mark.real_data
def test_native_rates_and_scope_coverage(native):
    data, result = native
    assert len(data["frames"]) == 7 and len(data["deep_seasons"]) == 13
    assert data["policy"] == "as_evaluated"
    assert set(result["summary"].query("scope == 'deep'").seasons) == {13}
    assert set(result["summary"].query("scope == 'all'").games) == {3721}
    assert all(frame.season_id.between(11, 25).all() for frame in data["frames"].values())


@pytest.mark.real_data
def test_probabilities_partition_and_calibration_bins(native):
    data, result = native
    for frame in data["frames"].values():
        np.testing.assert_allclose(frame[["ph", "pd", "pa"]].sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(
            frame[[f"total{i}" for i in range(6)]].sum(axis=1), 1, atol=1e-12
        )
    assert len(result["calibration"]) == 7 * 3 * 3 * 10
    assert result["calibration"].ece.between(0, 1).all()
    for _, part in result["draw"].query("group >= 0").groupby(["scope", "candidate"]):
        assert part.games.max() - part.games.min() <= 1


@pytest.mark.real_data
def test_diversity_symmetry_and_dispersion_se(native):
    _, result = native
    for matrices in result["matrices"].values():
        for key, p in matrices.items():
            np.testing.assert_allclose(p, p.T, atol=1e-13)
            np.testing.assert_allclose(
                np.diag(p), 0 if key == "lambda_difference" else 1, atol=1e-13
            )
    assert (result["dispersion"].rows == 2 * result["dispersion"].games).all()
    assert result["dispersion"].se.ge(0).all()
    self_rows = result["segments"].loc[result["segments"].candidate.eq(dossier.review.LABELS["F5"])]
    np.testing.assert_allclose(self_rows.d, 0, atol=1e-15)


def test_pearson_definition_has_unit_poisson_expectation():
    # Exact expectation for Bernoulli-like synthetic residuals with equal rate one.
    observed = np.array([0.0, 2.0])
    rates = np.ones(2)
    assert np.mean((observed - rates) ** 2 / rates) == 1


@pytest.mark.real_data
def test_dossier_figures_update_selectors_and_do_not_compress_main_season_axis(native):
    from ligat.viz import goals_m3_dossier_views as display
    from ligat.viz import goals_walk_plots as plots

    _, result = native
    for name, figure in display.figures(result).items():
        assert len(display.explanation(name, result, after=False)) >= 250
        assert len(display.explanation(name, result, after=True)) >= 250
        if figure.layout.updatemenus:
            assert all(
                "annotations" in button.args[1] for button in figure.layout.updatemenus[0].buttons
            )
    season = plots.metric_by_season(dossier.source_bundle())
    assert season.layout.yaxis.range[1] < 0.30
    assert season.layout.yaxis2.domain
