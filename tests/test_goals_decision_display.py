"""Dated decisions are a presentation overlay, never edits to signed evidence."""

import copy

import numpy as np
import pytest

from ligat.viz import goals_configuration_names as names
from ligat.viz import goals_decision_display as decision
from tests.test_goals_walk_review import evidence as evidence


@pytest.mark.real_data
def test_opening_matches_signed_selected_model_and_does_not_mutate(evidence):
    report = evidence["report"]
    before = copy.deepcopy(report)
    html = decision.opening(report)
    for number in ["24.50%", "24.83%", "104.5%", "107.7%", "0.000822", "0.000358"]:
        assert number in html
    assert "11 מתוך 14" in html and "4 מתוך 5" in html
    assert "ה-ש30–ה-ש34" in html and "ל-M3 עוברים" in html
    assert report == before


@pytest.mark.real_data
def test_three_rule10_flags_match_approved_independent_reference(evidence):
    result = decision.rule10_table(evidence).set_index("stage")
    assert len(result) == 3
    np.testing.assert_allclose(
        result.loc["S1-AB", ["nll_a", "nll_z", "rps_a", "rps_z"]].to_numpy(float),
        [1.4075728006771366, 1.4084734773498917, 0.20912226003579235, 0.20904046136672175],
        atol=1e-12,
        rtol=0,
    )
    assert result.nll_gap.gt(0).all() and result.rps_gap.lt(0).all()
    assert result.decision.eq("נסגר לפי NLL (ה-ש34)").all()
    assert abs(result.loc["base-without-remaining", "nll_gap"] - 0.00032744892542551626) < 1e-12


def test_stages_and_obsolete_visible_wording():
    for stage in ["upper", "middle", "lower"]:
        assert stage not in names.configuration_name(stage)
    for text in ["טיוטה", "המודל הסופי", "Poisson מספיק"]:
        assert text not in decision.clean(text)


@pytest.mark.real_data
def test_selected_win_rate_plot_reads_f5_and_summary_uses_common_deep(evidence):
    from ligat.viz import goals_qa_graphs as graphs

    figure = graphs.win_rate_by_season(evidence)
    expected = [
        r["better_non_tie_percent"]
        for r in evidence["report"]["readable_metrics"]["F5"]["by_season"]
    ]
    np.testing.assert_array_equal(np.asarray(figure.data[0].y), expected)
    summary = decision.final_summary(evidence)
    assert "1.4020" in summary and "107.7%" in summary and "ה-ש34" in summary
