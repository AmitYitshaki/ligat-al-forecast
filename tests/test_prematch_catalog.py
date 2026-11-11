"""The reader catalog covers the frozen schema and never invents model roles."""

from dataclasses import replace

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from ligat.analysis.prematch_catalog import EXPLANATIONS, prematch_feature_overview
from ligat.features.pipeline import PREMATCH_COLUMNS, TABLE_COLUMNS
from ligat.features.registry import FEATURES, IDENTIFICATION_COLUMNS
from ligat.viz.prematch import dead_games_chart, review_table


def test_catalog_covers_exactly_the_51_frozen_columns() -> None:
    assert len(PREMATCH_COLUMNS) == 51
    assert set(EXPLANATIONS) == set(PREMATCH_COLUMNS)
    catalog = prematch_feature_overview()
    assert catalog.feature.tolist() == list(PREMATCH_COLUMNS)
    assert catalog.notna().all().all()
    assert catalog.words.str.len().gt(10).all()
    assert catalog.notes.str.len().gt(10).all()
    assert not catalog.to_string().count("ה-ס")
    assert not catalog.to_string().count("ה-ת")
    assert not set(catalog) & {"block", "availability", "decision", "model_status"}
    assert set(catalog.group) == {
        "טבלה",
        "Elo",
        "כושר הקבוצה",
        "כושר היריבה",
        "הקשר עונתי",
        "הפרשים",
    }


def test_table_subset_is_identical_to_the_same_rows_of_full_catalog() -> None:
    expected = prematch_feature_overview().set_index("feature").loc[list(TABLE_COLUMNS)]
    actual = prematch_feature_overview(TABLE_COLUMNS).set_index("feature")
    assert len(actual) == 30
    pd.testing.assert_frame_equal(actual, expected)


def test_catalog_uses_current_roles_and_per_model_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for row in prematch_feature_overview().itertuples():
        for model in ("M1", "M2", "M4", "M5"):
            if row.feature in IDENTIFICATION_COLUMNS:
                assert getattr(row, model) == "—"
            else:
                spec = FEATURES[row.feature]
                expected = (
                    dict(spec.model_status).get(model, spec.status) if model in spec.models else "—"
                )
                assert getattr(row, model) == expected
    # Explicitly exercise the override, rather than comparing copied constants.
    monkeypatch.setitem(
        FEATURES, "elo_pre", replace(FEATURES["elo_pre"], model_status=(("M1", "✅"),))
    )
    assert prematch_feature_overview(["elo_pre"]).iloc[0].M1 == "✅"
    assert prematch_feature_overview(["nothing_to_play"]).iloc[0].M2 == "🔬"
    assert prematch_feature_overview(["prev_value_share_avg"]).iloc[0].M2 == "✅"


@pytest.mark.parametrize(
    "columns", [["referee"], ["goals_for", "goals_for"], ["prev_value_share_total"]]
)
def test_unknown_duplicate_or_archived_columns_cannot_enter_catalog(columns: list[str]) -> None:
    with pytest.raises(ValueError, match="unique, explicitly documented"):
        prematch_feature_overview(columns)


def test_unregistered_feature_cannot_be_silently_documented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(FEATURES, "goals_for")
    with pytest.raises(ValueError, match="Unregistered"):
        prematch_feature_overview(["goals_for"])


def test_review_table_formats_absent_rules_and_fractional_ranks_without_mutation() -> None:
    source = pd.DataFrame(
        {
            "playoff_format": ["none", "upper: 6 / 10"],
            "relegation_test": [None, 14],
            "upper_cut": [None, 6],
            "sporting_position": [3.0, 3.5],
        }
    )
    before = source.copy(deep=True)
    soup = BeautifulSoup(review_table(source)._repr_html_(), "html.parser")
    rows = [[cell.get_text() for cell in row.select("td")] for row in soup.select("tbody tr")]
    assert rows[0] == ["—", "—", "—", "3"]
    assert rows[1][-1] == "3.5"
    pd.testing.assert_frame_equal(source, before)


def test_dead_chart_omits_middle_only_in_presentation_and_preserves_other_points() -> None:
    source = pd.DataFrame(
        {
            "season_id": [14] * 4,
            "stage": ["regular", "upper", "middle", "lower"],
            "at_least_one_share": [0.01, 0.2, 1.0, 0.1],
            "games": [182, 30, 6, 28],
            "one": [1, 4, 0, 2],
            "both": [0, 2, 6, 1],
        }
    )
    before = source.copy(deep=True)
    figure = dead_games_chart(source)
    assert len(figure.data) == 3
    assert {float(trace.y[0]) for trace in figure.data} == {1.0, 20.0, 10.0}
    assert {trace.line.color for trace in figure.data} == {"#0072B2", "#D55E00", "#009E73"}
    assert not figure.layout.yaxis.range
    pd.testing.assert_frame_equal(source, before)
