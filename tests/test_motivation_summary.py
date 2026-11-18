"""Reader summary: every displayed number is checked, and technical names stay out of prose."""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from ligat.analysis import motivation_experiment_review as er
from ligat.analysis import motivation_summary as ms
from ligat.viz import motivation_summary as sv


@pytest.fixture
def display_claims():
    claims = defaultdict(lambda: (1.0, 0))
    claims.update(
        {
            "control_pct": (36.04, 2),
            "best_pct": (36.15, 2),
            "goals_book": ("05", None),
            "scores_book": ("06", None),
            "all_period": ("2009/2010–2023/2024", None),
        }
    )
    return claims


def test_summary_retains_all_draft_sections_and_has_no_technical_ids(display_claims):
    html = sv.reader_summary(display_claims)
    soup = BeautifulSoup(html, "html.parser")
    assert [h.get_text() for h in soup.find_all("h3")] == [
        "השורה התחתונה",
        "מושגים לסיכום הזה",
        "איך מדדנו מוטיבציה",
        "מה גילינו על העבר (חלק A)",
        "הניסוי: מה בדקנו (חלק B)",
        "מה יצא: השוואה בין הדרכים",
        "למה זה לא עבד",
        "איך יודעים שהתוצאה אמיתית",
        "מה עובר הלאה",
        "מה סגרנו, ומה לא נבדק",
    ]
    assert len(soup.find_all("table")) == 2
    assert len(soup.select("#bottom-line > ul > li")) == 4
    assert (
        soup.select_one("#bottom-line").decode()
        == BeautifulSoup(sv.bottom_line(display_claims), "html.parser").section.decode()
    )
    assert not ms.summary_text_violations(html)
    assert ms.verify_summary_numbers([html], display_claims).matches.all()


def test_every_number_in_opening_and_all_findings_is_registered(display_claims):
    fragments = [
        sv.bottom_line(display_claims),
        *[sv.findings(s, display_claims) for s in ("6.6", "7.5", "8.6")],
    ]
    assert ms.verify_summary_numbers(fragments, display_claims).matches.all()
    assert [
        len(BeautifulSoup(f, "html.parser").select("div > ul > li")) for f in fragments[1:]
    ] == [3, 4, 4]
    with pytest.raises(ValueError, match="Unknown"):
        sv.findings("9", display_claims)


@pytest.mark.parametrize(
    "fragment",
    [
        "<p>75 משחקים</p>",
        '<span data-claim="known">36.05</span>',
        '<span data-claim="missing">3</span>',
        '<span data-claim="known">36.04</span><span data-claim="known">36.05</span>',
    ],
)
def test_wrong_missing_or_unregistered_numbers_fail(fragment):
    with pytest.raises(AssertionError):
        ms.verify_summary_numbers([fragment], {"known": (36.04, 2)})


@pytest.mark.parametrize(
    "technical", ["L2-diff", "L3-diff-hl4", "L5-sides", "Lc-diff", "Comp-hl4", "C0", "-hl4"]
)
def test_summary_scanner_allows_only_the_identity_table_column(technical):
    assert ms.summary_text_violations(f"<p>{technical}</p>")
    assert not ms.summary_text_violations(
        f'<table><tr><td class="technical-identity">{technical}</td>'
        "<td>שלוש דרגות</td></tr></table>"
    )
    assert ms.summary_text_violations(f'<p class="technical-identity">{technical}</p>')


def test_claim_formatting_preserves_checked_dates_precision_and_escaping():
    assert ms.format_claim((36.041, 2)) == "36.04"
    assert ms.format_claim(("05", None)) == "05"
    assert "&lt;" in sv.claim_span({"test": ("<value>", None)}, "test")


def test_saved_summary_values_reproduce_reader_comparison_and_diagnostics(real_tables):
    review = er.load_experiment_review(Path.cwd(), real_tables)
    actual = ms.summary_values(
        review,
        {"flag_gap": (0.36, 2)},
        n_paths=2000,
        live_validation=pd.DataFrame({"loss_error": [0.0], "coefficient_error": [0.0]}),
        timing={"flag_mismatches": 0, "importance_as_of_mismatches": 0, "timing_mismatches": 0},
    )
    expected = {
        "configurations": "20",
        "ways": "5",
        "deep_count": "14",
        "recent_count": "5",
        "simulations": "2000",
        "control_pct": "36.04",
        "weighted_control_pct": "36.05",
        "three_pct": "36.14",
        "flag_pct": "36.09",
        "five_pct": "36.09",
        "continuous_pct": "36.06",
        "components_pct": "36.08",
        "best_pct": "36.15",
        "best_recent": "2",
        "noise_pass": "7",
        "recent_dead_games": "75",
        "almost_recent_games": "211",
        "dead_per_season": "15",
        "recent_games_per_season": "240",
        "flag_before": "1.42",
        "flag_after": "0.70",
        "expectations_correct": "2",
        "expectations_total": "3",
        "permutations": "200",
        "placebos_at_least_observed": "0",
        "live_error": "0",
        "timing_errors": "0",
        "flag_gap": "0.36",
    }
    assert {k: ms.format_claim(actual[k]) for k in expected} == expected
