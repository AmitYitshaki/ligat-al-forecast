"""R3 descriptive calculations from saved prematch and experiment outputs only."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import poisson  # type: ignore[import-untyped]

from ligat.analysis.prematch_b_review import WORDS
from ligat.analysis.prematch_form_review import (
    SELECTED_P,
    FormReview,
    candidate_table,
    draw_diagnostics,
    opening_coefficients,
    poisson_coefficients,
)
from ligat.evaluate.goal_metrics import poisson_outcomes


def poisson_example(rate: float = 1.5) -> pd.DataFrame:
    """Translate a fixed goal expectation into probabilities and per-result penalties."""
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Expected a finite positive Poisson rate")
    return pd.DataFrame(
        [
            {
                "שערים": goals,
                "probability": float(poisson.pmf(goals, rate)),
                "nll": float(-poisson.logpmf(goals, rate)),
                WORDS: f"אם הובקעו {goals} שערים, הקנס הוא מינוס לוג ההסתברות של התוצאה הזו",
            }
            for goals in range(5)
        ]
    )


def memory_curves() -> pd.DataFrame:
    """Ages count completed appearances; an abrupt five-game window is normalized."""
    rows = []
    for age in range(1, 36):
        for half_life in (5, 10, 20):
            alpha = 1 - 0.5 ** (1 / half_life)
            rows.append(
                {
                    "age": age,
                    "method": f"EWMA {half_life}",
                    "weight": alpha * (1 - alpha) ** (age - 1),
                }
            )
        rows.append({"age": age, "method": "חלון 5", "weight": 0.2 if age <= 5 else 0.0})
    return pd.DataFrame(rows)


def venue_prior_calendar(prematch: pd.DataFrame) -> pd.DataFrame:
    """Use the median saved home-season length in the 14-team era, excluding technicals."""
    frame = prematch.reset_index()
    if not frame.season_id.between(8, 25).all():
        raise ValueError("Only development prematch rows are allowed")
    counts = (
        frame.loc[frame.is_home & frame.season_id.ge(14)].groupby(["season_id", "team_id"]).size()
    )
    if counts.empty:
        raise ValueError("Need saved home appearances in the 14-team era")
    typical = float(counts.median())
    return pd.DataFrame(
        [
            {
                "נקודת זמן": label,
                "venue_games": games,
                "approx_rounds": 2 * games,
                "prior_mass": 0.5 ** (games / 5),
                WORDS: explanation,
            }
            for label, games, explanation in (
                ("חציית משקל הפתיחה", 5.0, "חמש הופעות באותו מגרש הן בערך עשרה מחזורי ליגה"),
                ("רבע ממשקל הפתיחה", 10.0, "עשר הופעות באותו מגרש הן בערך עשרים מחזורים"),
                (
                    "סוף עונה טיפוסית",
                    typical,
                    f"חציון {typical:g} משחקי בית לקבוצה־עונה בעידן 14; סדר הלוח אינו תמיד חלופי",
                ),
            )
        ]
    )


def promoted_opening_summary(review: FormReview) -> pd.DataFrame:
    """Arrange the last saved season's nine priors by metric and venue, without changing them."""
    source = opening_coefficients(review).loc[lambda x: x.season_id.eq(25)]
    rows = []
    for metric, word in (("gf", "הבקעה"), ("ga", "ספיגה"), ("pts", "נקודות")):
        subset = source.loc[source.metric.eq(metric)].set_index("stratum")
        row: dict[str, Any] = {"metric": metric}
        for stratum in ("all", "home", "away"):
            row[f"league_{stratum}"] = float(subset.at[stratum, "league_mean"])
            row[f"promoted_{stratum}"] = float(subset.at[stratum, "promoted_value"])
        difference = row["promoted_all"] - row["league_all"]
        row[WORDS] = f"{word}: עולה מתחילה בפער {difference:+.2f} למשחק מממוצע הליגה בכל המגרשים"
        rows.append(row)
    return pd.DataFrame(rows)


def readable_candidates(review: FormReview) -> pd.DataFrame:
    """Group the 22 saved P candidates by memory family, retaining their original ranks."""
    data = candidate_table(review, "P").copy()
    families = {
        "ewma": "דעיכה",
        "w": "חלון",
        "std": "כל העונה",
        "prev": "עונה קודמת",
        "hist": "כל העבר",
    }
    data["memory_family"] = data.candidate.map(
        lambda c: families[c.split("_")[0].rstrip("0123456789")]
    )
    order = {name: i for i, name in enumerate(families.values())}
    data["family_order"] = data.memory_family.map(order)
    data[WORDS] = [
        ("✅ נבחר: " if row.config_id == SELECTED_P else "") + row[WORDS]
        for _, row in data.iterrows()
    ]
    return data.sort_values(["family_order", "rank_all"]).drop(columns="family_order")


def coefficient_effects(review: FormReview) -> pd.DataFrame:
    """Saved P coefficients, goal multipliers and +100 raw home-minus-away Elo effects.

    Keep the predeclared gf=1.5 baseline, other goal rates at the training mean,
    and raw Elo=0. Raising the raw gap gives +100 to the home's perspective and
    -100 to the away perspective. The existing normalized 0..10 grid is used.
    """
    frame = poisson_coefficients(review)
    fits = review.fits.loc[review.fits.config_id.eq(SELECTED_P)].set_index("season_id")
    rows = []
    for row in frame.itertuples():
        c, gamma, a, d, b = np.asarray(fits.at[row.season_id, "coefficients"], float)
        home = np.exp(c + gamma + a * np.log(1.6) + d * np.log(row.training_mean + 0.1))
        away = np.exp(c + (a + d) * np.log(row.training_mean + 0.1))
        multiplier = np.exp(b / 4)
        outcomes, _ = poisson_outcomes([[home, away], [home * multiplier, away / multiplier]])
        rows.append(
            {
                "season_id": row.season_id,
                "c": c,
                "gamma": gamma,
                "a": a,
                "d": d,
                "b": b,
                "home_rate_percent": 100 * np.expm1(gamma),
                "elo_rate_percent": 100 * (multiplier - 1),
                "elo_win_pp": 100 * (outcomes[1, 0] - outcomes[0, 0]),
                "attack_change_percent": row.attack_change_percent,
                "defense_change_percent": row.defense_change_percent,
                "home_win_change_pp": row.home_win_change_pp,
                WORDS: (
                    f"בית מוסיף {100 * np.expm1(gamma):.2f}% לשערים; "
                    f"+100 Elo מוסיפים {100 * (multiplier - 1):.2f}%"
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_uncertainty(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Descriptive binomial rate SEs and exact-score count noise; no new model fit.

    Tertile SE uses observed p*(1-p)/n. Exact counts use sum q_i*(1-q_i)
    from the saved per-game grid, assuming independent games for this illustration.
    """
    outcomes, strata, exact, _ = draw_diagnostics(predictions)
    strata["difference_pp"] = 100 * (strata.observed - strata.predicted)
    strata["se_pp"] = 100 * np.sqrt(strata.observed * (1 - strata.observed) / strata.games)
    strata["descriptive_z"] = strata.difference_pp / strata.se_pp
    strata[WORDS] = [
        f"{row['סוג משחק']}: פער {row['difference_pp']:+.2f} נ״א, מול SE תיאורי {row['se_pp']:.2f}"
        for row in strata.to_dict("records")
    ]
    outcomes["difference_pp"] = 100 * (outcomes.observed - outcomes.predicted)
    outcomes[WORDS] = [
        f"{row['תוצאה']}: בפועל פחות חזוי {row['difference_pp']:+.2f} נ״א; חיובי הוא חוסר במודל"
        for row in outcomes.to_dict("records")
    ]
    rates = predictions[["lambda_home", "lambda_away"]].to_numpy(float)
    retained = poisson.cdf(10, rates[:, 0]) * poisson.cdf(10, rates[:, 1])
    exact["expected_count"] = exact.predicted * len(predictions)
    exact["observed_count"] = (exact.observed * len(predictions)).round().astype(int)
    exact["excess_count"] = exact.observed_count - exact.expected_count
    noises = []
    for goals in range(4):
        probabilities = poisson.pmf(goals, rates[:, 0]) * poisson.pmf(goals, rates[:, 1]) / retained
        noises.append(float(np.sqrt(np.sum(probabilities * (1 - probabilities)))))
    exact["count_noise"] = noises
    exact[WORDS] = [
        f"{row['תוצאה']}: עודף {row['excess_count']:.1f} משחקים; "
        f"סטיית תקן תיאורית {row['count_noise']:.1f}"
        for row in exact.to_dict("records")
    ]
    return outcomes, strata, exact
