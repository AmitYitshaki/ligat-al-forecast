"""Auditable numerical claims for notebook 04's standalone reader summary."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from bs4 import BeautifulSoup

from ligat.analysis.motivation_experiment_review import (
    GROUPS,
    ExperimentReview,
    entry_counts,
    group_decomposition,
    paired_games,
)
from ligat.evaluate.motivation_experiment import (
    DEEP_SEASONS,
    VALIDATION_SEASONS,
    load_motivation_experiment,
    motivation_configurations,
)
from ligat.viz.seasons import season_label

Claim = tuple[float | str, int | None]
SUMMARY_FORBIDDEN = re.compile(r"L2-|L3-|L5-|Lc-|Comp-|C0|-hl4")


def format_claim(claim: Claim) -> str:
    """Format a registered number at its declared precision, or preserve a checked text value."""
    value, precision = claim
    return str(value) if precision is None else f"{float(value):.{precision}f}"


def summary_values(
    review: ExperimentReview,
    a_metrics: Mapping[str, tuple[float, int]],
    *,
    n_paths: int,
    live_validation: pd.DataFrame | None = None,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Claim]:
    """Compute claims from saved results, shared definitions and unchanged A estimates."""
    summary = review.frames["summary"].set_index("config_id")
    configurations = motivation_configurations()
    recent_years = tuple(s for s in DEEP_SEASONS if s >= 21)
    evaluation = load_motivation_experiment().raw["evaluation"]
    values: dict[str, Claim] = dict(a_metrics)
    values.update(
        {
            "chapter": (9, 0),
            "goals_book": ("05", None),
            "scores_book": ("06", None),
            "configurations": (len(summary), 0),
            "ways": (len({c.name.split("-")[0] for c in configurations if c.name != "C0"}), 0),
            "scalar_ways": (
                len({c.name.split("-")[0] for c in configurations if "-" in c.name}),
                0,
            ),
            "forms": (len({c.name.split("-")[1] for c in configurations if "-" in c.name}), 0),
            "weightings": (len({c.half_life for c in configurations}), 0),
            "objectives": (len(next(c.columns for c in configurations if c.name == "Comp")), 0),
            "recent_count": (len(recent_years), 0),
            "deep_count": (len(DEEP_SEASONS), 0),
            "required_recent": (4, 0),
            "shallow_training": (min(VALIDATION_SEASONS) - evaluation["training_start"], 0),
            "simulations": (n_paths, 0),
            "almost_threshold_pct": (1, 0),
            "uniform_pct": (100 / 3, 1),
            "all_period": (
                f"{season_label(min(VALIDATION_SEASONS))}–{season_label(max(VALIDATION_SEASONS))}",
                None,
            ),
            "deep_period": (
                f"{season_label(min(DEEP_SEASONS))}–{season_label(max(DEEP_SEASONS))}",
                None,
            ),
            "recent_period": (
                f"{season_label(min(recent_years))}–{season_label(max(recent_years))}",
                None,
            ),
            "first_period": (
                f"{season_label(min(DEEP_SEASONS))}–{season_label(min(recent_years) - 1)}",
                None,
            ),
            "shallow_season": (season_label(min(VALIDATION_SEASONS)), None),
        }
    )
    names = {
        "control": "C0",
        "weighted_control": "C0-hl4",
        "flag": "L2-diff",
        "three": "L3-diff",
        "five": "L5-diff",
        "continuous": "Lc-diff",
        "components": "Comp",
        "best": "L3-diff-hl4",
    }
    for key, config_id in names.items():
        values[key + "_pct"] = (100 * float(np.exp(-summary.loc[config_id, "all15_log_loss"])), 2)
        if key not in ("control", "weighted_control"):
            values[key + "_better"] = (float(summary.loc[config_id, "log_loss_folds_better"]), 0)
            values[key + "_recent"] = (
                float(summary.loc[config_id, "log_loss_last5_folds_better"]),
                0,
            )
    if summary.all15_log_loss.idxmin() != names["best"]:
        raise AssertionError("Reader summary best-configuration description no longer holds")
    for representation in ("L2", "L3", "L5", "Lc"):
        for suffix in ("", "-hl4"):
            if (
                summary.loc[representation + "-sides" + suffix, "all15_log_loss"]
                < summary.loc[representation + "-diff" + suffix, "all15_log_loss"]
            ):
                raise AssertionError("Reader summary separate-sides comparison no longer holds")
    values["probability_gain_pp"] = (
        float(values["best_pct"][0]) - float(values["control_pct"][0]),
        2,
    )
    counts = entry_counts(review.frames["summary"])
    values["noise_pass"] = (counts["either"], 0)
    values["entered"] = (int(summary.log_loss_passes_rule8.sum()), 0)
    dead = paired_games(review, "L2-diff")
    deep = dead.loc[dead.season_id.isin(DEEP_SEASONS) & dead.group.isin(list(GROUPS)[:3])]
    recent = deep.loc[deep.season_id.isin(recent_years)]
    values["recent_dead_games"] = (len(recent), 0)
    values["recent_dead_improvement"] = (float(recent.log_loss_improvement.mean()), 3)
    values["dead_per_season"] = (len(deep) / len(DEEP_SEASONS), 0)
    folds = review.frames["fold_scores"]
    values["recent_games_per_season"] = (
        float(
            folds.loc[folds.config_id.eq("C0") & folds.season_id.isin(recent_years)].games.mean()
        ),
        0,
    )
    decomposition = group_decomposition(review).loc[lambda x: x.metric.eq("log_loss")]
    for key, c, period, group in (
        ("dead_almost_flag", "L2-diff", "deep", "dead_almost"),
        ("dead_almost_three", "L3-diff", "deep", "dead_almost"),
        ("almost_recent", "L3-diff", "last5", "almost"),
        ("almost_first", "L3-diff", "first9", "almost"),
    ):
        row = decomposition.loc[
            decomposition.config_id.eq(c)
            & decomposition.period.eq(period)
            & decomposition.group.eq(group)
        ].iloc[0]
        values[key + "_games"] = (float(row.games), 0)
        values[key + "_improvement"] = (float(row.mean_improvement), 3)
    coefficients = (
        review.frames["coefficients"].set_index(["config_id", "season_id", "coefficient"]).value
    )
    for key, c, season, coefficient, precision in (
        ("flag_before", "L2-diff", min(DEEP_SEASONS), "L2_m_diff", 2),
        ("flag_after", "L2-diff", max(DEEP_SEASONS), "L2_m_diff", 2),
        ("elo_without", "C0", max(DEEP_SEASONS), "b_elo", 1),
        ("elo_with", "Comp", max(DEEP_SEASONS), "b_elo", 1),
        ("title_weight", "Comp", max(DEEP_SEASONS), "Comp_title_diff", 1),
    ):
        values[key] = (float(coefficients.loc[(c, season, coefficient)]), precision)
    effects = (
        review.frames["expected_points"]
        .loc[lambda x: x.config_id.eq("L2-diff") & x.fit_scope.eq("last_fold") & x.elo_diff.eq(0)]
        .set_index("side")
    )
    for side in ("home", "away"):
        values["dead_" + side + "_points"] = (
            abs(float(effects.loc[side, "expected_points_change"])),
            2,
        )
    placebos = review.frames["audit_L2-diff_placebos"]
    observed = review.pilot["audit_L2-diff"]["observed"]
    values["permutations"] = (len(placebos), 0)
    values["placebos_at_least_observed"] = (
        int(placebos.log_loss_improvement.ge(observed).sum()),
        0,
    )
    controls = review.metadata["baseline_verification"]["controls"]
    values["baseline_error"] = (
        max(float(c["max_probability_error"]) for c in controls.values()),
        0,
    )
    values["live_error"] = (
        float(live_validation[["loss_error", "coefficient_error"]].to_numpy().max())
        if live_validation is not None
        else 0.0,
        0,
    )
    audit = timing if timing is not None else review.pilot["timing_audit"]
    values["timing_errors"] = (
        sum(
            int(audit[k])
            for k in ("flag_mismatches", "importance_as_of_mismatches", "timing_mismatches")
        ),
        0,
    )
    expectations = (
        summary.loc["L2-diff", "log_loss_last5_folds_better"] < 4,
        all(
            summary.loc[c, "log_loss_mean_improvement"]
            <= summary.loc["L2-diff", "log_loss_mean_improvement"]
            for c in ("L3-diff", "L5-diff", "Lc-diff", "Comp")
        ),
        not summary.log_loss_passes_rule8.any(),
    )
    values["expectations_correct"] = (sum(bool(v) for v in expectations), 0)
    values["expectations_total"] = (len(expectations), 0)
    return values


def verify_summary_numbers(fragments: Sequence[str], claims: Mapping[str, Claim]) -> pd.DataFrame:
    """Check every occurrence, reject unregistered digits and report the exact source claim."""
    rows = []
    for section, fragment in enumerate(fragments):
        soup = BeautifulSoup(fragment, "html.parser")
        for span in soup.select("[data-claim]"):
            key = str(span["data-claim"])
            if key not in claims:
                raise AssertionError(f"Unregistered summary claim: {key}")
            displayed = span.get_text().replace("−", "-")
            expected = format_claim(claims[key])
            if displayed != expected:
                raise AssertionError(
                    f"Summary numerical mismatch: {key}: {displayed} != {expected}"
                )
            rows.append(
                dict(
                    section=section,
                    claim=key,
                    displayed=displayed,
                    calculated=claims[key][0],
                    matches=True,
                )
            )
            span.decompose()
        for tag in soup.select(".reference"):
            tag.decompose()
        text = re.sub(
            r"(?:EXP-\d+|T-\d+|E\d+|M[0-6]|ה-[מכ]\d+)", "", soup.get_text(" ", strip=True)
        )
        if re.search(r"\d", text):
            raise AssertionError(f"Unregistered number in summary section {section}: {text}")
    return pd.DataFrame(rows)


def summary_text_violations(fragment: str) -> list[str]:
    """Technical configuration names are allowed only in an explicitly labelled identity cell."""
    soup = BeautifulSoup(fragment, "html.parser")
    for cell in soup.select("table td.technical-identity"):
        cell.decompose()
    return SUMMARY_FORBIDDEN.findall(soup.get_text(" ", strip=True))
