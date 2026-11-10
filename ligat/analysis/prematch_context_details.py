"""Small, descriptive R4 diagnostics on saved development artifacts; never fit a model."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prematch_context_review import WORDS


def _development(frame: pd.DataFrame) -> None:
    if frame.season_id.isna().any() or not set(frame.season_id) <= set(range(8, 26)):
        raise ValueError("Context details require development seasons 8–25")


def robust_value_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    """Equal-season average-minus-total Spearman contrast, with and without source season 10."""
    _development(paired)
    rows = []
    for omit, label in ((False, "כל עונות השווי"), (True, "בלי דף 2008/2009")):
        d = paired.loc[paired.season_id.ne(10) if omit else slice(None), "improvement"]
        if len(d) < 2 or not np.isfinite(d).all():
            raise ValueError("At least two finite paired season differences are required")
        mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        rows.append(
            dict(
                method=label,
                mean=mean,
                se=se,
                seasons=len(d),
                descriptive_z=mean / se if se else np.nan,
                **{WORDS: f"{label}: ממוצע השווי מוסיף {mean:.3f} בספירמן; {len(d)} עונות"},
            )
        )
    return pd.DataFrame(rows)


def source_value_medians(values: pd.DataFrame) -> pd.DataFrame:
    """Median source total over clubs in the two named pages, in million euros."""
    _development(values)
    rows = []
    for season in (9, 10):
        sample = values.loc[values.season_id.eq(season), "total_value_eur"].dropna()
        if sample.empty:
            raise ValueError("Missing source page for median comparison")
        median = float(sample.median()) / 1e6
        rows.append(
            dict(
                season_id=season,
                total_million=median,
                clubs=len(sample),
                **{WORDS: f"חצי מ-{len(sample)} השווים הידועים מתחת ל-{median:.2f} מיליון אירו"},
            )
        )
    return pd.DataFrame(rows)


def promoted_dispersion(seasons: pd.DataFrame) -> pd.DataFrame:
    """Observed sporting ppm across promotion events; sample SD uses ddof=1."""
    _development(seasons)
    observed = seasons.loc[seasons.promoted_in, "ppm"].astype(float)
    if len(observed) < 2 or not np.isfinite(observed).all():
        raise ValueError("At least two finite observed promotion events are required")
    quantities = [
        ("ממוצע", observed.mean(), "קצב הנקודות הממוצע של העולות בפועל"),
        (
            "סטיית תקן",
            observed.std(ddof=1),
            "פיזור קצב הנקודות בין עולות בודדות, ביחידות נקודות למשחק",
        ),
        ("מינימום", observed.min(), "העולה בעלת קצב הנקודות הנמוך ביותר במדגם"),
        ("מקסימום", observed.max(), "העולה בעלת קצב הנקודות הגבוה ביותר במדגם"),
    ]
    return pd.DataFrame(
        [
            dict(
                quantity=q, ppm=float(v), teams=len(observed), points_33=float(v) * 33, **{WORDS: w}
            )
            for q, v, w in quantities
        ]
    )


def promoted_value_example(values: pd.DataFrame, *, prior_season: int = 24) -> pd.DataFrame:
    """Expose the exact lower-half quantile inputs without changing any imputed feature."""
    _development(values)
    sample = values.loc[values.season_id.eq(prior_season)].copy()
    shares = sample.avg_value_eur.astype(float) / sample.avg_value_eur.astype(float).sum()
    ordered = shares.dropna().sort_values()
    lower = ordered.iloc[: len(ordered) // 2]
    if len(lower) < 2:
        raise ValueError("Not enough prior values for a lower-half quantile")
    quantile = float(lower.quantile(0.30))
    return pd.DataFrame(
        [
            dict(
                method="אחוזון 30 בשבע הזולות",
                value=quantile,
                clubs=len(lower),
                **{WORDS: f"מתוך {len(ordered)} שווים: {len(lower)} זולים; פתיחה {quantile:.5f}"},
            )
        ]
    )


def remaining_value_folds(folds: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Keep the original game-weighted judge while describing the fourteen other folds."""
    _development(folds)
    _development(scores)
    rows = []
    for route, part in folds.loc[folds.season_id.ne(11)].groupby("route", sort=True):
        config = route + ("_elo" if route == "O" else "_poisson")
        counts = scores.loc[scores.config_id.eq(config)].set_index("season_id").games
        weights = counts.reindex(part.season_id)
        if weights.isna().any() or not weights.gt(0).all():
            raise ValueError("Missing validation-game weights")
        improvement = float(np.average(part.improvement, weights=weights))
        rows.append(
            dict(
                route=route,
                improvement=improvement,
                folds=len(part),
                better=int(part.improvement.gt(0).sum()),
                **{WORDS: f"מסלול {route}: בלי 2009/2010, בשקלול משחקי הוולידציה"},
            )
        )
    return pd.DataFrame(rows)


def loss_attribution(
    predictions: pd.DataFrame,
    team_predictions: pd.DataFrame,
    identity: pd.DataFrame,
    club_ids: set[int],
    *,
    season: int = 11,
) -> pd.DataFrame:
    """Attribute loss to fixture groups; P first sums both team-row NLLs per fixture.

    A derby between the named clubs counts once. A share exceeding 100% means the
    other fixtures offset some loss. This is accounting, not a causal estimate.
    """
    for frame in (predictions, team_predictions, identity):
        _development(frame)
    sides = identity.reset_index() if "game_id" not in identity else identity.copy()
    sides = sides.loc[sides.season_id.eq(season)]
    if (
        sides.duplicated(["game_id", "team_id"]).any()
        or not sides.groupby("game_id").size().eq(2).all()
    ):
        raise ValueError("Fixture identity requires exactly two distinct team rows")
    selected = sides.groupby("game_id").team_id.agg(lambda teams: bool(set(teams) & club_ids))
    rows = []
    for route, frame, base, metric in (
        ("O", predictions, "O_elo", "log_loss"),
        ("P", team_predictions, "P_poisson", "poisson_nll"),
    ):
        own = route + "_value_avg"
        part = frame.loc[frame.season_id.eq(season) & frame.config_id.isin([base, own])]
        keys = ["config_id", "game_id", *(["team_id"] if route == "P" else [])]
        if part.duplicated(keys).any() or not np.isfinite(part[metric]).all():
            raise ValueError("Duplicate or nonfinite saved prediction loss")
        if route == "P":
            if not part.groupby(["config_id", "game_id"]).size().eq(2).all():
                raise ValueError("P requires both team losses for every fixture")
            pairs = part[["game_id", "team_id"]].drop_duplicates()
            if set(map(tuple, pairs.to_numpy())) != set(
                map(tuple, sides[["game_id", "team_id"]].to_numpy())
            ):
                raise ValueError("Prediction teams differ from fixture identity")
        losses = part.groupby(["game_id", "config_id"])[metric].sum().unstack()
        if (
            set(losses) != {base, own}
            or losses.isna().any().any()
            or set(losses.index) != set(selected.index)
        ):
            raise ValueError("References and candidates require identical validation fixtures")
        delta = losses[base] - losses[own]
        total = float(delta.sum())
        if np.isclose(total, 0):
            raise ValueError("Cannot attribute a zero net fold loss")
        mask = selected.reindex(delta.index)
        for included, label in ((True, "משחקי חיפה או הפועל ת״א"), (False, "יתר המשחקים")):
            values = delta.loc[mask.eq(included)]
            if values.empty:
                raise ValueError("Both fixture groups must be represented")
            mean, share = float(values.mean()), float(100 * values.sum() / total)
            rows.append(
                dict(
                    route=route,
                    method=label,
                    games=len(values),
                    improvement=mean,
                    loss_share=share,
                    loss_sum=float(values.sum()),
                    **{WORDS: f"{route}, {label}: {share:.1f}% מההפסד נטו; חיובי משפר"},
                )
            )
    return pd.DataFrame(rows)


def gap_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Explain every missing gap by rules/stage, retaining first-season context separately."""
    _development(frame)
    expected = {
        "gap_to_leader": frame.stage.isin(["lower", "middle"]),
        "gap_to_safety": frame.stage.isin(["upper", "middle"]),
        "gap_to_cut": frame.season_id.le(10) | frame.stage.ne("regular"),
    }
    rows = []
    for feature, condition, words in (
        ("gap_to_leader", "בפלייאוף התחתון והאמצעי", "קבוצה בקבוצות אלה אינה יכולה לזכות באליפות"),
        (
            "gap_to_safety",
            "בפלייאוף העליון והאמצעי",
            "אין ירידה משתי הקבוצות; האמצעי קיים ב-2009/2010–2010/2011 בלבד",
        ),
        (
            "gap_to_cut",
            "ללא פיצול ב-2006/2007–2008/2009, ובפלייאוף",
            "ללא פיצול אין קו; אחרי הפיצול הוא כבר אינו יעד",
        ),
    ):
        missing = frame[feature].isna()
        if not np.array_equal(missing.to_numpy(), expected[feature].to_numpy(dtype=bool)):
            raise ValueError(f"Missing gap contradicts the stated stage rule: {feature}")
        rows.append(
            dict(
                feature=feature,
                missing_when=condition,
                missing=int(missing.sum()),
                initial_three=int((missing & frame.season_id.le(10)).sum()),
                seasons_9_10=int((missing & frame.season_id.isin([9, 10])).sum()),
                **{WORDS: words},
            )
        )
    return pd.DataFrame(rows)
