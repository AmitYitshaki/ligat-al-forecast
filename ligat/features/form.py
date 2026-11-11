"""Within-season form candidates with causal opening values (ה-ת12).

The only exception is an explicit, injectable league-neutral value from season 8,
which is training-only. Everything else uses completed seasons and earlier days.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.features.registry import FORM_COLUMNS, validate_feature_columns
from ligat.features.table_state import game_rows, historical_groups
from ligat.simulate.rules import SeasonRules, sporting_table

METRICS = ("gf", "ga", "pts")
STRATA = ("all", "home", "away")
WINDOWS = (5, 10, 20)
HALF_LIVES = (5, 10, 20)
PRIOR_WEIGHTS = (3, 6, 10)
MIN_SEASON_PAIRS = 2
MIN_PROMOTED = 4
RHO_FALLBACK = 0.5


class RhoEstimate(TypedDict):
    rho: float
    rho_raw: float
    rho_clipped: bool
    rho_fallback: bool
    rho_reason: str
    season_pairs: int
    observations: int


COEFFICIENT_COLUMNS = (
    "season_id",
    "metric",
    "stratum",
    "rho",
    "rho_raw",
    "rho_clipped",
    "rho_fallback",
    "rho_reason",
    "season_pairs",
    "observations",
    "league_mean",
    "promoted_value",
    "promoted_observations",
    "promoted_fallback",
)


def _rows(tables: AnalysisTables) -> pd.DataFrame:
    rows = game_rows(tables)
    seasons = sorted(int(s) for s in rows.season_id.unique())
    if (
        not seasons
        or not set(seasons) <= set(DEVELOPMENT_SEASONS)
        or seasons != list(range(8, seasons[-1] + 1))
    ):
        raise ValueError("Form requires consecutive development seasons starting at 8")
    goals = rows[["goals_scored", "goals_conceded"]].to_numpy(float)
    if not np.isfinite(goals).all() or (goals < 0).any() or (goals % 1 != 0).any():
        raise ValueError("Form requires complete nonnegative integer goals")
    home = rows.loc[rows.is_home.eq(True)].set_index("game_id")
    away = rows.loc[rows.is_home.eq(False)].set_index("game_id").reindex(home.index)
    if (
        len(home) * 2 != len(rows)
        or not (
            home.team_id.eq(away.opponent_id)
            & home.opponent_id.eq(away.team_id)
            & home.goals_scored.eq(away.goals_conceded)
            & home.goals_conceded.eq(away.goals_scored)
            & home.as_of.eq(away.as_of)
            & home.season_id.eq(away.season_id)
        )
        .fillna(False)
        .all()
    ):
        raise ValueError("Inconsistent paired form results")
    rows["gf"] = rows.goals_scored.astype(float)
    rows["ga"] = rows.goals_conceded.astype(float)
    rows["pts"] = np.select([rows.gf.gt(rows.ga), rows.gf.eq(rows.ga)], [3.0, 1.0], 0.0)
    boundaries = rows.groupby("season_id").as_of.agg(["min", "max"]).sort_index()
    if (boundaries["min"].iloc[1:].to_numpy() <= boundaries["max"].iloc[:-1].to_numpy()).any():
        raise ValueError("Form requires completed, nonoverlapping seasons")
    return rows


def _stratum(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    return rows if name == "all" else rows.loc[rows.is_home.eq(name == "home")]


def neutral_form_values(tables: AnalysisTables) -> pd.DataFrame:
    """Return the explicit season-8 exception: game-weighted league means by venue.

    Hold this frame fixed when testing calendar causality, including in season 8.
    It contains no team-specific information, but it does use that entire season.
    """
    first = _rows(tables).loc[lambda frame: frame.season_id.eq(8)]
    return pd.DataFrame(
        [_stratum(first, name).loc[:, METRICS].mean() for name in STRATA],
        index=pd.Index(STRATA, name="stratum"),
        columns=METRICS,
    )


def _validate_neutral(neutral: pd.DataFrame) -> None:
    if set(neutral.index) != set(STRATA) or set(neutral.columns) != set(METRICS):
        raise ValueError("Neutral values require all/home/away rows and gf/ga/pts columns")
    if not np.isfinite(neutral.to_numpy(float)).all() or neutral.lt(0).any().any():
        raise ValueError("Neutral values must be finite and nonnegative")


def _rho(x: np.ndarray, y: np.ndarray, season_pairs: int, season_id: int) -> RhoEstimate:
    denominator = float(x @ x)
    varying = len(x) > 0 and float(np.ptp(x)) > 0
    raw = float(x @ y / denominator) if denominator > 0 and varying else np.nan
    reason = (
        "insufficient_season_pairs"
        if season_pairs < MIN_SEASON_PAIRS
        else ("zero_variation" if denominator == 0 or not varying else "none")
    )
    fallback = reason != "none"
    if fallback:
        if season_id >= 11:
            raise ValueError(f"Unexpected rho fallback in season {season_id}: {reason}")
        value, clipped = RHO_FALLBACK, False
    else:
        value = float(np.clip(raw, 0, 1))
        clipped = value != raw
        if clipped and season_id >= 11:
            raise ValueError(f"Unexpected rho clipping in season {season_id}: raw={raw}")
    return {
        "rho": value,
        "rho_raw": raw,
        "rho_clipped": clipped,
        "rho_fallback": fallback,
        "rho_reason": reason,
        "season_pairs": season_pairs,
        "observations": len(x),
    }


def _last_three(games: pd.DataFrame, rule: SeasonRules) -> list[int]:
    roster = sorted(int(t) for t in games.team_id.unique())
    if len(roster) < 3:
        raise ValueError("Bottom-three initialization requires at least three clubs")
    groups = historical_groups(games) if rule.groups else None
    table = sporting_table(games, roster, rule, groups=groups)
    boundary = len(table) - 3
    if boundary and table.position.iloc[boundary - 1] == table.position.iloc[boundary]:
        raise ValueError("Unresolved sporting tie across the bottom-three boundary")
    return [int(t) for t in table.index[-3:]]


def _opening_values(
    rows: pd.DataFrame, rules: Mapping[int, SeasonRules], neutral: pd.DataFrame
) -> pd.DataFrame:
    _validate_neutral(neutral)
    seasons = sorted(int(s) for s in rows.season_id.unique())
    rosters = {s: set(int(t) for t in rows.loc[rows.season_id.eq(s), "team_id"]) for s in seasons}
    # Season tables are descriptive sufficient statistics, never model features.
    means, counts, leagues = {}, {}, {}
    for s in seasons:
        for name in STRATA:
            part = _stratum(rows.loc[rows.season_id.eq(s)], name)
            means[s, name] = part.groupby("team_id")[list(METRICS)].mean()
            counts[s, name] = part.groupby("team_id").size()
            leagues[s, name] = part[list(METRICS)].mean()
    records = []
    for season in seasons:
        prior_seasons = [s for s in seasons if s < season]
        newcomer_events = [
            (s, t) for s in prior_seasons if s > 8 for t in sorted(rosters[s] - rosters[s - 1])
        ]
        promoted_fallback = season != 8 and len(newcomer_events) < MIN_PROMOTED
        bottom = (
            _last_three(rows.loc[rows.season_id.eq(season - 1)], rules[season - 1])
            if promoted_fallback
            else []
        )
        for name in STRATA:
            for metric in METRICS:
                xs, ys, pairs = [], [], 0
                for later in prior_seasons:
                    if later == 8:
                        continue
                    common = sorted(rosters[later] & rosters[later - 1])
                    if not common:
                        continue
                    try:
                        xs.extend(
                            (
                                means[later - 1, name].loc[common, metric]
                                - leagues[later - 1, name][metric]
                            ).tolist()
                        )
                        ys.extend(
                            (
                                means[later, name].loc[common, metric]
                                - leagues[later, name][metric]
                            ).tolist()
                        )
                    except KeyError as exc:
                        raise ValueError("Missing completed-season venue history") from exc
                    pairs += 1
                try:
                    coefficient = _rho(np.asarray(xs, float), np.asarray(ys, float), pairs, season)
                except ValueError as exc:
                    raise ValueError(f"{metric}/{name}: {exc}") from exc
                league = (
                    float(neutral.at[name, metric])
                    if season == 8
                    else float(leagues[season - 1, name][metric])
                )
                if season == 8:
                    promoted_value = league
                elif promoted_fallback:
                    promoted_value = float(
                        np.average(
                            means[season - 1, name].loc[bottom, metric],
                            weights=counts[season - 1, name].loc[bottom],
                        )
                    )
                else:
                    promoted_value = float(
                        np.average(
                            [means[s, name].at[t, metric] for s, t in newcomer_events],
                            weights=[counts[s, name].at[t] for s, t in newcomer_events],
                        )
                    )
                for team in sorted(rosters[season]):
                    promoted = season != 8 and team not in rosters[season - 1]
                    if season == 8:
                        prior = previous = historical = league
                    elif promoted:
                        prior = previous = historical = promoted_value
                    else:
                        if team not in means[season - 1, name].index:
                            raise ValueError("Missing previous-season venue history")
                        previous = float(means[season - 1, name].at[team, metric])
                        prior = league + float(coefficient["rho"]) * (previous - league)
                        history = [(s, int(counts[s, name].get(team, 0))) for s in prior_seasons]
                        total = sum(n for _, n in history)
                        historical = (
                            sum(float(means[s, name].at[team, metric]) * n for s, n in history if n)
                            / total
                        )
                    records.append(
                        {
                            "season_id": season,
                            "team_id": team,
                            "metric": metric,
                            "stratum": name,
                            **coefficient,
                            "league_mean": league,
                            "league_source_season": 8 if season == 8 else season - 1,
                            "prior": prior,
                            "prev": previous,
                            "hist": historical,
                            "promoted_value": promoted_value,
                            "promoted_observations": len(newcomer_events),
                            "promoted_fallback": promoted_fallback,
                            "is_promoted": promoted,
                            "neutral_fallback": season == 8,
                            "used_rho_fallback": season != 8
                            and not promoted
                            and coefficient["rho_fallback"],
                            "used_promoted_fallback": promoted and promoted_fallback,
                        }
                    )
    return (
        pd.DataFrame(records)
        .sort_values(["season_id", "team_id", "metric", "stratum"])
        .reset_index(drop=True)
    )


def form_opening_values(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules], *, neutral: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return every causal rho, promoted value and team opening, before within-season updates.

    League format rules are used only to identify the sporting bottom-three
    fallback, calculated from projected results; official tables are never read.
    """
    rows = _rows(tables)
    neutral = neutral_form_values(tables) if neutral is None else neutral
    return _opening_values(rows, rules, neutral)


def _fallback_counts(rows: pd.DataFrame, openings: pd.DataFrame) -> pd.DataFrame:
    identities = rows[["season_id", "game_id", "team_id", "is_home"]]
    expanded = pd.concat(
        [
            identities.assign(stratum="all"),
            identities.assign(stratum=np.where(identities.is_home, "home", "away")),
        ],
        ignore_index=True,
    )
    result = pd.DataFrame(index=pd.Index(sorted(rows.season_id.unique()), name="season_id"))
    for label, column in [
        ("neutral_rows", "neutral_fallback"),
        ("rho_fallback_rows", "used_rho_fallback"),
        ("promoted_fallback_rows", "used_promoted_fallback"),
    ]:
        used = openings.loc[openings[column], ["season_id", "team_id", "stratum"]].drop_duplicates()
        affected = expanded.merge(used, on=["season_id", "team_id", "stratum"]).drop_duplicates(
            ["game_id", "team_id"]
        )
        result[label] = affected.groupby("season_id").size().reindex(result.index, fill_value=0)
    return result.reset_index()


@dataclass(frozen=True)
class FormResult:
    """Candidates plus the complete descriptive opening/audit tables; no fitted match model."""

    features: pd.DataFrame
    openings: pd.DataFrame
    fallback_counts: pd.DataFrame

    @property
    def coefficients(self) -> pd.DataFrame:
        """Nine retention/promoted records per season, shared across that season's clubs."""
        return (
            self.openings.loc[:, COEFFICIENT_COLUMNS]
            .drop_duplicates()
            .sort_values(["season_id", "metric", "stratum"])
            .reset_index(drop=True)
        )


def compute_form(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules], *, neutral: pd.DataFrame | None = None
) -> FormResult:
    """Compute exactly 66 candidates from completed seasons and earlier calendar days."""
    rows = _rows(tables)
    openings = _opening_values(
        rows, rules, neutral_form_values(tables) if neutral is None else neutral
    )
    values = openings.set_index(["season_id", "team_id", "stratum", "metric"])
    output = []
    for (season, team), games in rows.groupby(["season_id", "team_id"], sort=True):
        histories: dict[str, list[np.ndarray]] = {name: [] for name in STRATA}
        ewmas = {
            name: {
                h: np.array([values.at[(season, team, name, m), "prior"] for m in METRICS], float)
                for h in HALF_LIVES
            }
            for name in STRATA
        }
        prior = {
            name: np.array([values.at[(season, team, name, m), "prior"] for m in METRICS], float)
            for name in STRATA
        }
        for _, day_games in games.groupby("as_of", sort=True):
            for game in day_games.itertuples():
                record = {"game_id": int(game.game_id), "team_id": int(team)}
                for variant, name in [
                    ("all", "all"),
                    ("venue", "home" if game.is_home else "away"),
                ]:
                    past = np.asarray(histories[name], float).reshape(-1, len(METRICS))
                    k = len(past)
                    for j, metric in enumerate(METRICS):
                        prefix = f"{metric}_{variant}_"
                        for w in WINDOWS:
                            n = min(k, w)
                            record[prefix + f"w{w}"] = (
                                (past[-n:, j].sum() + (w - n) * prior[name][j]) / w
                                if n
                                else prior[name][j]
                            )
                        for h in HALF_LIVES:
                            record[prefix + f"ewma{h}"] = ewmas[name][h][j]
                        for m in PRIOR_WEIGHTS:
                            record[prefix + f"std_m{m}"] = (
                                past[:, j].sum() + m * prior[name][j]
                            ) / (k + m)
                        record[prefix + "prev"] = values.at[(season, team, name, metric), "prev"]
                        record[prefix + "hist"] = values.at[(season, team, name, metric), "hist"]
                output.append(record)
            # Every fixture on this date uses the same morning state. Canonical
            # game_id order makes the next day's EWMA independent of input order.
            for game in day_games.sort_values("game_id").itertuples():
                observed = np.array([game.gf, game.ga, game.pts], float)
                for name in ("all", "home" if game.is_home else "away"):
                    histories[name].append(observed)
                    for h in HALF_LIVES:
                        alpha = 1 - 0.5 ** (1 / h)
                        ewmas[name][h] = alpha * observed + (1 - alpha) * ewmas[name][h]
    features = (
        pd.DataFrame(output).set_index(["game_id", "team_id"]).loc[:, FORM_COLUMNS].sort_index()
    )
    validate_feature_columns(features, "form")
    return FormResult(features, openings, _fallback_counts(rows, openings))


@dataclass(frozen=True)
class FormBlock:
    """FeatureBlock adapter; derived sporting tables are used only for early fallbacks."""

    rules: Mapping[int, SeasonRules]
    neutral: pd.DataFrame | None = None
    name: str = "form"
    football_rationale: str = (
        "Within-season attack, defense and points with a regressed previous-season opening."
    )

    def compute(self, tables: AnalysisTables) -> pd.DataFrame:
        """Return the candidate frame; opening audits are available through compute_form."""
        return compute_form(tables, self.rules, neutral=self.neutral).features
