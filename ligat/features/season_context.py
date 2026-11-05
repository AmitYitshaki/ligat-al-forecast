"""Season-fixed t-1 context from sporting results and prior roster values (ה-ת13)."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.features.form import MIN_PROMOTED, form_opening_values
from ligat.features.registry import SEASON_CONTEXT_COLUMNS, validate_feature_columns
from ligat.features.table_state import game_rows, historical_groups
from ligat.simulate.rules import SeasonRules, sporting_table

# ה-ת13: a rounded assumption, not a value fitted to the observed promoted clubs.
PROMOTED_POSITION_PCT = 0.75
VALUE_COLUMNS = ("avg_value_eur", "total_value_eur")


def _rows(tables: AnalysisTables) -> pd.DataFrame:
    rows = game_rows(tables)
    seasons = sorted(int(s) for s in rows.season_id.unique())
    if (
        not seasons
        or not set(seasons) <= set(DEVELOPMENT_SEASONS)
        or seasons != list(range(8, seasons[-1] + 1))
    ):
        raise ValueError("Season context requires consecutive development seasons 8–25 from 8")
    return rows


def season_value_inputs(tables: AnalysisTables) -> pd.DataFrame:
    """Project only source value columns; official standings and promotion flags are excluded."""
    source = tables.team_seasons
    source = source.copy() if "season_id" in source.columns else source.reset_index()
    values = source.loc[:, ["season_id", "team_id", *VALUE_COLUMNS]].copy()
    if values.season_id.isna().any() or not set(values.season_id) <= set(DEVELOPMENT_SEASONS):
        raise ValueError("Season context value inputs require development seasons 8–25")
    if values.duplicated(["season_id", "team_id"]).any():
        raise ValueError("Duplicate team-season source value")
    return values.sort_values(["season_id", "team_id"]).reset_index(drop=True)


def completed_sporting_positions(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules]
) -> pd.DataFrame:
    """Rebuild completed-season ranks from results and historical schedule groups only."""
    parts = []
    for season, games in _rows(tables).groupby("season_id", sort=True):
        season = int(season)
        roster = sorted(int(t) for t in games.team_id.unique())
        if len(roster) != rules[season].n_teams:
            raise ValueError(f"Sporting roster/rules size mismatch in season {season}")
        groups = historical_groups(games) if rules[season].groups else None
        table = sporting_table(games, roster, rules[season], groups=groups)
        part = table[["position"]].copy()
        part["position_pct"] = part.position / rules[season].n_teams
        part["season_id"] = season
        parts.append(part.reset_index())
    return pd.concat(parts, ignore_index=True).set_index(["season_id", "team_id"]).sort_index()


@dataclass(frozen=True)
class SeasonContextResult:
    """Registered prematch context and separate, expanding season-level audit values."""

    features: pd.DataFrame
    audit: pd.DataFrame


def compute_season_context(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules]
) -> SeasonContextResult:
    """Use completed seasons only; current-season scores cannot affect current context."""
    rows = _rows(tables)
    values = season_value_inputs(tables)
    positions = completed_sporting_positions(tables, rules)
    openings = form_opening_values(tables, rules)
    openings = openings.loc[openings.stratum.eq("all")].set_index(
        ["season_id", "team_id", "metric"]
    )
    rows["pts"] = np.select(
        [rows.goals_scored.gt(rows.goals_conceded), rows.goals_scored.eq(rows.goals_conceded)],
        [3.0, 1.0],
        0.0,
    )
    means = rows.groupby(["season_id", "team_id"])[["pts", "goals_scored", "goals_conceded"]].mean()
    rosters = {
        int(s): set(int(t) for t in games.team_id.unique())
        for s, games in rows.groupby("season_id")
    }
    shares = values.set_index(["season_id", "team_id"])
    for column in VALUE_COLUMNS:
        shares[column] = shares[column].astype("Float64")
        total = shares.groupby(level="season_id")[column].transform("sum", min_count=1)
        shares[column] = shares[column] / total.where(total.gt(0))
    per_team, audit = [], []
    for season in sorted(rosters):
        known = season > 8
        prior_roster = rosters.get(season - 1, set())
        promoted = rosters[season] - prior_roster if known else set()
        imputed = {column: np.nan for column in VALUE_COLUMNS}
        prior_events = [
            (s, t)
            for s in sorted(rosters)
            if 8 < s < season
            for t in sorted(rosters[s] - rosters[s - 1])
        ]
        position_mean = (
            float(np.mean([positions.at[event, "position_pct"] for event in prior_events]))
            if len(prior_events) >= MIN_PROMOTED
            else PROMOTED_POSITION_PCT
        )
        if known:
            prior_values = shares.xs(season - 1, level="season_id")
            for column in VALUE_COLUMNS:
                available = prior_values[column].dropna().sort_values()
                imputed[column] = float(available.iloc[: len(available) // 2].quantile(0.30))
            if not np.isfinite(list(imputed.values())).all():
                raise ValueError(f"Cannot initialize missing squad values for season {season}")
            example = min(rosters[season])
            promoted_ppm = float(openings.at[(season, example, "pts"), "promoted_value"])
            promoted_gd = float(
                openings.at[(season, example, "gf"), "promoted_value"]
                - openings.at[(season, example, "ga"), "promoted_value"]
            )
        else:
            promoted_ppm = promoted_gd = np.nan
        imputation_count = 0
        for team in sorted(rosters[season]):
            record = dict.fromkeys(SEASON_CONTEXT_COLUMNS[:5], np.nan)
            is_promoted = team in promoted
            filled = False
            if known:
                for source, output in zip(VALUE_COLUMNS, SEASON_CONTEXT_COLUMNS[:2], strict=True):
                    value = (
                        prior_values.at[team, source]
                        if not is_promoted and team in prior_values.index
                        else np.nan
                    )
                    if pd.isna(value):
                        value, filled = imputed[source], True
                    record[output] = float(value)
                record["prev_position_pct"] = (
                    PROMOTED_POSITION_PCT
                    if is_promoted
                    else float(positions.at[(season - 1, team), "position_pct"])
                )
                if is_promoted:
                    record["prev_ppm"], record["prev_gd_per_game"] = promoted_ppm, promoted_gd
                else:
                    previous = means.loc[(season - 1, team)]
                    record["prev_ppm"] = float(previous.pts)
                    record["prev_gd_per_game"] = float(
                        previous.goals_scored - previous.goals_conceded
                    )
            imputation_count += int(filled)
            per_team.append(
                {
                    "season_id": season,
                    "team_id": team,
                    **record,
                    "promoted_in": is_promoted,
                    "squad_value_imputed": filled,
                    "prev_season_known": known,
                    "league_size": rules[season].n_teams,
                }
            )
        audit.append(
            {
                "season_id": season,
                "imputed_value_share_avg": imputed["avg_value_eur"],
                "imputed_value_share_total": imputed["total_value_eur"],
                "imputed_ppm": promoted_ppm,
                "imputed_gd_per_game": promoted_gd,
                "promoted_position_mean": position_mean,
                "previous_promoted_events": len(prior_events),
                "promoted_events": len(promoted),
                "value_imputed_teams": imputation_count,
            }
        )
    features = (
        rows[["game_id", "team_id", "season_id"]]
        .astype("int64")
        .merge(pd.DataFrame(per_team), on=["season_id", "team_id"], validate="many_to_one")
        .set_index(["game_id", "team_id"])
        .loc[:, SEASON_CONTEXT_COLUMNS]
        .sort_index()
    )
    for column in SEASON_CONTEXT_COLUMNS[:5]:
        features[column] = features[column].astype(float)
    for column in SEASON_CONTEXT_COLUMNS[5:8]:
        features[column] = features[column].astype(bool)
    features["league_size"] = features.league_size.astype("int64")
    known_rows = features.prev_season_known
    if not np.isfinite(features.loc[known_rows, SEASON_CONTEXT_COLUMNS[:5]].to_numpy(float)).all():
        raise ValueError("Missing completed-season context above season 8")
    validate_feature_columns(features, "season_context")
    return SeasonContextResult(features, pd.DataFrame(audit))


@dataclass(frozen=True)
class SeasonContextBlock:
    """FeatureBlock adapter; all seasonal context is fixed before the season begins."""

    rules: Mapping[int, SeasonRules]
    name: str = "season_context"
    football_rationale: str = "Previous-season sporting strength and market-value context."

    def compute(self, tables: AnalysisTables) -> pd.DataFrame:
        """Return registered features; compute_season_context also exposes the audit."""
        return compute_season_context(tables, self.rules).features
