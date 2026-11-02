"""Engineering-only comparison with official tables; never a feature dependency.

Technical results are reconstructed in a separate validation ledger from the
already saved round responses and verified exclusions. They are NOT added to
AnalysisTables, the sporting feature inputs or any stored clean dataset.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from ligat.analysis.prepare import AnalysisTables
from ligat.data.parse import parse_round
from ligat.features.table_state import game_rows, historical_groups
from ligat.simulate.rules import SeasonRules, rank_table, split_playoffs, sporting_table

COUNTERS = {
    "games_played": "played",
    "wins": "won",
    "draws": "drawn",
    "losses": "lost",
    "goals_for": "goals_for",
    "goals_against": "goals_against",
    "points": "points_model",
    "goal_difference": "goal_difference",
}
# Amit's source-display policy, 2026-11-01 (T-010 §4).
MAX_DISPLAY_ORDER_SHARE = 0.01


class UnexplainedStandingsError(ValueError):
    """Stop the phase rather than conceal an unexplained official-table mismatch."""

    def __init__(self, rows: pd.DataFrame) -> None:
        self.rows = rows
        super().__init__(f"Unexplained standings discrepancies:\n{rows.to_string(index=False)}")


@dataclass(frozen=True)
class PrematchValidation:
    """Audit frames, separated from model inputs and development-only research."""

    checks: pd.DataFrame
    differences: pd.DataFrame
    split_differences: pd.DataFrame
    ties: pd.DataFrame


def _counter_discrepancies(
    calculated: pd.Series, official: pd.Series, season_id: int
) -> tuple[list[str], bool]:
    """Permit only the verified equal GF/GA offset in season 8; GD stays strict."""
    differences = [key for key, value in COUNTERS.items() if calculated[key] != official[value]]
    delta_for = calculated.goals_for - official.goals_for
    delta_against = calculated.goals_against - official.goals_against
    artifact = season_id == 8 and delta_for != 0 and delta_for == delta_against
    if artifact:
        differences = [key for key in differences if key not in ("goals_for", "goals_against")]
    return differences, bool(artifact)


def _position_comparison(
    table: pd.DataFrame, team: int, official_position: int, intermediate: bool
) -> tuple[bool, bool]:
    """Check all criteria at endpoints; intermediate display may reorder equal P/GD.

    The permitted interval is derived from the entire P/GD block, so a row cannot
    cross a better/worse team's position. Fixed playoff offsets remain in force.
    """
    row = table.loc[team]
    tied = table.loc[table.position.eq(row.position)]
    strict_ok = abs(float(official_position) - float(row.position)) <= (len(tied) - 1) / 2
    if strict_ok or not intermediate:
        return bool(strict_ok), False
    peers = table.loc[table.table_group.eq(row.table_group)]
    equal = peers.points.eq(row.points) & peers.goal_difference.eq(row.goal_difference)
    block = peers.loc[equal]
    lower = float(block.position.min())
    upper = float(block.position.max())
    # A complete final-criterion tie has average positions rather than integers.
    first = table.loc[table.position.eq(lower)]
    last = table.loc[table.position.eq(upper)]
    lower -= (len(first) - 1) / 2
    upper += (len(last) - 1) / 2
    display_order = lower <= official_position <= upper
    return bool(display_order), bool(display_order)


def _check_display_order_limit(checks: pd.DataFrame) -> None:
    intermediate = checks.loc[checks.scope.eq("round") & checks.intermediate.eq(True)]
    display_rows = int(intermediate.display_order.sum())
    if len(intermediate) and display_rows / len(intermediate) > MAX_DISPLAY_ORDER_SHARE:
        raise UnexplainedStandingsError(
            pd.DataFrame(
                [
                    dict(
                        fields="display order exceeds 1% of intermediate rows",
                        display_rows=display_rows,
                        intermediate_rows=len(intermediate),
                        share=display_rows / len(intermediate),
                    )
                ]
            )
        )


def load_technical_ledger(
    seasons: list[int],
    overrides_path: Path = Path("config/match_overrides.yaml"),
    raw_root: Path = Path("data/raw/ifa/rounds"),
) -> pd.DataFrame:
    """Read only verified dropped matches, for validation of administrative effects.

    A void is counted as a draw in IFA W/D/L but earns no point, except for the
    explicitly deferred 2024/25 correction already recorded in the config.
    """
    overrides = yaml.safe_load(overrides_path.read_text(encoding="utf-8"))["matches"]
    rows = []
    for game_id, item in overrides.items():
        if item["season_id"] not in seasons or not item["verified"] or item.get("action") != "drop":
            continue
        path = raw_root / str(item["season_id"]) / f"round_{item['round']:02d}.xml"
        game = next(row for row in parse_round(path.read_bytes()).games if row.game_id == game_id)
        if (game.home_team_id, game.away_team_id) != (item["home_team_id"], item["away_team_id"]):
            raise ValueError(f"Technical-match identity mismatch: {game_id}")
        no_point_draw = item["status"] == "void" and item.get("adjust_points", True)
        for team, opponent, gf, ga, home in [
            (game.home_team_id, game.away_team_id, game.home_goals, game.away_goals, True),
            (game.away_team_id, game.home_team_id, game.away_goals, game.home_goals, False),
        ]:
            rows.append(
                dict(
                    game_id=game_id,
                    season_id=item["season_id"],
                    round=item["round"],
                    stage=item["stage"],
                    date=pd.Timestamp(game.date),
                    team_id=team,
                    opponent_id=opponent,
                    goals_scored=gf,
                    goals_conceded=ga,
                    is_home=home,
                    no_point_draw=int(no_point_draw),
                )
            )
    return pd.DataFrame(rows)


def _reference_table(
    games: pd.DataFrame,
    roster: list[int],
    rule: SeasonRules,
    groups: Mapping[int, str] | None,
) -> pd.DataFrame:
    table = sporting_table(games, roster, rule, groups=groups)
    voids = games.groupby("team_id").no_point_draw.sum().reindex(roster, fill_value=0)
    if groups is not None and rule.points_halved:
        regular = games.loc[games.stage.eq("regular")]
        reg_voids = regular.groupby("team_id").no_point_draw.sum().reindex(roster, fill_value=0)
        reg_points = (
            regular.assign(
                p=3 * regular.goals_scored.gt(regular.goals_conceded)
                + regular.goals_scored.eq(regular.goals_conceded)
            )
            .groupby("team_id")
            .p.sum()
            .reindex(roster, fill_value=0)
        )
        table["points"] += (
            (reg_points - reg_voids + 1) // 2 - (reg_points + 1) // 2 - voids + reg_voids
        )
    else:
        table["points"] -= voids
    if groups is None:
        table = rank_table(table, rule, games)
    else:
        parts = []
        for group in rule.groups:
            part = rank_table(
                table.loc[[team for team in roster if groups[team] == group.name]], rule, games
            )
            part["position"] += rule.offset(group.name)
            parts.append(part)
        table = pd.concat(parts)
    return table


def validate_prematch_tables(
    tables: AnalysisTables,
    rules: Mapping[int, SeasonRules],
    round_tables: pd.DataFrame,
    final_tables: pd.DataFrame,
    technical: pd.DataFrame,
) -> PrematchValidation:
    """Validate clean rounds/finals; stop on any discrepancy not explained independently.

    Raw sporting differences caused by exclusions are checked against the separate
    technical ledger. Dirty rounds are reported, excluded from the numeric gate.
    Official ranks are checked only without deductions. At intermediate rounds,
    equal-points/equal-GD display blocks may be reordered, up to the 1% gate.
    Season 8 alone permits the verified equal GF/GA source offset, never a GD offset.
    """
    source = game_rows(tables)
    checks: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    split_changes: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    previous: dict[int, float] | None = None
    for season, games in source.groupby("season_id", sort=True):
        sid = int(season)
        rule = rules[sid]
        roster = sorted(int(team) for team in games.team_id.unique())
        groups = historical_groups(games) if rule.groups else None
        admin = (
            technical.loc[technical.season_id.eq(sid)].copy()
            if not technical.empty
            else pd.DataFrame()
        )
        reference = pd.concat([games.assign(no_point_draw=0), admin], ignore_index=True)
        official_rounds = round_tables.loc[round_tables.season_id.eq(sid)].copy()
        official_rounds["goal_difference"] = (
            official_rounds.goals_for - official_rounds.goals_against
        )
        for rnd, official in official_rounds.groupby("round", sort=True):
            intermediate = rnd not in (rule.regular_rounds, official_rounds["round"].max())
            active_groups = groups if rnd > rule.regular_rounds else None
            observed = games.loc[games["round"].le(rnd)]
            past_reference = reference.loc[reference["round"].le(rnd)]
            future = reference.loc[reference["round"].gt(rnd)]
            dirty = (
                not future.empty
                and past_reference.date.max().normalize() > future.date.min().normalize()
            )
            calculated = sporting_table(observed, roster, rule, groups=active_groups)
            corrected = _reference_table(past_reference, roster, rule, active_groups)
            official = official.set_index("team_id")
            deductions = official.deduction.ne(0).any()
            for team, row in official.iterrows():
                raw_diff = [
                    key for key, value in COUNTERS.items() if calculated.at[team, key] != row[value]
                ]
                unexplained, artifact = _counter_discrepancies(corrected.loc[team], row, sid)
                rank = float(corrected.at[team, "position"])
                tied = corrected.loc[corrected.position.eq(rank)]
                rank_ok, display_order = _position_comparison(
                    corrected, int(team), int(row.position), intermediate
                )
                display_order = display_order and not dirty and not deductions
                if len(tied) > 1 and int(team) == int(tied.index.min()):
                    ties.append(
                        dict(
                            season_id=sid,
                            round=int(rnd),
                            teams=tuple(int(t) for t in tied.index),
                            position=rank,
                        )
                    )
                if not dirty and (unexplained or (not deductions and not rank_ok)):
                    errors.append(
                        dict(
                            season_id=sid,
                            round=int(rnd),
                            team_id=int(team),
                            fields=", ".join(unexplained or ["position"]),
                            calculated={key: corrected.at[team, key] for key in unexplained},
                            official={key: row[COUNTERS[key]] for key in unexplained},
                        )
                    )
                category = (
                    "postponed"
                    if dirty
                    else "source_artifact"
                    if artifact
                    else "technical"
                    if raw_diff
                    else "match"
                )
                checks.append(
                    dict(
                        scope="round",
                        season_id=sid,
                        round=int(rnd),
                        team_id=int(team),
                        category=category,
                        counters_checked=not dirty,
                        rank_checked=not dirty and not deductions,
                        rank_match=rank_ok,
                        intermediate=intermediate,
                        display_order=display_order,
                        source_artifact=artifact and not dirty,
                    )
                )
                if raw_diff or deductions:
                    differences.append(
                        dict(
                            season_id=sid,
                            round=int(rnd),
                            team_id=int(team),
                            category=category if raw_diff else "deduction",
                            fields=", ".join(raw_diff),
                            deduction=int(row.deduction),
                            delta_goals_for=int(corrected.at[team, "goals_for"] - row.goals_for),
                            delta_goals_against=int(
                                corrected.at[team, "goals_against"] - row.goals_against
                            ),
                        )
                    )
                if display_order:
                    differences.append(
                        dict(
                            season_id=sid,
                            round=int(rnd),
                            team_id=int(team),
                            category="display_order",
                            fields="position",
                            calculated_position=rank,
                            official_position=int(row.position),
                        )
                    )
        final = final_tables.loc[final_tables.season_id.eq(sid)].set_index("team_id")
        last = sporting_table(games, roster, rule, groups=groups)
        ref_last = _reference_table(reference, roster, rule, groups)
        for team, row in final.iterrows():
            if ref_last.at[team, "points"] != row.points_model:
                errors.append(dict(season_id=sid, team_id=int(team), fields="final points_model"))
            if groups is not None and groups[int(team)] != row.final_stage:
                errors.append(dict(season_id=sid, team_id=int(team), fields="final_stage"))
            rank = float(ref_last.at[team, "position"])
            tied_count = int(ref_last.position.eq(rank).sum())
            rank_ok = abs(float(row.position) - rank) <= (tied_count - 1) / 2
            if not final.deduction.ne(0).any() and not rank_ok:
                errors.append(dict(season_id=sid, team_id=int(team), fields="final position"))
            checks.append(
                dict(
                    scope="final",
                    season_id=sid,
                    team_id=int(team),
                    category="technical"
                    if last.at[team, "points"] != row.points_model
                    else "match",
                    counters_checked=True,
                    rank_checked=not final.deduction.ne(0).any(),
                    rank_match=rank_ok,
                    intermediate=False,
                    display_order=False,
                    source_artifact=False,
                )
            )
        if final.relegated.notna().all() and int(final.relegated.sum()) != rule.relegated:
            errors.append(dict(season_id=sid, fields="relegated count"))
        if groups is not None:
            regular = games.loc[games.stage.eq("regular")]
            sporting_split = split_playoffs(
                sporting_table(regular, roster, rule), rule, regular, previous_positions=previous
            )
            reg_ref = reference.loc[reference.stage.eq("regular")]
            corrected_regular = _reference_table(reg_ref, roster, rule, None)
            admin_split = split_playoffs(
                corrected_regular, rule, reg_ref, previous_positions=previous
            )
            official_regular = official_rounds.loc[
                official_rounds["round"].eq(rule.regular_rounds)
            ].set_index("team_id")
            deducted = corrected_regular.copy()
            deducted["points"] += official_regular.deduction
            official_split = split_playoffs(deducted, rule, reg_ref, previous_positions=previous)
            for team in roster:
                if sporting_split[team] == groups[team]:
                    continue
                cause = (
                    "technical"
                    if admin_split[team] == groups[team]
                    else "deduction"
                    if official_split[team] == groups[team]
                    else "unexplained"
                )
                split_changes.append(
                    dict(
                        season_id=sid,
                        team_id=team,
                        sporting_group=sporting_split[team],
                        actual_group=groups[team],
                        cause=cause,
                    )
                )
                if cause == "unexplained":
                    errors.append(dict(season_id=sid, team_id=team, fields="split assignment"))
        previous = {int(team): float(row.position) for team, row in last.iterrows()}
    result = PrematchValidation(
        pd.DataFrame(checks),
        pd.DataFrame(differences),
        pd.DataFrame(split_changes),
        pd.DataFrame(ties),
    )
    if errors:
        raise UnexplainedStandingsError(pd.DataFrame(errors))
    _check_display_order_limit(result.checks)
    return result
