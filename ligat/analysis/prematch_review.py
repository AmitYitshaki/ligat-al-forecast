"""Development-only notebook 03 diagnostics, separated from feature construction."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict

import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prematch_validation import PrematchValidation
from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.features.registry import FEATURES
from ligat.features.table_state import game_rows, historical_groups
from ligat.simulate.rules import SeasonRules, sporting_table


def _development(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(frame.season_id.unique()) <= set(DEVELOPMENT_SEASONS):
        raise ValueError("Notebook research requires development seasons 8–25")
    return frame.copy()


def rules_overview(rules: Mapping[int, SeasonRules], seasons: Sequence[int]) -> pd.DataFrame:
    """Show the season format and sporting objectives without reading results."""
    rows = []
    for season in seasons:
        rule = rules[season]
        rows.append(
            dict(
                season_id=season,
                n_teams=rule.n_teams,
                regular_rounds=rule.regular_rounds,
                playoff_format="; ".join(f"{g.name}: {g.teams} / {g.rounds}" for g in rule.groups)
                or "none",
                points_halved=rule.points_halved,
                relegated=rule.relegated,
                relegation_test=rule.relegation_playoff_position,
                upper_cut=rule.upper_size,
                last_safe=rule.last_safe_position,
                tiebreak=" → ".join(rule.tiebreak_rules),
            )
        )
    return pd.DataFrame(rows)


def validation_overview(validation: PrematchValidation) -> pd.DataFrame:
    """Report raw matches, independently explained differences and excluded dirty rounds."""
    return (
        validation.checks.groupby(["scope", "category"], observed=True)
        .agg(
            rows=("team_id", "size"),
            counter_rows=("counters_checked", "sum"),
            rank_rows=("rank_checked", "sum"),
            display_rows=("display_order", "sum"),
        )
        .reset_index()
    )


def registry_overview() -> pd.DataFrame:
    """Expose every approved derived feature and its per-model status in one table."""
    rows = []
    for feature, spec in FEATURES.items():
        row = asdict(spec)
        row["feature"] = feature
        row["models"] = ", ".join(spec.models)
        row["model_status"] = "; ".join(f"{model}: {status}" for model, status in spec.model_status)
        rows.append(row)
    return pd.DataFrame(rows)


def postponed_example(
    tables: AnalysisTables, features: pd.DataFrame, official_rounds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find a real earlier-round fixture played after the target fixture's morning.

    The official end-of-round table also includes the target round's results;
    this is explicitly a leakage illustration, not an equal-time comparison.
    """
    games = _development(tables.match_table.reset_index()).sort_values(
        ["season_id", "date"], ascending=[False, True]
    )
    for target in games.itertuples():
        later = games.loc[
            games.season_id.eq(target.season_id)
            & games["round"].lt(target.round)
            & games.date.dt.normalize().gt(target.date.normalize())
        ]
        if later.empty:
            continue
        delayed = later.iloc[0]
        participants = {int(delayed.home_team_id), int(delayed.away_team_id)}
        # Show a participant in the target game, so its prematch feature is available.
        common = participants & {int(target.home_team_id), int(target.away_team_id)}
        if not common:
            continue
        team = sorted(common)[0]
        before = features.loc[(target.game_id, team)]
        official = official_rounds.loc[
            official_rounds.season_id.eq(target.season_id)
            & official_rounds["round"].eq(target.round)
            & official_rounds.team_id.eq(team)
        ].iloc[0]
        team_name = target.home_team_name if team == target.home_team_id else target.away_team_name
        context = pd.DataFrame(
            [
                dict(
                    season_id=target.season_id,
                    example="target",
                    date=target.date.strftime("%d.%m.%Y"),
                    round=target.round,
                    fixture=f"{target.home_team_name} – {target.away_team_name}",
                ),
                dict(
                    season_id=target.season_id,
                    example="delayed",
                    date=delayed.date.strftime("%d.%m.%Y"),
                    round=int(delayed["round"]),
                    fixture=f"{delayed.home_team_name} – {delayed.away_team_name}",
                ),
            ]
        )
        comparison = pd.DataFrame(
            [
                dict(
                    team_name=team_name,
                    basis="morning",
                    games_played=int(before.games_played),
                    points=int(before.points),
                ),
                dict(
                    team_name=team_name,
                    basis="official_round_end",
                    games_played=int(official.played),
                    points=int(official.points_model),
                ),
            ]
        )
        return context, comparison
    raise ValueError("No real postponed-game illustration found")


def sporting_official_comparison(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules], finals: pd.DataFrame
) -> pd.DataFrame:
    """Diagnose official sanctions/exclusions at season end; never return model inputs."""
    games = _development(game_rows(tables))
    names = tables.team_matches.reset_index().groupby("team_id").team_name.first()
    rows = []
    for season, season_games in games.groupby("season_id", sort=True):
        rule = rules[int(season)]
        roster = sorted(season_games.team_id.unique())
        groups = historical_groups(season_games) if rule.groups else None
        table = sporting_table(season_games, roster, rule, groups=groups)
        official = finals.loc[finals.season_id.eq(season)].set_index("team_id")
        if not official.deduction.ne(0).any() and table.points.eq(official.points_model).all():
            continue
        for team in roster:
            row = official.loc[team]
            rows.append(
                dict(
                    season_id=int(season),
                    team_name=names.loc[team],
                    sporting_points=int(table.at[team, "points"]),
                    official_points=int(row.points_official),
                    deduction=int(row.deduction),
                    technical_points=int(row.points_model - table.at[team, "points"]),
                    sporting_position=float(table.at[team, "position"]),
                    official_position=int(row.position),
                    rank_changed=table.at[team, "position"] != row.position,
                    officially_relegated=bool(row.relegated),
                )
            )
    return pd.DataFrame(rows)


def split_comparison(tables: AnalysisTables, validation: PrematchValidation) -> pd.DataFrame:
    """List every development split discrepancy, with club names and verified cause."""
    frame = validation.split_differences
    frame = frame.loc[frame.season_id.isin(DEVELOPMENT_SEASONS)].copy()
    names = tables.team_matches.reset_index().groupby("team_id").team_name.first()
    frame["team_name"] = frame.team_id.map(names)
    return frame.drop(columns="team_id")


def dead_game_summary(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count matches with zero/one/two dead teams, not paired rows as independent games."""
    frame = _development(features.reset_index())
    matches = (
        frame.groupby(["season_id", "stage", "game_id"], observed=True)
        .agg(
            dead_teams=("nothing_to_play", "sum"),
            team_rows=("team_id", "size"),
        )
        .reset_index()
    )
    if not matches.team_rows.eq(2).all():
        raise ValueError("Dead-game summary requires both team rows")
    for count, name in [(0, "neither"), (1, "one"), (2, "both")]:
        matches[name] = matches.dead_teams.eq(count).astype(int)
    summary = (
        matches.groupby(["season_id", "stage"], observed=True)
        .agg(
            games=("game_id", "size"),
            neither=("neither", "sum"),
            one=("one", "sum"),
            both=("both", "sum"),
        )
        .reset_index()
    )
    summary["at_least_one_share"] = (summary.one + summary.both) / summary.games
    summary["both_share"] = summary.both / summary.games
    return summary, matches


def championship_examples(tables: AnalysisTables, features: pd.DataFrame) -> pd.DataFrame:
    """First observed champion's own prematch row with a proven title; a lower bound.

    Clinching can happen between its fixtures and conservative bounds may detect
    it late. This does not claim the actual earliest mathematical clinching date.
    """
    frame = _development(features.reset_index())
    frame = frame.loc[frame.title_decided & frame.position.eq(1)].sort_values(["date", "game_id"])
    frame = frame.groupby("season_id", observed=True).head(1).copy()
    names = tables.team_matches.reset_index().groupby("team_id").team_name.first()
    frame["team_name"] = frame.team_id.map(names)
    frame["date"] = frame.date.dt.strftime("%d.%m.%Y")
    return frame[
        ["season_id", "team_name", "date", "round", "stage", "rounds_to_stage_end"]
    ].sort_values(["rounds_to_stage_end", "season_id"], ascending=[False, True])


def stage_end_distribution(features: pd.DataFrame) -> pd.DataFrame:
    """Count target games by stage and games remaining AFTER that target game."""
    frame = _development(features.reset_index())
    if frame.rounds_to_stage_end.lt(0).any():
        raise ValueError("Negative games remaining after target game")
    return (
        frame.groupby(["stage", "rounds_to_stage_end"], observed=True)
        .agg(
            team_games=("team_id", "size"),
        )
        .reset_index()
        .sort_values(["stage", "rounds_to_stage_end"])
    )


def place_line_contradictions(
    tables: AnalysisTables, features: pd.DataFrame, rules: Mapping[int, SeasonRules]
) -> pd.DataFrame:
    """Audit every claimed top-2/3/4 decision against the final sporting table.

    This is a development-only retrospective check, never a feature input. The
    morning side uses group_position with the fixed league offset translated
    into the line. Thus first in the lower playoff is not first in the league;
    the middle/lower groups remain structurally outside all three place lines.
    """
    games = _development(game_rows(tables))
    frame = _development(features.reset_index())
    errors = []
    for season, season_games in games.groupby("season_id", sort=True):
        rule = rules[int(season)]
        groups = historical_groups(season_games) if rule.groups else None
        final = sporting_table(
            season_games, sorted(season_games.team_id.unique()), rule, groups=groups
        )
        claimed = frame.loc[frame.season_id.eq(season) & frame.places_decided]
        for row in claimed.itertuples():
            offset = float(row.position - row.group_position)
            final_position = float(final.at[row.team_id, "position"])
            for line in (2, 3, 4):
                before_inside = row.group_position <= line - offset
                final_inside = final_position <= line
                if before_inside != final_inside:
                    errors.append(
                        dict(
                            season_id=int(season),
                            game_id=int(row.game_id),
                            team_id=int(row.team_id),
                            stage=str(row.stage),
                            line=line,
                            group_position=float(row.group_position),
                            group_offset=offset,
                            final_position=final_position,
                            before_inside=bool(before_inside),
                            final_inside=bool(final_inside),
                        )
                    )
    return pd.DataFrame(
        errors,
        columns=[
            "season_id",
            "game_id",
            "team_id",
            "stage",
            "line",
            "group_position",
            "group_offset",
            "final_position",
            "before_inside",
            "final_inside",
        ],
    )
