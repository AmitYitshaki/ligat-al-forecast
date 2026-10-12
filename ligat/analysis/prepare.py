"""Pure analysis views and health checks; no publication or held-out analysis."""

from dataclasses import dataclass, fields
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml
from statsmodels.stats.proportion import proportion_confint  # type: ignore[import-untyped]

from ligat.data.details import load_exceptions
from ligat.data.repository import Repository

ANALYSIS_SEASONS = range(8, 26)
BEFORE_METRICS = [
    "matches",
    "points",
    "goals_scored",
    "goals_conceded",
    "goal_difference",
]
COLUMN_GROUPS: dict[str, list[str]] = {
    "identity": [
        "season_id",
        "season",
        "round",
        "stage",
        "era",
        "date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "venue",
        "venue_raw",
        "is_playoff",
    ],
    "pre_match": [
        "home_coach_id",
        "home_coach_name",
        "away_coach_id",
        "away_coach_name",
        "referee",
        "home_prev_season_total_value_eur",
        "home_prev_season_avg_value_eur",
        "away_prev_season_total_value_eur",
        "away_prev_season_avg_value_eur",
        *[f"{side}_{metric}_before" for side in ("home", "away") for metric in BEFORE_METRICS],
    ],
    "in_match": [
        "home_goals",
        "away_goals",
        "total_goals",
        "goal_difference",
        "result",
        "ht_home_goals",
        "ht_away_goals",
        "ht_result",
        "home_goals_first_half",
        "away_goals_first_half",
        "home_red_cards",
        "away_red_cards",
        "first_goal_side",
        "first_goal_minute",
    ],
    "data_quality": [
        "is_home_venue",
        "is_postponed",
        "ht_conflict",
        "goals_available",
        "has_timeline",
        "has_lineups",
        "has_coaches",
    ],
}


def _outcome(difference: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    result = pd.Series(pd.NA, index=difference.index, dtype="string")
    for condition, label in zip(
        (difference.gt(0), difference.eq(0), difference.lt(0)), labels, strict=True
    ):
        result.loc[condition.fillna(False)] = label
    return result.astype(pd.CategoricalDtype(labels))


def _before(frame: pd.DataFrame) -> pd.DataFrame:
    """Accumulate complete timestamp batches; simultaneous games never see each other."""
    sides = []
    for side, opponent in (("home", "away"), ("away", "home")):
        scored, conceded = frame[f"{side}_goals"], frame[f"{opponent}_goals"]
        sides.append(
            pd.DataFrame(
                {
                    "game_id": frame.game_id,
                    "team_id": frame[f"{side}_team_id"],
                    "season_id": frame.season_id,
                    "date": frame.date,
                    "matches": 1,
                    "points": scored.gt(conceded).astype(int) * 3 + scored.eq(conceded).astype(int),
                    "goals_scored": scored,
                    "goals_conceded": conceded,
                    "goal_difference": scored - conceded,
                }
            )
        )
    team_rows = pd.concat(sides, ignore_index=True)
    keys = ["season_id", "team_id", "date"]
    batches = team_rows.groupby(keys, sort=True)[BEFORE_METRICS].sum()
    previous = batches.groupby(level=["season_id", "team_id"]).cumsum() - batches
    return team_rows[keys + ["game_id"]].join(previous, on=keys, validate="many_to_one")


def _goal_summary(frame: pd.DataFrame, goals: pd.DataFrame, reds: pd.DataFrame) -> None:
    """Count credited goals and cards, preserving unavailable source information."""
    halftime = goals.loc[goals.half.eq(1)].groupby(["game_id", "team_id"]).size()
    cards = reds.groupby(["game_id", "team_id"]).size()
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([frame.game_id, frame[f"{side}_team_id"]])
        frame[f"{side}_goals_first_half"] = pd.array(
            halftime.reindex(keys, fill_value=0), dtype="Int64"
        )
        unavailable = frame.ht_conflict | ~frame.goals_available
        frame.loc[unavailable, f"{side}_goals_first_half"] = pd.NA
        frame[f"{side}_red_cards"] = pd.array(cards.reindex(keys, fill_value=0), dtype="Int64")
        frame.loc[~frame.has_timeline, f"{side}_red_cards"] = pd.NA
    frame["first_goal_side"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["first_goal_minute"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[frame.goals_available & frame.total_goals.eq(0), "first_goal_side"] = "none"
    for game_id, events in goals.groupby("game_id", sort=False):
        positions = frame.index[frame.game_id.eq(game_id)]
        if not len(positions):
            continue
        position = positions[0]
        if not frame.at[position, "goals_available"]:
            continue
        ordered = events.sort_values(["half", "minute", "added_time"], na_position="last")
        first = ordered.iloc[0]
        tied = ordered.loc[
            ordered.half.fillna(0).eq(0 if pd.isna(first.half) else first.half)
            & ordered.minute.eq(first.minute)
            & ordered.added_time.fillna(0).eq(0 if pd.isna(first.added_time) else first.added_time)
        ]
        frame.at[position, "first_goal_minute"] = first.minute
        if tied.team_id.nunique() == 1:
            frame.at[position, "first_goal_side"] = (
                "home" if first.team_id == frame.at[position, "home_team_id"] else "away"
            )
    frame["first_goal_side"] = frame.first_goal_side.astype(
        pd.CategoricalDtype(["home", "away", "none"])
    )


def build_match_table(
    repository: Repository,
    seasons: range | list[int] = ANALYSIS_SEASONS,
    teams_path: Path = Path("config/teams.yaml"),
    seasons_path: Path = Path("config/seasons.yaml"),
) -> pd.DataFrame:
    """Join approved games and source details; pre-match totals use strictly earlier dates."""
    selected = list(seasons)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(ANALYSIS_SEASONS)
    ):
        raise ValueError("Analysis seasons must be distinct IDs within 8–25")
    frame = repository.read_filtered("matches", [("season_id", "in", selected)]).copy()
    game_ids = frame.game_id.tolist()
    inputs = {
        name: repository.read_filtered(name, [("game_id", "in", game_ids)])
        for name in ("game_details", "goals", "red_cards", "coaches")
    }
    details = inputs["game_details"]
    assert set(details.game_id) == set(game_ids), "Missing game_details rows for analysis games"
    frame = frame.merge(details, on="game_id", how="left", validate="one_to_one")
    teams = yaml.safe_load(teams_path.read_text(encoding="utf-8"))["teams"]
    names = {row["team_id"]: row["name_he"] for row in teams}
    formats = yaml.safe_load(seasons_path.read_text(encoding="utf-8"))["seasons"]
    eras = {row["season_id"]: f"{row['n_teams']}_teams" for row in formats}
    frame["era"] = frame.season_id.map(eras)
    assert frame.era.notna().all(), "Missing season format for analysis era"
    frame["era"] = frame.era.astype(pd.CategoricalDtype(["12_teams", "16_teams", "14_teams"]))
    assert frame.era.notna().all(), "Unsupported n_teams in season format"
    frame["is_playoff"] = frame.stage.ne("regular").astype("boolean")
    frame["total_goals"] = frame.home_goals + frame.away_goals
    frame["goal_difference"] = frame.home_goals - frame.away_goals
    frame["result"] = _outcome(frame.goal_difference, ("home_win", "draw", "away_win"))
    frame["ht_result"] = _outcome(
        frame.ht_home_goals - frame.ht_away_goals, ("home_win", "draw", "away_win")
    )
    previous = repository.read_filtered(
        "squad_values", [("season_id", "in", sorted({season - 1 for season in selected}))]
    ).copy()
    previous["season_id"] += 1
    cumulative = _before(frame)
    for side in ("home", "away"):
        frame[f"{side}_team_name"] = frame[f"{side}_team_id"].map(names).astype("string")
        assert frame[f"{side}_team_name"].notna().all(), f"Missing {side} canonical team name"
        coaches = inputs["coaches"][["game_id", "team_id", "coach_id", "coach_name"]].rename(
            columns={
                "team_id": f"{side}_team_id",
                "coach_id": f"{side}_coach_id",
                "coach_name": f"{side}_coach_name",
            }
        )
        frame = frame.merge(
            coaches, on=["game_id", f"{side}_team_id"], how="left", validate="one_to_one"
        )
        values = previous[["season_id", "team_id", "total_value_eur", "avg_value_eur"]].rename(
            columns={
                "team_id": f"{side}_team_id",
                **{
                    name: f"{side}_prev_season_{name}"
                    for name in ("total_value_eur", "avg_value_eur")
                },
            }
        )
        frame = frame.merge(
            values, on=["season_id", f"{side}_team_id"], how="left", validate="many_to_one"
        )
        before = cumulative.rename(
            columns={
                "team_id": f"{side}_team_id",
                **{metric: f"{side}_{metric}_before" for metric in BEFORE_METRICS},
            }
        )
        frame = frame.merge(
            before.drop(columns=["season_id", "date"]),
            on=["game_id", f"{side}_team_id"],
            validate="one_to_one",
        )
    next_round = frame.groupby(["season_id", "stage", "round"], observed=True).date.min()
    keys = pd.MultiIndex.from_arrays([frame.season_id, frame.stage, frame["round"] + 1])
    dates = pd.Series(next_round.reindex(keys).array, index=frame.index)
    frame["is_postponed"] = frame.date.gt(dates).astype("boolean")
    _goal_summary(frame, inputs["goals"], inputs["red_cards"])
    columns = [column for group in COLUMN_GROUPS.values() for column in group]
    frame = (
        frame[columns + ["game_id"]]
        .sort_values(["date", "game_id"])
        .set_index("game_id", drop=True)
    )
    for column in frame:
        if column.endswith(("_id", "_before", "_eur", "_goals", "_cards", "_half")) or column in (
            "round",
            "total_goals",
            "goal_difference",
            "first_goal_minute",
        ):
            frame[column] = frame[column].astype("Int64")
        elif column.startswith(("is_", "has_")) or column in ("ht_conflict", "goals_available"):
            frame[column] = frame[column].astype("boolean")
    assert frame.index.is_unique, "match_table game_id index must be unique"
    assert frame.date.is_monotonic_increasing, "match_table must be chronological"
    assert len(columns) == len(set(columns)) and set(columns) == set(frame.columns), (
        "COLUMN_GROUPS must partition match_table columns exactly"
    )
    return frame


def to_team_perspective(match_table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team/game, translating outcomes without altering source rows."""
    assert match_table.index.name == "game_id" and match_table.index.is_unique, (
        "Expected unique game_id index"
    )
    source = match_table.reset_index()
    sides = []
    shared = [
        "game_id",
        "date",
        "round",
        "season_id",
        "season",
        "stage",
        "era",
        "venue",
        "venue_raw",
        "is_playoff",
        "referee",
        *COLUMN_GROUPS["data_quality"],
        "total_goals",
        "first_goal_minute",
    ]
    for side, opponent in (("home", "away"), ("away", "home")):
        frame = source[shared].copy()
        translations = {
            "team_id": f"{side}_team_id",
            "team_name": f"{side}_team_name",
            "opponent_id": f"{opponent}_team_id",
            "opponent_name": f"{opponent}_team_name",
            "goals_scored": f"{side}_goals",
            "goals_conceded": f"{opponent}_goals",
            "ht_goals_scored": f"ht_{side}_goals",
            "ht_goals_conceded": f"ht_{opponent}_goals",
            "goals_first_half": f"{side}_goals_first_half",
            "opponent_goals_first_half": f"{opponent}_goals_first_half",
            "red_cards": f"{side}_red_cards",
            "opponent_red_cards": f"{opponent}_red_cards",
            "coach_id": f"{side}_coach_id",
            "coach_name": f"{side}_coach_name",
            "opponent_coach_id": f"{opponent}_coach_id",
            "opponent_coach_name": f"{opponent}_coach_name",
            **{
                f"prev_season_{name}": f"{side}_prev_season_{name}"
                for name in ("total_value_eur", "avg_value_eur")
            },
            **{f"{metric}_before": f"{side}_{metric}_before" for metric in BEFORE_METRICS},
        }
        for target, origin in translations.items():
            frame[target] = source[origin]
        frame["is_home"] = pd.Series(side == "home", index=frame.index, dtype="boolean")
        frame["goal_difference"] = frame.goals_scored - frame.goals_conceded
        frame["result"] = _outcome(frame.goal_difference, ("win", "draw", "loss"))
        frame["points_earned"] = frame.result.map({"win": 3, "draw": 1, "loss": 0}).astype("Int64")
        frame["is_clean_sheet"] = frame.goals_conceded.eq(0).astype("boolean")
        frame["failed_to_score"] = frame.goals_scored.eq(0).astype("boolean")
        frame["ht_state"] = _outcome(
            frame.ht_goals_scored - frame.ht_goals_conceded, ("leading", "level", "trailing")
        )
        frame.loc[frame.ht_conflict, "ht_state"] = pd.NA
        frame["scored_first"] = source.first_goal_side.eq(side).astype("boolean")
        frame.loc[
            source.first_goal_side.isna()
            | source.first_goal_side.eq("none")
            | ~source.goals_available,
            "scored_first",
        ] = pd.NA
        sides.append(frame)
    combined = pd.concat(sides, ignore_index=True).sort_values(["team_id", "date", "game_id"])
    combined["match_number"] = (combined.groupby(["team_id", "season_id"]).cumcount() + 1).astype(
        "Int64"
    )
    first = combined.groupby(["team_id", "season_id"], sort=False).head(1)
    assert first[[f"{metric}_before" for metric in BEFORE_METRICS]].eq(0).all().all(), (
        "First team/season game must have zero prior totals"
    )
    combined = combined.set_index(["team_id", "game_id"], drop=True)
    assert combined.index.is_unique, "team_matches index must be unique"
    assert len(combined) == 2 * len(match_table), "Every match must have two team rows"
    assert (
        combined.groupby(level="team_id")
        .date.apply(lambda dates: dates.is_monotonic_increasing)
        .all()
    ), "team_matches must be chronological within teams"
    return combined


def team_history(
    team_matches: pd.DataFrame,
    team: int | str,
    teams_path: Path = Path("config/teams.yaml"),
) -> pd.DataFrame:
    """Resolve an ID or Hebrew canonical name/alias and return an independent history."""
    if isinstance(team, str):
        entries: list[dict[str, Any]] = yaml.safe_load(teams_path.read_text(encoding="utf-8"))[
            "teams"
        ]
        matches = {
            row["team_id"]
            for row in entries
            if team.strip()
            in [
                str(row["name_he"]).strip(),
                *[str(alias).strip() for alias in row.get("aliases", [])],
            ]
        }
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous team name/alias: {team}")
        team_id = matches.pop()
    else:
        team_id = team
    if team_id not in team_matches.index.get_level_values("team_id"):
        raise ValueError(f"No analysis history for team_id={team_id}")
    return team_matches.loc[[team_id]].sort_values(["date", "game_id"]).copy()


@dataclass(frozen=True)
class AnalysisTables:
    """Six independent in-memory views for the approved analysis seasons."""

    match_table: pd.DataFrame
    team_matches: pd.DataFrame
    goal_events: pd.DataFrame
    team_seasons: pd.DataFrame
    season_summary: pd.DataFrame
    coach_spells: pd.DataFrame


MINUTE_BUCKETS = [
    "0-15",
    "16-30",
    "31-45",
    "45+ (first-half stoppage)",
    "46-60",
    "61-75",
    "76-90",
    "90+",
]


def _minute_bucket(half: Any, minute: int) -> Any:
    if pd.isna(half) and minute > 45:
        return pd.NA
    if not pd.isna(half) and half == 1 and minute > 45:
        return "45+ (first-half stoppage)"
    if not pd.isna(half) and half == 2 and minute <= 60:
        return "46-60"  # source minute 45 can denote the start of the second half
    for maximum, label in [
        (15, "0-15"),
        (30, "16-30"),
        (45, "31-45"),
        (60, "46-60"),
        (75, "61-75"),
        (90, "76-90"),
    ]:
        if minute <= maximum:
            return label
    return "90+"


def build_goal_events(match_table: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    """Attach match identity to credited goals; retain uncertainty in simultaneous events."""
    frame = goals.loc[goals.game_id.isin(match_table.index)].copy()
    context = match_table[
        [
            "season_id",
            "season",
            "date",
            "home_team_id",
            "away_team_id",
            "home_team_name",
            "away_team_name",
            "ht_conflict",
            "goals_available",
        ]
    ]
    frame = frame.join(context, on="game_id", validate="many_to_one")
    frame = frame.loc[frame.goals_available].copy()
    assert (frame.team_id.eq(frame.home_team_id) | frame.team_id.eq(frame.away_team_id)).all(), (
        "Goal credited to a nonparticipant"
    )
    frame["is_home"] = frame.team_id.eq(frame.home_team_id).astype("boolean")
    frame["team_name"] = frame.home_team_name.where(frame.is_home, frame.away_team_name)
    frame["opponent_id"] = frame.away_team_id.where(frame.is_home, frame.home_team_id).astype(
        "Int64"
    )
    frame = frame.sort_values(
        ["game_id", "half", "minute", "added_time", "team_id"], kind="stable", na_position="last"
    )
    frame["goal_number"] = (frame.groupby("game_id").cumcount() + 1).astype("Int64")
    frame["minute_bucket"] = pd.Categorical(
        [_minute_bucket(row.half, row.minute) for row in frame.itertuples()],
        categories=MINUTE_BUCKETS,
        ordered=True,
    )
    frame["score_before"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for _, events in frame.groupby("game_id", sort=False):
        if events.ht_conflict.any():
            continue
        scores = {True: 0, False: 0}
        # The batch is a half/minute; adding explicit stoppage does not invent ordering.
        for _, batch in events.groupby(["half", "minute"], sort=True, dropna=False):
            if len(batch) == 1:
                row = batch.iloc[0]
                difference = scores[bool(row.is_home)] - scores[not bool(row.is_home)]
                frame.at[row.name, "score_before"] = (
                    "leading" if difference > 0 else "trailing" if difference < 0 else "level"
                )
            for row in batch.itertuples():
                scores[bool(row.is_home)] += 1
    frame["score_before"] = frame.score_before.astype(
        pd.CategoricalDtype(["leading", "level", "trailing"])
    )
    frame = frame.drop(
        columns=[
            "home_team_id",
            "away_team_id",
            "home_team_name",
            "away_team_name",
            "ht_conflict",
            "goals_available",
        ]
    )
    return frame.set_index(["game_id", "goal_number"], drop=True)


def _longest_run(flags: Any) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _mean(values: pd.Series) -> Any:
    return values.mean() if values.notna().any() else pd.NA


def _coach_count(frame: pd.DataFrame) -> Any:
    ids = set(frame.coach_id.dropna())
    names_with_ids = set(frame.loc[frame.coach_id.notna(), "coach_name"].dropna())
    names_without_ids = (
        set(frame.loc[frame.coach_id.isna(), "coach_name"].dropna()) - names_with_ids
    )
    return len(ids) + len(names_without_ids) if ids or names_without_ids else pd.NA


def _typed_view(
    frame: pd.DataFrame, integers: list[str], booleans: tuple[str, ...] | list[str] = ()
) -> pd.DataFrame:
    for column in integers:
        frame[column] = frame[column].astype("Int64")
    for column in booleans:
        frame[column] = frame[column].astype("boolean")
    return frame


def build_team_seasons(
    team_matches: pd.DataFrame,
    final_tables: pd.DataFrame,
    squad_values: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate sporting results and join official descriptive season information."""
    rows = []
    source = team_matches.reset_index().sort_values(["team_id", "date", "game_id"])
    for (season, team), games in source.groupby(["season_id", "team_id"], sort=True):
        home, away = games.loc[games.is_home], games.loc[~games.is_home]
        hppm, appm = _mean(home.points_earned), _mean(away.points_earned)
        rows.append(
            {
                "season_id": season,
                "team_id": team,
                "team_name": games.team_name.iloc[0],
                "matches_played": len(games),
                "wins": games.result.eq("win").sum(),
                "draws": games.result.eq("draw").sum(),
                "losses": games.result.eq("loss").sum(),
                "points_from_matches": games.points_earned.sum(),
                "points_per_match": _mean(games.points_earned),
                "goals_scored": games.goals_scored.sum(),
                "goals_conceded": games.goals_conceded.sum(),
                "goals_scored_per_match": _mean(games.goals_scored),
                "goals_conceded_per_match": _mean(games.goals_conceded),
                "home_points_per_match": hppm,
                "away_points_per_match": appm,
                "home_away_gap": hppm - appm,
                "clean_sheet_rate": _mean(games.is_clean_sheet),
                "failed_to_score_rate": _mean(games.failed_to_score),
                "draw_rate": games.result.eq("draw").mean(),
                "points_per_match_when_leading_ht": _mean(
                    games.loc[games.ht_state.eq("leading"), "points_earned"]
                ),
                "points_per_match_when_trailing_ht": _mean(
                    games.loc[games.ht_state.eq("trailing"), "points_earned"]
                ),
                "red_cards_per_match": _mean(games.loc[games.has_timeline, "red_cards"]),
                "longest_unbeaten_run": _longest_run(games.result.ne("loss")),
                "longest_winless_run": _longest_run(games.result.ne("win")),
                "n_coaches": _coach_count(games),
            }
        )
    frame = pd.DataFrame(rows)
    finals = final_tables[
        [
            "season_id",
            "team_id",
            "position",
            "points_model",
            "points_official",
            "deduction",
            "final_stage",
            "relegated",
            "promoted_in",
        ]
    ].rename(columns={"position": "final_position"})
    frame = frame.merge(finals, on=["season_id", "team_id"], how="left", validate="one_to_one")
    values = squad_values[
        ["season_id", "team_id", "squad_size", "avg_value_eur", "total_value_eur"]
    ].copy()
    totals = values.groupby("season_id").total_value_eur.transform(
        lambda group: group.sum(min_count=1)
    )
    values["value_share_of_league"] = values.total_value_eur / totals.where(totals.gt(0))
    previous = values[["season_id", "team_id", "value_share_of_league"]].rename(
        columns={"value_share_of_league": "prev_season_value_share"}
    )
    previous["season_id"] += 1
    values["squad_row_available"] = True
    frame = frame.merge(values, on=["season_id", "team_id"], how="left", validate="one_to_one")
    frame["squad_row_available"] = frame.squad_row_available.eq(True).astype("boolean")
    frame = frame.merge(previous, on=["season_id", "team_id"], how="left", validate="one_to_one")
    frame.loc[frame.season_id.eq(8) | frame.promoted_in.eq(True), "prev_season_value_share"] = pd.NA
    integers = [
        "season_id",
        "team_id",
        "matches_played",
        "wins",
        "draws",
        "losses",
        "points_from_matches",
        "goals_scored",
        "goals_conceded",
        "longest_unbeaten_run",
        "longest_winless_run",
        "n_coaches",
        "final_position",
        "points_model",
        "points_official",
        "deduction",
        "squad_size",
        "avg_value_eur",
        "total_value_eur",
    ]
    frame = _typed_view(frame, integers, ["relegated", "promoted_in", "squad_row_available"])
    for column in frame:
        if column.endswith(("_per_match", "_rate", "_share")) or column in (
            "home_away_gap",
            "value_share_of_league",
            "points_per_match_when_leading_ht",
            "points_per_match_when_trailing_ht",
        ):
            frame[column] = frame[column].astype("Float64")
    frame["final_stage"] = frame.final_stage.astype(
        pd.CategoricalDtype(["regular", "upper", "middle", "lower"])
    )
    return frame.sort_values(["season_id", "final_position", "team_id"]).set_index(
        ["season_id", "team_id"], drop=True
    )


def build_season_summary(
    match_table: pd.DataFrame,
    goal_events: pd.DataFrame,
    team_seasons: pd.DataFrame,
    season_formats: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    """Compare seasons using sporting points rates, not administrative/halved totals."""
    rows = []
    for season, games in match_table.groupby("season_id", sort=True):
        teams = team_seasons.xs(season, level="season_id")
        goals = goal_events.loc[goal_events.season_id.eq(season)]
        known_half = goals.loc[goals.half.notna()]
        cards = games.loc[games.has_timeline]
        champion = teams.loc[teams.final_position.eq(1), "points_per_match"]
        relegated = teams.loc[teams.relegated.eq(True)].sort_values("final_position")
        total = teams.total_value_eur.sum(min_count=1)
        rows.append(
            {
                "season_id": season,
                "n_teams": season_formats[int(season)]["n_teams"],
                "n_matches": len(games),
                "goals_per_match": _mean(games.total_goals),
                "home_goals_per_match": _mean(games.home_goals),
                "away_goals_per_match": _mean(games.away_goals),
                "home_win_rate": games.result.eq("home_win").mean(),
                "draw_rate": games.result.eq("draw").mean(),
                "away_win_rate": games.result.eq("away_win").mean(),
                "zero_zero_rate": games.total_goals.eq(0).mean(),
                "second_half_goal_share": known_half.half.eq(2).mean()
                if len(known_half)
                else pd.NA,
                "first_half_stoppage_goals": goals.is_first_half_stoppage.sum(),
                "penalties_per_match": goals.is_penalty.sum() / games.goals_available.sum()
                if games.goals_available.any()
                else pd.NA,
                "red_cards_per_match": _mean(cards.home_red_cards + cards.away_red_cards),
                "champion_points_per_match": champion.iloc[0] if len(champion) == 1 else pd.NA,
                "relegation_line_points_per_match": relegated.points_per_match.iloc[0]
                if len(relegated)
                else pd.NA,
                "league_total_value_eur": total,
                "top3_value_share": teams.total_value_eur.nlargest(3).sum(min_count=1) / total
                if not pd.isna(total) and total > 0
                else pd.NA,
                "n_postponed": games.is_postponed.sum(),
            }
        )
    frame = _typed_view(
        pd.DataFrame(rows),
        [
            "season_id",
            "n_teams",
            "n_matches",
            "first_half_stoppage_goals",
            "league_total_value_eur",
            "n_postponed",
        ],
    )
    for column in frame:
        if column not in (
            "season_id",
            "n_teams",
            "n_matches",
            "first_half_stoppage_goals",
            "league_total_value_eur",
            "n_postponed",
        ):
            frame[column] = frame[column].astype("Float64")
    return frame.set_index("season_id", drop=True).sort_index()


def build_coach_spells(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Group observed coach runs; unknown-coach games bridge runs but are never assigned."""
    rows: list[dict[str, Any]] = []
    source = team_matches.reset_index().sort_values(["team_id", "date", "game_id"])
    source = source.loc[source.season_id.ge(11)]
    known = source.loc[source.coach_id.notna() | source.coach_name.notna()]
    first_observed = known.groupby(["team_id", "season_id"]).game_id.first().to_dict()
    for team, games in source.groupby("team_id", sort=True):
        spell: list[Any] = []
        number = 0

        def finish(observed: list[Any], team_id: int) -> None:
            nonlocal number
            if not observed:
                return
            number += 1
            identifiers = [g.coach_id for g in observed if not pd.isna(g.coach_id)]
            names = [g.coach_name for g in observed if not pd.isna(g.coach_name)]
            rows.append(
                {
                    "team_id": team_id,
                    "spell_number": number,
                    "coach_id": identifiers[0] if identifiers else pd.NA,
                    "coach_name": names[-1] if names else pd.NA,
                    "season_id_start": observed[0].season_id,
                    "season_id_end": observed[-1].season_id,
                    "first_date": observed[0].date,
                    "last_date": observed[-1].date,
                    "matches": len(observed),
                    "points_per_match": sum(g.points_earned for g in observed) / len(observed),
                    "is_mid_season_start": observed[0].match_number > 1,
                    "is_left_censored": observed[0].match_number > 1
                    and observed[0].game_id == first_observed[(team_id, observed[0].season_id)],
                }
            )

        for game in games.itertuples():
            if pd.isna(game.coach_id) and pd.isna(game.coach_name):
                continue
            if spell:
                prior = spell[-1]
                same = (
                    game.coach_id == prior.coach_id
                    if not pd.isna(game.coach_id) and not pd.isna(prior.coach_id)
                    else (
                        not pd.isna(game.coach_name)
                        and not pd.isna(prior.coach_name)
                        and game.coach_name == prior.coach_name
                    )
                )
                if not same:
                    finish(spell, int(team))
                    spell = []
            spell.append(game)
        finish(spell, int(team))
    columns = [
        "team_id",
        "spell_number",
        "coach_id",
        "coach_name",
        "season_id_start",
        "season_id_end",
        "first_date",
        "last_date",
        "matches",
        "points_per_match",
        "is_mid_season_start",
        "is_left_censored",
    ]
    frame = _typed_view(
        pd.DataFrame(rows, columns=columns),
        ["team_id", "spell_number", "coach_id", "season_id_start", "season_id_end", "matches"],
        ["is_mid_season_start", "is_left_censored"],
    )
    frame["coach_name"] = frame.coach_name.astype("string")
    frame["points_per_match"] = frame.points_per_match.astype("Float64")
    for column in ("first_date", "last_date"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert("Asia/Jerusalem")
    frame["is_flicker"] = pd.Series(False, index=frame.index, dtype="boolean")
    for _, group in frame.groupby("team_id", sort=False):
        for position in range(1, len(group) - 1):
            before, current, after = (group.iloc[position + offset] for offset in (-1, 0, 1))
            same = (
                before.coach_id == after.coach_id
                if pd.notna(before.coach_id) and pd.notna(after.coach_id)
                else pd.notna(before.coach_name)
                and pd.notna(after.coach_name)
                and before.coach_name == after.coach_name
            )
            frame.at[current.name, "is_flicker"] = bool(same) and current.matches <= 2
    return frame.set_index(["team_id", "spell_number"], drop=True)


def preview_pre_match_row(
    tables: AnalysisTables,
    game_id: int,
    window: int = 10,
) -> pd.DataFrame:
    """Prototype features from strictly earlier games; this is not a trained model.

    Both venue points rates use the same last-window games, possibly across seasons.
    Missing/conflicting halftimes are excluded from the halftime mean, not imputed.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if game_id not in tables.match_table.index:
        raise ValueError(f"Unknown game_id={game_id}")
    game = tables.match_table.loc[game_id]
    row: dict[str, Any] = {}
    for side in ("home", "away"):
        team_id = game[f"{side}_team_id"]
        history = tables.team_matches.xs(team_id, level="team_id")
        history = (
            history.loc[history.date.lt(game.date)].sort_values(["date", "game_id"]).tail(window)
        )
        for column in ("goals_scored", "goals_conceded", "points_earned", "ht_goals_scored"):
            values = history[column]
            if column == "ht_goals_scored":
                values = values.where(~history.ht_conflict)
            row[f"feature_{side}_{column}_mean"] = _mean(values)
        for venue, is_home in (("home", True), ("away", False)):
            row[f"feature_{side}_{venue}_points_per_match"] = _mean(
                history.loc[history.is_home.eq(is_home), "points_earned"]
            )
        row[f"feature_{side}_prev_season_value_share"] = tables.team_seasons.loc[
            (game.season_id, team_id), "prev_season_value_share"
        ]
    row.update(
        target_home_goals=game.home_goals,
        target_away_goals=game.away_goals,
        target_result=game.result,
    )
    frame = pd.DataFrame([row], index=pd.Index([game_id], name="game_id"))
    for column in frame:
        if column.startswith("feature_"):
            frame[column] = frame[column].astype("Float64")
    return frame


def outcome_base_rates(match_table: pd.DataFrame) -> pd.DataFrame:
    """Descriptive B0 home/draw/away rates and marginal 95% Wilson intervals.

    These full-sample rates describe EDA; evaluating future matches must fit B0
    on training data only. The three marginal intervals are not simultaneous.
    """
    labels = ("home_win", "draw", "away_win")
    if match_table.empty or not match_table.result.isin(labels).all():
        raise ValueError("B0 requires nonempty matches with known home/draw/away outcomes")
    rows = []
    groups = [
        ("overall", pd.NA, match_table),
        *[("season", season, games) for season, games in match_table.groupby("season_id")],
    ]
    for scope, season, games in groups:
        for outcome in labels:
            count = int(games.result.eq(outcome).sum())
            lower, upper = proportion_confint(count, len(games), alpha=0.05, method="wilson")
            rows.append(
                dict(
                    scope=scope,
                    season_id=season,
                    outcome=outcome,
                    n_matches=len(games),
                    n_outcomes=count,
                    rate=count / len(games),
                    ci_lower=float(lower),
                    ci_upper=float(upper),
                )
            )
    frame = pd.DataFrame(rows)
    frame["season_id"] = frame.season_id.astype("Int64")
    return frame


def load_analysis_tables(
    repository: Repository | None = None,
    seasons: range | list[int] = ANALYSIS_SEASONS,
    teams_path: Path = Path("config/teams.yaml"),
    seasons_path: Path = Path("config/seasons.yaml"),
) -> AnalysisTables:
    """Build the six views from filtered approved inputs without writing any dataset."""
    repo = repository or Repository()
    selected = list(seasons)
    matches = build_match_table(repo, selected, teams_path, seasons_path)
    team_matches = to_team_perspective(matches)
    goals = repo.read_filtered("goals", [("game_id", "in", matches.index.tolist())])
    finals = repo.read_filtered("final_tables", [("season_id", "in", selected)])
    values = repo.read_filtered(
        "squad_values", [("season_id", "in", sorted(set(selected) | {s - 1 for s in selected}))]
    )
    formats = {
        row["season_id"]: row
        for row in yaml.safe_load(seasons_path.read_text(encoding="utf-8"))["seasons"]
    }
    events = build_goal_events(matches, goals)
    team_seasons = build_team_seasons(team_matches, finals, values)
    summary = build_season_summary(matches, events, team_seasons, formats)
    return AnalysisTables(
        matches, team_matches, events, team_seasons, summary, build_coach_spells(team_matches)
    )


DOCUMENTED_MISSING_REASONS: dict[tuple[str, str], str] = {}
for _table, _columns, _reason in (
    (
        "match_table",
        [f"{side}_{field}" for side in ("home", "away") for field in ("coach_id", "coach_name")],
        "No coaches in 2006/07–2008/09; partial coverage in 2009/10; later source omissions. "
        "IDs may be absent when names have no linked identifier (48 matches).",
    ),
    (
        "team_matches",
        ["coach_id", "coach_name", "opponent_coach_id", "opponent_coach_name"],
        "Historical coach coverage and source omissions; a name may have no linked identifier.",
    ),
    (
        "match_table",
        [
            f"{side}_prev_season_{field}_value_eur"
            for side in ("home", "away")
            for field in ("avg", "total")
        ],
        "First analysis season has no predecessor; promoted teams; "
        "Hapoel Petah Tikva 2008/09 has no source value. No imputation.",
    ),
    (
        "team_matches",
        ["prev_season_avg_value_eur", "prev_season_total_value_eur"],
        "No previous league season for promoted clubs or 2006/07; "
        "Hapoel Petah Tikva 2008/09 source value missing.",
    ),
    ("match_table", ["referee"], "Referee absent in source (27 matches)."),
    ("team_matches", ["referee"], "Referee absent in source (27 matches, two team rows each)."),
    ("team_matches", ["first_goal_minute"], "No first goal in 0:0; unreliable scorers in 864982."),
    (
        "match_table",
        ["first_goal_side", "first_goal_minute"],
        "Unknown first scorer when both teams score in the same minute; "
        "unreliable scorers in 864982. First minute also absent for 0:0.",
    ),
    (
        "match_table",
        ["home_goals_first_half", "away_goals_first_half"],
        "13 documented halftime/timeline conflicts and unavailable scorers in 864982.",
    ),
    (
        "team_matches",
        ["goals_first_half", "opponent_goals_first_half", "ht_state"],
        "13 documented halftime conflicts; first-half event counts also unavailable for 864982.",
    ),
    (
        "team_matches",
        ["scored_first"],
        "Not applicable to 0:0; tied first-goal minute or unavailable scorers in 864982.",
    ),
    (
        "goal_events",
        ["added_time"],
        "Source uses running minutes instead of + notation; "
        "first-half stoppage identified using half and minute.",
    ),
    (
        "goal_events",
        ["half", "minute_bucket", "score_before"],
        "13 documented halftime conflicts; unknown half after minute 45; "
        "simultaneous goals have unknown prior score.",
    ),
    ("team_seasons", ["promoted_in"], "2006/07 has no previous league season."),
    ("season_summary", ["second_half_goal_share"], "No goals with a known half in this season."),
    (
        "team_seasons",
        ["prev_season_value_share"],
        "Previous league's share unavailable for first season, "
        "promoted teams or missing source value. No imputation.",
    ),
    (
        "team_seasons",
        ["avg_value_eur", "total_value_eur", "value_share_of_league"],
        "Hapoel Petah Tikva 2008/09 has no market value in the source.",
    ),
    (
        "team_seasons",
        ["n_coaches"],
        "No observed coaches in early seasons; unknown coaches are not invented.",
    ),
    (
        "team_seasons",
        ["points_per_match_when_leading_ht", "points_per_match_when_trailing_ht"],
        "No observed games with that halftime state; conflicting halftimes excluded.",
    ),
    (
        "coach_spells",
        ["coach_id", "coach_name"],
        "Source can provide a coach name without a linked ID, or an ID without a name. "
        "Coverage/flicker flags document source quality without imputing coaches.",
    ),
):
    DOCUMENTED_MISSING_REASONS.update({(_table, column): _reason for column in _columns})


def describe_missing_values(tables: AnalysisTables) -> pd.DataFrame:
    """Inventory every column; unexplained high missingness blocks analysis."""
    rows = []
    for field in fields(tables):
        frame = getattr(tables, field.name)
        for column in frame:
            count = int(frame[column].isna().sum())
            share = count / len(frame) if len(frame) else 0.0
            status = "fail" if share > 0.30 else "warn" if share > 0.05 else "ok"
            reason = DOCUMENTED_MISSING_REASONS.get((field.name, column))
            assert status != "fail" or reason is not None, (
                f"Undocumented missing values: {field.name}.{column} ({share:.1%})"
            )
            rows.append(
                {
                    "table": field.name,
                    "column": column,
                    "n_missing": count,
                    "share": share,
                    "status": status,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def flag_unusual_matches(tables: AnalysisTables) -> pd.DataFrame:
    """Flag unusual observations without dropping or changing any match."""
    source = tables.match_table
    flags = pd.DataFrame(index=source.index)
    flags["seven_or_more_goals"] = source.total_goals.ge(7)
    flags["three_or_more_red_cards"] = (source.home_red_cards + source.away_red_cards).ge(3)
    flags["is_postponed"] = source.is_postponed
    flags["borrowed_venue"] = ~source.is_home_venue
    return (
        source[["season_id", "date", "home_team_name", "away_team_name"]]
        .join(flags)
        .loc[flags.any(axis=1)]
        .copy()
    )


def _scheduled_games(season_id: int, n_teams: int, final_stage: str) -> int:
    if n_teams == 12:
        return 33
    if n_teams == 16:
        if season_id == 13:
            return 37
        return 33 if final_stage == "middle" else 35
    assert n_teams == 14, f"Unknown league format: {n_teams} teams"
    return 36 if final_stage == "upper" else 33


def check_analysis_tables(
    tables: AnalysisTables,
    seasons_path: Path = Path("config/seasons.yaml"),
    overrides_path: Path = Path("config/match_overrides.yaml"),
    teams_path: Path = Path("config/teams.yaml"),
    exceptions_path: Path = Path("config/game_detail_exceptions.yaml"),
) -> list[str]:
    """Check sporting, schedule, source coverage and join invariants without network I/O."""
    matches, team_matches, seasons = tables.match_table, tables.team_matches, tables.team_seasons
    selected = set(matches.season_id)
    assert selected <= set(ANALYSIS_SEASONS), "Held-out seasons in analysis"
    for field in fields(tables):
        frame = getattr(tables, field.name)
        assert frame.index.is_unique, f"Duplicate index in {field.name}"
        assert not set(frame.columns) & set(frame.index.names), f"Ambiguous index in {field.name}"
        if "season_id" in frame or "season_id" in frame.index.names:
            ids = (
                frame.season_id
                if "season_id" in frame
                else frame.index.get_level_values("season_id")
            )
            assert set(ids) <= selected, f"Unexpected seasons in {field.name}"
    expected_teams = to_team_perspective(matches)
    assert set(team_matches.index) == set(expected_teams.index) and len(team_matches) == 2 * len(
        matches
    ), "Every game must have exactly two team rows"
    for column in (
        "season_id",
        "round",
        "opponent_id",
        "goals_scored",
        "goals_conceded",
        "points_earned",
    ):
        assert team_matches[column].sort_index().equals(expected_teams[column].sort_index()), (
            f"Incorrect team_matches.{column}"
        )
    assert not team_matches.reset_index().duplicated(["season_id", "team_id", "round"]).any(), (
        "Team appears twice in a round"
    )
    expected_stats = team_matches.reset_index().groupby(["season_id", "team_id"])
    for column, origin, aggregation in (
        ("matches_played", "game_id", "count"),
        ("goals_scored", "goals_scored", "sum"),
        ("goals_conceded", "goals_conceded", "sum"),
        ("points_from_matches", "points_earned", "sum"),
    ):
        expected = expected_stats[origin].agg(aggregation)
        assert seasons[column].sort_index().eq(expected.sort_index()).all() and set(
            seasons.index
        ) == set(expected.index), f"Incorrect team_seasons.{column}"
    for column, result in (("wins", "win"), ("draws", "draw"), ("losses", "loss")):
        expected = expected_stats.result.agg(lambda group, label=result: group.eq(label).sum())
        assert seasons[column].sort_index().eq(expected.sort_index()).all(), (
            f"Incorrect team_seasons.{column}"
        )
    assert seasons.points_per_match.eq(
        seasons.points_from_matches / seasons.matches_played
    ).all(), "Incorrect sporting points_per_match"
    formats = {
        row["season_id"]: row
        for row in yaml.safe_load(seasons_path.read_text(encoding="utf-8"))["seasons"]
    }
    overrides = yaml.safe_load(overrides_path.read_text(encoding="utf-8"))["matches"]
    drops = []
    for game_id, rule in overrides.items():
        if rule.get("action") != "drop":
            continue
        assert game_id not in matches.index, f"Dropped match {game_id} present"
        assert "season_id" in rule, f"Missing schedule metadata for override {game_id}"
        if rule["season_id"] in selected:
            assert all(key in rule for key in ("home_team_id", "away_team_id", "stage")), (
                f"Incomplete override metadata: {game_id}"
            )
            drops.append(rule)
    for season_id, games in matches.groupby("season_id"):
        teams = seasons.xs(season_id, level="season_id")
        n_teams = formats[int(season_id)]["n_teams"]
        assert len(teams) == n_teams, f"Wrong team count: season {season_id}"
        assert teams.relegated.sum() == (3 if season_id == 13 else 2), (
            f"Wrong relegation count: season {season_id}"
        )
        assert teams.squad_row_available.all(), f"Missing squad_values row: season {season_id}"
        assert teams.goals_scored.sum() == teams.goals_conceded.sum(), (
            f"Unbalanced goals: season {season_id}"
        )
        assert (
            teams.points_from_matches.sum()
            == 3 * games.result.ne("draw").sum() + 2 * games.result.eq("draw").sum()
        ), f"Incorrect sporting points: season {season_id}"
        excluded = [rule for rule in drops if rule["season_id"] == season_id]
        for team_id, row in teams.iterrows():
            count = _scheduled_games(int(season_id), n_teams, row.final_stage)
            missing = sum(
                team_id in (rule["home_team_id"], rule["away_team_id"]) for rule in excluded
            )
            assert row.matches_played == count - missing, (
                f"Wrong schedule length: season {season_id}, team {team_id}"
            )
        regular = games.loc[games.stage.eq("regular")]
        for home in teams.index:
            for away in teams.index:
                if home >= away:
                    continue
                pair = regular.loc[
                    ((regular.home_team_id == home) & (regular.away_team_id == away))
                    | ((regular.home_team_id == away) & (regular.away_team_id == home))
                ]
                removed = [
                    rule
                    for rule in excluded
                    if rule["stage"] == "regular"
                    and {home, away} == {rule["home_team_id"], rule["away_team_id"]}
                ]
                assert len(pair) == (3 if n_teams == 12 else 2) - len(removed), (
                    f"Wrong regular pair count: season {season_id}, {home}/{away}"
                )
                if n_teams != 12:
                    for host in (home, away):
                        assert pair.home_team_id.eq(host).sum() == 1 - sum(
                            rule["home_team_id"] == host for rule in removed
                        ), f"Wrong home/away rotation: season {season_id}, {home}/{away}"
        summary = tables.season_summary.loc[season_id]
        assert summary.n_matches == len(games) and summary.n_teams == n_teams, (
            f"Incorrect season_summary counts: {season_id}"
        )
        assert summary.n_postponed == games.is_postponed.sum(), (
            f"Incorrect season_summary.n_postponed: {season_id}"
        )
        champion = teams.loc[teams.final_position.eq(1), "points_per_match"]
        line = teams.loc[teams.relegated].sort_values("final_position").points_per_match.iloc[0]
        assert (
            len(champion) == 1
            and summary.champion_points_per_match == champion.iloc[0]
            and summary.relegation_line_points_per_match == line
        ), f"Incorrect sporting points benchmark: {season_id}"
    exceptions = load_exceptions(exceptions_path)
    expected_conflicts = {
        game
        for game, kind in exceptions.items()
        if kind == "halftime_timeline_conflict" and game in matches.index
    }
    assert set(matches.index[matches.ht_conflict]) == expected_conflicts, (
        "Undocumented or missing halftime conflict flag"
    )
    events = tables.goal_events.reset_index()
    assert set(events.game_id) <= set(matches.index), "Unknown goal_events game"
    assert events.loc[events.game_id.isin(expected_conflicts), "half"].isna().all(), (
        "Conflicting halftime must have unknown goal half"
    )
    assert set(zip(events.game_id, events.team_id, strict=True)) <= set(
        team_matches.index.swaplevel()
    ), "Unknown scoring team"
    counts = events.groupby(["game_id", "team_id"]).size()
    halves = events.loc[events.half.eq(1)].groupby(["game_id", "team_id"]).size()
    for game_id, game in matches.iterrows():
        if game_id == 864982:
            assert not game.goals_available and not events.game_id.eq(game_id).any(), (
                "Unreliable goals must be excluded"
            )
            continue
        assert game.goals_available, f"Undocumented unavailable goals: {game_id}"
        for side in ("home", "away"):
            key = (game_id, game[f"{side}_team_id"])
            assert counts.get(key, 0) == game[f"{side}_goals"], (
                f"Goal count mismatch: {game_id}/{side}"
            )
            if not game.ht_conflict:
                assert halves.get(key, 0) == game[f"ht_{side}_goals"], (
                    f"Halftime goal count mismatch: {game_id}/{side}"
                )
    spells = tables.coach_spells.reset_index().sort_values(["team_id", "first_date"])
    assert set(spells.season_id_start) <= selected and set(spells.season_id_end) <= selected, (
        "Unexpected coach spell seasons"
    )
    assert spells.first_date.le(spells.last_date).all() and spells.matches.gt(0).all(), (
        "Invalid coach spell interval"
    )
    previous_end = spells.groupby("team_id").last_date.shift()
    assert (previous_end.isna() | spells.first_date.gt(previous_end)).all(), (
        "Overlapping coach spells"
    )
    warnings = []
    entries = yaml.safe_load(teams_path.read_text(encoding="utf-8"))["teams"]
    active = set(seasons.index.get_level_values("team_id"))
    for entry in entries:
        if entry["team_id"] not in active:
            continue
        aliases = entry.get("aliases", [])
        if any(SequenceMatcher(None, entry["name_he"], alias).ratio() < 0.45 for alias in aliases):
            warnings.append(f"Documented team aliases vary greatly: team_id={entry['team_id']}")
    missing = describe_missing_values(tables)
    for row in missing.loc[missing.n_missing.gt(0)].itertuples():
        assert pd.notna(row.reason), f"Undocumented missing values: {row.table}.{row.column}"
        if row.status != "ok":
            warnings.append(
                f"{row.table}.{row.column}: {row.n_missing} missing ({row.share:.1%}); {row.reason}"
            )
    return warnings
