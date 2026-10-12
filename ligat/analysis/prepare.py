"""Pure stage-A analysis views; no publication, imputation or held-out analysis."""

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

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
