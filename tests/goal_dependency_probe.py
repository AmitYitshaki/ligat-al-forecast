"""Executed in a clean Python subprocess, never imported by the pytest process."""

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goals_experiment import GoalUnit
from ligat.evaluate.goals_runner import UNIT_CODE_MODULES, goal_provenance, run_goal_unit
from ligat.models.m2_catboost import CatBoostGoalConfig
from ligat.models.m2_glm import GLMGoalConfig
from ligat.models.m2_maher import MaherGoalConfig
from ligat.models.m2_xgb import XGBGoalConfig

rows, goals = [], []
for season in range(8, 26):
    for number, (home, away) in enumerate(((1, 2), (2, 3), (3, 4), (4, 1))):
        date = pd.Timestamp(1998 + season, 8, 1, tz="Asia/Jerusalem") + pd.Timedelta(
            days=number, hours=12
        )
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            rows.append(
                dict(
                    game_id=season * 100 + number,
                    team_id=team,
                    season_id=season,
                    date=date,
                    as_of=date.normalize(),
                    is_home=is_home,
                    opponent_id=opponent,
                    promoted_in=False,
                )
            )
            goals.append(2 + (number + int(is_home)) % 3)
frame = pd.DataFrame(rows).set_index(["game_id", "team_id"])
target = pd.Series(goals, index=frame.index, name="goals_scored").sort_index()
frame = frame.sort_index()
context = (
    frame.reset_index()
    .set_index(["game_id", "team_id"], drop=False)
    .loc[:, ["game_id", "team_id", "season_id", "date", "as_of", "is_home"]]
)
data = GoalData(
    frame.assign(team_id=frame.index.get_level_values("team_id")).loc[
        :, ["team_id", "opponent_id", "is_home", "promoted_in"]
    ],
    target,
    context,
)
configs = [
    ("xgb", XGBGoalConfig(2, 1, 8, 0.1, 0.8, 0.8, 1), ("is_home",)),
    ("catboost", CatBoostGoalConfig(4, 1, 8, 0.1), ("is_home",)),
    ("glm", GLMGoalConfig("r0"), ("is_home",)),
    ("maher", MaherGoalConfig(), ("team_id", "opponent_id", "is_home")),
]
with tempfile.TemporaryDirectory() as directory:
    for backend, cfg, columns in configs:
        result = run_goal_unit(
            GoalUnit(backend, asdict(cfg), columns, 11),
            data,
            goal_provenance(data),
            Path(directory),
            stage="synthetic",
        )
        assert result.evaluation.per_team["lambda"].gt(0).all()
loaded = {
    name
    for name, module in sys.modules.items()
    if (name == "ligat" or name.startswith("ligat.")) and getattr(module, "__file__", None)
}
assert loaded <= set(UNIT_CODE_MODULES), sorted(loaded - set(UNIT_CODE_MODULES))
print(json.dumps({"loaded": sorted(loaded), "declared": list(UNIT_CODE_MODULES), "backends": 4}))
