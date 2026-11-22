"""EXP-005 declarations: fixed grids, registered features and a score-free pilot plan."""

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import yaml

from ligat.features import registry
from ligat.models.m2_catboost import CatBoostGoalConfig
from ligat.models.m2_glm import GLMGoalConfig
from ligat.models.m2_maher import MaherGoalConfig
from ligat.models.m2_maher_prior import PromotedPrior, load_promoted_prior
from ligat.models.m2_xgb import XGBGoalConfig, _digest

GoalBackend = Literal["xgb", "catboost", "glm", "maher"]


@dataclass(frozen=True)
class GoalUnit:
    """One model/configuration/features/weight/fold; role is reporting-only."""

    backend: GoalBackend
    parameters: dict[str, Any]
    features: tuple[str, ...]
    season: int
    half_life: float | None = None
    role: str = "unit"

    def __post_init__(self) -> None:
        if self.backend not in ("xgb", "catboost", "glm", "maher") or self.season not in range(
            11, 26
        ):
            raise ValueError("Goal unit requires a declared backend and development fold")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Goal unit requires distinct explicit features")
        if self.half_life not in (None, 1, 2, 4) or (
            self.backend == "maher" and self.half_life is not None
        ):
            raise ValueError("Invalid judge weights for goal unit")
        self.configuration()

    def configuration(self) -> XGBGoalConfig | CatBoostGoalConfig | GLMGoalConfig | MaherGoalConfig:
        if self.backend == "xgb":
            return XGBGoalConfig(**self.parameters)
        if self.backend == "catboost":
            return CatBoostGoalConfig(**self.parameters)
        if self.backend == "glm":
            return GLMGoalConfig(**self.parameters)
        parameters = dict(self.parameters)
        if parameters.get("prior") is not None:
            parameters["prior"] = PromotedPrior(**parameters["prior"])
        return MaherGoalConfig(**parameters)

    def identity(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "config_id": self.configuration().config_id,
            "parameters": self.parameters,
            "features": self.features,
            "season": self.season,
            "half_life": self.half_life,
        }

    @property
    def unit_id(self) -> str:
        return "unit-" + _digest(self.identity())

    @property
    def label(self) -> str:
        return self.configuration().label


@dataclass(frozen=True)
class GoalExperimentSpec:
    """Validated fixed declaration; authorization gates remain explicit values."""

    raw: dict[str, Any]

    @property
    def n_jobs(self) -> int:
        return int(self.raw["n_jobs"])

    @property
    def output(self) -> Path:
        return Path(self.raw["output"])

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(self.raw["evaluation"]["validation_seasons"])

    @property
    def pilot_seasons(self) -> tuple[int, ...]:
        return tuple(self.raw["evaluation"]["pilot_seasons"])

    def guard_full(self) -> None:
        if self.raw["full_run_approved"] is not True:
            raise ValueError(
                "Full EXP-005 is not approved; full_run_approved must be explicitly true"
            )
        if self.raw["fast_path"] not in ("full", "short"):
            raise ValueError("Full run requires Amit's explicit full/short path decision")
        if not isinstance(self.raw["correlation_drops"], list):
            raise ValueError("Full run requires an approved fixed correlation_drops list")


def load_goal_experiment(path: Path = Path("config/exp005.yaml")) -> GoalExperimentSpec:
    """Fail closed on altered grids, seasons, resource limits or unresolved declarations."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("experiment_id") != "EXP-005":
        raise ValueError("Expected EXP-005 declaration")
    expected = {
        "max_depth": [2, 3, 4, 6, 8],
        "min_child_weight": [1, 10, 30],
        "n_estimators": [150, 400, 1000],
        "learning_rate": [0.05, 0.1],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1,
        "library_default": True,
    }
    if raw["xgboost"]["stage_a"] != expected or raw["xgboost"]["stage_b"] != {
        "reg_lambda": [0, 1, 10],
        "subsample": [0.7, 1.0],
        "half_life": [None, 1, 2, 4],
    }:
        raise ValueError("XGBoost grids differ from the preregistration")
    if raw["catboost"] != {
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 10],
        "iterations": 1000,
        "learning_rate": 0.05,
        "one_hot_max_size": 4,
        "library_default": True,
    }:
        raise ValueError("CatBoost grid differs from the preregistration")
    if raw["glm"] != {"recipe": "base", "training_start": 9, "half_life": [None, 1, 2, 4]} or raw[
        "maher"
    ] != {"half_life": [None, 0.5, 1, 2]}:
        raise ValueError("Baseline grids differ from the preregistration")
    if raw["evaluation"] != {
        "training_start": 8,
        "validation_seasons": list(range(11, 26)),
        "pilot_seasons": [12, 18, 25],
    }:
        raise ValueError("Only preregistered development folds are permitted")
    if raw["features"] != {
        "s1": "M2_S1_COLUMNS",
        "base": "M2_BASE_COLUMNS",
        "families": "M2_FAMILY_COLUMNS",
        "base_groups": "M2_BASE_GROUPS",
    }:
        raise ValueError("Feature lists must resolve through the registry")
    if (
        isinstance(raw["n_jobs"], bool)
        or not isinstance(raw["n_jobs"], int)
        or not 1 <= raw["n_jobs"] <= 15
    ):
        raise ValueError("EXP-005 permits one to fifteen model workers")
    if (
        raw["seed"] != 0
        or raw["catboost_f6"] != "conditional"
        or not isinstance(raw["full_run_approved"], bool)
    ):
        raise ValueError("Invalid seed, conditional F6 policy or approval flag")
    if not isinstance(raw.get("catboost_f6_requested", False), bool):
        raise ValueError("catboost_f6_requested must be an explicit boolean")
    return GoalExperimentSpec(raw)


def xgb_stage_a(spec: GoalExperimentSpec) -> tuple[XGBGoalConfig, ...]:
    raw = spec.raw["xgboost"]["stage_a"]
    configs = [
        XGBGoalConfig(d, m, n, lr, 0.8, 0.8, 1, seed=0)
        for d, m, n, lr in product(
            raw["max_depth"], raw["min_child_weight"], raw["n_estimators"], raw["learning_rate"]
        )
    ]
    return tuple([*configs, XGBGoalConfig.library_default()])


def xgb_stage_b(
    spec: GoalExperimentSpec, anchor: XGBGoalConfig
) -> tuple[tuple[XGBGoalConfig, float | None], ...]:
    raw = spec.raw["xgboost"]["stage_b"]
    return tuple(
        (
            XGBGoalConfig(
                anchor.max_depth,
                anchor.min_child_weight,
                anchor.n_estimators,
                anchor.learning_rate,
                ss,
                0.8,
                rl,
                seed=0,
            ),
            hl,
        )
        for rl, ss, hl in product(raw["reg_lambda"], raw["subsample"], raw["half_life"])
    )


def catboost_grid(spec: GoalExperimentSpec) -> tuple[CatBoostGoalConfig, ...]:
    raw = spec.raw["catboost"]
    return tuple(
        [
            *(
                CatBoostGoalConfig(d, l2, 1000, 0.05)
                for d, l2 in product(raw["depth"], raw["l2_leaf_reg"])
            ),
            CatBoostGoalConfig.library_default(),
        ]
    )


def family_features(current: tuple[str, ...], family: str) -> tuple[str, ...]:
    """F1-R replaces the two venue form columns; every other family adds its fields."""
    base = tuple(c for c in current if family != "F1-R" or c not in registry.M2_BASE_GROUPS["form"])
    return tuple(dict.fromkeys((*base, *registry.M2_FAMILY_COLUMNS[family])))


def pilot_units(spec: GoalExperimentSpec) -> tuple[GoalUnit, ...]:
    """Exactly D0's fixed timing-only plan; no data or scores select these units."""
    base = registry.M2_BASE_COLUMNS
    units: list[GoalUnit] = []
    anchor = XGBGoalConfig(4, 10, 400, 0.05, 0.8, 0.8, 1)
    frozen = load_promoted_prior(Path(spec.raw["inputs"]["maher_prior"]))
    width = [
        XGBGoalConfig(2, 1, 150, 0.1, 0.8, 0.8, 1),
        anchor,
        XGBGoalConfig(8, 30, 1000, 0.05, 0.8, 0.8, 1),
    ]
    for season in spec.pilot_seasons:
        units.extend(
            GoalUnit("xgb", asdict(c), base, season, role="S2-A") for c in xgb_stage_a(spec)
        )
        units.extend(
            GoalUnit("xgb", asdict(c), base, season, hl, "timing-B-anchor")
            for c, hl in xgb_stage_b(spec, anchor)
        )
        units.extend(
            GoalUnit("catboost", asdict(c), base, season, role="catboost-base")
            for c in catboost_grid(spec)
        )
        units.extend(
            GoalUnit("glm", asdict(GLMGoalConfig("base", 9)), base, season, hl, "glm-base")
            for hl in spec.raw["glm"]["half_life"]
        )
        units.extend(
            GoalUnit(
                "maher",
                asdict(MaherGoalConfig(hl, frozen)),
                ("team_id", "opponent_id", "is_home", "promoted_in"),
                season,
                role="maher",
            )
            for hl in spec.raw["maher"]["half_life"]
        )
        units.extend(
            GoalUnit("xgb", asdict(c), family_features(base, "F1-A"), season, role="width-F1-A")
            for c in width
        )
    return tuple(units)
