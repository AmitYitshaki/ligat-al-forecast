"""Fixed M2 Poisson GLM recipes and reproducible historical reference rulers."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.linalg import qr  # type: ignore[import-untyped]

from ligat.features.registry import M2_BASE_COLUMNS
from ligat.models.form_models import PoissonFit, fit_poisson
from ligat.models.m2_xgb import STAGE_DTYPE, _alignment, _digest
from ligat.models.training_log import TrainingJournal

_FORMAT_VERSION = 2
_COLUMNS = {
    "r0": ("is_home",),
    "r1": ("is_home", "elo_diff"),
    "ruler": ("is_home", "gf_venue_ewma5", "opp_ga_venue_ewma5", "elo_diff"),
    "base": tuple(M2_BASE_COLUMNS),
}


@dataclass(frozen=True)
class GLMGoalConfig:
    """Recipe and earliest training season; observation weights belong to the judge."""

    recipe: Literal["r0", "r1", "ruler", "base"]
    training_start: int = 8

    def __post_init__(self) -> None:
        if self.recipe not in _COLUMNS:
            raise ValueError("Unknown GLM recipe")
        if (
            isinstance(self.training_start, (bool, np.bool_))
            or not isinstance(self.training_start, (int, np.integer))
            or self.training_start not in (8, 9)
        ):
            raise ValueError("GLM training_start must be 8 or 9")
        object.__setattr__(self, "training_start", int(self.training_start))
        if self.recipe == "base" and self.training_start != 9:
            raise ValueError("The base recipe requires training_start=9")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return _COLUMNS[self.recipe]

    @property
    def config_id(self) -> str:
        return "glm-" + _digest({"format_version": _FORMAT_VERSION, "config": asdict(self)})

    @property
    def label(self) -> str:
        names = {
            "r0": "R0 · בית",
            "r1": "R1 · Elo ובית",
            "ruler": "GLM · סרגל",
            "base": "GLM · בסיס",
        }
        return f"{names[self.recipe]} · מעונה {self.training_start}"


def glm_design(X: pd.DataFrame, config: GLMGoalConfig) -> pd.DataFrame:
    """Use the original EXP-002 ruler order and Q37's fixed base transforms."""
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or X.empty:
        raise ValueError("GLM requires nonempty distinct feature columns")
    if not set(config.feature_columns) <= set(X):
        raise ValueError("Missing registered GLM recipe columns")
    result = pd.DataFrame({"intercept": np.ones(len(X))}, index=X.index)
    for name in config.feature_columns:
        values = X[name]
        if name == "stage":
            if values.isna().any() or not values.isin(STAGE_DTYPE.categories).all():
                raise ValueError("GLM stage must contain four registered nonmissing categories")
            for stage in ("upper", "middle", "lower"):
                result[f"stage_{stage}"] = values.eq(stage).astype(float)
            continue
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ValueError(f"GLM {name} must be numeric or boolean")
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(numbers).all():
            raise ValueError(f"GLM {name} contains NaN or nonfinite values")
        if name in ("gf_venue_ewma5", "opp_ga_venue_ewma5"):
            if (numbers < 0).any():
                raise ValueError("GLM form goals must be nonnegative")
            numbers = np.log(numbers + 0.1)
        elif name in ("elo_diff", "elo_pre"):
            numbers = numbers / 400
        result[name] = numbers
    return result


class GLMGoalModel(TrainingJournal):
    """Full-rank unpenalized GLM, with explicit zero coefficients for absent stages."""

    refit_policy: ClassVar[Literal["per_fold"]] = "per_fold"
    supports_sample_weight: ClassVar[bool] = True

    def __init__(self, config: GLMGoalConfig, *, record_training_log: bool = True) -> None:
        self.record_training_log = record_training_log
        self._training_log = None
        self._config = config
        self._coefficients: pd.Series | None = None
        self.dropped_columns: tuple[str, ...] = ()
        self.training_seasons: tuple[int, ...] = ()
        self.fit_result: PoissonFit | None = None

    @property
    def config(self) -> GLMGoalConfig:
        return self._config

    @property
    def coefficients(self) -> pd.Series:
        if self._coefficients is None:
            raise RuntimeError("Fit or load GLM before reading coefficients")
        return self._coefficients.copy()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        context: pd.DataFrame,
    ) -> Self:
        started = perf_counter()
        self._training_log = None
        self._coefficients, self.fit_result = None, None
        self.dropped_columns, self.training_seasons = (), ()
        _alignment(X, context)
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise ValueError("GLM targets must align with features")
        if "season_id" not in context or context.season_id.isna().any():
            raise ValueError("GLM requires complete season_id context")
        selected = context.season_id.ge(self.config.training_start)
        if not selected.any():
            raise ValueError("GLM training_start leaves no training rows")
        weights = None
        if sample_weight is not None:
            if not isinstance(sample_weight, pd.Series) or not X.index.equals(sample_weight.index):
                raise ValueError("GLM weights must align with features")
            weights = sample_weight.loc[selected].to_numpy(float, na_value=np.nan)
            if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
                raise ValueError("GLM weights must be finite nonnegative with positive mass")
        design = glm_design(X.loc[selected], self.config)
        effective = design if weights is None else design.loc[weights > 0]
        dropped = tuple(c for c in design if c.startswith("stage_") and not effective[c].any())
        active = design.drop(columns=list(dropped))
        rank_design = active if weights is None else active.loc[weights > 0]
        rank = np.linalg.matrix_rank(rank_design.to_numpy())
        if rank != len(active.columns):
            _, _, pivots = qr(rank_design.to_numpy(), mode="economic", pivoting=True)
            dependent = [str(active.columns[i]) for i in pivots[rank:]]
            raise ValueError(f"GLM rank-deficient design; dependent columns: {dependent}")
        # Unit weights are exactly the historical unweighted likelihood.
        native_weights = None if weights is None or np.all(weights == 1) else weights
        losses: list[float] = []
        fitted = fit_poisson(
            active.to_numpy(),
            y.loc[selected].to_numpy(float),
            native_weights,
            callback=losses.append if self.record_training_log else None,
        )
        self._training_log = {
            "model": "glm",
            "config_id": self.config.config_id,
            "label": self.config.label,
            "actual_params": {
                "method": "BFGS",
                "gtol": 1e-8,
                "maxiter": 1000,
                **asdict(self.config),
            },
            "n_rows": int(selected.sum()),
            "n_features": len(active.columns),
            "fit_seconds": perf_counter() - started,
            "dataset": "training",
            "metric": "mean_poisson_without_log_factorial"
            if native_weights is None
            else "weighted_poisson_nll",
            "train_loss": losses,
            "iterations": fitted.iterations,
            "converged": fitted.converged,
            "message": "Accepted by precision-loss gradient check"
            if fitted.accepted_by_gradient_check
            else "Converged",
            "recorded": self.record_training_log,
        }
        coefficients = pd.Series(0.0, index=design.columns, name="coefficient")
        coefficients.loc[active.columns] = fitted.coefficients
        self._coefficients, self.fit_result = coefficients, fitted
        self.dropped_columns = dropped
        self.training_seasons = tuple(
            sorted(int(s) for s in context.loc[selected, "season_id"].unique())
        )
        return self

    def predict(self, X: pd.DataFrame, *, context: pd.DataFrame) -> pd.Series:
        _alignment(X, context)
        coefficients = self.coefficients
        design = glm_design(X, self.config)
        if not design.columns.equals(coefficients.index):
            raise ValueError("GLM prediction schema differs from the fitted matrix")
        values = np.exp(design.to_numpy() @ coefficients.to_numpy())
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("GLM must return finite positive lambda")
        return pd.Series(values, index=X.index.copy(), name="lambda")

    def save(self, path: Path) -> None:
        coefficients = self.coefficients
        payload = {
            "training_log": self.training_log,
            "format_version": _FORMAT_VERSION,
            "config": asdict(self.config),
            "config_id": self.config.config_id,
            "columns": coefficients.index.tolist(),
            "coefficients": coefficients.tolist(),
            "dropped_columns": self.dropped_columns,
            "training_seasons": self.training_seasons,
        }
        path.write_text(
            json.dumps({"payload": payload, "sha256": _digest(payload)}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or set(saved) != {"payload", "sha256"}:
            raise ValueError("Invalid GLM saved schema")
        payload = saved["payload"]
        if payload["format_version"] != _FORMAT_VERSION:
            raise ValueError(f"Unsupported GLM format_version; expected {_FORMAT_VERSION}")
        if saved["sha256"] != _digest(payload):
            raise ValueError("Invalid GLM signature or format version")
        config = GLMGoalConfig(**payload["config"])
        if payload["config_id"] != config.config_id:
            raise ValueError("GLM configuration signature mismatch")
        result = cls(config)
        result._training_log = payload["training_log"]
        result._coefficients = pd.Series(
            payload["coefficients"], index=payload["columns"], dtype=float, name="coefficient"
        )
        if not result._coefficients.index.is_unique or not np.isfinite(result._coefficients).all():
            raise ValueError("Invalid GLM saved coefficients")
        result.dropped_columns = tuple(payload["dropped_columns"])
        result.training_seasons = tuple(payload["training_seasons"])
        return result
