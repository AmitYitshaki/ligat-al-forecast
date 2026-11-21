"""M2 Poisson trees: explicit configurations, fixed stage categories and model IO."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from xgboost import XGBRegressor
from xgboost import __version__ as XGBOOST_VERSION

from ligat.evaluate.goal_metrics import poisson_nll
from ligat.features.prematch_v2 import PREMATCH_V2_COLUMNS

STAGE_DTYPE = pd.CategoricalDtype(["regular", "upper", "middle", "lower"])
_FORMAT_VERSION = 1
_FIXED = {
    "objective": "count:poisson",
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 1,
    "enable_categorical": True,
}
_DEFAULTS = {
    "max_depth": 6,
    "min_child_weight": 1.0,
    "n_estimators": 100,
    "learning_rate": 0.3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "max_delta_step": 0.7,
}
_Schema = tuple[tuple[str, str, str], ...]


def _digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class XGBGoalConfig:
    """All tunable parameters are explicit; a library-default preset cannot be partial.

    IDs include the complete parameters, preset, fixed execution settings, schema
    version and XGBoost version. Observation weights belong to the judge/run,
    not to this estimator configuration, and are never rescaled by the adapter.
    """

    max_depth: int
    min_child_weight: float
    n_estimators: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    max_delta_step: float = 0.7
    seed: int = 0
    preset: Literal["grid", "library_default"] = "grid"

    def __post_init__(self) -> None:
        for name in ("max_depth", "n_estimators", "seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "seed" else 1) or value > 2**31 - 1:
                raise ValueError(f"{name} is outside its supported range")
            object.__setattr__(self, name, int(value))
        for name in (
            "min_child_weight",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "max_delta_step",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if any(
            not 0 < getattr(self, c) <= 1
            for c in ("learning_rate", "subsample", "colsample_bytree")
        ):
            raise ValueError("learning_rate, subsample and colsample_bytree must be in (0, 1]")
        if self.max_delta_step != 0.7 or self.preset not in ("grid", "library_default"):
            raise ValueError("M2 requires max_delta_step=0.7 and an explicit supported preset")
        if self.preset == "library_default" and any(
            getattr(self, k) != v for k, v in _DEFAULTS.items()
        ):
            raise ValueError("library_default requires all registered library-default values")

    @classmethod
    def library_default(cls, *, seed: int = 0) -> Self:
        """The registered complete default combination, with an explicit experiment seed."""
        return cls(**_DEFAULTS, seed=seed, preset="library_default")  # type: ignore[arg-type]

    @property
    def config_id(self) -> str:
        """Stable full SHA-256 of canonical parameters and execution settings, never hash()."""
        return f"xgb-{self.preset}-" + _digest(
            {
                "format_version": _FORMAT_VERSION,
                "xgboost_version": XGBOOST_VERSION,
                "fixed": _FIXED,
                "config": asdict(self),
            }
        )


def _features(X: pd.DataFrame) -> tuple[pd.DataFrame, _Schema]:
    if (
        not isinstance(X, pd.DataFrame)
        or X.empty
        or not X.columns.is_unique
        or not all(isinstance(c, str) for c in X.columns)
        or not set(X) <= set(PREMATCH_V2_COLUMNS)
        or set(X) & {"team_id", "opponent_id"}
        or not X.index.is_unique
        or X.index.to_frame(index=False).isna().any().any()
    ):
        raise ValueError(
            "C1a requires distinct registered feature columns and complete unique keys; no team IDs"
        )
    result, schema = X.copy(deep=True), []
    for column in X:
        values = X[column]
        if column == "stage":
            if not values.dropna().isin(STAGE_DTYPE.categories).all():
                raise ValueError("stage must be regular, upper, middle or lower")
            result[column] = values.astype(STAGE_DTYPE)
            schema.append((column, "category", "stage"))
        else:
            if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_complex_dtype(
                values.dtype
            ):
                raise ValueError(f"{column} must be numeric or boolean")
            kind = "boolean" if pd.api.types.is_bool_dtype(values.dtype) else "numeric"
            numbers = values.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(numbers).any():
                raise ValueError(f"{column} contains infinite feature values")
            result[column] = numbers
            schema.append((column, str(values.dtype), kind))
    return result, tuple(schema)


def _alignment(X: pd.DataFrame, context: pd.DataFrame) -> None:
    if not isinstance(context, pd.DataFrame) or not X.index.equals(context.index):
        raise ValueError("Goal feature and context keys must align in exact order")


class XGBGoalModel:
    """One positive lambda per team row; the judge alone supplies history and weights."""

    refit_policy: ClassVar[Literal["per_fold"]] = "per_fold"
    supports_sample_weight: ClassVar[bool] = True

    def __init__(self, config: XGBGoalConfig) -> None:
        self._config = config
        self._estimator: XGBRegressor | None = None
        self._schema: _Schema | None = None

    @property
    def config(self) -> XGBGoalConfig:
        return self._config

    def _new_estimator(self) -> XGBRegressor:
        parameters = asdict(self.config)
        parameters.pop("preset")
        parameters["random_state"] = parameters.pop("seed")
        return XGBRegressor(**_FIXED, **parameters)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        context: pd.DataFrame,
    ) -> Self:
        """Fit past rows, consuming raw aligned weights; no validation data or early stopping."""
        self._estimator, self._schema = None, None
        prepared, schema = _features(X)
        _alignment(X, context)
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise ValueError("Goal targets must align with feature keys in exact order")
        if (
            not pd.api.types.is_numeric_dtype(y.dtype)
            or pd.api.types.is_bool_dtype(y.dtype)
            or pd.api.types.is_complex_dtype(y.dtype)
        ):
            raise ValueError("Goal targets must be numeric nonnegative integers")
        goals = y.to_numpy(dtype=float, na_value=np.nan)
        poisson_nll(goals, np.ones(len(goals)))
        weights = None
        if sample_weight is not None:
            if (
                not isinstance(sample_weight, pd.Series)
                or not X.index.equals(sample_weight.index)
                or not pd.api.types.is_numeric_dtype(sample_weight.dtype)
                or pd.api.types.is_bool_dtype(sample_weight.dtype)
                or pd.api.types.is_complex_dtype(sample_weight.dtype)
            ):
                raise ValueError("sample_weight must be an aligned numeric Series")
            weights = sample_weight.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
                raise ValueError("sample_weight must be finite nonnegative with positive mass")
        if not ((goals > 0) & (True if weights is None else weights > 0)).any():
            raise ValueError("Poisson trees require positive weighted goal mass")
        estimator = self._new_estimator()
        estimator.fit(prepared, goals, sample_weight=weights)
        self._estimator, self._schema = estimator, schema
        return self

    def predict(self, X: pd.DataFrame, *, context: pd.DataFrame) -> pd.Series:
        """Return the native Poisson mean directly, preserving the requested index order."""
        if self._estimator is None or self._schema is None:
            raise RuntimeError("Fit or load the goal model before prediction")
        prepared, schema = _features(X)
        _alignment(X, context)
        if schema != self._schema:
            raise ValueError("Prediction columns, order or dtypes differ from the fitted schema")
        values = np.asarray(self._estimator.predict(prepared), dtype=float)
        if values.shape != (len(X),) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Poisson trees must return one positive finite lambda per row")
        return pd.Series(values, index=X.index.copy(), name="lambda")

    def save(self, path: Path) -> None:
        """Save native JSON plus a schema/config sidecar bound to that exact model."""
        if self._estimator is None or self._schema is None:
            raise RuntimeError("Fit or load the goal model before saving")
        if path.suffix != ".json":
            raise ValueError("M2 XGBoost models require a .json path")
        metadata = {
            "format_version": _FORMAT_VERSION,
            "xgboost_version": XGBOOST_VERSION,
            "config": asdict(self.config),
            "config_id": self.config.config_id,
            "features": [list(f) for f in self._schema],
            "stage_categories": list(STAGE_DTYPE.categories),
            "stage_ordered": False,
        }
        self._estimator.get_booster().set_attr(ligat_m2_schema_sha256=_digest(metadata))
        self._estimator.save_model(path)
        metadata["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reject mismatched sidecars, model files, schemas or library versions before use."""
        if path.suffix != ".json":
            raise ValueError("M2 XGBoost models require a .json path")
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        keys = {
            "format_version",
            "xgboost_version",
            "config",
            "config_id",
            "features",
            "stage_categories",
            "stage_ordered",
            "model_sha256",
        }
        if not isinstance(metadata, dict) or set(metadata) != keys:
            raise ValueError("Invalid M2 model sidecar schema")
        if metadata.pop("model_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Model JSON does not match its sidecar")
        if (
            metadata["format_version"] != _FORMAT_VERSION
            or metadata["xgboost_version"] != XGBOOST_VERSION
            or metadata["stage_categories"] != list(STAGE_DTYPE.categories)
            or metadata["stage_ordered"] is not False
        ):
            raise ValueError("Incompatible M2 schema or XGBoost version")
        try:
            config = XGBGoalConfig(**metadata["config"])
            features = metadata["features"]
            if (
                not isinstance(features, list)
                or not features
                or any(
                    not isinstance(row, list)
                    or len(row) != 3
                    or not all(isinstance(v, str) for v in row)
                    for row in features
                )
            ):
                raise ValueError("Invalid M2 feature schema")
            schema = tuple(tuple(row) for row in features)
        except TypeError as exc:
            raise ValueError("Invalid M2 configuration in sidecar") from exc
        if metadata["config_id"] != config.config_id:
            raise ValueError("Model config_id does not match its configuration")
        result = cls(config)
        estimator = result._new_estimator()
        estimator.load_model(path)
        booster = estimator.get_booster()
        if (
            booster.attr("ligat_m2_schema_sha256") != _digest(metadata)
            or booster.feature_names != [row[0] for row in schema]
            or booster.feature_types != ["c" if row[2] == "stage" else "float" for row in schema]
        ):
            raise ValueError("Model feature schema does not match its sidecar")
        result._estimator, result._schema = estimator, schema
        return result
