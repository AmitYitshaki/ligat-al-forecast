"""M2 CatBoost Poisson adapter with shared categories and auditable native defaults."""

import hashlib
import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from catboost import CatBoostRegressor  # type: ignore[import-untyped]
from catboost import __version__ as CATBOOST_VERSION

from ligat.evaluate.goal_metrics import poisson_nll
from ligat.models.m2_identity import IDENTITY_COLUMNS, TeamIdentity
from ligat.models.m2_xgb import STAGE_DTYPE, _alignment, _digest, _features, _Schema
from ligat.models.training_log import TrainingJournal

_FORMAT_VERSION = 3
_FIXED = {
    "loss_function": "Poisson",
    "task_type": "CPU",
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
}
_TUNABLE = ("depth", "l2_leaf_reg", "iterations", "learning_rate", "one_hot_max_size")


@dataclass(frozen=True)
class CatBoostGoalConfig:
    """Explicit grid values or an explicit preset that leaves defaults to CatBoost.

    None is allowed only in library_default, where all five tunable parameters
    are omitted from the native constructor. Fitted defaults are recorded separately.
    """

    depth: int | None
    l2_leaf_reg: float | None
    iterations: int | None
    learning_rate: float | None
    seed: int = 0
    preset: Literal["grid", "library_default"] = "grid"
    one_hot_max_size: int | None = 4

    def __post_init__(self) -> None:
        if self.preset not in ("grid", "library_default"):
            raise ValueError("An explicit supported CatBoost preset is required")
        value = self.one_hot_max_size
        if value is not None:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError("one_hot_max_size must be an integer >= 1")
            object.__setattr__(self, "one_hot_max_size", int(value))
        elif self.preset == "grid":
            raise ValueError("one_hot_max_size must be an integer >= 1 in grid configs")
        for name in ("seed", "depth", "iterations"):
            value = getattr(self, name)
            if value is None and name != "seed" and self.preset == "library_default":
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            upper = 16 if name == "depth" else 2**31 - 1
            if not (0 if name == "seed" else 1) <= value <= upper:
                raise ValueError(f"{name} is outside its supported range")
            object.__setattr__(self, name, int(value))
        for name in ("l2_leaf_reg", "learning_rate"):
            value = getattr(self, name)
            if value is None and self.preset == "library_default":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value < 0
                or (name == "learning_rate" and not 0 < value <= 1)
            ):
                raise ValueError(f"{name} is outside its supported range")
            object.__setattr__(self, name, float(value))
        if self.preset == "library_default" and any(
            getattr(self, name) is not None for name in _TUNABLE
        ):
            raise ValueError("library_default must omit all five tunable parameters")

    @classmethod
    def library_default(cls, *, seed: int = 0) -> Self:
        return cls(
            None, None, None, None, seed=seed, preset="library_default", one_hot_max_size=None
        )

    @property
    def config_id(self) -> str:
        """Canonical full SHA-256 including backend version and fixed execution settings."""
        return f"catboost-{self.preset}-" + _digest(
            {
                "format_version": _FORMAT_VERSION,
                "catboost_version": CATBOOST_VERSION,
                "fixed": _FIXED,
                "config": asdict(self),
            }
        )

    @property
    def label(self) -> str:
        if self.preset == "library_default":
            label = "ברירת המחדל של CatBoost"
            return label if self.seed == 0 else f"{label} · seed{self.seed}"
        fields = (
            ("d", self.depth),
            ("l2=", self.l2_leaf_reg),
            ("it", self.iterations),
            ("lr", self.learning_rate),
            ("oh", self.one_hot_max_size),
            ("seed", self.seed),
        )
        return "catboost · " + " · ".join(
            f"{name}{int(value) if float(value).is_integer() else repr(value)}"
            for name, value in fields
            if value is not None
        )


def _cat_features(prepared: pd.DataFrame, schema: _Schema) -> list[str]:
    columns = [name for name, _, kind in schema if kind in ("stage", "identity")]
    if prepared[columns].isna().any().any():
        raise ValueError("CatBoost categorical values must be present; numeric NaN is supported")
    return columns


class CatBoostGoalModel(TrainingJournal):
    """Positive Poisson means, raw judge weights and immutable shared identity history."""

    refit_policy: ClassVar[Literal["per_fold"]] = "per_fold"
    supports_sample_weight: ClassVar[bool] = True

    def __init__(
        self,
        config: CatBoostGoalConfig,
        *,
        team_ids: Sequence[int] | None = None,
        record_training_log: bool = True,
        identity_columns: Sequence[str] = IDENTITY_COLUMNS,
    ) -> None:
        if (
            not identity_columns
            or len(set(identity_columns)) != len(identity_columns)
            or not set(identity_columns) <= set(IDENTITY_COLUMNS)
        ):
            raise ValueError("Select a nonempty subset of the shared identity columns")
        self._identity_columns = tuple(identity_columns)
        self.record_training_log = record_training_log
        self._training_log = None
        self._config = config
        self._team_ids = TeamIdentity(tuple(team_ids)).team_ids if team_ids is not None else None
        self._identity: TeamIdentity | None = None
        self._estimator: CatBoostRegressor | None = None
        self._schema: _Schema | None = None
        self._actual_params: dict[str, object] | None = None

    @property
    def config(self) -> CatBoostGoalConfig:
        return self._config

    @property
    def identity(self) -> TeamIdentity | None:
        return self._identity

    @property
    def actual_params(self) -> dict[str, object]:
        """A defensive copy of the complete fitted get_all_params(), including defaults."""
        if self._actual_params is None:
            raise RuntimeError("Fit or load the goal model before reading actual parameters")
        return deepcopy(self._actual_params)

    def _new_estimator(self) -> CatBoostRegressor:
        parameters = {name: getattr(self.config, name) for name in _TUNABLE}
        if self.config.preset == "library_default":
            parameters = {}
        return CatBoostRegressor(**_FIXED, random_seed=self.config.seed, **parameters)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        context: pd.DataFrame,
    ) -> Self:
        """Fit supplied past rows only, without validation data or early stopping."""
        started = perf_counter()
        self._training_log = None
        self._estimator, self._schema, self._identity, self._actual_params = None, None, None, None
        identity = None
        features = X
        if self._team_ids is not None:
            identity = TeamIdentity.from_context(context, team_ids=self._team_ids)
            features = identity.transform(X, context=context, columns=self._identity_columns)
        prepared, schema = _features(
            features, with_identity=identity is not None, identity_columns=self._identity_columns
        )
        categories = _cat_features(prepared, schema)
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
        estimator.fit(prepared, goals, sample_weight=weights, cat_features=categories)
        self._actual_params = deepcopy(estimator.get_all_params())
        self._training_log = {
            "model": "catboost",
            "config_id": self.config.config_id,
            "label": self.config.label,
            "actual_params": self.actual_params,
            "n_rows": len(X),
            "n_features": prepared.shape[1],
            "fit_seconds": perf_counter() - started,
            "dataset": "training",
            "metric": "Poisson",
            "train_loss": estimator.get_evals_result()["learn"]["Poisson"]
            if self.record_training_log
            else [],
            "iterations": estimator.tree_count_,
            "converged": True,
            "recorded": self.record_training_log,
        }
        self._estimator, self._schema, self._identity = estimator, schema, identity
        return self

    def predict(self, X: pd.DataFrame, *, context: pd.DataFrame) -> pd.Series:
        """Ask explicitly for exp(raw Poisson score), exactly once inside CatBoost."""
        if self._estimator is None or self._schema is None:
            raise RuntimeError("Fit or load the goal model before prediction")
        features = (
            self._identity.transform(X, context=context, columns=self._identity_columns)
            if self._identity is not None
            else X
        )
        prepared, schema = _features(
            features,
            with_identity=self._identity is not None,
            identity_columns=self._identity_columns,
        )
        _alignment(X, context)
        _cat_features(prepared, schema)
        if schema != self._schema:
            raise ValueError("Prediction columns, order or dtypes differ from the fitted schema")
        values = np.asarray(
            self._estimator.predict(prepared, prediction_type="Exponent", thread_count=1),
            dtype=float,
        )
        if values.shape != (len(X),) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Poisson trees must return one positive finite lambda per row")
        return pd.Series(values, index=X.index.copy(), name="lambda")

    def save(self, path: Path) -> None:
        """Bind native CBM and its complete configuration/schema/defaults sidecar both ways."""
        if self._estimator is None or self._schema is None:
            raise RuntimeError("Fit or load the goal model before saving")
        if path.suffix != ".cbm":
            raise ValueError("M2 CatBoost models require a .cbm path")
        metadata = {
            "training_log": self.training_log,
            "format_version": _FORMAT_VERSION,
            "catboost_version": CATBOOST_VERSION,
            "config": asdict(self.config),
            "config_id": self.config.config_id,
            "features": [list(f) for f in self._schema],
            "stage_categories": list(STAGE_DTYPE.categories),
            "stage_ordered": False,
            "actual_params": self.actual_params,
        }
        if self._identity is not None:
            metadata["identity"] = self._identity.to_metadata()
        self._estimator.get_metadata()["ligat_m2_schema_sha256"] = _digest(metadata)
        self._estimator.save_model(str(path), format="cbm")
        metadata["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reject mismatched native model, sidecar, actual defaults or library version."""
        if path.suffix != ".cbm":
            raise ValueError("M2 CatBoost models require a .cbm path")
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if isinstance(metadata, dict) and metadata.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported catboost format_version {metadata.get('format_version')}; "
                f"expected {_FORMAT_VERSION}"
            )
        keys = {
            "training_log",
            "format_version",
            "catboost_version",
            "config",
            "config_id",
            "features",
            "stage_categories",
            "stage_ordered",
            "actual_params",
            "model_sha256",
        }
        if not isinstance(metadata, dict) or set(metadata) not in (keys, keys | {"identity"}):
            raise ValueError("Invalid M2 CatBoost sidecar schema")
        if metadata.pop("model_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Model CBM does not match its sidecar")
        if metadata["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported CatBoost sidecar format_version {metadata['format_version']}; "
                f"expected {_FORMAT_VERSION}. Retrain with the explicit categorical encoding."
            )
        if (
            metadata["catboost_version"] != CATBOOST_VERSION
            or metadata["stage_categories"] != list(STAGE_DTYPE.categories)
            or metadata["stage_ordered"] is not False
        ):
            raise ValueError("Incompatible M2 schema or CatBoost version")
        try:
            config = CatBoostGoalConfig(**metadata["config"])
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
        identity = (
            TeamIdentity.from_metadata(metadata["identity"]) if "identity" in metadata else None
        )
        selected = tuple(row[0] for row in schema if row[2] == "identity")
        if (identity is None and selected) or (
            identity is not None and (not selected or not set(selected) <= set(IDENTITY_COLUMNS))
        ):
            raise ValueError("Identity schema and saved history disagree")
        result = cls(
            config,
            team_ids=identity.team_ids if identity is not None else None,
            identity_columns=selected if identity is not None else IDENTITY_COLUMNS,
        )
        estimator = result._new_estimator()
        estimator.load_model(str(path), format="cbm")
        # CBM rounds get_all_params() floats on reload (e.g. 0.10000000149011612
        # becomes 0.1000000015). The embedded digest binds the original complete
        # parameters exactly; preserve those original values from the sidecar.
        if (
            estimator.get_metadata().get("ligat_m2_schema_sha256") != _digest(metadata)
            or estimator.feature_names_ != [row[0] for row in schema]
            or estimator.get_cat_feature_indices()
            != [i for i, row in enumerate(schema) if row[2] in ("stage", "identity")]
        ):
            raise ValueError("Model schema or actual parameters do not match its sidecar")
        result._estimator, result._schema, result._identity = estimator, schema, identity
        result._actual_params = deepcopy(metadata["actual_params"])
        result._training_log = metadata["training_log"]
        return result
