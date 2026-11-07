"""M0 assembly from four frozen blocks; never recompute a feature block or target."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa
import yaml
from pandas.testing import assert_frame_equal, assert_series_equal  # type: ignore[import-untyped]

from ligat.features.registry import (
    FORM_COLUMNS,
    PREMATCH_CONTRAST_COLUMNS,
    PREMATCH_OPPONENT_COLUMNS,
    SEASON_CONTEXT_COLUMNS,
    validate_prematch_feature_columns,
)

DEVELOPMENT_SEASONS = tuple(range(8, 26))
PREMATCH_ROWS = 8630
TABLE_COLUMNS = (
    "season_id",
    "date",
    "as_of",
    "round",
    "stage",
    "is_home",
    "opponent_id",
    "games_played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
    "position",
    "group_position",
    "points_per_game",
    "position_pct",
    "gap_to_leader",
    "gap_to_safety",
    "gap_to_cut",
    "games_remaining_stage",
    "games_behind",
    "title_decided",
    "cut_decided",
    "relegation_decided",
    "places_decided",
    "nothing_to_play",
    "rounds_to_stage_end",
)
ELO_COLUMNS = ("elo_pre", "elo_opponent_pre", "elo_diff", "elo_expected")
SELECTED_FORM_COLUMNS = ("gf_venue_ewma5", "ga_venue_ewma5", "pts_venue_ewma5")
SELECTED_CONTEXT_COLUMNS = tuple(c for c in SEASON_CONTEXT_COLUMNS if c != "prev_value_share_total")
PREVIOUS_VALUE_COLUMNS = SELECTED_CONTEXT_COLUMNS[:4]
PREMATCH_COLUMNS = (
    *TABLE_COLUMNS,
    *ELO_COLUMNS,
    *SELECTED_FORM_COLUMNS,
    *PREMATCH_OPPONENT_COLUMNS,
    *SELECTED_CONTEXT_COLUMNS,
    *PREMATCH_CONTRAST_COLUMNS,
)
INPUT_COLUMNS = {
    "table_state": TABLE_COLUMNS,
    "elo": ("season_id", "as_of", *ELO_COLUMNS),
    "form": FORM_COLUMNS,
    "season_context": SEASON_CONTEXT_COLUMNS,
}
_NUMERIC_COLUMNS = (
    *ELO_COLUMNS,
    *SELECTED_FORM_COLUMNS,
    *PREMATCH_OPPONENT_COLUMNS,
    *PREVIOUS_VALUE_COLUMNS,
    *PREMATCH_CONTRAST_COLUMNS,
)
PREMATCH_SCHEMA = pa.DataFrameSchema(
    {
        column: pa.Column(
            float
            if column in _NUMERIC_COLUMNS
            else bool
            if column in ("promoted_in", "squad_value_imputed", "prev_season_known")
            else int
            if column == "league_size"
            else None,
            nullable=column
            in (*TABLE_COLUMNS, *PREVIOUS_VALUE_COLUMNS, *PREMATCH_CONTRAST_COLUMNS),
        )
        for column in PREMATCH_COLUMNS
    },
    index=pa.MultiIndex([pa.Index(int, name="game_id"), pa.Index(int, name="team_id")]),
    strict=True,
    ordered=True,
    coerce=False,
)


@dataclass(frozen=True)
class FrozenPrematchInput:
    """A source artifact and its separately pinned sidecar."""

    path: Path
    sha256: str
    metadata_sha256: str

    def record(self) -> dict[str, str]:
        """Return the portable provenance stored in the unified artifact."""
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "metadata_sha256": self.metadata_sha256,
        }


@dataclass(frozen=True)
class PrematchConfig:
    """Exactly four declared, immutable inputs; no model or training settings."""

    inputs: dict[str, FrozenPrematchInput]
    sha256: str

    def input_records(self) -> dict[str, dict[str, str]]:
        """Include both artifact and sidecar identities in downstream provenance."""
        return {name: self.inputs[name].record() for name in INPUT_COLUMNS}

    def input_fingerprint(self) -> str:
        """Hash the four complete input identities, independently of YAML comments."""
        return hashlib.sha256(json.dumps(self.input_records(), sort_keys=True).encode()).hexdigest()


def load_prematch_config(path: Path = Path("config/prematch.yaml")) -> PrematchConfig:
    """Load the fixed block contract, rejecting undeclared blocks or incomplete hashes."""
    content = path.read_bytes()
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid prematch YAML: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"inputs"}:
        raise ValueError("Prematch config requires only inputs")
    entries = raw["inputs"]
    if not isinstance(entries, dict) or set(entries) != set(INPUT_COLUMNS):
        raise ValueError("Prematch requires exactly table_state, elo, form, season_context")
    inputs = {}
    for name, item in entries.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "metadata_sha256"}:
            raise ValueError(f"Prematch {name}: require path, sha256 and metadata_sha256")
        if not isinstance(item["path"], str) or not item["path"].endswith(".parquet"):
            raise ValueError(f"Prematch {name}: expected a Parquet path")
        for field in ("sha256", "metadata_sha256"):
            value = item[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"Prematch {name}: invalid {field}")
        inputs[name] = FrozenPrematchInput(
            Path(item["path"]), item["sha256"], item["metadata_sha256"]
        )
    return PrematchConfig(inputs, hashlib.sha256(content).hexdigest())


def load_prematch_inputs(spec: PrematchConfig) -> dict[str, pd.DataFrame]:
    """Verify all bytes and sidecars before reading any DataFrame; only derived inputs."""
    for name, item in spec.inputs.items():
        if not item.path.resolve().is_relative_to(Path("data/derived").resolve()):
            raise ValueError(f"Prematch {name}: input must be inside data/derived")
        sidecar = item.path.with_suffix(".metadata.json")
        for path, digest in ((item.path, item.sha256), (sidecar, item.metadata_sha256)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Prematch {name}: checksum mismatch for {path}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            metadata.get("component") != name
            or metadata.get("artifact_sha256") != item.sha256
            or metadata.get("seasons") != list(DEVELOPMENT_SEASONS)
            or metadata.get("rows") != PREMATCH_ROWS
            or metadata.get("columns") != list(INPUT_COLUMNS[name])
            or metadata.get("index_columns") != ["game_id", "team_id"]
        ):
            raise ValueError(
                f"Prematch {name}: metadata rows, columns, seasons or identity mismatch"
            )
        if name == "table_state" and metadata.get("version") != "2":
            raise ValueError("Prematch requires table_state version 2")
    frames = {name: pd.read_parquet(item.path) for name, item in spec.inputs.items()}
    _validate_blocks(frames)
    return frames


def _validate_identity(frame: pd.DataFrame) -> None:
    if (
        frame.index.names != ["game_id", "team_id"]
        or not frame.index.is_unique
        or frame.index.to_frame().isna().any().any()
        or len(frame) != PREMATCH_ROWS
        or frame.index.get_level_values("game_id").nunique() != PREMATCH_ROWS // 2
        or not frame.groupby(level="game_id").size().eq(2).all()
    ):
        raise ValueError(
            "Prematch requires 8630 unique (game_id, team_id) rows, two per 4315 games"
        )
    if frame.season_id.isna().any() or set(frame.season_id) != set(DEVELOPMENT_SEASONS):
        raise ValueError("Prematch requires exactly development seasons 8–25; 26–28 are forbidden")
    for column in ("date", "as_of"):
        if (
            not isinstance(frame[column].dtype, pd.DatetimeTZDtype)
            or str(frame[column].dt.tz) != "Asia/Jerusalem"
            or frame[column].isna().any()
        ):
            raise ValueError(f"Prematch {column} requires nonmissing Asia/Jerusalem timestamps")
    if not frame.as_of.equals(frame.date.dt.normalize()):
        raise ValueError("Prematch as_of must be the calendar date of the fixture")


def _opponents(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    keys = pd.MultiIndex.from_arrays(
        [table.index.get_level_values("game_id"), table.opponent_id], names=table.index.names
    )
    if table.opponent_id.isna().any() or (frame.index.get_indexer(keys) < 0).any():
        raise ValueError("Prematch opponent_id does not identify the other row of its game")
    if np.equal(keys.get_level_values("team_id"), table.index.get_level_values("team_id")).any():
        raise ValueError("Prematch opponent_id cannot be the team's own ID")
    paired = frame.reindex(keys).copy()
    paired.index = table.index
    return paired


def _validate_previous_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    first = frame.season_id.eq(8)
    if (
        not frame.loc[first, list(columns)].isna().all().all()
        or not np.isfinite(frame.loc[~first, list(columns)].to_numpy(float)).all()
        or frame.prev_season_known.isna().any()
        or not frame.prev_season_known.eq(~first).all()
    ):
        raise ValueError("Prematch t-1 values and contrasts must be missing in season 8 only")


def _validate_pairing(table: pd.DataFrame) -> pd.DataFrame:
    opponent = _opponents(table, table)
    if (
        not pd.api.types.is_bool_dtype(table.is_home.dtype)
        or table.is_home.isna().any()
        or not table.groupby(level="game_id").is_home.sum().eq(1).all()
    ):
        raise ValueError("Prematch requires exactly one home and one away row per game")
    identity = ["season_id", "date", "as_of"]
    if not table.loc[:, identity].equals(opponent.loc[:, identity]):
        raise ValueError("Prematch paired rows disagree on season or fixture date")
    return opponent


def _validate_blocks(frames: dict[str, pd.DataFrame]) -> None:
    table = frames["table_state"]
    for name, frame in frames.items():
        if tuple(frame.columns) != INPUT_COLUMNS[name]:
            raise ValueError(f"Prematch {name}: columns do not match the frozen block contract")
        if (
            frame.index.names != ["game_id", "team_id"]
            or not frame.index.is_unique
            or not frame.index.equals(table.index)
        ):
            raise ValueError(f"Prematch {name}: expected exactly the same ordered, unique keys")
    _validate_identity(table)
    _validate_pairing(table)
    for column in ("season_id", "as_of"):
        try:
            # Season IDs are Int64 in table_state and int64 in the frozen Elo block.
            assert_series_equal(
                table[column],
                frames["elo"][column],
                check_exact=True,
                check_dtype=column != "season_id",
            )
        except AssertionError as exc:
            raise ValueError(f"Prematch Elo overlap mismatch: {column}") from exc
    for name, columns in (("elo", ELO_COLUMNS), ("form", FORM_COLUMNS)):
        if not np.isfinite(frames[name].loc[:, list(columns)].to_numpy(float)).all():
            raise ValueError(f"Prematch {name}: nonfinite values in the frozen block")
    context = frames["season_context"].assign(season_id=table.season_id)
    _validate_previous_values(context, SEASON_CONTEXT_COLUMNS[:5])
    if (
        not context.loc[table.season_id.gt(8), ["prev_value_share_avg", "prev_position_pct"]]
        .gt(0)
        .all()
        .all()
    ):
        raise ValueError("Prematch ratios require positive prior-season values")


def validate_prematch_frame(frame: pd.DataFrame) -> None:
    """Validate the unified schema, pairing, missingness and antisymmetric contrasts."""
    validate_prematch_feature_columns(frame)
    if tuple(frame.columns) != PREMATCH_COLUMNS:
        raise ValueError("Prematch output requires exactly its ordered 51 columns")
    try:
        PREMATCH_SCHEMA.validate(frame, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        raise ValueError(f"Prematch schema validation failed: {exc}") from exc
    _validate_identity(frame)
    _validate_previous_values(frame, (*PREVIOUS_VALUE_COLUMNS, *PREMATCH_CONTRAST_COLUMNS))
    finite = (*ELO_COLUMNS, *SELECTED_FORM_COLUMNS, *PREMATCH_OPPONENT_COLUMNS)
    if not np.isfinite(frame.loc[:, list(finite)].to_numpy(float)).all():
        raise ValueError("Prematch dynamic features must be finite")
    opponent = _validate_pairing(frame)
    for own, other in zip(SELECTED_FORM_COLUMNS[:2], PREMATCH_OPPONENT_COLUMNS, strict=True):
        if not frame[other].equals(opponent[own].rename(other)):
            raise ValueError(f"Prematch opponent feature mismatch: {other}")
    contrasts = frame.loc[:, list(PREMATCH_CONTRAST_COLUMNS)].to_numpy(float)
    opposite = opponent.loc[:, list(PREMATCH_CONTRAST_COLUMNS)].to_numpy(float)
    if not np.allclose(contrasts, -opposite, atol=1e-12, rtol=0, equal_nan=True):
        raise ValueError("Prematch contrasts are not antisymmetric within games")


def build_prematch(
    table_state: pd.DataFrame, elo: pd.DataFrame, form: pd.DataFrame, season_context: pd.DataFrame
) -> pd.DataFrame:
    """Pure M0 union plus opponent attack/defense and four prior-strength contrasts."""
    frames = {
        "table_state": table_state,
        "elo": elo,
        "form": form,
        "season_context": season_context,
    }
    _validate_blocks(frames)
    opponent_form = _opponents(form, table_state)
    context_opponent = _opponents(season_context, table_state)
    result = table_state.copy(deep=True)
    for source, columns in ((elo, ELO_COLUMNS), (form, SELECTED_FORM_COLUMNS)):
        result = result.join(source.loc[:, list(columns)], validate="one_to_one")
    result = result.join(
        opponent_form.loc[:, list(SELECTED_FORM_COLUMNS[:2])].set_axis(
            PREMATCH_OPPONENT_COLUMNS, axis=1
        ),
        validate="one_to_one",
    ).join(season_context.loc[:, list(SELECTED_CONTEXT_COLUMNS)], validate="one_to_one")
    for target, source in (
        ("value_avg_log_ratio", "prev_value_share_avg"),
        ("position_log_ratio", "prev_position_pct"),
    ):
        result[target] = np.log(season_context[source] / context_opponent[source])
    for target, source in (("prev_ppm_diff", "prev_ppm"), ("prev_gd_diff", "prev_gd_per_game")):
        result[target] = season_context[source] - context_opponent[source]
    validate_prematch_frame(result)
    # Exact reconstruction, including nullable dtypes, categories and all original NA.
    for source, copied_columns in (
        (table_state, TABLE_COLUMNS),
        (elo, ELO_COLUMNS),
        (form, SELECTED_FORM_COLUMNS),
        (season_context, SELECTED_CONTEXT_COLUMNS),
    ):
        assert_frame_equal(
            result.loc[:, list(copied_columns)],
            source.loc[:, list(copied_columns)],
            check_exact=True,
        )
    return result
