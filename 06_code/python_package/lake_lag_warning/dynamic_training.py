"""Leakage-closed assembly and nested cross-fitting for the lifecycle model.

This module defines infrastructure shared by the trainable environment-evidence
branch, biomass-evidence branch, continuous-stage module, and flat ablation. It
does not fit a gate, choose a dynamic weight, or produce model scores.

The public vocabulary deliberately uses ``branch`` rather than ``expert``. The
branches are internal parameter paths of one lifecycle model, not external
models or human judgements.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TargetKind = Literal["stage", "event_30_day"]
EstimatorFamily = Literal["logistic", "hgb"]

KEY_COLUMNS = ("lake_id", "origin_date")
DEFAULT_OUTER_TARGET_YEARS = (2016, 2017, 2018, 2019, 2020)
FIRST_INNER_TARGET_YEAR = 2013
PERMANENT_LAKE_FOLDS = 5
DEVELOPMENT_ORIGIN_CUTOFF = pd.Timestamp("2020-12-31")
MATURITY_DAYS: Mapping[str, int] = {"stage": 67, "event_30_day": 37}

DEFAULT_LEDGER_COLUMNS = (
    "candidate_row_key",
    "lake_id",
    "origin_date",
    "outer_lake_fold",
    "origin_year",
    "rolling_target_year",
    "eligible_common_development_candidate",
    "eligible_rolling_evaluation_candidate",
    "stage_label_observed",
    "ordered_stage_numeric",
    "incident_onset_event_30_day_observed",
)

BACKBONE_NUMERIC_PREDICTORS = (
    "origin_dayofyear_sin",
    "origin_dayofyear_cos",
    "lake_area_ha",
    "lake_maxdepth_m",
    "landuse_agriculture_pct",
)
BACKBONE_CATEGORICAL_PREDICTORS = ("ecoregion",)

ALLOWLIST_COLUMNS = (
    "feature_name",
    "evidence_stream",
    "source_product",
    "freeze_decision",
)
ALLOWED_EVIDENCE_STREAMS = {"environment_precursor", "biomass_response"}
ALLOWED_FREEZE_DECISIONS = {"ALLOW_PREDICTOR", "ALLOW_RELIABILITY"}
RELIABILITY_AGE_NAME_TOKENS = (
    "days_since",
    "source_age",
    "observation_age",
)

HARD_BLOCKED_MODEL_COLUMNS = {
    "candidate_row_key",
    "lake_id",
    "origin_date",
    "period_start",
    "hab_weekly_split",
    "rolling_target_year",
    "target_year",
    "origin_year",
    "outer_lake_fold",
    "eligible_common_development_candidate",
    "eligible_rolling_evaluation_candidate",
    "stage_label_observed",
    "ordered_stage_label",
    "ordered_stage_numeric",
    "incident_onset_event_30_day_observed",
    "same_day_incident_onset",
    "antecedent_61_90d_diagnostic",
}
PROHIBITED_MODEL_TOKEN = re.compile(
    r"(?:^|_)(?:future|next|stage|incident|event)(?:_|$)", re.IGNORECASE
)


class DynamicTrainingContractError(RuntimeError):
    """Base class for fail-closed dynamic-training contract violations."""


class FeatureContractError(DynamicTrainingContractError):
    """Raised when a feature can cross evidence streams or expose outcomes."""


class SplitContractError(DynamicTrainingContractError):
    """Raised when a temporal or permanent-lake split can leak information."""


@dataclass(frozen=True)
class BranchColumns:
    """Columns admitted to one internal parameter branch of the model."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


@dataclass(frozen=True)
class FeatureContract:
    """Frozen feature routing reconstructed from the G0 allowlist."""

    backbone_numeric: tuple[str, ...]
    backbone_categorical: tuple[str, ...]
    reliability_numeric: tuple[str, ...]
    environment_evidence_numeric: tuple[str, ...]
    biomass_evidence_numeric: tuple[str, ...]
    source_product_by_feature: Mapping[str, str]
    environment_reliability_numeric: tuple[str, ...] = field(default_factory=tuple)
    biomass_reliability_numeric: tuple[str, ...] = field(default_factory=tuple)

    def reliability_for(self, evidence_stream: str) -> tuple[str, ...]:
        """Return gate inputs from one source stream only."""

        if evidence_stream == "environment":
            specific = self.environment_reliability_numeric
        elif evidence_stream == "biomass":
            specific = self.biomass_reliability_numeric
        else:
            raise FeatureContractError(
                f"Unknown reliability evidence stream: {evidence_stream!r}"
            )
        # Synthetic unit-test contracts created before the explicit split use
        # the shared collection. Allowlist-backed contracts always populate it.
        return specific or self.reliability_numeric

    def columns_for(self, component: str) -> BranchColumns:
        """Return model columns for one named internal component.

        ``stage_module`` and ``flat_full`` use the union of both evidence
        streams. The two evidence branches share only the backbone.
        """

        aliases = {
            "p0": "backbone",
            "environment": "environment_branch",
            "p_env": "environment_branch",
            "biomass": "biomass_branch",
            "p_bio": "biomass_branch",
            "stage": "stage_module",
            "s": "stage_module",
            "flat": "flat_full",
        }
        name = aliases.get(component, component)
        backbone = _unique(self.backbone_numeric + self.reliability_numeric)
        if name == "backbone":
            numeric = backbone
        elif name == "environment_branch":
            numeric = _unique(backbone + self.environment_evidence_numeric)
        elif name == "biomass_branch":
            numeric = _unique(backbone + self.biomass_evidence_numeric)
        elif name in {"stage_module", "flat_full"}:
            numeric = _unique(
                backbone
                + self.environment_evidence_numeric
                + self.biomass_evidence_numeric
            )
        else:
            raise FeatureContractError(f"Unknown model component: {component!r}")
        assert_prior_only_feature_names(numeric + self.backbone_categorical)
        return BranchColumns(numeric=numeric, categorical=self.backbone_categorical)


@dataclass(frozen=True)
class RowIndexSplit:
    """Integer row positions for one leakage-closed fit/score cell."""

    level: str
    target_kind: str
    target_year: int
    heldout_lake_fold: int
    maturity_days: int
    train_positions: np.ndarray
    score_positions: np.ndarray
    train_row_key_sha256: str
    score_row_key_sha256: str
    outer_target_year: int | None = None
    outer_heldout_lake_fold: int | None = None

    def audit_record(self) -> dict[str, int | str | None]:
        return {
            "level": self.level,
            "target_kind": self.target_kind,
            "target_year": self.target_year,
            "heldout_lake_fold": self.heldout_lake_fold,
            "maturity_days": self.maturity_days,
            "train_rows": int(len(self.train_positions)),
            "score_rows": int(len(self.score_positions)),
            "train_row_key_sha256": self.train_row_key_sha256,
            "score_row_key_sha256": self.score_row_key_sha256,
            "outer_target_year": self.outer_target_year,
            "outer_heldout_lake_fold": self.outer_heldout_lake_fold,
        }


@dataclass(frozen=True)
class NestedOOFPlan:
    """Inner time-by-lake OOF cells for one outer training cell."""

    outer_split: RowIndexSplit
    cells: tuple[RowIndexSplit, ...]
    oof_positions: np.ndarray
    uncovered_outer_training_positions: np.ndarray

    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame([cell.audit_record() for cell in self.cells])


@dataclass(frozen=True)
class FoldPreprocessingAudit:
    """Training-fold-only column decisions for one branch."""

    estimator_family: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    dropped_columns: Mapping[str, str]
    output_storage: str

    def as_dict(self) -> dict[str, object]:
        return {
            "estimator_family": self.estimator_family,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "dropped_columns": dict(self.dropped_columns),
            "output_storage": self.output_storage,
        }


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value)
        if item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def assert_prior_only_feature_names(feature_names: Iterable[str]) -> None:
    """Reject identifiers, outcome fields, and future-looking feature tokens."""

    violations: list[str] = []
    for raw_name in feature_names:
        name = str(raw_name).strip()
        if not name or name in HARD_BLOCKED_MODEL_COLUMNS or PROHIBITED_MODEL_TOKEN.search(name):
            violations.append(name or "<empty>")
    if violations:
        raise FeatureContractError(
            "Prohibited model features requested: " + ", ".join(sorted(set(violations)))
        )


def load_feature_contract(
    allowlist_path: Path | str,
    *,
    require_complete_backbone: bool = True,
) -> FeatureContract:
    """Read and fail-closed validate the frozen G0 model-safe allowlist."""

    path = Path(allowlist_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = [column for column in ALLOWLIST_COLUMNS if column not in frame]
    if missing_columns:
        raise FeatureContractError(
            f"Allowlist is missing required columns: {missing_columns}"
        )
    if frame.empty:
        raise FeatureContractError("Feature allowlist is empty.")
    for column in ALLOWLIST_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    if frame["feature_name"].eq("").any():
        raise FeatureContractError("Allowlist contains an empty feature name.")
    if frame["feature_name"].duplicated().any():
        duplicated = sorted(frame.loc[frame["feature_name"].duplicated(), "feature_name"])
        raise FeatureContractError(f"Allowlist feature names are duplicated: {duplicated}")
    unknown_streams = set(frame["evidence_stream"]) - ALLOWED_EVIDENCE_STREAMS
    unknown_decisions = set(frame["freeze_decision"]) - ALLOWED_FREEZE_DECISIONS
    if unknown_streams:
        raise FeatureContractError(f"Unknown evidence streams: {sorted(unknown_streams)}")
    if unknown_decisions:
        raise FeatureContractError(f"Unknown freeze decisions: {sorted(unknown_decisions)}")
    if frame["source_product"].eq("").any():
        raise FeatureContractError("Every allowed feature must name one source product.")

    assert_prior_only_feature_names(frame["feature_name"])
    feature_names = set(frame["feature_name"])
    expected_backbone = set(BACKBONE_NUMERIC_PREDICTORS) | set(
        BACKBONE_CATEGORICAL_PREDICTORS
    )
    if require_complete_backbone and not expected_backbone.issubset(feature_names):
        raise FeatureContractError(
            "Allowlist is missing registered backbone fields: "
            + ", ".join(sorted(expected_backbone - feature_names))
        )

    backbone_numeric = tuple(
        name for name in BACKBONE_NUMERIC_PREDICTORS if name in feature_names
    )
    backbone_categorical = tuple(
        name for name in BACKBONE_CATEGORICAL_PREDICTORS if name in feature_names
    )
    backbone_rows = frame[frame["feature_name"].isin(expected_backbone)]
    if not backbone_rows.empty and not backbone_rows["freeze_decision"].eq(
        "ALLOW_PREDICTOR"
    ).all():
        raise FeatureContractError("Backbone fields must be ALLOW_PREDICTOR rows.")

    reliability_rows = frame[frame["freeze_decision"].eq("ALLOW_RELIABILITY")]
    reliability = tuple(reliability_rows["feature_name"])
    environment_reliability = tuple(
        reliability_rows.loc[
            reliability_rows["evidence_stream"].eq("environment_precursor"),
            "feature_name",
        ]
    )
    biomass_reliability = tuple(
        reliability_rows.loc[
            reliability_rows["evidence_stream"].eq("biomass_response"),
            "feature_name",
        ]
    )
    if set(environment_reliability) & set(biomass_reliability):
        raise FeatureContractError("Reliability fields cross evidence streams.")
    if set(environment_reliability) | set(biomass_reliability) != set(reliability):
        raise FeatureContractError(
            "Every reliability field must belong to exactly one evidence stream."
        )
    categorical_reliability = set(reliability) & set(backbone_categorical)
    if categorical_reliability:
        raise FeatureContractError(
            f"Categorical reliability fields are unsupported: {sorted(categorical_reliability)}"
        )

    predictor = frame[frame["freeze_decision"].eq("ALLOW_PREDICTOR")]
    env = tuple(
        predictor.loc[
            predictor["evidence_stream"].eq("environment_precursor")
            & ~predictor["feature_name"].isin(expected_backbone),
            "feature_name",
        ]
    )
    bio = tuple(
        predictor.loc[
            predictor["evidence_stream"].eq("biomass_response"), "feature_name"
        ]
    )
    overlap = set(env) & set(bio)
    if overlap:
        raise FeatureContractError(
            f"Numeric evidence streams overlap outside the backbone: {sorted(overlap)}"
        )
    if set(reliability) & (set(env) | set(bio)):
        raise FeatureContractError("A feature cannot be both predictor and reliability input.")

    sources = dict(zip(frame["feature_name"], frame["source_product"], strict=True))
    return FeatureContract(
        backbone_numeric=backbone_numeric,
        backbone_categorical=backbone_categorical,
        reliability_numeric=reliability,
        environment_evidence_numeric=env,
        biomass_evidence_numeric=bio,
        source_product_by_feature=sources,
        environment_reliability_numeric=environment_reliability,
        biomass_reliability_numeric=biomass_reliability,
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise DynamicTrainingContractError(f"{label} is missing columns: {missing}")


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def _read_parquet_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    available = _parquet_columns(path)
    missing = [column for column in columns if column not in available]
    if missing:
        raise FeatureContractError(f"{path} is missing required columns: {missing}")
    return pd.read_parquet(path, columns=list(columns))


def _validate_keys(frame: pd.DataFrame, label: str) -> None:
    _require_columns(frame, KEY_COLUMNS, label)
    if frame[list(KEY_COLUMNS)].isna().any().any():
        raise FeatureContractError(f"{label} contains null lake/origin keys.")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise FeatureContractError(f"{label} contains duplicate lake/origin keys.")


def _resolve_registered_source(project_root: Path, registered: str) -> Path:
    windows = PureWindowsPath(registered)
    portable = Path(*windows.parts)
    candidate = portable if portable.is_absolute() else project_root / portable
    root = project_root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FeatureContractError(
            f"Registered source escapes the project root: {registered!r}"
        ) from exc
    if not resolved.is_file():
        raise FeatureContractError(f"Registered source does not exist: {resolved}")
    return resolved


def _restore_reliability_age_missing_values(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    reliability_feature_names: frozenset[str],
) -> None:
    """Restore the frozen ``-1`` age sentinel to a real missing value in place."""

    for feature in feature_names:
        normalized = str(feature).lower()
        if feature not in reliability_feature_names or not any(
            token in normalized for token in RELIABILITY_AGE_NAME_TOKENS
        ):
            continue
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        frame[feature] = frame[feature].mask(numeric.eq(-1))


@dataclass(frozen=True)
class FeatureFrameAssembler:
    """Load one branch at a time from allowlisted prior-only products.

    Loading branch-by-branch prevents simultaneous materialisation of several
    753-column dense matrices. Product origins are checked before any label
    columns are read, so a drifted post-2020 ledger fails closed.
    """

    project_root: Path
    candidate_ledger_path: Path
    feature_contract: FeatureContract

    @classmethod
    def from_allowlist(
        cls,
        *,
        project_root: Path | str,
        candidate_ledger_path: Path | str,
        allowlist_path: Path | str,
    ) -> "FeatureFrameAssembler":
        return cls(
            project_root=Path(project_root).resolve(),
            candidate_ledger_path=Path(candidate_ledger_path).resolve(),
            feature_contract=load_feature_contract(allowlist_path),
        )

    def _assert_pre2021_product(self, path: Path, label: str) -> None:
        origins = _read_parquet_columns(path, ["origin_date"])["origin_date"]
        parsed = pd.to_datetime(origins, errors="raise")
        if parsed.isna().any() or parsed.max() > DEVELOPMENT_ORIGIN_CUTOFF:
            raise FeatureContractError(f"{label} exposes post-2020 or null origins.")

    def load_branch_frame(
        self,
        component: str,
        *,
        ledger_columns: Sequence[str] = DEFAULT_LEDGER_COLUMNS,
    ) -> pd.DataFrame:
        """Assemble one branch while preserving candidate-ledger row order."""

        branch = self.feature_contract.columns_for(component)
        controls = _unique(tuple(ledger_columns))
        if not set(KEY_COLUMNS).issubset(controls):
            raise FeatureContractError("Ledger columns must include lake_id and origin_date.")
        if "candidate_row_key" not in controls:
            raise FeatureContractError("Ledger columns must include candidate_row_key.")

        self._assert_pre2021_product(self.candidate_ledger_path, "candidate ledger")
        ledger = _read_parquet_columns(self.candidate_ledger_path, controls)
        ledger["origin_date"] = pd.to_datetime(ledger["origin_date"], errors="raise")
        _validate_keys(ledger, "candidate ledger")
        if ledger["candidate_row_key"].isna().any() or ledger[
            "candidate_row_key"
        ].astype(str).duplicated().any():
            raise FeatureContractError("candidate_row_key must be nonnull and unique.")
        ledger = ledger.copy()
        ledger["__row_order__"] = np.arange(len(ledger), dtype=np.int64)

        assignments: dict[str, list[str]] = {}
        for feature in branch.all:
            registered = self.feature_contract.source_product_by_feature[feature]
            assignments.setdefault(registered, []).append(feature)

        joined = ledger
        reliability_features = frozenset(self.feature_contract.reliability_numeric)
        for number, (registered, features) in enumerate(assignments.items()):
            source = _resolve_registered_source(self.project_root, registered)
            self._assert_pre2021_product(source, f"feature product {registered}")
            columns = _unique(KEY_COLUMNS + tuple(features))
            product = _read_parquet_columns(source, columns)
            _restore_reliability_age_missing_values(
                product,
                feature_names=features,
                reliability_feature_names=reliability_features,
            )
            product["origin_date"] = pd.to_datetime(
                product["origin_date"], errors="raise"
            )
            _validate_keys(product, f"feature product {registered}")
            if len(product) != len(ledger):
                raise FeatureContractError(
                    f"Feature product row count differs from candidate ledger: {registered}"
                )
            marker = f"__source_{number}__"
            joined = joined.merge(
                product,
                on=list(KEY_COLUMNS),
                how="left",
                validate="one_to_one",
                indicator=marker,
                sort=False,
            )
            if not joined[marker].eq("both").all():
                raise FeatureContractError(
                    f"Feature product does not cover every candidate key: {registered}"
                )
            joined = joined.drop(columns=marker)

        joined = joined.sort_values("__row_order__", kind="mergesort").drop(
            columns="__row_order__"
        )
        expected = list(controls) + list(branch.all)
        if list(joined.columns) != expected:
            joined = joined.loc[:, expected]
        return joined.reset_index(drop=True)


def validate_split_ledger(frame: pd.DataFrame) -> None:
    """Validate permanent folds, row identities, and the development cutoff."""

    required = [
        "candidate_row_key",
        "lake_id",
        "origin_date",
        "outer_lake_fold",
        "origin_year",
        "rolling_target_year",
        "eligible_common_development_candidate",
        "eligible_rolling_evaluation_candidate",
        "stage_label_observed",
        "ordered_stage_numeric",
        "incident_onset_event_30_day_observed",
    ]
    _require_columns(frame, required, "split ledger")
    if frame["candidate_row_key"].isna().any() or frame[
        "candidate_row_key"
    ].astype(str).str.strip().eq("").any():
        raise SplitContractError("candidate_row_key is null or empty.")
    if frame["candidate_row_key"].astype(str).duplicated().any():
        raise SplitContractError("candidate_row_key is not unique.")

    lake = frame["lake_id"].astype(str)
    origin = pd.to_datetime(frame["origin_date"], errors="raise")
    if lake.str.strip().eq("").any() or origin.isna().any():
        raise SplitContractError("Lake/origin keys must be nonempty.")
    identity = pd.DataFrame({"lake_id": lake, "origin_date": origin})
    if identity.duplicated().any():
        raise SplitContractError("Lake/origin identities are duplicated.")
    if origin.max() > DEVELOPMENT_ORIGIN_CUTOFF:
        raise SplitContractError("Post-2020 candidate origins are prohibited.")

    folds = pd.to_numeric(frame["outer_lake_fold"], errors="raise")
    if folds.isna().any() or not np.equal(folds, np.floor(folds)).all():
        raise SplitContractError("Permanent lake folds must be integers.")
    if not folds.between(0, PERMANENT_LAKE_FOLDS - 1).all():
        raise SplitContractError("Permanent lake fold is outside 0..4.")
    fold_frame = pd.DataFrame({"lake_id": lake, "fold": folds.astype(int)})
    if fold_frame.groupby("lake_id", sort=False)["fold"].nunique().gt(1).any():
        raise SplitContractError("A lake changes permanent fold across origins.")

    origin_year = pd.to_numeric(frame["origin_year"], errors="raise")
    if not origin_year.eq(origin.dt.year).all():
        raise SplitContractError("origin_year does not match origin_date.")
    rolling = pd.to_numeric(frame["rolling_target_year"], errors="coerce")
    expected = origin_year.where(origin_year.isin(DEFAULT_OUTER_TARGET_YEARS))
    if not rolling.fillna(-1).eq(expected.fillna(-1)).all():
        raise SplitContractError("rolling_target_year does not match the frozen years.")

    stage_observed = frame["stage_label_observed"].fillna(False).astype(bool)
    stage = pd.to_numeric(frame["ordered_stage_numeric"], errors="coerce")
    if (stage_observed & ~stage.isin([0, 1, 2, 3])).any():
        raise SplitContractError("Observed stage labels must be integers 0..3.")
    event_30_day = pd.to_numeric(frame["incident_onset_event_30_day_observed"], errors="coerce")
    if (event_30_day.notna() & ~event_30_day.isin([0, 1])).any():
        raise SplitContractError("Observed event_30_day labels must be binary.")


def _normalize_target_kind(target_kind: str) -> TargetKind:
    if target_kind not in MATURITY_DAYS:
        raise SplitContractError(f"Unknown target kind: {target_kind!r}")
    return target_kind  # type: ignore[return-value]


def _target_observed_mask(frame: pd.DataFrame, target_kind: TargetKind) -> pd.Series:
    if target_kind == "stage":
        return frame["stage_label_observed"].fillna(False).astype(bool) & pd.to_numeric(
            frame["ordered_stage_numeric"], errors="coerce"
        ).isin([0, 1, 2, 3])
    return pd.to_numeric(
        frame["incident_onset_event_30_day_observed"], errors="coerce"
    ).isin([0, 1])


def outer_training_mask(
    frame: pd.DataFrame,
    *,
    target_year: int,
    heldout_lake_fold: int,
    target_kind: TargetKind,
) -> pd.Series:
    """Return other-lake rows whose requested label matured before target year."""

    kind = _normalize_target_kind(target_kind)
    # The same maturity mask is used by the 2013--2015 inner rolling OOF
    # cells.  Only *scoring* is restricted to the registered 2016--2020 outer
    # evaluation years.
    if not FIRST_INNER_TARGET_YEAR <= int(target_year) <= DEFAULT_OUTER_TARGET_YEARS[-1]:
        raise SplitContractError(f"Training target year is not registered: {target_year}")
    if int(heldout_lake_fold) not in range(PERMANENT_LAKE_FOLDS):
        raise SplitContractError("Held-out lake fold must be in 0..4.")
    origin = pd.to_datetime(frame["origin_date"], errors="raise")
    boundary = pd.Timestamp(year=int(target_year), month=1, day=1)
    mature = origin + pd.to_timedelta(MATURITY_DAYS[kind], unit="D") < boundary
    return (
        frame["eligible_common_development_candidate"].fillna(False).astype(bool)
        & _target_observed_mask(frame, kind)
        & pd.to_numeric(frame["outer_lake_fold"], errors="raise").ne(
            int(heldout_lake_fold)
        )
        & mature
    )


def outer_scoring_mask(
    frame: pd.DataFrame, *, target_year: int, heldout_lake_fold: int
) -> pd.Series:
    """Return all deployable candidates in one outer target-year/lake-fold cell."""

    if int(target_year) not in DEFAULT_OUTER_TARGET_YEARS:
        raise SplitContractError(f"Outer target year is not registered: {target_year}")
    if int(heldout_lake_fold) not in range(PERMANENT_LAKE_FOLDS):
        raise SplitContractError("Held-out lake fold must be in 0..4.")
    return (
        frame["eligible_rolling_evaluation_candidate"].fillna(False).astype(bool)
        & pd.to_numeric(frame["rolling_target_year"], errors="coerce").eq(
            int(target_year)
        )
        & pd.to_numeric(frame["outer_lake_fold"], errors="raise").eq(
            int(heldout_lake_fold)
        )
    )


def stable_row_key_hash(values: Iterable[object]) -> str:
    """Hash a row-key set in a deterministic order."""

    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _readonly_positions(mask: pd.Series | np.ndarray) -> np.ndarray:
    values = mask.to_numpy(dtype=bool) if isinstance(mask, pd.Series) else np.asarray(mask)
    positions = np.flatnonzero(values).astype(np.int64, copy=False)
    positions.setflags(write=False)
    return positions


def _make_split(
    frame: pd.DataFrame,
    *,
    level: str,
    target_kind: TargetKind,
    target_year: int,
    heldout_lake_fold: int,
    train_mask: pd.Series | np.ndarray,
    score_mask: pd.Series | np.ndarray,
    outer_target_year: int | None = None,
    outer_heldout_lake_fold: int | None = None,
) -> RowIndexSplit:
    train = _readonly_positions(train_mask)
    score = _readonly_positions(score_mask)
    if np.intersect1d(train, score).size:
        raise SplitContractError("Training and score positions overlap.")
    train_lakes = set(frame.iloc[train]["lake_id"].astype(str))
    score_lakes = set(frame.iloc[score]["lake_id"].astype(str))
    if train_lakes & score_lakes:
        raise SplitContractError("Training and score rows share permanent lakes.")
    return RowIndexSplit(
        level=level,
        target_kind=target_kind,
        target_year=int(target_year),
        heldout_lake_fold=int(heldout_lake_fold),
        maturity_days=int(MATURITY_DAYS[target_kind]),
        train_positions=train,
        score_positions=score,
        train_row_key_sha256=stable_row_key_hash(
            frame.iloc[train]["candidate_row_key"]
        ),
        score_row_key_sha256=stable_row_key_hash(
            frame.iloc[score]["candidate_row_key"]
        ),
        outer_target_year=outer_target_year,
        outer_heldout_lake_fold=outer_heldout_lake_fold,
    )


def build_outer_split(
    frame: pd.DataFrame,
    *,
    target_year: int,
    heldout_lake_fold: int,
    target_kind: TargetKind,
    validate: bool = True,
) -> RowIndexSplit:
    """Build one 2016-2020 rolling-year by permanent-lake outer cell."""

    if validate:
        validate_split_ledger(frame)
    kind = _normalize_target_kind(target_kind)
    train = outer_training_mask(
        frame,
        target_year=target_year,
        heldout_lake_fold=heldout_lake_fold,
        target_kind=kind,
    )
    score = outer_scoring_mask(
        frame, target_year=target_year, heldout_lake_fold=heldout_lake_fold
    )
    split = _make_split(
        frame,
        level="outer",
        target_kind=kind,
        target_year=target_year,
        heldout_lake_fold=heldout_lake_fold,
        train_mask=train,
        score_mask=score,
    )
    if not len(split.train_positions) or not len(split.score_positions):
        raise SplitContractError(
            f"Empty outer fit/score cell: year={target_year}, fold={heldout_lake_fold}"
        )
    return split


def build_inner_oof_plan(
    frame: pd.DataFrame,
    outer_split: RowIndexSplit,
    *,
    inner_target_years: Sequence[int] | None = None,
    validate: bool = True,
) -> NestedOOFPlan:
    """Generate nested time-by-lake OOF indices inside an outer training cell.

    Each inner score row belongs to one earlier calendar year and one of the
    four non-outer lake folds. Its component fit uses only labels matured before
    that inner year and excludes both the outer and inner held-out lake folds.
    """

    if validate:
        validate_split_ledger(frame)
    if outer_split.level != "outer":
        raise SplitContractError("An inner OOF plan requires an outer split.")
    if len(frame) <= max(
        np.concatenate(
            [outer_split.train_positions, outer_split.score_positions], dtype=np.int64
        ),
        default=-1,
    ):
        raise SplitContractError("Outer split positions do not belong to this ledger.")

    origin = pd.to_datetime(frame["origin_date"], errors="raise")
    origin_year = origin.dt.year
    outer_train_mask = np.zeros(len(frame), dtype=bool)
    outer_train_mask[outer_split.train_positions] = True
    available_years = sorted(
        set(origin_year.iloc[outer_split.train_positions].astype(int))
    )
    if inner_target_years is None:
        years = [
            year
            for year in available_years
            if year >= FIRST_INNER_TARGET_YEAR
            and year < outer_split.target_year
        ]
    else:
        years = sorted(set(int(year) for year in inner_target_years))
    if not years:
        raise SplitContractError("No inner target years are available for OOF fitting.")
    if any(year >= outer_split.target_year for year in years):
        raise SplitContractError("Inner target years must precede the outer target year.")

    folds = pd.to_numeric(frame["outer_lake_fold"], errors="raise").astype(int)
    cells: list[RowIndexSplit] = []
    all_score_positions: list[np.ndarray] = []
    for target_year in years:
        for inner_fold in range(PERMANENT_LAKE_FOLDS):
            if inner_fold == outer_split.heldout_lake_fold:
                continue
            inner_train = outer_training_mask(
                frame,
                target_year=target_year,
                heldout_lake_fold=outer_split.heldout_lake_fold,
                target_kind=_normalize_target_kind(outer_split.target_kind),
            ) & folds.ne(inner_fold)
            inner_score = (
                pd.Series(outer_train_mask, index=frame.index)
                & origin_year.eq(target_year)
                & folds.eq(inner_fold)
            )
            split = _make_split(
                frame,
                level="inner_oof",
                target_kind=_normalize_target_kind(outer_split.target_kind),
                target_year=target_year,
                heldout_lake_fold=inner_fold,
                train_mask=inner_train,
                score_mask=inner_score,
                outer_target_year=outer_split.target_year,
                outer_heldout_lake_fold=outer_split.heldout_lake_fold,
            )
            if len(split.score_positions) and not len(split.train_positions):
                raise SplitContractError(
                    f"Inner score cell has no earlier mature fit rows: {target_year}/{inner_fold}"
                )
            boundary = pd.Timestamp(year=target_year, month=1, day=1)
            if len(split.train_positions):
                matured = origin.iloc[split.train_positions] + pd.to_timedelta(
                    split.maturity_days, unit="D"
                )
                if not matured.lt(boundary).all():
                    raise SplitContractError("Inner fitting admitted an immature label.")
            cells.append(split)
            if len(split.score_positions):
                all_score_positions.append(split.score_positions)

    if all_score_positions:
        concatenated = np.concatenate(all_score_positions).astype(np.int64, copy=False)
        if len(np.unique(concatenated)) != len(concatenated):
            raise SplitContractError("An outer-training row receives multiple OOF scores.")
        oof = np.sort(concatenated)
    else:
        oof = np.array([], dtype=np.int64)
    if not set(oof).issubset(set(outer_split.train_positions)):
        raise SplitContractError("Inner OOF rows are not a subset of outer training rows.")
    uncovered = np.setdiff1d(outer_split.train_positions, oof, assume_unique=True)
    oof.setflags(write=False)
    uncovered.setflags(write=False)
    return NestedOOFPlan(
        outer_split=outer_split,
        cells=tuple(cells),
        oof_positions=oof,
        uncovered_outer_training_positions=uncovered,
    )


def build_fold_preprocessor(
    training_frame: pd.DataFrame,
    columns: BranchColumns,
    *,
    estimator_family: EstimatorFamily,
) -> tuple[ColumnTransformer, FoldPreprocessingAudit]:
    """Create an unfitted preprocessing object using training-fold decisions.

    Numeric medians and missing indicators are fitted only when the returned
    transformer is fitted on this training frame. Logistic output is sparse
    whenever possible; HGB output is dense because sklearn HGB requires it.
    """

    if estimator_family not in {"logistic", "hgb"}:
        raise FeatureContractError(
            f"Unsupported estimator family: {estimator_family!r}"
        )
    assert_prior_only_feature_names(columns.all)
    _require_columns(training_frame, columns.all, "training branch frame")

    keep_numeric: list[str] = []
    keep_categorical: list[str] = []
    dropped: dict[str, str] = {}
    for column in columns.numeric:
        raw = training_frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any():
            raise FeatureContractError(f"Numeric feature contains text: {column}")
        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise FeatureContractError(f"Numeric feature contains infinity: {column}")
        observed = numeric.dropna()
        if observed.empty:
            dropped[column] = "all_missing_in_training_fold"
        elif observed.nunique(dropna=True) <= 1:
            dropped[column] = "constant_in_training_fold"
        else:
            keep_numeric.append(column)

    for column in columns.categorical:
        normalized = training_frame[column].astype("object").where(
            training_frame[column].notna(), "__MISSING__"
        )
        if normalized.nunique(dropna=False) <= 1:
            dropped[column] = "constant_in_training_fold"
        else:
            keep_categorical.append(column)
    if not keep_numeric and not keep_categorical:
        raise FeatureContractError("No informative features remain in this training fold.")

    numeric_steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if estimator_family == "logistic":
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    numeric = Pipeline(steps=numeric_steps)
    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=estimator_family == "logistic",
                ),
            ),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = []
    if keep_numeric:
        transformers.append(("numeric", numeric, keep_numeric))
    if keep_categorical:
        transformers.append(("categorical", categorical, keep_categorical))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0 if estimator_family == "logistic" else 0.0,
        verbose_feature_names_out=True,
    )
    audit = FoldPreprocessingAudit(
        estimator_family=estimator_family,
        numeric_columns=tuple(keep_numeric),
        categorical_columns=tuple(keep_categorical),
        dropped_columns=dropped,
        output_storage="sparse_when_possible" if estimator_family == "logistic" else "dense",
    )
    return preprocessor, audit
