"""Nested OOF execution for the lifecycle dynamic-evidence model.

This module joins the already frozen feature/split contract to the core
dynamic-fusion API.  It deliberately keeps the evidence branches internal to
one lifecycle model: the staged fitting is solely a leakage-control device.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .dynamic_evidence_weighting import (
    CumulativeHGBStageEstimator,
    CumulativeOrdinalLogistic,
    DynamicEvidenceFusionModel,
    fit_dynamic_evidence_fusion,
    probability_logit,
    project_cumulative_probabilities,
)
from .dynamic_training import (
    BranchColumns,
    FeatureContract,
    FoldPreprocessingAudit,
    NestedOOFPlan,
    RowIndexSplit,
    build_fold_preprocessor,
    build_inner_oof_plan,
    build_outer_split,
)


class NestedOOFError(RuntimeError):
    """Raised when an OOF component or fusion input violates the contract."""


@dataclass
class FittedComponent:
    """One fold-fitted internal component with its fold-local transformer."""

    component: str
    family: str
    preprocessor: Any
    estimator: Any
    preprocessing_audit: FoldPreprocessingAudit
    fit_audit: Mapping[str, Any] = field(default_factory=dict)

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame)
        probability = self.estimator.predict_proba(transformed)[:, 1]
        return _validate_probability(probability, self.component)

    def predict_stage_score(self, frame: pd.DataFrame) -> np.ndarray:
        cumulative = self.predict_stage_cumulative(frame)
        values = cumulative.mean(axis=1)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise NestedOOFError("Continuous stage score is outside [0, 1].")
        return values

    def predict_stage_cumulative(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame)
        if self.family == "S2":
            transformed = _dense(transformed)
        values = np.asarray(
            self.estimator.predict_cumulative_proba(transformed), dtype=float
        )
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not np.all(values[:, 0] >= values[:, 1] - 1e-12)
            or not np.all(values[:, 1] >= values[:, 2] - 1e-12)
        ):
            raise NestedOOFError("Cumulative stage probabilities violate their order contract.")
        return values


@dataclass(frozen=True)
class SigmoidProbabilityCalibrator:
    """Positive-slope Platt calibration fitted only from inner OOF scores."""

    intercept: float
    log_slope: float

    @property
    def slope(self) -> float:
        return float(np.exp(self.log_slope))

    def predict(self, probability: Sequence[float] | np.ndarray) -> np.ndarray:
        raw_logit = probability_logit(probability, name="component probability")
        calibrated = expit(self.intercept + self.slope * raw_logit)
        return _validate_probability(calibrated, "calibrated component")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "model_type": "positive_slope_sigmoid_calibration",
            "intercept": float(self.intercept),
            "slope": self.slope,
        }


@dataclass
class OuterCellResult:
    """Scores and audits for one outer year-by-lake-fold deployment cell."""

    outer_split: RowIndexSplit
    oof_components: pd.DataFrame
    final_scores: pd.DataFrame
    fusion_models: Mapping[str, DynamicEvidenceFusionModel]
    components: Mapping[str, FittedComponent]
    component_calibrators: Mapping[str, SigmoidProbabilityCalibrator]
    stage_calibrators: Mapping[str, SigmoidProbabilityCalibrator]
    split_audit: pd.DataFrame
    fit_audit: pd.DataFrame
    audit: Mapping[str, Any]


def _dense(values: Any) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values, dtype=float)


def _validate_probability(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise NestedOOFError(f"{name} did not produce finite probabilities in [0, 1].")
    return result


def fit_sigmoid_probability_calibrator(
    probability: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
) -> SigmoidProbabilityCalibrator:
    """Calibrate an honest OOF probability without allowing rank reversal."""

    raw_logit = probability_logit(probability, name="OOF component probability")
    label = np.asarray(outcome, dtype=float)
    if label.shape != raw_logit.shape or not np.isin(label, [0.0, 1.0]).all():
        raise NestedOOFError("Calibration outcomes must be aligned binary labels.")
    if np.unique(label).size != 2:
        raise NestedOOFError("Calibration requires both outcome classes.")
    prevalence = float(np.clip(label.mean(), 1e-6, 1.0 - 1e-6))
    initial_intercept = float(np.log(prevalence) - np.log1p(-prevalence) - raw_logit.mean())

    def objective(theta: np.ndarray) -> float:
        intercept, log_slope = theta
        calibrated_logit = intercept + np.exp(log_slope) * raw_logit
        loss = np.logaddexp(0.0, calibrated_logit) - label * calibrated_logit
        return float(loss.mean() + 1e-6 * log_slope**2)

    fitted = minimize(
        objective,
        np.asarray([initial_intercept, 0.0], dtype=float),
        method="L-BFGS-B",
        bounds=((-20.0, 20.0), (-5.0, 5.0)),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise NestedOOFError(f"Component calibration failed: {fitted.message}")
    return SigmoidProbabilityCalibrator(
        intercept=float(fitted.x[0]),
        log_slope=float(fitted.x[1]),
    )


def _binary_labels(frame: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    label = pd.to_numeric(frame.iloc[positions]["incident_onset_event_30_day_observed"], errors="coerce")
    if label.isna().any() or not label.isin([0, 1]).all() or label.nunique() != 2:
        raise NestedOOFError("A event_30_day training cell must have two observed binary classes.")
    return label.to_numpy(dtype=np.int8)


def _binary_training_weights(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    loss_kind: str,
) -> np.ndarray | None:
    if loss_kind == "standard_event_30_day_log_loss":
        return None
    if loss_kind != "fixed_lead_aware_event_30_day_log_loss":
        raise NestedOOFError(f"Unsupported registered training loss: {loss_kind}")
    lead_column = "days_to_next_future_incident_onset"
    if lead_column not in frame:
        raise NestedOOFError(
            "Lead-aware loss requires its frozen target-only lead metadata."
        )
    label = _binary_labels(frame, positions)
    lead = pd.to_numeric(frame.iloc[positions][lead_column], errors="coerce").to_numpy(
        dtype=float
    )
    early_positive = (label == 1) & np.isfinite(lead) & (lead >= 28.0) & (lead <= 30.0)
    weights = np.ones(len(positions), dtype=float)
    weights[early_positive] = 2.0
    return weights / float(weights.mean())


def _stage_labels(frame: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    observed = frame.iloc[positions]["stage_label_observed"].fillna(False).astype(bool)
    label = pd.to_numeric(frame.iloc[positions]["ordered_stage_numeric"], errors="coerce")
    if not observed.all() or label.isna().any() or not label.isin([0, 1, 2, 3]).all():
        raise NestedOOFError("A stage training cell must have observed labels in {0,1,2,3}.")
    return label.to_numpy(dtype=np.int8)


def _hgb_settings(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    values = dict(config["branch_candidates"]["hist_gradient_boosting"]["common"])
    values.update(config["branch_candidates"]["hist_gradient_boosting"]["configurations"][key])
    return values


def _iteration_audit(estimator: Any, *, family: str) -> dict[str, Any]:
    fitted = estimator
    if isinstance(fitted, Pipeline):
        fitted = fitted.named_steps.get("classifier", fitted)
    iterations = getattr(fitted, "n_iter_", None)
    if iterations is None:
        values: list[int] = []
    else:
        values = np.asarray(iterations, dtype=int).ravel().tolist()
    maximum = getattr(fitted, "max_iter", None)
    return {
        "family": family,
        "iterations": values,
        "configured_max_iter": None if maximum is None else int(maximum),
    }


def _fit_audit(
    estimator: Any,
    *,
    family: str,
    convergence_warnings: Sequence[warnings.WarningMessage],
) -> dict[str, Any]:
    if isinstance(estimator, CumulativeOrdinalLogistic):
        iterations = [
            _iteration_audit(model, family=family) for model in estimator._models_
        ]
    elif isinstance(estimator, CumulativeHGBStageEstimator):
        iterations = [
            _iteration_audit(model, family=family) for model in estimator._models_
        ]
    else:
        iterations = [_iteration_audit(estimator, family=family)]
    warning_text = [str(item.message) for item in convergence_warnings]
    return {
        "family": family,
        "converged": not warning_text,
        "convergence_warning_count": int(len(warning_text)),
        "convergence_warnings": warning_text,
        "iteration_audit": iterations,
    }


def fit_binary_component(
    *,
    component: str,
    family: str,
    frame: pd.DataFrame,
    columns: BranchColumns,
    train_positions: np.ndarray,
    configuration: Mapping[str, Any],
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> FittedComponent:
    """Fit a prior-only event_30_day component with fold-local preprocessing."""

    estimator_family = "logistic" if family == "logistic" else "hgb"
    train = frame.iloc[train_positions]
    transformer, audit = build_fold_preprocessor(
        train, columns, estimator_family=estimator_family  # type: ignore[arg-type]
    )
    design = transformer.fit_transform(train)
    outcome = _binary_labels(frame, train_positions)
    if family == "logistic":
        estimator: Any = LogisticRegression(
            C=float(configuration.get("C", 0.2)),
            solver=str(configuration.get("solver", "liblinear")),
            max_iter=int(configuration.get("max_iter", 1000)),
            tol=float(configuration.get("tol", 1e-4)),
            random_state=int(configuration.get("random_state", 20260730)),
        )
    elif family == "hgb":
        estimator = HistGradientBoostingClassifier(**dict(configuration))
        design = _dense(design)
    else:
        raise NestedOOFError(f"Unsupported binary component family: {family}")
    fitted_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if fitted_weight is not None and (
        fitted_weight.shape != outcome.shape
        or not np.isfinite(fitted_weight).all()
        or (fitted_weight <= 0).any()
    ):
        raise NestedOOFError("Binary component sample weights are invalid or misaligned.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(design, outcome, sample_weight=fitted_weight)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    return FittedComponent(
        component,
        family,
        transformer,
        estimator,
        audit,
        _fit_audit(estimator, family=family, convergence_warnings=convergence),
    )


def fit_stage_component(
    *,
    frame: pd.DataFrame,
    columns: BranchColumns,
    train_positions: np.ndarray,
    family: str,
    configuration: Mapping[str, Any],
) -> FittedComponent:
    """Fit S1 or S2 from prior-only features and observed ordered stages."""

    estimator_family = "logistic" if family == "S1" else "hgb"
    train = frame.iloc[train_positions]
    transformer, audit = build_fold_preprocessor(
        train, columns, estimator_family=estimator_family  # type: ignore[arg-type]
    )
    design = transformer.fit_transform(train)
    outcome = _stage_labels(frame, train_positions)
    if family == "S1":
        estimator: Any = CumulativeOrdinalLogistic(
            C=float(configuration.get("C", 0.2)),
            max_iter=int(configuration.get("max_iter", 1000)),
            solver=str(configuration.get("solver", "liblinear")),
            tol=float(configuration.get("tol", 1e-4)),
            random_state=int(configuration.get("random_state", 20260730)),
            preprocessed=True,
        )
    elif family == "S2":
        estimator = CumulativeHGBStageEstimator(**dict(configuration))
        design = _dense(design)
    else:
        raise NestedOOFError(f"Unsupported stage family: {family}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(design, outcome, feature_names=[f"feature_{i}" for i in range(design.shape[1])])
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    return FittedComponent(
        "continuous_stage_module",
        family,
        transformer,
        estimator,
        audit,
        _fit_audit(estimator, family=family, convergence_warnings=convergence),
    )


def _require_aligned_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    required = {"backbone", "environment_branch", "biomass_branch", "stage_module"}
    missing = required - set(frames)
    if missing:
        raise NestedOOFError(f"Missing component frames: {sorted(missing)}")
    ledger = frames["backbone"]
    keys = ledger["candidate_row_key"].astype(str).tolist()
    for name in required - {"backbone"}:
        candidate = frames[name]
        if candidate["candidate_row_key"].astype(str).tolist() != keys:
            raise NestedOOFError(f"{name} row order differs from the backbone frame.")
    return ledger


def _component_columns(contract: FeatureContract) -> dict[str, BranchColumns]:
    return {
        "p0": contract.columns_for("backbone"),
        "p_env": contract.columns_for("environment_branch"),
        "p_bio": contract.columns_for("biomass_branch"),
        "stage": contract.columns_for("stage_module"),
    }


def _component_frame_name(component: str) -> str:
    return {"p0": "backbone", "p_env": "environment_branch", "p_bio": "biomass_branch", "stage": "stage_module"}[component]


def _fit_components_for_positions(
    *,
    frames: Mapping[str, pd.DataFrame],
    contract: FeatureContract,
    event_30_day_train_positions: np.ndarray,
    stage_train_positions: np.ndarray,
    backbone_family: str,
    backbone_configuration: Mapping[str, Any],
    environment_family: str,
    environment_configuration: Mapping[str, Any],
    biomass_family: str,
    biomass_configuration: Mapping[str, Any],
    stage_family: str,
    stage_configuration: Mapping[str, Any],
    risk_sample_weight: Sequence[float] | np.ndarray | None = None,
    fit_stage_module: bool = True,
) -> dict[str, FittedComponent]:
    columns = _component_columns(contract)
    result = {
        "p0": fit_binary_component(component="p0", family=backbone_family, frame=frames["backbone"], columns=columns["p0"], train_positions=event_30_day_train_positions, configuration=backbone_configuration, sample_weight=risk_sample_weight),
        "p_env": fit_binary_component(component="p_env", family=environment_family, frame=frames["environment_branch"], columns=columns["p_env"], train_positions=event_30_day_train_positions, configuration=environment_configuration, sample_weight=risk_sample_weight),
        "p_bio": fit_binary_component(component="p_bio", family=biomass_family, frame=frames["biomass_branch"], columns=columns["p_bio"], train_positions=event_30_day_train_positions, configuration=biomass_configuration, sample_weight=risk_sample_weight),
    }
    if fit_stage_module:
        result["stage"] = fit_stage_component(frame=frames["stage_module"], columns=columns["stage"], train_positions=stage_train_positions, family=stage_family, configuration=stage_configuration)
    return result


def _predict_components(
    components: Mapping[str, FittedComponent],
    frames: Mapping[str, pd.DataFrame],
    positions: np.ndarray,
    *,
    default_stage_score: float | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame({"row_position": positions.astype(np.int64)})
    for name in ("p0", "p_env", "p_bio"):
        result[name] = components[name].predict_probability(frames[_component_frame_name(name)].iloc[positions])
    if "stage" in components:
        cumulative = components["stage"].predict_stage_cumulative(
            frames["stage_module"].iloc[positions]
        )
    elif default_stage_score is not None:
        if not 0.0 <= default_stage_score <= 1.0:
            raise NestedOOFError("Default stage score must lie in [0, 1].")
        cumulative = np.full((len(positions), 3), default_stage_score, dtype=float)
    else:
        raise NestedOOFError("Stage predictions are unavailable.")
    result[["stage_ge1_probability", "stage_ge2_probability", "stage_ge3_probability"]] = cumulative
    result["continuous_stage_score"] = cumulative.mean(axis=1)
    return result


def _reliability(
    frame: pd.DataFrame,
    contract: FeatureContract,
    positions: np.ndarray,
    *,
    evidence_stream: str | None = None,
    fields: Sequence[str] | None = None,
) -> pd.DataFrame:
    if fields is None:
        source = (
            contract.reliability_numeric
            if evidence_stream is None
            else contract.reliability_for(evidence_stream)
        )
        fields = list(source)
    else:
        fields = list(fields)
    if not fields:
        raise NestedOOFError("The frozen contract contains no reliability fields.")
    return frame.iloc[positions].loc[:, fields].reset_index(drop=True)


def _select_gate_reliability(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Drop only fold-level all-missing reliability inputs before gate fitting."""

    selected = tuple(column for column in frame if frame[column].notna().any())
    if not selected:
        raise NestedOOFError("All registered reliability fields are missing in the OOF gate table.")
    return frame.loc[:, list(selected)], selected


def _fusion_reliability(
    environment: pd.DataFrame, biomass: pd.DataFrame
) -> pd.DataFrame:
    if len(environment) != len(biomass):
        raise NestedOOFError("Environment and biomass reliability rows differ.")
    return pd.concat(
        [
            environment.add_prefix("environment__"),
            biomass.add_prefix("biomass__"),
        ],
        axis=1,
    )


def _inner_cell_map(
    plan: NestedOOFPlan, *, require_score_rows: bool
) -> dict[tuple[int, int], RowIndexSplit]:
    return {
        (cell.target_year, cell.heldout_lake_fold): cell
        for cell in plan.cells
        if (len(cell.score_positions) if require_score_rows else len(cell.train_positions))
    }


def run_outer_cell(
    *,
    frames: Mapping[str, pd.DataFrame],
    contract: FeatureContract,
    config: Mapping[str, Any],
    target_year: int,
    heldout_lake_fold: int,
    branch_family: str = "logistic",
    branch_configuration: Mapping[str, Any] | None = None,
    backbone_family: str | None = None,
    backbone_configuration: Mapping[str, Any] | None = None,
    environment_family: str | None = None,
    environment_configuration: Mapping[str, Any] | None = None,
    biomass_family: str | None = None,
    biomass_configuration: Mapping[str, Any] | None = None,
    stage_family: str = "S1",
    stage_configuration: Mapping[str, Any] | None = None,
    training_loss: str = "standard_event_30_day_log_loss",
    w1_penalty: float = 0.1,
    w2_penalty: float = 0.1,
    fusion_kinds: Sequence[str] = ("W0", "W1", "W2"),
    fit_stage_module: bool = True,
) -> OuterCellResult:
    """Run requested fusion variants for one leakage-closed outer cell."""

    requested_fusion_kinds = tuple(fusion_kinds)
    allowed_fusion_kinds = {"W0", "W1", "W2"}
    if (
        not requested_fusion_kinds
        or len(set(requested_fusion_kinds)) != len(requested_fusion_kinds)
        or not set(requested_fusion_kinds).issubset(allowed_fusion_kinds)
    ):
        raise NestedOOFError(
            "fusion_kinds must be a non-empty unique subset of W0, W1, and W2."
        )
    if not fit_stage_module and set(requested_fusion_kinds) != {"W0"}:
        raise NestedOOFError("The stage module can be skipped only for W0-only fitting.")
    ledger = _require_aligned_frames(frames)
    event_30_day_outer = build_outer_split(ledger, target_year=target_year, heldout_lake_fold=heldout_lake_fold, target_kind="event_30_day")
    stage_outer = build_outer_split(ledger, target_year=target_year, heldout_lake_fold=heldout_lake_fold, target_kind="stage")
    event_30_day_plan = build_inner_oof_plan(ledger, event_30_day_outer)
    stage_plan = build_inner_oof_plan(ledger, stage_outer)
    event_30_day_cells = _inner_cell_map(event_30_day_plan, require_score_rows=True)
    stage_cells = _inner_cell_map(stage_plan, require_score_rows=False)
    common_keys = sorted(set(event_30_day_cells) & set(stage_cells))
    if not common_keys:
        raise NestedOOFError("No common inner time-by-lake OOF cells exist for fusion training.")
    shared_branch_configuration = dict(
        branch_configuration or {"C": 0.2, "random_state": 20260730}
    )
    resolved_backbone_family = backbone_family or branch_family
    resolved_environment_family = environment_family or branch_family
    resolved_biomass_family = biomass_family or branch_family
    resolved_backbone_configuration = dict(
        backbone_configuration or shared_branch_configuration
    )
    resolved_environment_configuration = dict(
        environment_configuration or shared_branch_configuration
    )
    resolved_biomass_configuration = dict(
        biomass_configuration or shared_branch_configuration
    )
    oof_parts: list[pd.DataFrame] = []
    fit_audit_rows: list[dict[str, Any]] = []
    for key in common_keys:
        event_30_day_cell, stage_cell = event_30_day_cells[key], stage_cells[key]
        # Stage labels are needed to fit the prior-only stage module, not to
        # obtain its prediction. Score every mature event_30_day OOF row so the fusion
        # module does not discard otherwise usable training information.
        positions = event_30_day_cell.score_positions
        if not len(positions):
            continue
        components = _fit_components_for_positions(
            frames=frames, contract=contract,
            event_30_day_train_positions=event_30_day_cell.train_positions,
            stage_train_positions=stage_cell.train_positions,
            backbone_family=resolved_backbone_family,
            backbone_configuration=resolved_backbone_configuration,
            environment_family=resolved_environment_family,
            environment_configuration=resolved_environment_configuration,
            biomass_family=resolved_biomass_family,
            biomass_configuration=resolved_biomass_configuration,
            stage_family=stage_family,
            stage_configuration=dict(stage_configuration or {"C": 0.2, "random_state": 20260730}),
            risk_sample_weight=_binary_training_weights(
                ledger, event_30_day_cell.train_positions, loss_kind=training_loss
            ),
            fit_stage_module=fit_stage_module,
        )
        for component_name, fitted_component in components.items():
            fit_audit_rows.append(
                {
                    "fit_level": "inner_oof",
                    "fit_target_year": int(key[0]),
                    "fit_heldout_lake_fold": int(key[1]),
                    "component": component_name,
                    **dict(fitted_component.fit_audit),
                }
            )
        part = _predict_components(
            components,
            frames,
            positions,
            default_stage_score=0.5 if not fit_stage_module else None,
        )
        identity = ledger.iloc[positions][
            ["candidate_row_key", "lake_id", "origin_date"]
        ].reset_index(drop=True)
        part = pd.concat([part, identity], axis=1)
        part["inner_target_year"], part["inner_lake_fold"] = key
        oof_parts.append(part)
    if not oof_parts:
        raise NestedOOFError("Inner OOF scoring produced no common rows.")
    oof = pd.concat(oof_parts, ignore_index=True)
    if oof["row_position"].duplicated().any():
        raise NestedOOFError("A gate-training row received more than one OOF component score.")
    oof = oof.sort_values("row_position", kind="mergesort").reset_index(drop=True)
    oof_positions = oof["row_position"].to_numpy(dtype=np.int64)
    oof_label = _binary_labels(ledger, oof_positions)
    component_calibrators: dict[str, SigmoidProbabilityCalibrator] = {}
    for component in ("p0", "p_env", "p_bio"):
        raw_column = f"{component}_raw"
        oof[raw_column] = oof[component]
        calibrator = fit_sigmoid_probability_calibrator(oof[component], oof_label)
        oof[component] = calibrator.predict(oof[component])
        component_calibrators[component] = calibrator
    stage_columns = (
        "stage_ge1_probability",
        "stage_ge2_probability",
        "stage_ge3_probability",
    )
    stage_calibrators: dict[str, SigmoidProbabilityCalibrator] = {}
    if fit_stage_module:
        stage_observed = (
            ledger.iloc[oof_positions]["stage_label_observed"]
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        observed_stage = pd.to_numeric(
            ledger.iloc[oof_positions]["ordered_stage_numeric"], errors="coerce"
        ).to_numpy(dtype=float)
        if not stage_observed.any():
            raise NestedOOFError("No observed stage labels exist for OOF calibration.")
        calibrated_cumulative: list[np.ndarray] = []
        for threshold, column in enumerate(stage_columns, start=1):
            raw_column = f"{column}_raw"
            oof[raw_column] = oof[column]
            calibrator = fit_sigmoid_probability_calibrator(
                oof.loc[stage_observed, column],
                (observed_stage[stage_observed] >= threshold).astype(np.int8),
            )
            stage_calibrators[column] = calibrator
            calibrated_cumulative.append(calibrator.predict(oof[column]))
        calibrated_stage = project_cumulative_probabilities(
            np.column_stack(calibrated_cumulative)
        )
        oof.loc[:, list(stage_columns)] = calibrated_stage
        oof["continuous_stage_score"] = calibrated_stage.mean(axis=1)
    else:
        for column in stage_columns:
            oof[f"{column}_raw"] = oof[column]
    env_reliability, env_reliability_fields = _select_gate_reliability(
        _reliability(
            frames["environment_branch"],
            contract,
            oof_positions,
            evidence_stream="environment",
        )
    )
    bio_reliability, bio_reliability_fields = _select_gate_reliability(
        _reliability(
            frames["biomass_branch"],
            contract,
            oof_positions,
            evidence_stream="biomass",
        )
    )
    fusion_reliability = _fusion_reliability(env_reliability, bio_reliability)
    fusion_models: dict[str, DynamicEvidenceFusionModel] = {}
    fusion_specs = {
        "W0": ({}, None),
        "W1": ({"smoothness_penalty": float(w1_penalty)}, None),
        "W2": (
            {
                "knots": tuple(config["fusion"]["W2"]["stage_knots"]),
                "smoothness_penalty": float(w2_penalty),
                "l2_regularization": 0.1,
                "monotonic": True,
            },
            fusion_reliability,
        ),
    }
    for kind in requested_fusion_kinds:
        options, fusion_values = fusion_specs[kind]
        fusion_models[kind] = fit_dynamic_evidence_fusion(
            backbone_probability=oof["p0"], environment_branch_probability=oof["p_env"], biomass_branch_probability=oof["p_bio"],
            continuous_stage_score_values=oof["continuous_stage_score"], outcome=oof_label,
            environment_reliability_values=env_reliability, biomass_reliability_values=bio_reliability,
            environment_reliability_feature_names=tuple(env_reliability.columns), biomass_reliability_feature_names=tuple(bio_reliability.columns),
            weight_kind=kind, fusion_reliability_values=fusion_values,
            fusion_reliability_feature_names=(
                tuple(fusion_reliability.columns)
                if fusion_values is not None
                else None
            ),
            reliability_l2_regularization=0.1, weight_options=options,
        )
    final_components = _fit_components_for_positions(
        frames=frames, contract=contract,
        event_30_day_train_positions=event_30_day_outer.train_positions, stage_train_positions=stage_outer.train_positions,
        backbone_family=resolved_backbone_family,
        backbone_configuration=resolved_backbone_configuration,
        environment_family=resolved_environment_family,
        environment_configuration=resolved_environment_configuration,
        biomass_family=resolved_biomass_family,
        biomass_configuration=resolved_biomass_configuration,
        stage_family=stage_family,
        stage_configuration=dict(stage_configuration or {"C": 0.2, "random_state": 20260730}),
        risk_sample_weight=_binary_training_weights(
            ledger, event_30_day_outer.train_positions, loss_kind=training_loss
        ),
        fit_stage_module=fit_stage_module,
    )
    for component_name, fitted_component in final_components.items():
        fit_audit_rows.append(
            {
                "fit_level": "outer_final",
                "fit_target_year": int(target_year),
                "fit_heldout_lake_fold": int(heldout_lake_fold),
                "component": component_name,
                **dict(fitted_component.fit_audit),
            }
        )
    score_pos = event_30_day_outer.score_positions
    base = _predict_components(
        final_components,
        frames,
        score_pos,
        default_stage_score=0.5 if not fit_stage_module else None,
    )
    final = ledger.iloc[score_pos][["candidate_row_key", "lake_id", "origin_date", "outer_lake_fold", "rolling_target_year", "incident_onset_event_30_day_observed"]].reset_index(drop=True)
    final = pd.concat([final, base.drop(columns="row_position")], axis=1)
    for component, calibrator in component_calibrators.items():
        raw_column = f"{component}_raw"
        final[raw_column] = final[component]
        final[component] = calibrator.predict(final[component])
    if fit_stage_module:
        final_stage: list[np.ndarray] = []
        for column, calibrator in stage_calibrators.items():
            raw_column = f"{column}_raw"
            final[raw_column] = final[column]
            final_stage.append(calibrator.predict(final[column]))
        calibrated_final_stage = project_cumulative_probabilities(
            np.column_stack(final_stage)
        )
        final.loc[:, list(stage_columns)] = calibrated_final_stage
        final["continuous_stage_score"] = calibrated_final_stage.mean(axis=1)
    else:
        for column in stage_columns:
            final[f"{column}_raw"] = final[column]
    env_score_reliability = _reliability(
        frames["environment_branch"],
        contract,
        score_pos,
        evidence_stream="environment",
        fields=env_reliability_fields,
    )
    bio_score_reliability = _reliability(
        frames["biomass_branch"],
        contract,
        score_pos,
        evidence_stream="biomass",
        fields=bio_reliability_fields,
    )
    fusion_score_reliability = _fusion_reliability(
        env_score_reliability, bio_score_reliability
    )
    for kind, fusion in fusion_models.items():
        prediction = fusion.predict(
            backbone_probability=final["p0"], environment_branch_probability=final["p_env"], biomass_branch_probability=final["p_bio"],
            continuous_stage_score_values=final["continuous_stage_score"], environment_reliability_values=env_score_reliability,
            biomass_reliability_values=bio_score_reliability,
            fusion_reliability_values=(
                fusion_score_reliability if kind == "W2" else None
            ),
        )
        final[f"{kind}_risk"] = prediction.final_probability
        final[f"{kind}_biomass_weight"] = prediction.biomass_weight
        final[f"{kind}_environment_reliability"] = prediction.environment_reliability
        final[f"{kind}_biomass_reliability"] = prediction.biomass_reliability
    split_audit = pd.concat(
        [
            pd.DataFrame(
                [
                    {**event_30_day_outer.audit_record(), "split_role": "outer_event_30_day"},
                    {**stage_outer.audit_record(), "split_role": "outer_stage"},
                ]
            ),
            event_30_day_plan.audit_frame().assign(split_role="inner_event_30_day"),
            stage_plan.audit_frame().assign(split_role="inner_stage"),
        ],
        ignore_index=True,
    )
    fit_audit_frame = pd.DataFrame(fit_audit_rows)
    audit = {
        "target_year": int(target_year), "heldout_lake_fold": int(heldout_lake_fold),
        "outer_event_30_day_train_rows": int(len(event_30_day_outer.train_positions)), "outer_stage_train_rows": int(len(stage_outer.train_positions)),
        "outer_score_rows": int(len(score_pos)), "inner_oof_rows": int(len(oof)),
        "inner_event_30_day_cells": int(len(event_30_day_cells)), "inner_stage_cells": int(len(stage_cells)),
        "inner_common_cells": int(len(common_keys)),
        "environment_gate_reliability_fields": list(env_reliability_fields),
        "biomass_gate_reliability_fields": list(bio_reliability_fields),
        "component_calibration_source": "inner_oof_only",
        "training_loss": training_loss,
        "component_families": {
            "p0": resolved_backbone_family,
            "p_env": resolved_environment_family,
            "p_bio": resolved_biomass_family,
            "stage": stage_family if fit_stage_module else "skipped_W0_only",
        },
        "component_fit_records": int(len(fit_audit_frame)),
        "all_component_fits_converged": bool(
            fit_audit_frame["converged"].fillna(False).all()
        ),
        "component_calibrators": {
            name: calibrator.to_dict()
            for name, calibrator in component_calibrators.items()
        },
        "stage_calibrators": {
            name: calibrator.to_dict()
            for name, calibrator in stage_calibrators.items()
        },
        "stage_module_fitted": bool(fit_stage_module),
        "gate_fit_inputs": "inner_oof_only",
        "weights": list(requested_fusion_kinds),
    }
    return OuterCellResult(
        outer_split=event_30_day_outer,
        oof_components=oof,
        final_scores=final,
        fusion_models=fusion_models,
        components=final_components,
        component_calibrators=component_calibrators,
        stage_calibrators=stage_calibrators,
        split_audit=split_audit,
        fit_audit=fit_audit_frame,
        audit=audit,
    )


__all__ = [
    "FittedComponent",
    "NestedOOFError",
    "OuterCellResult",
    "SigmoidProbabilityCalibrator",
    "fit_binary_component",
    "fit_sigmoid_probability_calibrator",
    "fit_stage_component",
    "run_outer_cell",
]
