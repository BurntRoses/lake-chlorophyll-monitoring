"""Core components for one lifecycle dynamic-evidence warning model.

The environment and biomass objects in this module are trainable evidence
branches inside one model.  They are not human or external experts.  The
module supports prior-only continuous-stage estimation, reliability-aware
branch shrinkage, learned dynamic fusion, and one final risk probability.

The calling training script remains responsible for nested, time-by-lake
cross-fitting.  In particular, every probability passed to a fusion fitter
must be an out-of-fold prediction from the corresponding internal branch.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class DynamicEvidenceContractError(RuntimeError):
    """Raised when dynamic-evidence inputs violate the model contract."""


PRIOR_ONLY_FORBIDDEN_NAME_PATTERNS: tuple[str, ...] = (
    r"(^|_)future($|_)",
    r"(^|_)next($|_)",
    r"(^|_)stage($|_)",
    r"(^|_)incident($|_)",
    r"(^|_)event($|_)",
    r"(^|_)episode($|_)",
    r"(^|_)onset($|_)",
    r"(^|_)target($|_)",
    r"(^|_)outcome($|_)",
    r"(^|_)label($|_)",
    r"(^|_)lead_days($|_)",
    r"(^|_)days_to($|_)",
    r"^y_true$",
    r"^y\d+$",
)

_PROBABILITY_FLOOR = 1e-6
_PARAMETER_LIMIT = 20.0


def find_forbidden_prior_only_feature_names(
    feature_names: Sequence[str],
    *,
    patterns: Sequence[str] = PRIOR_ONLY_FORBIDDEN_NAME_PATTERNS,
) -> list[str]:
    """Return input names that imply future outcomes or event-stage labels."""

    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    flagged: list[str] = []
    for raw_name in feature_names:
        name = str(raw_name).strip()
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
        if any(pattern.search(normalized) for pattern in compiled):
            flagged.append(name)
    return flagged


def assert_prior_only_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Fail closed when a model input name can encode future event state."""

    names = tuple(str(name).strip() for name in feature_names)
    if not names or any(not name for name in names):
        raise DynamicEvidenceContractError("Feature names must be nonempty.")
    if len(set(names)) != len(names):
        raise DynamicEvidenceContractError("Feature names must be unique.")
    flagged = find_forbidden_prior_only_feature_names(names)
    if flagged:
        raise DynamicEvidenceContractError(
            f"Prior-only inputs contain forbidden future/event fields: {flagged}"
        )
    return names


def _as_feature_matrix(
    values: pd.DataFrame | np.ndarray,
    *,
    feature_names: Sequence[str] | None,
    context: str,
) -> tuple[np.ndarray | scipy_sparse.spmatrix, tuple[str, ...]]:
    if isinstance(values, pd.DataFrame):
        names = assert_prior_only_feature_names(tuple(str(column) for column in values.columns))
        if feature_names is not None and tuple(feature_names) != names:
            raise DynamicEvidenceContractError(
                f"{context} DataFrame columns do not match the registered feature names."
            )
        matrix = values.to_numpy(dtype=float)
    elif scipy_sparse.issparse(values):
        matrix = values.astype(float)
        if feature_names is None:
            names = tuple(f"x_{index}" for index in range(matrix.shape[1]))
        else:
            names = assert_prior_only_feature_names(feature_names)
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2:
            raise DynamicEvidenceContractError(f"{context} must be a two-dimensional matrix.")
        if feature_names is None:
            names = tuple(f"x_{index}" for index in range(matrix.shape[1]))
        else:
            names = assert_prior_only_feature_names(feature_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise DynamicEvidenceContractError(
            f"{context} matrix width does not match its feature names."
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise DynamicEvidenceContractError(f"{context} must be nonempty.")
    finite_values = matrix.data if scipy_sparse.issparse(matrix) else matrix
    if np.isinf(finite_values).any():
        raise DynamicEvidenceContractError(
            f"{context} contains infinite values; NaN must remain explicit instead."
        )
    return matrix, names


def _prediction_feature_matrix(
    values: pd.DataFrame | np.ndarray,
    *,
    feature_names: Sequence[str],
    context: str,
) -> np.ndarray | scipy_sparse.spmatrix:
    names = tuple(feature_names)
    if isinstance(values, pd.DataFrame):
        supplied = tuple(str(column) for column in values.columns)
        assert_prior_only_feature_names(supplied)
        if set(supplied) != set(names) or len(supplied) != len(names):
            raise DynamicEvidenceContractError(
                f"{context} columns do not match the fitted feature contract."
            )
        matrix = values.loc[:, list(names)].to_numpy(dtype=float)
    elif scipy_sparse.issparse(values):
        matrix = values.astype(float)
    else:
        matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise DynamicEvidenceContractError(
            f"{context} matrix width does not match the fitted feature contract."
        )
    finite_values = matrix.data if scipy_sparse.issparse(matrix) else matrix
    if np.isinf(finite_values).any():
        raise DynamicEvidenceContractError(
            f"{context} contains infinite values; NaN must remain explicit instead."
        )
    return matrix


def _binary_outcome(values: Sequence[int] | np.ndarray) -> np.ndarray:
    outcome = np.asarray(values, dtype=float)
    if outcome.ndim != 1 or not np.isfinite(outcome).all():
        raise DynamicEvidenceContractError("The outcome must be a finite one-dimensional array.")
    if not np.isin(outcome, [0.0, 1.0]).all() or np.unique(outcome).size != 2:
        raise DynamicEvidenceContractError("The outcome must contain both binary classes.")
    return outcome


def _ordered_stage(values: Sequence[int] | np.ndarray) -> np.ndarray:
    stage = np.asarray(values, dtype=float)
    if stage.ndim != 1 or not np.isfinite(stage).all():
        raise DynamicEvidenceContractError("Stage labels must be finite and one-dimensional.")
    if not np.equal(stage, np.floor(stage)).all() or not np.isin(stage, [0, 1, 2, 3]).all():
        raise DynamicEvidenceContractError("Stage labels must be integers in {0, 1, 2, 3}.")
    stage = stage.astype(np.int8)
    for threshold in (1, 2, 3):
        if np.unique(stage >= threshold).size != 2:
            raise DynamicEvidenceContractError(
                f"Stage training data must contain both classes for stage >= {threshold}."
            )
    return stage


def _normalized_sample_weight(
    sample_weight: Sequence[float] | np.ndarray | None,
    *,
    rows: int,
) -> np.ndarray:
    if sample_weight is None:
        return np.ones(rows, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (rows,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise DynamicEvidenceContractError(
            "sample_weight must be finite, one-dimensional, and strictly positive."
        )
    return weights / float(weights.mean())


def _probability_array(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    probability_floor: float = _PROBABILITY_FLOOR,
) -> np.ndarray:
    if not 0 < probability_floor < 0.5:
        raise DynamicEvidenceContractError("probability_floor must lie in (0, 0.5).")
    probability = np.asarray(values, dtype=float)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise DynamicEvidenceContractError(f"{name} must be a finite one-dimensional array.")
    if (probability < 0).any() or (probability > 1).any():
        raise DynamicEvidenceContractError(f"{name} must contain probabilities in [0, 1].")
    return np.clip(probability, probability_floor, 1.0 - probability_floor)


def probability_logit(
    values: Sequence[float] | np.ndarray,
    *,
    name: str = "probability",
    probability_floor: float = _PROBABILITY_FLOOR,
) -> np.ndarray:
    """Convert probabilities to finite logits with a registered numeric floor."""

    probability = _probability_array(
        values,
        name=name,
        probability_floor=probability_floor,
    )
    return np.log(probability) - np.log1p(-probability)


def project_cumulative_probabilities(
    cumulative_probabilities: np.ndarray,
) -> np.ndarray:
    """Project P(stage >= 1/2/3) onto q1 >= q2 >= q3 in Euclidean distance."""

    values = np.asarray(cumulative_probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise DynamicEvidenceContractError(
            "Cumulative stage probabilities must have shape (n_rows, 3)."
        )
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise DynamicEvidenceContractError(
            "Cumulative stage probabilities must be finite values in [0, 1]."
        )

    first, second, third = values.T
    projected = values.copy()

    violation_12 = first < second
    pooled_12 = (first + second) / 2.0
    all_from_12 = violation_12 & (pooled_12 < third)
    pair_12 = violation_12 & ~all_from_12
    projected[pair_12, 0] = pooled_12[pair_12]
    projected[pair_12, 1] = pooled_12[pair_12]
    pooled_all = (first + second + third) / 3.0
    projected[all_from_12, :] = pooled_all[all_from_12, None]

    no_12_violation = ~violation_12
    violation_23 = no_12_violation & (second < third)
    pooled_23 = (second + third) / 2.0
    all_from_23 = violation_23 & (first < pooled_23)
    pair_23 = violation_23 & ~all_from_23
    projected[pair_23, 1] = pooled_23[pair_23]
    projected[pair_23, 2] = pooled_23[pair_23]
    projected[all_from_23, :] = pooled_all[all_from_23, None]

    projected = np.clip(projected, 0.0, 1.0)
    if not (
        np.all(projected[:, 0] >= projected[:, 1] - 1e-12)
        and np.all(projected[:, 1] >= projected[:, 2] - 1e-12)
    ):
        raise DynamicEvidenceContractError("Cumulative-probability projection failed.")
    return projected


def cumulative_probabilities_to_stage_probabilities(
    cumulative_probabilities: np.ndarray,
) -> np.ndarray:
    """Convert ordered cumulative probabilities to probabilities for stages 0..3."""

    cumulative = project_cumulative_probabilities(cumulative_probabilities)
    stage_probability = np.column_stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2],
        ]
    )
    stage_probability = np.clip(stage_probability, 0.0, 1.0)
    stage_probability /= stage_probability.sum(axis=1, keepdims=True)
    return stage_probability


def continuous_stage_score(cumulative_probabilities: np.ndarray) -> np.ndarray:
    """Return E(stage)/3 = (q1 + q2 + q3)/3 on the [0, 1] scale."""

    return project_cumulative_probabilities(cumulative_probabilities).mean(axis=1)


class _CumulativeStageMixin(ClassifierMixin):
    _models_: list[BaseEstimator]
    feature_names_in_: np.ndarray

    def _predict_matrix(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_models_", "feature_names_in_"])
        return _prediction_feature_matrix(
            values,
            feature_names=tuple(str(name) for name in self.feature_names_in_),
            context="stage prediction input",
        )

    def predict_cumulative_proba(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = self._predict_matrix(values)
        raw = np.column_stack(
            [np.asarray(model.predict_proba(matrix), dtype=float)[:, 1] for model in self._models_]
        )
        return project_cumulative_probabilities(raw)

    def predict_proba(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return cumulative_probabilities_to_stage_probabilities(
            self.predict_cumulative_proba(values)
        )

    def predict_stage_score(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return continuous_stage_score(self.predict_cumulative_proba(values))

    def predict(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(values), axis=1).astype(np.int8)


class CumulativeOrdinalLogistic(_CumulativeStageMixin, BaseEstimator):
    """S1: three cumulative logistic models for a prior-only stage score."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 500,
        solver: str = "lbfgs",
        tol: float = 1e-4,
        class_weight: str | dict[int, float] | None = None,
        random_state: int = 20260730,
        preprocessed: bool = False,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.preprocessed = preprocessed

    def fit(
        self,
        values: pd.DataFrame | np.ndarray,
        stage: Sequence[int] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> "CumulativeOrdinalLogistic":
        matrix, names = _as_feature_matrix(
            values,
            feature_names=feature_names,
            context="S1 stage training input",
        )
        ordered = _ordered_stage(stage)
        if len(ordered) != matrix.shape[0]:
            raise DynamicEvidenceContractError("Stage labels and S1 inputs have different lengths.")
        weights = _normalized_sample_weight(sample_weight, rows=len(ordered))
        if self.C <= 0 or self.max_iter < 1 or self.tol <= 0:
            raise DynamicEvidenceContractError("S1 C, max_iter, and tol must be positive.")

        if self.preprocessed:
            finite_values = matrix.data if scipy_sparse.issparse(matrix) else matrix
            if np.isnan(finite_values).any():
                raise DynamicEvidenceContractError(
                    "A preprocessed S1 matrix must not contain NaN."
                )

        models: list[BaseEstimator] = []
        for threshold in (1, 2, 3):
            classifier = LogisticRegression(
                C=float(self.C),
                solver=str(self.solver),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                class_weight=self.class_weight,
                random_state=int(self.random_state),
            )
            if self.preprocessed:
                classifier.fit(
                    matrix,
                    (ordered >= threshold).astype(np.int8),
                    sample_weight=weights,
                )
                models.append(classifier)
            else:
                pipeline = Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                                keep_empty_features=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                        ("classifier", classifier),
                    ]
                )
                pipeline.fit(
                    matrix,
                    (ordered >= threshold).astype(np.int8),
                    classifier__sample_weight=weights,
                )
                models.append(pipeline)
        self._models_ = models
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.classes_ = np.arange(4, dtype=np.int8)
        return self

    def to_parameter_dict(self) -> dict[str, object]:
        """Return JSON-compatible fitted parameters; use joblib for full persistence."""

        check_is_fitted(self, attributes=["_models_", "feature_names_in_"])
        cumulative_models: list[dict[str, object]] = []
        for threshold, fitted in zip((1, 2, 3), self._models_, strict=True):
            if isinstance(fitted, Pipeline):
                classifier = fitted.named_steps["classifier"]
                imputer = fitted.named_steps["imputer"]
                scaler = fitted.named_steps["scaler"]
                preprocessing: dict[str, object] = {
                    "imputation_statistics": imputer.statistics_.astype(float).tolist(),
                    "scale_mean": scaler.mean_.astype(float).tolist(),
                    "scale": scaler.scale_.astype(float).tolist(),
                }
            else:
                classifier = fitted
                preprocessing = {"preprocessing": "external_fold_preprocessor"}
            cumulative_models.append(
                {
                    "threshold": threshold,
                    "intercept": classifier.intercept_.astype(float).tolist(),
                    "coefficients": classifier.coef_.astype(float).tolist(),
                    **preprocessing,
                }
            )
        return {
            "model_type": "S1_cumulative_ordinal_logistic",
            "feature_names": self.feature_names_in_.astype(str).tolist(),
            "C": float(self.C),
            "max_iter": int(self.max_iter),
            "solver": str(self.solver),
            "tol": float(self.tol),
            "preprocessed": bool(self.preprocessed),
            "random_state": int(self.random_state),
            "cumulative_models": cumulative_models,
        }


class CumulativeHGBStageEstimator(_CumulativeStageMixin, BaseEstimator):
    """S2: three cumulative HGB classifiers with ordered-probability projection."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 150,
        max_leaf_nodes: int = 15,
        max_depth: int | None = 4,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        class_weight: str | dict[int, float] | None = None,
        random_state: int = 20260730,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(
        self,
        values: pd.DataFrame | np.ndarray,
        stage: Sequence[int] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> "CumulativeHGBStageEstimator":
        matrix, names = _as_feature_matrix(
            values,
            feature_names=feature_names,
            context="S2 stage training input",
        )
        ordered = _ordered_stage(stage)
        if len(ordered) != matrix.shape[0]:
            raise DynamicEvidenceContractError("Stage labels and S2 inputs have different lengths.")
        weights = _normalized_sample_weight(sample_weight, rows=len(ordered))
        if (
            self.learning_rate <= 0
            or self.max_iter < 1
            or self.max_leaf_nodes < 2
            or self.min_samples_leaf < 1
            or self.l2_regularization < 0
        ):
            raise DynamicEvidenceContractError("S2 hyperparameters are outside their valid ranges.")

        models: list[HistGradientBoostingClassifier] = []
        for threshold in (1, 2, 3):
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=float(self.learning_rate),
                max_iter=int(self.max_iter),
                max_leaf_nodes=int(self.max_leaf_nodes),
                max_depth=self.max_depth,
                min_samples_leaf=int(self.min_samples_leaf),
                l2_regularization=float(self.l2_regularization),
                class_weight=self.class_weight,
                early_stopping=False,
                random_state=int(self.random_state) + threshold,
            )
            model.fit(matrix, (ordered >= threshold).astype(np.int8), sample_weight=weights)
            models.append(model)
        self._models_ = models
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.classes_ = np.arange(4, dtype=np.int8)
        return self

    def to_parameter_dict(self) -> dict[str, object]:
        """Return JSON-compatible fit metadata; use joblib for tree persistence."""

        check_is_fitted(self, attributes=["_models_", "feature_names_in_"])
        return {
            "model_type": "S2_three_cumulative_HGB",
            "feature_names": self.feature_names_in_.astype(str).tolist(),
            "learning_rate": float(self.learning_rate),
            "max_iter": int(self.max_iter),
            "max_leaf_nodes": int(self.max_leaf_nodes),
            "max_depth": self.max_depth,
            "min_samples_leaf": int(self.min_samples_leaf),
            "l2_regularization": float(self.l2_regularization),
            "random_state": int(self.random_state),
            "fitted_iterations": [int(model.n_iter_) for model in self._models_],
        }


@dataclass(frozen=True)
class ReliabilityFeatureTransform:
    """Fold-fitted median, scale, and missing-indicator transform."""

    feature_names: tuple[str, ...]
    fill_values: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]

    @property
    def design_feature_names(self) -> tuple[str, ...]:
        return self.feature_names + tuple(f"{name}__missing" for name in self.feature_names)

    def transform(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = _prediction_feature_matrix(
            values,
            feature_names=self.feature_names,
            context="reliability prediction input",
        )
        missing = np.isnan(matrix)
        filled = np.where(missing, np.asarray(self.fill_values, dtype=float), matrix)
        standardized = (filled - np.asarray(self.centers, dtype=float)) / np.asarray(
            self.scales, dtype=float
        )
        design = np.column_stack([standardized, missing.astype(float)])
        if not np.isfinite(design).all():
            raise DynamicEvidenceContractError("Reliability transformation produced nonfinite values.")
        return design

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_reliability_feature_transform(
    values: pd.DataFrame | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[ReliabilityFeatureTransform, np.ndarray]:
    """Fit reliability preprocessing without treating a missing value as zero."""

    matrix, names = _as_feature_matrix(
        values,
        feature_names=feature_names,
        context="reliability training input",
    )
    finite_count = np.isfinite(matrix).sum(axis=0)
    if (finite_count == 0).any():
        failed = [names[index] for index in np.flatnonzero(finite_count == 0)]
        raise DynamicEvidenceContractError(
            f"Reliability features are all missing in training: {failed}"
        )
    fill = np.nanmedian(matrix, axis=0)
    missing = np.isnan(matrix)
    filled = np.where(missing, fill, matrix)
    centers = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    transform = ReliabilityFeatureTransform(
        feature_names=names,
        fill_values=tuple(float(value) for value in fill),
        centers=tuple(float(value) for value in centers),
        scales=tuple(float(value) for value in scales),
    )
    return transform, transform.transform(
        pd.DataFrame(matrix, columns=list(names)) if isinstance(values, pd.DataFrame) else matrix
    )


@dataclass(frozen=True)
class ReliabilityShrinkageModel:
    """Learned reliability gate that shrinks one evidence branch toward the backbone."""

    branch_name: Literal["environment", "biomass"]
    transform: ReliabilityFeatureTransform
    intercept: float
    coefficients: tuple[float, ...]
    l2_regularization: float
    objective_value: float
    optimization_method: str = "L-BFGS-B"

    def predict_reliability(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        design = self.transform.transform(values)
        coefficient = np.asarray(self.coefficients, dtype=float)
        if design.shape[1] != len(coefficient):
            raise DynamicEvidenceContractError("Reliability design does not match its model.")
        return expit(np.clip(self.intercept + design @ coefficient, -40.0, 40.0))

    def adjusted_logit(
        self,
        backbone_probability: Sequence[float] | np.ndarray,
        branch_probability: Sequence[float] | np.ndarray,
        reliability_values: pd.DataFrame | np.ndarray,
        *,
        probability_floor: float = _PROBABILITY_FLOOR,
    ) -> np.ndarray:
        backbone_logit = probability_logit(
            backbone_probability,
            name="backbone_probability",
            probability_floor=probability_floor,
        )
        branch_logit = probability_logit(
            branch_probability,
            name=f"{self.branch_name}_branch_probability",
            probability_floor=probability_floor,
        )
        if backbone_logit.shape != branch_logit.shape:
            raise DynamicEvidenceContractError("Backbone and branch probabilities differ in length.")
        reliability = self.predict_reliability(reliability_values)
        if reliability.shape != backbone_logit.shape:
            raise DynamicEvidenceContractError("Reliability rows and branch probabilities differ.")
        return backbone_logit + reliability * (branch_logit - backbone_logit)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["transform"] = self.transform.to_dict()
        return result


def fit_reliability_shrinkage(
    backbone_probability: Sequence[float] | np.ndarray,
    branch_probability: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    reliability_values: pd.DataFrame | np.ndarray,
    *,
    branch_name: Literal["environment", "biomass"],
    feature_names: Sequence[str] | None = None,
    l2_regularization: float = 0.1,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    probability_floor: float = _PROBABILITY_FLOOR,
) -> ReliabilityShrinkageModel:
    """Learn how reliability controls a branch departure from the shared backbone."""

    if branch_name not in {"environment", "biomass"}:
        raise DynamicEvidenceContractError("branch_name must be environment or biomass.")
    if l2_regularization < 0:
        raise DynamicEvidenceContractError("l2_regularization must be nonnegative.")
    y = _binary_outcome(outcome)
    z0 = probability_logit(
        backbone_probability,
        name="backbone_probability",
        probability_floor=probability_floor,
    )
    z_branch = probability_logit(
        branch_probability,
        name=f"{branch_name}_branch_probability",
        probability_floor=probability_floor,
    )
    if z0.shape != y.shape or z_branch.shape != y.shape:
        raise DynamicEvidenceContractError("Reliability-fit arrays have incompatible lengths.")
    transform, design = fit_reliability_feature_transform(
        reliability_values,
        feature_names=feature_names,
    )
    if len(design) != len(y):
        raise DynamicEvidenceContractError("Reliability rows and outcomes differ in length.")
    weights = _normalized_sample_weight(sample_weight, rows=len(y))
    delta = z_branch - z0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        reliability = expit(np.clip(theta[0] + design @ theta[1:], -40.0, 40.0))
        final_logit = z0 + reliability * delta
        residual = weights * (expit(final_logit) - y)
        loss = np.mean(weights * (np.logaddexp(0.0, final_logit) - y * final_logit))
        loss += 0.5 * l2_regularization * float(np.sum(theta[1:] ** 2))
        reliability_gradient = residual * delta * reliability * (1.0 - reliability)
        gradient = np.r_[
            reliability_gradient.mean(),
            design.T @ reliability_gradient / len(y)
            + l2_regularization * theta[1:],
        ]
        return float(loss), gradient

    initial = np.zeros(design.shape[1] + 1, dtype=float)
    result = minimize(
        lambda theta: objective(theta)[0],
        initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[(-_PARAMETER_LIMIT, _PARAMETER_LIMIT)] * len(initial),
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise DynamicEvidenceContractError(
            f"Reliability shrinkage fit failed for {branch_name}: {result.message}"
        )
    return ReliabilityShrinkageModel(
        branch_name=branch_name,
        transform=transform,
        intercept=float(result.x[0]),
        coefficients=tuple(float(value) for value in result.x[1:]),
        l2_regularization=float(l2_regularization),
        objective_value=float(result.fun),
    )


def _fusion_fit_arrays(
    environment_adjusted_logit: Sequence[float] | np.ndarray,
    biomass_adjusted_logit: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    environment = np.asarray(environment_adjusted_logit, dtype=float)
    biomass = np.asarray(biomass_adjusted_logit, dtype=float)
    y = _binary_outcome(outcome)
    if (
        environment.shape != y.shape
        or biomass.shape != y.shape
        or not np.isfinite(environment).all()
        or not np.isfinite(biomass).all()
    ):
        raise DynamicEvidenceContractError("Adjusted branch logits and outcomes are incompatible.")
    weights = _normalized_sample_weight(sample_weight, rows=len(y))
    return environment, biomass, y, weights


def _initial_calibration_intercept(
    environment_logit: np.ndarray,
    biomass_logit: np.ndarray,
    outcome: np.ndarray,
) -> float:
    prevalence = float(np.clip(outcome.mean(), _PROBABILITY_FLOOR, 1 - _PROBABILITY_FLOOR))
    prevalence_logit = float(np.log(prevalence) - np.log1p(-prevalence))
    return float(np.clip(prevalence_logit - np.mean((environment_logit + biomass_logit) / 2), -10, 10))


@dataclass(frozen=True)
class ConstantWeightModel:
    """W0/A2: one globally learned biomass-branch weight."""

    biomass_weight: float
    calibration_intercept: float
    objective_value: float
    optimization_method: str = "L-BFGS-B"
    model_type: str = "W0_constant"

    def predict_weight(self, stage_score: Sequence[float] | np.ndarray) -> np.ndarray:
        stage = _stage_score_array(stage_score)
        return np.full(len(stage), self.biomass_weight, dtype=float)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_constant_weight(
    environment_adjusted_logit: Sequence[float] | np.ndarray,
    biomass_adjusted_logit: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> ConstantWeightModel:
    """Fit W0 from OOF branch logits; no stage proportion is preset."""

    environment, biomass, y, weights = _fusion_fit_arrays(
        environment_adjusted_logit,
        biomass_adjusted_logit,
        outcome,
        sample_weight,
    )
    delta = biomass - environment

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        final_logit = environment + theta[0] * delta + theta[1]
        residual = weights * (expit(final_logit) - y)
        loss = np.mean(weights * (np.logaddexp(0.0, final_logit) - y * final_logit))
        gradient = np.array(
            [np.mean(residual * delta), residual.mean()],
            dtype=float,
        )
        return float(loss), gradient

    initial = np.array(
        [0.5, _initial_calibration_intercept(environment, biomass, y)],
        dtype=float,
    )
    result = minimize(
        lambda theta: objective(theta)[0],
        initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[(0.0, 1.0), (-_PARAMETER_LIMIT, _PARAMETER_LIMIT)],
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise DynamicEvidenceContractError(f"W0 fit failed: {result.message}")
    return ConstantWeightModel(
        biomass_weight=float(result.x[0]),
        calibration_intercept=float(result.x[1]),
        objective_value=float(result.fun),
    )


def _stage_score_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    stage = np.asarray(values, dtype=float)
    if stage.ndim != 1 or not np.isfinite(stage).all():
        raise DynamicEvidenceContractError("Continuous stage scores must be finite and 1D.")
    if (stage < 0).any() or (stage > 1).any():
        raise DynamicEvidenceContractError("Continuous stage scores must lie in [0, 1].")
    return stage


def _anchor_interpolation_matrix(
    stage_score: np.ndarray,
    anchor_positions: np.ndarray,
) -> np.ndarray:
    rows = len(stage_score)
    design = np.zeros((rows, len(anchor_positions)), dtype=float)
    upper = np.searchsorted(anchor_positions, stage_score, side="right")
    upper = np.clip(upper, 1, len(anchor_positions) - 1)
    lower = upper - 1
    span = anchor_positions[upper] - anchor_positions[lower]
    fraction = (stage_score - anchor_positions[lower]) / span
    left_edge = stage_score <= anchor_positions[0]
    right_edge = stage_score >= anchor_positions[-1]
    fraction[left_edge] = 0.0
    lower[left_edge] = 0
    upper[left_edge] = 1
    fraction[right_edge] = 1.0
    lower[right_edge] = len(anchor_positions) - 2
    upper[right_edge] = len(anchor_positions) - 1
    index = np.arange(rows)
    design[index, lower] = 1.0 - fraction
    design[index, upper] += fraction
    return design


@dataclass(frozen=True)
class AnchorWeightModel:
    """W1/P1: four learned stage-anchor weights with smooth interpolation."""

    anchor_positions: tuple[float, ...]
    anchor_logits: tuple[float, ...]
    calibration_intercept: float
    smoothness_penalty: float
    objective_value: float
    optimization_method: str = "L-BFGS-B"
    model_type: str = "W1_four_anchor"

    @property
    def anchor_weights(self) -> tuple[float, ...]:
        return tuple(float(value) for value in expit(np.asarray(self.anchor_logits)))

    def predict_weight(self, stage_score: Sequence[float] | np.ndarray) -> np.ndarray:
        stage = _stage_score_array(stage_score)
        position = np.asarray(self.anchor_positions, dtype=float)
        design = _anchor_interpolation_matrix(stage, position)
        return design @ expit(np.asarray(self.anchor_logits, dtype=float))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["anchor_weights"] = list(self.anchor_weights)
        return result


def fit_anchor_weight(
    environment_adjusted_logit: Sequence[float] | np.ndarray,
    biomass_adjusted_logit: Sequence[float] | np.ndarray,
    stage_score: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    *,
    anchor_positions: Sequence[float] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    smoothness_penalty: float = 0.1,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> AnchorWeightModel:
    """Fit four free W1 anchors; positions encode stage scale, not fixed weights."""

    environment, biomass, y, weights = _fusion_fit_arrays(
        environment_adjusted_logit,
        biomass_adjusted_logit,
        outcome,
        sample_weight,
    )
    stage = _stage_score_array(stage_score)
    if stage.shape != y.shape:
        raise DynamicEvidenceContractError("Stage scores and outcomes differ in length.")
    position = np.asarray(anchor_positions, dtype=float)
    if (
        position.shape != (4,)
        or not np.isfinite(position).all()
        or position[0] != 0.0
        or position[-1] != 1.0
        or not np.all(np.diff(position) > 0)
    ):
        raise DynamicEvidenceContractError(
            "W1 requires four strictly increasing anchor positions from 0 to 1."
        )
    if smoothness_penalty < 0:
        raise DynamicEvidenceContractError("smoothness_penalty must be nonnegative.")
    interpolation = _anchor_interpolation_matrix(stage, position)
    second_difference = np.array([[1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0]])
    delta = biomass - environment

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        anchor_weight = expit(theta[:4])
        row_weight = interpolation @ anchor_weight
        final_logit = environment + row_weight * delta + theta[4]
        residual = weights * (expit(final_logit) - y)
        curvature = second_difference @ anchor_weight
        loss = np.mean(weights * (np.logaddexp(0.0, final_logit) - y * final_logit))
        loss += 0.5 * smoothness_penalty * float(curvature @ curvature)
        gradient_weight = interpolation.T @ (residual * delta) / len(y)
        gradient_weight += smoothness_penalty * second_difference.T @ curvature
        gradient_anchor_logit = gradient_weight * anchor_weight * (1.0 - anchor_weight)
        gradient = np.r_[gradient_anchor_logit, residual.mean()]
        return float(loss), gradient

    initial = np.r_[
        np.zeros(4, dtype=float),
        _initial_calibration_intercept(environment, biomass, y),
    ]
    result = minimize(
        lambda theta: objective(theta)[0],
        initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0)] * 4 + [(-_PARAMETER_LIMIT, _PARAMETER_LIMIT)],
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise DynamicEvidenceContractError(f"W1 fit failed: {result.message}")
    return AnchorWeightModel(
        anchor_positions=tuple(float(value) for value in position),
        anchor_logits=tuple(float(value) for value in result.x[:4]),
        calibration_intercept=float(result.x[4]),
        smoothness_penalty=float(smoothness_penalty),
        objective_value=float(result.fun),
    )


def _smooth_logistic_basis(
    stage_score: np.ndarray,
    knots: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    raw = expit((stage_score[:, None] - knots[None, :]) / bandwidth)
    lower = expit(-knots / bandwidth)
    upper = expit((1.0 - knots) / bandwidth)
    denominator = upper - lower
    if (denominator <= 1e-10).any():
        raise DynamicEvidenceContractError("W2 basis knots and bandwidth are numerically invalid.")
    return (raw - lower[None, :]) / denominator[None, :]


@dataclass(frozen=True)
class SmoothLogisticWeightModel:
    """W2/P2: learned smooth stage weight, monotone for the primary candidate."""

    knots: tuple[float, ...]
    bandwidth: float
    stage_intercept: float
    stage_coefficients: tuple[float, ...]
    reliability_transform: ReliabilityFeatureTransform | None
    reliability_coefficients: tuple[float, ...]
    calibration_intercept: float
    monotonic: bool
    smoothness_penalty: float
    l2_regularization: float
    objective_value: float
    optimization_method: str = "L-BFGS-B"
    model_type: str = "W2_smooth_logistic"

    def predict_weight(
        self,
        stage_score: Sequence[float] | np.ndarray,
        reliability_values: pd.DataFrame | np.ndarray | None = None,
    ) -> np.ndarray:
        stage = _stage_score_array(stage_score)
        basis = _smooth_logistic_basis(
            stage,
            np.asarray(self.knots, dtype=float),
            float(self.bandwidth),
        )
        linear = self.stage_intercept + basis @ np.asarray(
            self.stage_coefficients, dtype=float
        )
        if self.reliability_transform is not None:
            if reliability_values is None:
                raise DynamicEvidenceContractError("W2 requires its registered reliability inputs.")
            reliability_design = self.reliability_transform.transform(reliability_values)
            if len(reliability_design) != len(stage):
                raise DynamicEvidenceContractError("W2 reliability rows and stage scores differ.")
            linear += reliability_design @ np.asarray(
                self.reliability_coefficients, dtype=float
            )
        elif reliability_values is not None:
            raise DynamicEvidenceContractError("W2 was fitted without reliability inputs.")
        return expit(np.clip(linear, -40.0, 40.0))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.reliability_transform is not None:
            result["reliability_transform"] = self.reliability_transform.to_dict()
        return result


def fit_smooth_logistic_weight(
    environment_adjusted_logit: Sequence[float] | np.ndarray,
    biomass_adjusted_logit: Sequence[float] | np.ndarray,
    stage_score: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    *,
    reliability_values: pd.DataFrame | np.ndarray | None = None,
    reliability_feature_names: Sequence[str] | None = None,
    knots: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    bandwidth: float = 0.12,
    monotonic: bool = True,
    smoothness_penalty: float = 0.1,
    l2_regularization: float = 0.01,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> SmoothLogisticWeightModel:
    """Fit W2 on OOF components with a smooth learned stage/reliability weight."""

    environment, biomass, y, weights = _fusion_fit_arrays(
        environment_adjusted_logit,
        biomass_adjusted_logit,
        outcome,
        sample_weight,
    )
    stage = _stage_score_array(stage_score)
    if stage.shape != y.shape:
        raise DynamicEvidenceContractError("Stage scores and outcomes differ in length.")
    knot_array = np.asarray(knots, dtype=float)
    if (
        knot_array.ndim != 1
        or len(knot_array) < 2
        or not np.isfinite(knot_array).all()
        or (knot_array <= 0).any()
        or (knot_array >= 1).any()
        or not np.all(np.diff(knot_array) > 0)
    ):
        raise DynamicEvidenceContractError("W2 knots must be strictly increasing inside (0, 1).")
    if bandwidth <= 0 or smoothness_penalty < 0 or l2_regularization < 0:
        raise DynamicEvidenceContractError("W2 penalties and bandwidth are invalid.")
    stage_basis = _smooth_logistic_basis(stage, knot_array, bandwidth)
    if reliability_values is None:
        if reliability_feature_names is not None:
            raise DynamicEvidenceContractError(
                "Reliability feature names were supplied without reliability values."
            )
        reliability_transform = None
        reliability_design = np.empty((len(stage), 0), dtype=float)
    else:
        reliability_transform, reliability_design = fit_reliability_feature_transform(
            reliability_values,
            feature_names=reliability_feature_names,
        )
        if len(reliability_design) != len(stage):
            raise DynamicEvidenceContractError("W2 reliability rows and outcomes differ.")

    grid = np.linspace(0.0, 1.0, 41)
    grid_basis = _smooth_logistic_basis(grid, knot_array, bandwidth)
    second_difference = np.diff(np.eye(len(grid)), n=2, axis=0)
    curvature_design = second_difference @ grid_basis
    delta = biomass - environment
    stage_count = stage_basis.shape[1]
    reliability_count = reliability_design.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        stage_intercept = theta[0]
        stage_coefficient = theta[1 : 1 + stage_count]
        reliability_coefficient = theta[
            1 + stage_count : 1 + stage_count + reliability_count
        ]
        calibration_intercept = theta[-1]
        weight_logit = stage_intercept + stage_basis @ stage_coefficient
        if reliability_count:
            weight_logit += reliability_design @ reliability_coefficient
        biomass_weight = expit(np.clip(weight_logit, -40.0, 40.0))
        final_logit = environment + biomass_weight * delta + calibration_intercept
        residual = weights * (expit(final_logit) - y)
        loss = np.mean(weights * (np.logaddexp(0.0, final_logit) - y * final_logit))
        curvature = curvature_design @ stage_coefficient
        loss += 0.5 * smoothness_penalty * float(np.mean(curvature**2))
        loss += 0.5 * l2_regularization * float(
            np.sum(stage_coefficient**2) + np.sum(reliability_coefficient**2)
        )

        weight_gradient = residual * delta * biomass_weight * (1.0 - biomass_weight)
        gradient_stage_intercept = weight_gradient.mean()
        gradient_stage = stage_basis.T @ weight_gradient / len(y)
        gradient_stage += (
            smoothness_penalty
            * curvature_design.T
            @ curvature
            / max(len(curvature), 1)
        )
        gradient_stage += l2_regularization * stage_coefficient
        if reliability_count:
            gradient_reliability = reliability_design.T @ weight_gradient / len(y)
            gradient_reliability += l2_regularization * reliability_coefficient
        else:
            gradient_reliability = np.empty(0, dtype=float)
        gradient = np.r_[
            gradient_stage_intercept,
            gradient_stage,
            gradient_reliability,
            residual.mean(),
        ]
        return float(loss), gradient

    initial = np.r_[
        0.0,
        np.zeros(stage_count, dtype=float),
        np.zeros(reliability_count, dtype=float),
        _initial_calibration_intercept(environment, biomass, y),
    ]
    stage_bounds = (
        [(0.0, _PARAMETER_LIMIT)] * stage_count
        if monotonic
        else [(-_PARAMETER_LIMIT, _PARAMETER_LIMIT)] * stage_count
    )
    bounds = (
        [(-_PARAMETER_LIMIT, _PARAMETER_LIMIT)]
        + stage_bounds
        + [(-_PARAMETER_LIMIT, _PARAMETER_LIMIT)] * reliability_count
        + [(-_PARAMETER_LIMIT, _PARAMETER_LIMIT)]
    )
    result = minimize(
        lambda theta: objective(theta)[0],
        initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise DynamicEvidenceContractError(f"W2 fit failed: {result.message}")
    stage_coefficient = result.x[1 : 1 + stage_count]
    if monotonic and (stage_coefficient < -1e-10).any():
        raise DynamicEvidenceContractError("Monotone W2 returned a negative stage coefficient.")
    reliability_coefficient = result.x[
        1 + stage_count : 1 + stage_count + reliability_count
    ]
    return SmoothLogisticWeightModel(
        knots=tuple(float(value) for value in knot_array),
        bandwidth=float(bandwidth),
        stage_intercept=float(result.x[0]),
        stage_coefficients=tuple(float(max(value, 0.0) if monotonic else value) for value in stage_coefficient),
        reliability_transform=reliability_transform,
        reliability_coefficients=tuple(float(value) for value in reliability_coefficient),
        calibration_intercept=float(result.x[-1]),
        monotonic=bool(monotonic),
        smoothness_penalty=float(smoothness_penalty),
        l2_regularization=float(l2_regularization),
        objective_value=float(result.fun),
    )


WeightModel = ConstantWeightModel | AnchorWeightModel | SmoothLogisticWeightModel


@dataclass(frozen=True)
class DynamicEvidencePrediction:
    """Diagnostics and the single final risk output from one lifecycle model."""

    final_probability: np.ndarray
    biomass_weight: np.ndarray
    environment_reliability: np.ndarray
    biomass_reliability: np.ndarray
    environment_adjusted_logit: np.ndarray
    biomass_adjusted_logit: np.ndarray


@dataclass(frozen=True)
class DynamicEvidenceFusionModel:
    """Dynamic fusion module joining two internal evidence branches into one risk."""

    environment_shrinkage: ReliabilityShrinkageModel
    biomass_shrinkage: ReliabilityShrinkageModel
    weight_model: WeightModel
    probability_floor: float = _PROBABILITY_FLOOR
    model_identity: str = "lifecycle_dynamic_evidence_weighting"

    def predict(
        self,
        *,
        backbone_probability: Sequence[float] | np.ndarray,
        environment_branch_probability: Sequence[float] | np.ndarray,
        biomass_branch_probability: Sequence[float] | np.ndarray,
        continuous_stage_score_values: Sequence[float] | np.ndarray,
        environment_reliability_values: pd.DataFrame | np.ndarray,
        biomass_reliability_values: pd.DataFrame | np.ndarray,
        fusion_reliability_values: pd.DataFrame | np.ndarray | None = None,
    ) -> DynamicEvidencePrediction:
        environment_logit = self.environment_shrinkage.adjusted_logit(
            backbone_probability,
            environment_branch_probability,
            environment_reliability_values,
            probability_floor=self.probability_floor,
        )
        biomass_logit = self.biomass_shrinkage.adjusted_logit(
            backbone_probability,
            biomass_branch_probability,
            biomass_reliability_values,
            probability_floor=self.probability_floor,
        )
        stage = _stage_score_array(continuous_stage_score_values)
        if environment_logit.shape != stage.shape or biomass_logit.shape != stage.shape:
            raise DynamicEvidenceContractError("Fusion inputs differ in row count.")
        if isinstance(self.weight_model, SmoothLogisticWeightModel):
            biomass_weight = self.weight_model.predict_weight(
                stage,
                fusion_reliability_values,
            )
        else:
            if fusion_reliability_values is not None:
                raise DynamicEvidenceContractError(
                    "Fusion reliability inputs are only registered for W2."
                )
            biomass_weight = self.weight_model.predict_weight(stage)
        final_logit = (
            environment_logit
            + biomass_weight * (biomass_logit - environment_logit)
            + self.weight_model.calibration_intercept
        )
        final_probability = expit(np.clip(final_logit, -40.0, 40.0))
        environment_reliability = self.environment_shrinkage.predict_reliability(
            environment_reliability_values
        )
        biomass_reliability = self.biomass_shrinkage.predict_reliability(
            biomass_reliability_values
        )
        return DynamicEvidencePrediction(
            final_probability=final_probability,
            biomass_weight=biomass_weight,
            environment_reliability=environment_reliability,
            biomass_reliability=biomass_reliability,
            environment_adjusted_logit=environment_logit,
            biomass_adjusted_logit=biomass_logit,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_identity": self.model_identity,
            "model_flow": (
                "prior-only inputs -> shared backbone -> environment evidence branch / "
                "biomass evidence branch -> continuous stage score + reliability -> "
                "learned dynamic fusion -> one risk score"
            ),
            "probability_floor": float(self.probability_floor),
            "environment_shrinkage": self.environment_shrinkage.to_dict(),
            "biomass_shrinkage": self.biomass_shrinkage.to_dict(),
            "weight_model": self.weight_model.to_dict(),
        }


def fit_dynamic_evidence_fusion(
    *,
    backbone_probability: Sequence[float] | np.ndarray,
    environment_branch_probability: Sequence[float] | np.ndarray,
    biomass_branch_probability: Sequence[float] | np.ndarray,
    continuous_stage_score_values: Sequence[float] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    environment_reliability_values: pd.DataFrame | np.ndarray,
    biomass_reliability_values: pd.DataFrame | np.ndarray,
    environment_reliability_feature_names: Sequence[str] | None = None,
    biomass_reliability_feature_names: Sequence[str] | None = None,
    weight_kind: Literal["W0", "W1", "W2"] = "W2",
    fusion_reliability_values: pd.DataFrame | np.ndarray | None = None,
    fusion_reliability_feature_names: Sequence[str] | None = None,
    reliability_l2_regularization: float = 0.1,
    weight_options: dict[str, object] | None = None,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    probability_floor: float = _PROBABILITY_FLOOR,
) -> DynamicEvidenceFusionModel:
    """Fit the reliability and weight modules from one OOF component table."""

    y = _binary_outcome(outcome)
    environment_shrinkage = fit_reliability_shrinkage(
        backbone_probability,
        environment_branch_probability,
        y,
        environment_reliability_values,
        branch_name="environment",
        feature_names=environment_reliability_feature_names,
        l2_regularization=reliability_l2_regularization,
        sample_weight=sample_weight,
        probability_floor=probability_floor,
    )
    biomass_shrinkage = fit_reliability_shrinkage(
        backbone_probability,
        biomass_branch_probability,
        y,
        biomass_reliability_values,
        branch_name="biomass",
        feature_names=biomass_reliability_feature_names,
        l2_regularization=reliability_l2_regularization,
        sample_weight=sample_weight,
        probability_floor=probability_floor,
    )
    environment_logit = environment_shrinkage.adjusted_logit(
        backbone_probability,
        environment_branch_probability,
        environment_reliability_values,
        probability_floor=probability_floor,
    )
    biomass_logit = biomass_shrinkage.adjusted_logit(
        backbone_probability,
        biomass_branch_probability,
        biomass_reliability_values,
        probability_floor=probability_floor,
    )
    options = dict(weight_options or {})
    if weight_kind == "W0":
        if fusion_reliability_values is not None or fusion_reliability_feature_names is not None:
            raise DynamicEvidenceContractError("W0 does not accept fusion reliability inputs.")
        weight_model: WeightModel = fit_constant_weight(
            environment_logit,
            biomass_logit,
            y,
            sample_weight=sample_weight,
            **options,
        )
    elif weight_kind == "W1":
        if fusion_reliability_values is not None or fusion_reliability_feature_names is not None:
            raise DynamicEvidenceContractError("W1 does not accept fusion reliability inputs.")
        weight_model = fit_anchor_weight(
            environment_logit,
            biomass_logit,
            continuous_stage_score_values,
            y,
            sample_weight=sample_weight,
            **options,
        )
    elif weight_kind == "W2":
        weight_model = fit_smooth_logistic_weight(
            environment_logit,
            biomass_logit,
            continuous_stage_score_values,
            y,
            reliability_values=fusion_reliability_values,
            reliability_feature_names=fusion_reliability_feature_names,
            sample_weight=sample_weight,
            **options,
        )
    else:
        raise DynamicEvidenceContractError("weight_kind must be W0, W1, or W2.")
    return DynamicEvidenceFusionModel(
        environment_shrinkage=environment_shrinkage,
        biomass_shrinkage=biomass_shrinkage,
        weight_model=weight_model,
        probability_floor=float(probability_floor),
    )


__all__ = [
    "AnchorWeightModel",
    "ConstantWeightModel",
    "CumulativeHGBStageEstimator",
    "CumulativeOrdinalLogistic",
    "DynamicEvidenceContractError",
    "DynamicEvidenceFusionModel",
    "DynamicEvidencePrediction",
    "PRIOR_ONLY_FORBIDDEN_NAME_PATTERNS",
    "ReliabilityFeatureTransform",
    "ReliabilityShrinkageModel",
    "SmoothLogisticWeightModel",
    "assert_prior_only_feature_names",
    "continuous_stage_score",
    "cumulative_probabilities_to_stage_probabilities",
    "find_forbidden_prior_only_feature_names",
    "fit_anchor_weight",
    "fit_constant_weight",
    "fit_dynamic_evidence_fusion",
    "fit_reliability_feature_transform",
    "fit_reliability_shrinkage",
    "fit_smooth_logistic_weight",
    "probability_logit",
    "project_cumulative_probabilities",
]
