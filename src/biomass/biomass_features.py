from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WINDOWS = (7, 14, 30, 60, 90)

class BiomassContractError(RuntimeError):
    pass

def source_feature_name(source: str, window: int, statistic: str) -> str:
    return f"{source}_lag{window}_{statistic}"

def canonical_dates(values: pd.Series | np.ndarray) -> np.ndarray:
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed, pd.Series):
        return parsed.to_numpy(dtype="datetime64[D]")
    return np.asarray(parsed, dtype="datetime64[D]")

def expected_day_index(origin_days: np.ndarray, max_window: int) -> np.ndarray:
    offsets = np.arange(max_window, 0, -1, dtype=np.int64)
    return origin_days.astype("datetime64[D]").astype(np.int64)[:, None] - offsets[None, :]

def gather_by_availability_day(
    availability_days: np.ndarray,
    expected_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_int = availability_days.astype("datetime64[D]").astype(np.int64)
    positions = np.searchsorted(source_int, expected_days, side="left")
    clipped = np.clip(positions, 0, max(len(source_int) - 1, 0))
    if len(source_int) == 0:
        matched = np.zeros(expected_days.shape, dtype=bool)
    else:
        matched = source_int[clipped] == expected_days
    return clipped, matched

def safe_row_stat(values: np.ndarray, statistic: str) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if statistic == "mean":
            out = np.nanmean(values, axis=1)
        elif statistic == "max":
            out = np.nanmax(values, axis=1)
        elif statistic == "p90":
            out = np.nanquantile(values, 0.90, axis=1)
        elif statistic == "std":
            out = np.nanstd(values, axis=1)
        else:
            raise BiomassContractError(f"Unsupported row statistic: {statistic}")
    out[counts == 0] = np.nan
    return out

def rowwise_slope(values: np.ndarray) -> np.ndarray:
    x = np.broadcast_to(
        np.arange(values.shape[1], dtype=np.float64),
        values.shape,
    )
    valid = np.isfinite(values)
    n = valid.sum(axis=1).astype(np.float64)
    x_masked = np.where(valid, x, 0.0)
    y_masked = np.where(valid, values, 0.0)
    sum_x = x_masked.sum(axis=1)
    sum_y = y_masked.sum(axis=1)
    sum_xx = (x_masked * x_masked).sum(axis=1)
    sum_xy = (x_masked * y_masked).sum(axis=1)
    denominator = n * sum_xx - sum_x * sum_x
    return np.divide(
        n * sum_xy - sum_x * sum_y,
        denominator,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=(n >= 2) & (denominator > 0),
    )

def latest_from_matrix(
    values: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    any_valid = valid.any(axis=1)
    last_position = values.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    row = np.arange(len(values))
    latest_value = np.full(len(values), np.nan, dtype=np.float64)
    latest_date = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
    latest_value[any_valid] = values[row[any_valid], last_position[any_valid]]
    latest_date[any_valid] = dates[row[any_valid], last_position[any_valid]].astype(
        "datetime64[ns]"
    )
    return latest_value, latest_date

def compute_source_features(
    *,
    source: str,
    origins: np.ndarray,
    history: pd.DataFrame,
    availability_col: str,
    observation_col: str,
    value_col: str,
    quality_col: str,
    censor_col: str | None,
    site_count_col: str | None,
    positive_threshold: float,
    windows: Sequence[int] = WINDOWS,
) -> dict[str, np.ndarray]:
    origin_days = canonical_dates(origins)
    if np.isnat(origin_days).any():
        raise BiomassContractError("Origins contain invalid dates.")
    history = history.copy()
    history[availability_col] = pd.to_datetime(
        history[availability_col], errors="coerce"
    ).dt.normalize()
    history[observation_col] = pd.to_datetime(
        history[observation_col], errors="coerce"
    ).dt.normalize()
    if history[[availability_col, observation_col]].isna().any().any():
        raise BiomassContractError(f"{source} history contains invalid dates.")
    history = history.sort_values(observation_col, kind="stable")
    if history[observation_col].duplicated().any():
        raise BiomassContractError(
            f"{source} observation date must be unique within lake."
        )
    observation_days = history[observation_col].to_numpy(dtype="datetime64[D]")
    availability_days = history[availability_col].to_numpy(dtype="datetime64[D]")
    if len(observation_days) > 1 and np.any(
        np.diff(observation_days.astype(np.int64)) <= 0
    ):
        raise BiomassContractError(
            f"{source} observation dates are not strictly increasing."
        )

    max_window = int(max(windows))
    expected = expected_day_index(origin_days, max_window)
    positions, observed_in_window = gather_by_availability_day(
        observation_days, expected
    )
    availability_matrix = np.full(
        expected.shape, np.datetime64("NaT"), dtype="datetime64[D]"
    )
    if len(history):
        availability_matrix[observed_in_window] = availability_days[
            positions[observed_in_window]
        ]
    available_before_origin = (
        observed_in_window
        & ~np.isnat(availability_matrix)
        & (
            availability_matrix.astype(np.int64)
            < origin_days.astype(np.int64)[:, None]
        )
    )
    value_source = pd.to_numeric(history[value_col], errors="coerce").to_numpy(float)
    quality_source = pd.to_numeric(history[quality_col], errors="coerce").to_numpy(float)
    value_matrix = np.full(expected.shape, np.nan, dtype=np.float64)
    quality_matrix = np.full(expected.shape, np.nan, dtype=np.float64)
    observation_matrix = np.full(expected.shape, np.datetime64("NaT"), dtype="datetime64[D]")
    if len(history):
        value_matrix[available_before_origin] = value_source[
            positions[available_before_origin]
        ]
        quality_matrix[available_before_origin] = quality_source[
            positions[available_before_origin]
        ]
        observation_matrix[available_before_origin] = observation_days[
            positions[available_before_origin]
        ]

    censor_matrix: np.ndarray | None = None
    if censor_col is not None:
        censor_source = pd.to_numeric(history[censor_col], errors="coerce").to_numpy(float)
        censor_matrix = np.full(expected.shape, np.nan, dtype=np.float64)
        if len(history):
            censor_matrix[available_before_origin] = censor_source[
                positions[available_before_origin]
            ]
    site_matrix: np.ndarray | None = None
    if site_count_col is not None:
        site_source = pd.to_numeric(
            history[site_count_col], errors="coerce"
        ).to_numpy(float)
        site_matrix = np.full(expected.shape, np.nan, dtype=np.float64)
        if len(history):
            site_matrix[available_before_origin] = site_source[
                positions[available_before_origin]
            ]

    result: dict[str, np.ndarray] = {}
    for window in windows:
        observed_present = observed_in_window[:, -window:]
        source_present = available_before_origin[:, -window:]
        values = value_matrix[:, -window:]
        quality = quality_matrix[:, -window:]
        observation_dates = observation_matrix[:, -window:]
        availability_dates = availability_matrix[:, -window:]
        valid = np.isfinite(values)
        observed_days = observed_present.sum(axis=1).astype(np.int16)
        source_days = source_present.sum(axis=1).astype(np.int16)
        unavailable_days = (observed_present & ~source_present).sum(axis=1).astype(
            np.int16
        )
        valid_days = valid.sum(axis=1).astype(np.int16)

        prefix = lambda statistic: source_feature_name(source, window, statistic)
        result[prefix("observation_record_days")] = observed_days
        result[prefix("source_record_days")] = source_days
        result[prefix("unavailable_record_days")] = unavailable_days
        result[prefix("valid_value_days")] = valid_days
        result[prefix("valid_day_fraction")] = (
            valid_days.astype(np.float32) / np.float32(window)
        )
        result[prefix("mean")] = safe_row_stat(values, "mean").astype(np.float32)
        result[prefix("max")] = safe_row_stat(values, "max").astype(np.float32)
        result[prefix("p90")] = safe_row_stat(values, "p90").astype(np.float32)
        result[prefix("variability")] = safe_row_stat(values, "std").astype(np.float32)
        result[prefix("slope_per_day")] = rowwise_slope(values).astype(np.float32)

        positive_count = (valid & (values > positive_threshold)).sum(axis=1)
        result[prefix("positive_days")] = positive_count.astype(np.int16)
        result[prefix("positive_fraction_among_valid")] = np.divide(
            positive_count,
            valid_days,
            out=np.full(len(values), np.nan, dtype=np.float32),
            where=valid_days > 0,
        ).astype(np.float32)
        result[prefix("quality_mean")] = safe_row_stat(
            np.where(valid, quality, np.nan), "mean"
        ).astype(np.float32)

        latest_value, latest_availability = latest_from_matrix(
            values, availability_dates
        )
        _, latest_observation = latest_from_matrix(
            values, observation_dates
        )
        result[prefix("latest_value")] = latest_value.astype(np.float32)
        result[prefix("latest_availability_date")] = latest_availability
        result[prefix("latest_observation_date")] = latest_observation
        latest_available_day = latest_availability.astype("datetime64[D]")
        latest_observed_day = latest_observation.astype("datetime64[D]")
        result[prefix("days_since_latest_availability")] = np.where(
            np.isnat(latest_available_day),
            -1,
            origin_days.astype(np.int64) - latest_available_day.astype(np.int64),
        ).astype(np.int16)
        result[prefix("days_since_latest_observation")] = np.where(
            np.isnat(latest_observed_day),
            -1,
            origin_days.astype(np.int64) - latest_observed_day.astype(np.int64),
        ).astype(np.int16)

        reason = np.full(len(values), 2, dtype=np.int8)
        reason[(observed_days > 0) & (source_days == 0)] = 4
        reason[(source_days > 0) & (valid_days == 0)] = 1
        reason[valid_days > 0] = 0
        result[prefix("missing_reason_code")] = reason

        if censor_matrix is not None:
            censor_values = censor_matrix[:, -window:]
            result[prefix("censored_fraction_mean")] = safe_row_stat(
                np.where(valid, censor_values, np.nan), "mean"
            ).astype(np.float32)
        if site_matrix is not None:
            site_values = site_matrix[:, -window:]
            result[prefix("site_count_mean")] = safe_row_stat(
                np.where(valid, site_values, np.nan), "mean"
            ).astype(np.float32)
    return result
