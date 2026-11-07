from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WINDOWS = (7, 14, 30, 60, 90)

ITER_BATCH_ROWS = 131_072

VARIABLE_SPECS: dict[str, dict[str, Any]] = {
    "air_temp_mean_c": {
        "source": "air_temp_2m_mean_c",
        "unit": "degC",
        "stats": ("mean", "p90", "slope"),
    },
    "air_temp_min_c": {
        "source": "air_temp_2m_min_c",
        "unit": "degC",
        "stats": ("min",),
    },
    "air_temp_max_c": {
        "source": "air_temp_2m_max_c",
        "unit": "degC",
        "stats": ("max", "count_ge_25"),
    },
    "precipitation_mm": {
        "source": "precipitation_mm",
        "unit": "mm/day",
        "stats": ("sum", "max", "p90", "slope", "count_gt_0", "count_ge_25"),
    },
    "wind_speed_mean_ms": {
        "source": "wind_speed_10m_mean_ms",
        "unit": "m/s",
        "stats": ("mean", "max", "p90", "slope", "count_le_2"),
    },
    "wind_u_mean_ms": {
        "source": "wind_u_10m_mean_ms",
        "unit": "m/s",
        "stats": ("mean",),
    },
    "wind_v_mean_ms": {
        "source": "wind_v_10m_mean_ms",
        "unit": "m/s",
        "stats": ("mean",),
    },
    "shortwave_radiation_j_m2": {
        "source": "shortwave_radiation_j_m2",
        "unit": "J/m2/day",
        "stats": ("mean", "sum", "p90", "slope"),
    },
    "surface_pressure_mean_pa": {
        "source": "surface_pressure_mean_pa",
        "unit": "Pa",
        "stats": ("mean", "p90", "slope"),
    },
    "dewpoint_mean_c": {
        "source": "dewpoint_2m_mean_c",
        "unit": "degC",
        "stats": ("mean", "p90", "slope"),
    },
    "dewpoint_min_c": {
        "source": "dewpoint_2m_min_c",
        "unit": "degC",
        "stats": ("min",),
    },
    "dewpoint_max_c": {
        "source": "dewpoint_2m_max_c",
        "unit": "degC",
        "stats": ("max",),
    },
    "vpd_mean_kpa": {
        "source": "vpd_mean_kpa",
        "unit": "kPa",
        "stats": ("mean", "p90", "slope", "count_ge_1_5"),
    },
    "vpd_max_kpa": {
        "source": "vpd_max_kpa",
        "unit": "kPa",
        "stats": ("max",),
    },
}

class MeteorologyContractError(RuntimeError):
    pass

def canonical_dates(values: pd.Series | np.ndarray) -> np.ndarray:
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed, pd.Series):
        parsed = parsed.to_numpy(dtype="datetime64[D]")
    else:
        parsed = np.asarray(parsed, dtype="datetime64[D]")
    return parsed

def feature_name(variable: str, window: int, statistic: str) -> str:
    return f"met_{variable}_lag{window}_{statistic}"

def coverage_name(window: int, field: str) -> str:
    return f"met_lag{window}_{field}"

def variable_valid_name(variable: str, window: int) -> str:
    return f"met_{variable}_lag{window}_valid_days"

def safe_row_stat(values: np.ndarray, statistic: str) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if statistic == "mean":
            out = np.nanmean(values, axis=1)
        elif statistic == "sum":
            sums = np.nansum(values, axis=1)
            out[counts > 0] = sums[counts > 0]
        elif statistic == "min":
            out = np.nanmin(values, axis=1)
        elif statistic == "max":
            out = np.nanmax(values, axis=1)
        elif statistic == "p90":
            out = np.nanquantile(values, 0.90, axis=1)
        else:
            raise MeteorologyContractError(f"Unsupported statistic: {statistic}")
    out[counts == 0] = np.nan
    return out

def rowwise_slope(values: np.ndarray) -> np.ndarray:
    """OLS slope per real day, ignoring missing values."""

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
    slope = np.divide(
        n * sum_xy - sum_x * sum_y,
        denominator,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=(n >= 2) & (denominator > 0),
    )
    return slope

def threshold_count(values: np.ndarray, statistic: str) -> np.ndarray:
    valid = np.isfinite(values)
    if statistic == "count_ge_25":
        selected = valid & (values >= 25.0)
    elif statistic == "count_gt_0":
        selected = valid & (values > 0.0)
    elif statistic == "count_le_2":
        selected = valid & (values <= 2.0)
    elif statistic == "count_ge_1_5":
        selected = valid & (values >= 1.5)
    else:
        raise MeteorologyContractError(f"Unsupported threshold statistic: {statistic}")
    return selected.sum(axis=1).astype(np.float32)

def expected_day_index(origin_days: np.ndarray, max_window: int) -> np.ndarray:
    offsets = np.arange(max_window, 0, -1, dtype=np.int64)
    return origin_days.astype("datetime64[D]").astype(np.int64)[:, None] - offsets[None, :]

def gather_expected_days(
    daily_days: np.ndarray,
    expected_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    daily_int = daily_days.astype("datetime64[D]").astype(np.int64)
    positions = np.searchsorted(daily_int, expected_days, side="left")
    clipped = np.clip(positions, 0, max(len(daily_int) - 1, 0))
    if len(daily_int) == 0:
        matched = np.zeros(expected_days.shape, dtype=bool)
    else:
        matched = daily_int[clipped] == expected_days
    return clipped, matched

def compute_group_features(
    origins: np.ndarray,
    daily: pd.DataFrame,
    windows: Sequence[int] = WINDOWS,
) -> dict[str, np.ndarray]:
    """Compute strict [origin-W, origin) features for one ERA5 grid."""

    origin_days = canonical_dates(origins)
    if np.isnat(origin_days).any():
        raise MeteorologyContractError("Origin dates contain null or invalid values.")

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    if daily["date"].isna().any():
        raise MeteorologyContractError("Daily source contains invalid dates.")
    daily = daily.sort_values("date", kind="stable")
    daily_days = daily["date"].to_numpy(dtype="datetime64[D]")
    if len(daily_days) > 1:
        deltas = np.diff(daily_days.astype(np.int64))
        if np.any(deltas <= 0):
            raise MeteorologyContractError(
                "Daily grid dates must be unique and strictly increasing."
            )

    max_window = int(max(windows))
    expected = expected_day_index(origin_days, max_window)
    positions, matched = gather_expected_days(daily_days, expected)

    result: dict[str, np.ndarray] = {}
    for window in windows:
        mask = matched[:, -window:]
        available = mask.sum(axis=1).astype(np.int16)
        result[coverage_name(window, "expected_days")] = np.full(
            len(origin_days), window, dtype=np.int16
        )
        result[coverage_name(window, "available_days")] = available
        result[coverage_name(window, "coverage_fraction")] = (
            available.astype(np.float32) / np.float32(window)
        )
        latest = np.full(len(origin_days), np.datetime64("NaT"), dtype="datetime64[ns]")
        age = np.full(len(origin_days), -1, dtype=np.int16)
        for row_index in np.flatnonzero(available > 0):
            last_expected = expected[row_index, -window:][mask[row_index]][-1]
            latest[row_index] = np.datetime64(int(last_expected), "D").astype(
                "datetime64[ns]"
            )
            age[row_index] = int(
                origin_days[row_index].astype(np.int64) - int(last_expected)
            )
        result[coverage_name(window, "latest_source_date")] = latest
        result[coverage_name(window, "days_since_latest")] = age
        reason = np.full(len(origin_days), 4, dtype=np.int8)
        reason[available == 0] = 3
        reason[available == window] = 0
        result[coverage_name(window, "missing_reason_code")] = reason

    for variable, spec in VARIABLE_SPECS.items():
        source = str(spec["source"])
        if source not in daily.columns:
            raise MeteorologyContractError(f"Daily source is missing required column {source}.")
        raw = pd.to_numeric(daily[source], errors="coerce").to_numpy(dtype=np.float64)
        gathered = np.full(expected.shape, np.nan, dtype=np.float64)
        if len(raw):
            gathered[matched] = raw[positions[matched]]
        for window in windows:
            values = gathered[:, -window:]
            valid_days = np.isfinite(values).sum(axis=1).astype(np.int16)
            result[variable_valid_name(variable, window)] = valid_days
            for statistic in spec["stats"]:
                name = feature_name(variable, window, statistic)
                if statistic == "slope":
                    out = rowwise_slope(values)
                elif statistic.startswith("count_"):
                    out = threshold_count(values, statistic)
                    out[valid_days == 0] = np.nan
                else:
                    out = safe_row_stat(values, statistic)
                result[name] = out.astype(np.float32)
    return result

def allocate_outputs(n_rows: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for window in WINDOWS:
        arrays[coverage_name(window, "expected_days")] = np.full(
            n_rows, window, dtype=np.int16
        )
        arrays[coverage_name(window, "available_days")] = np.zeros(n_rows, dtype=np.int16)
        arrays[coverage_name(window, "coverage_fraction")] = np.zeros(
            n_rows, dtype=np.float32
        )
        arrays[coverage_name(window, "latest_source_date")] = np.full(
            n_rows, np.datetime64("NaT"), dtype="datetime64[ns]"
        )
        arrays[coverage_name(window, "days_since_latest")] = np.full(
            n_rows, -1, dtype=np.int16
        )
        arrays[coverage_name(window, "missing_reason_code")] = np.full(
            n_rows, 1, dtype=np.int8
        )
    for variable, spec in VARIABLE_SPECS.items():
        for window in WINDOWS:
            arrays[variable_valid_name(variable, window)] = np.zeros(
                n_rows, dtype=np.int16
            )
            for statistic in spec["stats"]:
                arrays[feature_name(variable, window, statistic)] = np.full(
                    n_rows, np.nan, dtype=np.float32
                )
    return arrays

def iter_contiguous_grid_frames(
    path: Path,
    required_grids: set[str],
    columns: Sequence[str],
    batch_size: int = ITER_BATCH_ROWS,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Stream a grid-sorted parquet file without retaining the 8.6M-row table."""

    parquet = pq.ParquetFile(path)
    current_grid: str | None = None
    current_chunks: list[pd.DataFrame] = []
    seen_closed: set[str] = set()

    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        frame = batch.to_pandas()
        frame["era5_grid_id"] = frame["era5_grid_id"].astype("string")
        ids = frame["era5_grid_id"].to_numpy()
        if len(ids) == 0:
            continue
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(frame)]))
        for start, end in zip(starts, ends, strict=True):
            grid = str(ids[start])
            if current_grid is None:
                current_grid = grid
            if grid != current_grid:
                if current_grid in seen_closed:
                    raise MeteorologyContractError(
                        f"Grid {current_grid} appears in multiple non-contiguous blocks."
                    )
                seen_closed.add(current_grid)
                if current_grid in required_grids:
                    yield current_grid, pd.concat(current_chunks, ignore_index=True)
                current_grid = grid
                current_chunks = []
            if grid in required_grids:
                current_chunks.append(frame.iloc[start:end].copy())

    if current_grid is not None:
        if current_grid in seen_closed:
            raise MeteorologyContractError(
                f"Grid {current_grid} appears in multiple non-contiguous blocks."
            )
        if current_grid in required_grids:
            yield current_grid, pd.concat(current_chunks, ignore_index=True)
