from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

CANONICAL_UNITS = {
    "chla": "ug/L",
    "tn": "ug/L",
    "tp": "ug/L",
    "do": "mg/L",
    "ph": "pH",
    "conductivity": "uS/cm",
    "turbidity": "NTU",
    "secchi_depth": "m",
}

PLAUSIBLE_RULES = {
    "chla": {"min": 0.0, "max": 10_000.0},
    "tn": {"min": 0.0, "max": 1_000_000.0},
    "tp": {"min": 0.0, "max": 1_000_000.0},
    "do": {"min": 0.0, "max": 100.0},
    "ph": {"min": 0.0, "max": 14.0},
    "conductivity": {"min": 0.0, "max": 200_000.0},
    "turbidity": {"min": 0.0, "max": 100_000.0},
    "secchi_depth": {"min": 0.0, "max": 100.0},
}

RAW_COLUMNS = [
    "lake_id",
    "site_id",
    "source_site_id",
    "sample_date",
    "sample_id",
    "canonical_variable",
    "standard_value",
    "standard_unit_inferred",
    "raw_unit",
    "censor_code",
    "standard_detection_limit_value",
    "source_detection_limit_condition",
    "source_value_qualifier_code",
    "sample_depth_m",
    "source_id",
    "parameter_conversion_factor",
    "package_id",
    "package_doi",
]

class WaterQualityContractError(RuntimeError):
    pass

def classify_censor_codes(codes: pd.Series) -> pd.DataFrame:
    normalized = codes.fillna("").astype(str).str.strip().str.upper()
    family = np.select(
        [normalized.str.startswith("NC"), normalized.str.startswith("LE")],
        ["NOT_CENSORED", "LEFT_CENSORED"],
        default="UNKNOWN",
    )
    return pd.DataFrame(
        {
            "censor_code_normalized": normalized,
            "censor_family": pd.Series(family, index=codes.index, dtype="string"),
            "is_censored": normalized.str.startswith("LE"),
            "is_unknown_censor": ~normalized.str.startswith(("NC", "LE")),
        },
        index=codes.index,
    )

def classify_qc(
    variable: pd.Series,
    value: pd.Series,
    standardized_unit: pd.Series,
) -> pd.DataFrame:
    numeric = pd.to_numeric(value, errors="coerce")
    expected_unit = variable.map(CANONICAL_UNITS)
    missing_key = (variable.isna() | ~variable.isin(CANONICAL_UNITS)).fillna(True)
    missing_value = numeric.isna().fillna(True)
    unit_mismatch = (
        standardized_unit.astype("string")
        .ne(expected_unit.astype("string"))
        .fillna(True)
    )
    below_min = pd.Series(False, index=variable.index)
    above_max = pd.Series(False, index=variable.index)
    for name, rules in PLAUSIBLE_RULES.items():
        selected = variable.eq(name)
        below_min |= selected & numeric.lt(rules["min"])
        above_max |= selected & numeric.gt(rules["max"])
    below_min = below_min.fillna(False)
    above_max = above_max.fillna(False)
    valid = ~(missing_key | missing_value | unit_mismatch | below_min | above_max)
    primary_flag = np.select(
        [
            missing_key.to_numpy(dtype=bool),
            missing_value.to_numpy(dtype=bool),
            unit_mismatch.to_numpy(dtype=bool),
            below_min.to_numpy(dtype=bool),
            above_max.to_numpy(dtype=bool),
        ],
        [
            "UNKNOWN_VARIABLE",
            "MISSING_VALUE",
            "STANDARD_UNIT_MISMATCH",
            "BELOW_PHYSICAL_MINIMUM",
            "ABOVE_PRESPECIFIED_MAXIMUM",
        ],
        default="OK",
    )
    return pd.DataFrame(
        {
            "qc_valid": valid,
            "qc_primary_flag": pd.Series(
                primary_flag, index=variable.index, dtype="string"
            ),
            "flag_missing_value": missing_value,
            "flag_unit_mismatch": unit_mismatch,
            "flag_below_min": below_min,
            "flag_above_max": above_max,
        },
        index=variable.index,
    )

def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise WaterQualityContractError(f"Raw table missing columns: {sorted(missing)}")
    frame = raw.loc[raw["canonical_variable"].isin(CANONICAL_UNITS)].copy()
    frame["sample_date"] = pd.to_datetime(
        frame["sample_date"], errors="coerce"
    ).dt.normalize()
    frame["missing_lake_id"] = frame["lake_id"].isna()
    frame["missing_site_id"] = frame["site_id"].isna()
    frame["missing_sample_date"] = frame["sample_date"].isna()
    frame["site_id"] = frame["site_id"].fillna("__MISSING_SITE_ID__")
    frame["key_eligible"] = ~(
        frame["missing_lake_id"] | frame["missing_sample_date"]
    )
    frame["standard_value"] = pd.to_numeric(
        frame["standard_value"], errors="coerce"
    )
    frame["standard_detection_limit_value"] = pd.to_numeric(
        frame["standard_detection_limit_value"], errors="coerce"
    )
    censor = classify_censor_codes(frame["censor_code"])
    qc = classify_qc(
        frame["canonical_variable"],
        frame["standard_value"],
        frame["standard_unit_inferred"],
    )
    frame = pd.concat([frame, censor, qc], axis=1)
    frame["is_detection_limit_only"] = frame["censor_code_normalized"].eq("LE5")
    frame["is_zero_without_detection_limit"] = frame[
        "censor_code_normalized"
    ].eq("LE6")
    frame["is_nonquantitative_censor"] = (
        frame["is_detection_limit_only"] | frame["is_zero_without_detection_limit"]
    )
    frame["numeric_aggregate_eligible"] = (
        frame["qc_valid"]
        & frame["key_eligible"]
        & ~frame["is_nonquantitative_censor"]
    )
    frame["value_qc_valid"] = frame["standard_value"].where(
        frame["numeric_aggregate_eligible"]
    )
    frame["value_qc_valid_uncensored"] = frame["standard_value"].where(
        frame["numeric_aggregate_eligible"] & ~frame["is_censored"]
    )
    frame["value_qc_valid_censored"] = frame["standard_value"].where(
        frame["numeric_aggregate_eligible"] & frame["is_censored"]
    )
    frame["detection_limit_qc_valid_censored"] = frame[
        "standard_detection_limit_value"
    ].where(frame["qc_valid"] & frame["key_eligible"] & frame["is_censored"])
    frame["valid_censored"] = (
        frame["numeric_aggregate_eligible"] & frame["is_censored"]
    )
    frame["valid_uncensored"] = (
        frame["numeric_aggregate_eligible"] & ~frame["is_censored"]
    )
    return frame

def aggregate_site_date(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame["key_eligible"]].copy()
    group = ["lake_id", "site_id", "sample_date", "canonical_variable"]
    site = (
        frame.groupby(group, sort=False, dropna=False)
        .agg(
            value_median=("value_qc_valid", "median"),
            value_min=("value_qc_valid", "min"),
            value_max_sensitivity=("value_qc_valid", "max"),
            uncensored_value_median=("value_qc_valid_uncensored", "median"),
            censored_reported_value_median=("value_qc_valid_censored", "median"),
            censored_detection_limit_median=(
                "detection_limit_qc_valid_censored",
                "median",
            ),
            raw_record_count=("sample_id", "size"),
            unique_sample_count=("sample_id", "nunique"),
            qc_valid_record_count=("qc_valid", "sum"),
            numeric_aggregate_eligible_count=("numeric_aggregate_eligible", "sum"),
            detection_limit_only_record_count=("is_detection_limit_only", "sum"),
            zero_without_detection_limit_record_count=(
                "is_zero_without_detection_limit",
                "sum",
            ),
            censored_valid_record_count=("valid_censored", "sum"),
            uncensored_valid_record_count=("valid_uncensored", "sum"),
            unknown_censor_record_count=("is_unknown_censor", "sum"),
            missing_value_record_count=("flag_missing_value", "sum"),
            unit_mismatch_record_count=("flag_unit_mismatch", "sum"),
            below_min_record_count=("flag_below_min", "sum"),
            above_max_record_count=("flag_above_max", "sum"),
            missing_site_id_record_count=("missing_site_id", "sum"),
            raw_unit_count=("raw_unit", "nunique"),
            source_count=("source_id", "nunique"),
            source_site_count=("source_site_id", "nunique"),
            conversion_factor_count=("parameter_conversion_factor", "nunique"),
            sample_depth_median=("sample_depth_m", "median"),
        )
        .reset_index()
        .rename(columns={"canonical_variable": "variable"})
    )
    site["canonical_unit"] = site["variable"].map(CANONICAL_UNITS)
    site["physical_qc_pass_fraction"] = np.divide(
        site["qc_valid_record_count"],
        site["raw_record_count"],
        out=np.zeros(len(site), dtype=float),
        where=site["raw_record_count"].gt(0),
    )
    site["numeric_aggregate_eligible_fraction"] = np.divide(
        site["numeric_aggregate_eligible_count"],
        site["raw_record_count"],
        out=np.zeros(len(site), dtype=float),
        where=site["raw_record_count"].gt(0),
    )
    site["censored_fraction_among_numeric_valid"] = np.divide(
        site["censored_valid_record_count"],
        site["numeric_aggregate_eligible_count"],
        out=np.zeros(len(site), dtype=float),
        where=site["numeric_aggregate_eligible_count"].gt(0),
    )
    site["aggregation_method"] = "median_within_lake_site_date_variable"
    site["sensitivity_aggregation"] = "maximum_within_lake_site_date_variable"
    return site

def aggregate_lake_date(site: pd.DataFrame) -> pd.DataFrame:
    site = site.copy()
    site["site_has_valid_value"] = site["value_median"].notna()
    group = ["lake_id", "sample_date", "variable"]
    lake = (
        site.groupby(group, sort=False, dropna=False)
        .agg(
            value_median=("value_median", "median"),
            value_min=("value_min", "min"),
            value_max_sensitivity=("value_max_sensitivity", "max"),
            uncensored_value_median=("uncensored_value_median", "median"),
            censored_reported_value_median=(
                "censored_reported_value_median",
                "median",
            ),
            censored_detection_limit_median=(
                "censored_detection_limit_median",
                "median",
            ),
            site_count=("site_id", "nunique"),
            site_with_valid_value_count=("site_has_valid_value", "sum"),
            raw_record_count=("raw_record_count", "sum"),
            unique_sample_count=("unique_sample_count", "sum"),
            qc_valid_record_count=("qc_valid_record_count", "sum"),
            numeric_aggregate_eligible_count=(
                "numeric_aggregate_eligible_count",
                "sum",
            ),
            detection_limit_only_record_count=(
                "detection_limit_only_record_count",
                "sum",
            ),
            zero_without_detection_limit_record_count=(
                "zero_without_detection_limit_record_count",
                "sum",
            ),
            censored_valid_record_count=("censored_valid_record_count", "sum"),
            uncensored_valid_record_count=("uncensored_valid_record_count", "sum"),
            unknown_censor_record_count=("unknown_censor_record_count", "sum"),
            missing_value_record_count=("missing_value_record_count", "sum"),
            unit_mismatch_record_count=("unit_mismatch_record_count", "sum"),
            below_min_record_count=("below_min_record_count", "sum"),
            above_max_record_count=("above_max_record_count", "sum"),
            missing_site_id_record_count=("missing_site_id_record_count", "sum"),
            raw_unit_count_max=("raw_unit_count", "max"),
            source_count_sum=("source_count", "sum"),
        )
        .reset_index()
    )
    lake["canonical_unit"] = lake["variable"].map(CANONICAL_UNITS)
    lake["physical_qc_pass_fraction"] = np.divide(
        lake["qc_valid_record_count"],
        lake["raw_record_count"],
        out=np.zeros(len(lake), dtype=float),
        where=lake["raw_record_count"].gt(0),
    )
    lake["numeric_aggregate_eligible_fraction"] = np.divide(
        lake["numeric_aggregate_eligible_count"],
        lake["raw_record_count"],
        out=np.zeros(len(lake), dtype=float),
        where=lake["raw_record_count"].gt(0),
    )
    lake["censored_fraction_among_numeric_valid"] = np.divide(
        lake["censored_valid_record_count"],
        lake["numeric_aggregate_eligible_count"],
        out=np.zeros(len(lake), dtype=float),
        where=lake["numeric_aggregate_eligible_count"].gt(0),
    )
    # Outcome-blind record-quality term. Censored records retain half credit
    # rather than being treated as exact measurements or deleted.
    lake["record_quality_fraction"] = (
        lake["numeric_aggregate_eligible_fraction"]
        * (1.0 - 0.5 * lake["censored_fraction_among_numeric_valid"])
    ).clip(0.0, 1.0)
    lake["aggregation_method"] = (
        "median_of_site_medians_within_lake_date_variable"
    )
    lake["sensitivity_aggregation"] = "maximum_across_valid_site_maxima"
    return lake
