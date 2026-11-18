from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

DEVELOPMENT_OUTCOME_CUTOFF = pd.Timestamp("2020-12-31")

PRIMARY_THRESHOLD = 75.0

PRIMARY_WASHOUT_DAYS = 60

STAGE_HORIZON_DAYS = 60

DIAGNOSTIC_HORIZON_DAYS = 90

FIELD_RESULT_LAG_DAYS = 7

ROLLING_TARGET_YEARS = (2016, 2017, 2018, 2019, 2020)

OUTER_FOLDS = 5

FOLD_SALT = "LDEW_OUTER_LAKE_FOLD_V1_20260730"

CANDIDATE_SALT = "LDEW_CANDIDATE_V1_20260730"

class EndpointError(RuntimeError):
    """Raised when endpoint or stage construction violates the frozen contract."""

def deterministic_outer_fold(lake_id: str) -> int:
    payload = f"{FOLD_SALT}|{lake_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % OUTER_FOLDS

def candidate_key(lake_id: str, origin_date: pd.Timestamp) -> str:
    payload = (
        f"{CANDIDATE_SALT}|{lake_id}|{origin_date.strftime('%Y-%m-%d')}"
    ).encode("utf-8")
    return "candidate__" + hashlib.sha256(payload).hexdigest()[:24]

def episode_key(
    *,
    aggregation: str,
    washout_days: int,
    lake_id: str,
    event_date: pd.Timestamp,
) -> str:
    payload = (
        f"LDEW_EPISODE_V1|{aggregation}|washout{washout_days}|"
        f"{lake_id}|{event_date.strftime('%Y-%m-%d')}"
    ).encode("utf-8")
    return "episode__" + hashlib.sha256(payload).hexdigest()[:24]

def load_chla_observations(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_parquet(
        path,
        filters=[
            ("variable", "==", "chla"),
            (
                "sample_date",
                "<=",
                DEVELOPMENT_OUTCOME_CUTOFF.to_pydatetime(),
            ),
        ],
        columns=[
            "lake_id",
            "sample_date",
            "variable",
            "value_median",
            "value_max_sensitivity",
            "site_count",
            "site_with_valid_value_count",
            "numeric_aggregate_eligible_count",
            "record_quality_fraction",
            "canonical_unit",
        ],
    )
    source["lake_id"] = source["lake_id"].astype(str)
    source["sample_date"] = pd.to_datetime(
        source["sample_date"], errors="coerce"
    ).dt.normalize()
    source["value_median"] = pd.to_numeric(
        source["value_median"], errors="coerce"
    )
    source["value_max_sensitivity"] = pd.to_numeric(
        source["value_max_sensitivity"], errors="coerce"
    )
    if source["sample_date"].isna().any():
        raise EndpointError("Versioned Chl ledger has invalid dates.")
    if source.duplicated(["lake_id", "sample_date"]).any():
        raise EndpointError(
            "Versioned lake-day Chl key is not unique."
        )
    if not source["canonical_unit"].dropna().eq("ug/L").all():
        raise EndpointError("Chl canonical unit is not ug/L.")
    if source["sample_date"].gt(DEVELOPMENT_OUTCOME_CUTOFF).any():
        raise EndpointError("2021+ outcome entered development.")
    valid = source.loc[source["value_median"].notna()].copy()
    valid["primary_severe"] = valid["value_median"].ge(PRIMARY_THRESHOLD)
    valid["max_sensitivity_severe"] = valid["value_max_sensitivity"].ge(
        PRIMARY_THRESHOLD
    )
    valid["assumed_result_available_date"] = valid[
        "sample_date"
    ] + pd.to_timedelta(FIELD_RESULT_LAG_DAYS, unit="D")
    valid = valid.sort_values(["lake_id", "sample_date"], kind="stable").reset_index(
        drop=True
    )
    contract = {
        "predicate_filtered_chla_rows_through_cutoff": int(len(source)),
        "admitted_date_min": source["sample_date"].min().isoformat(),
        "admitted_date_max": source["sample_date"].max().isoformat(),
        "development_rows_through_cutoff": int(len(source)),
        "development_valid_primary_rows": int(len(valid)),
        "development_date_max": valid["sample_date"].max().isoformat(),
        "development_outcome_cutoff": DEVELOPMENT_OUTCOME_CUTOFF.isoformat(),
    }
    return valid, contract

def build_episode_starts(
    observations: pd.DataFrame,
    *,
    value_column: str,
    aggregation: str,
    washout_days: int,
) -> pd.DataFrame:
    severe = observations.loc[
        pd.to_numeric(observations[value_column], errors="coerce").ge(
            PRIMARY_THRESHOLD
        ),
        [
            "lake_id",
            "sample_date",
            value_column,
            "site_count",
            "record_quality_fraction",
        ],
    ].copy()
    severe = severe.rename(
        columns={
            "sample_date": "severe_observation_date",
            value_column: "severe_value_ug_l",
        }
    )
    severe = severe.sort_values(
        ["lake_id", "severe_observation_date"], kind="stable"
    ).reset_index(drop=True)
    severe["previous_severe_gap_days"] = (
        severe.groupby("lake_id", sort=False)["severe_observation_date"]
        .diff()
        .dt.days
    )
    starts = severe.loc[
        severe["previous_severe_gap_days"].isna()
        | severe["previous_severe_gap_days"].gt(washout_days)
    ].copy()
    starts["aggregation"] = aggregation
    starts["threshold_ug_l"] = PRIMARY_THRESHOLD
    starts["washout_days"] = int(washout_days)
    starts["event_date"] = starts.pop("severe_observation_date")
    starts["episode_id"] = [
        episode_key(
            aggregation=aggregation,
            washout_days=washout_days,
            lake_id=str(lake),
            event_date=pd.Timestamp(date),
        )
        for lake, date in zip(starts["lake_id"], starts["event_date"])
    ]
    columns = [
        "episode_id",
        "lake_id",
        "event_date",
        "aggregation",
        "threshold_ug_l",
        "washout_days",
        "severe_value_ug_l",
        "site_count",
        "record_quality_fraction",
        "previous_severe_gap_days",
    ]
    return starts[columns].reset_index(drop=True)

def _date_to_day(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_datetime(values, errors="coerce").to_numpy(dtype="datetime64[D]")

def count_interval(
    sorted_dates: np.ndarray,
    origins: np.ndarray,
    lower_days: int,
    upper_days: int,
) -> np.ndarray:
    """Count dates in (origin+lower_days, origin+upper_days], vectorised."""

    left_dates = origins + np.timedelta64(lower_days, "D")
    right_dates = origins + np.timedelta64(upper_days, "D")
    left = np.searchsorted(sorted_dates, left_dates, side="right")
    right = np.searchsorted(sorted_dates, right_dates, side="right")
    return (right - left).astype(np.int32)



def build_candidate_ledger(
    risk: pd.DataFrame,
    observations: pd.DataFrame,
    primary_episodes: pd.DataFrame,
) -> pd.DataFrame:
    candidate = risk.copy().reset_index(drop=True)
    candidate["lake_id"] = candidate["lake_id"].astype(str)
    candidate["origin_date"] = pd.to_datetime(
        candidate["origin_date"], errors="coerce"
    ).dt.normalize()
    if candidate[["lake_id", "origin_date"]].isna().any().any():
        raise EndpointError("Risk candidate key contains nulls.")
    if candidate.duplicated(["lake_id", "origin_date"]).any():
        raise EndpointError("Risk candidate key is not unique.")
    candidate["candidate_row_key"] = [
        candidate_key(str(lake), pd.Timestamp(origin))
        for lake, origin in zip(candidate["lake_id"], candidate["origin_date"])
    ]
    if candidate["candidate_row_key"].duplicated().any():
        raise EndpointError("Candidate hashes are not unique.")
    fold_map = {
        lake: deterministic_outer_fold(lake)
        for lake in candidate["lake_id"].unique()
    }
    candidate["outer_lake_fold"] = candidate["lake_id"].map(fold_map).astype(
        np.int8
    )
    candidate["origin_year"] = candidate["origin_date"].dt.year.astype(np.int16)
    candidate["rolling_target_year"] = candidate["origin_year"].where(
        candidate["origin_year"].isin(ROLLING_TARGET_YEARS)
    )
    candidate["full_stage_window_within_development_cutoff"] = (
        candidate["origin_date"]
        + pd.to_timedelta(STAGE_HORIZON_DAYS, unit="D")
    ).le(DEVELOPMENT_OUTCOME_CUTOFF)

    observation_lookup = {
        lake: _date_to_day(group["sample_date"])
        for lake, group in observations.groupby("lake_id", sort=False)
    }
    severe = observations.loc[observations["primary_severe"]].copy()
    severe_sample_lookup = {
        lake: _date_to_day(group["sample_date"])
        for lake, group in severe.groupby("lake_id", sort=False)
    }
    severe_available_lookup = {
        lake: _date_to_day(group["assumed_result_available_date"])
        for lake, group in severe.groupby("lake_id", sort=False)
    }
    episode_lookup = {
        lake: _date_to_day(group["event_date"])
        for lake, group in primary_episodes.groupby("lake_id", sort=False)
    }

    n = len(candidate)
    last_known_sample = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    last_known_available = np.full(
        n, np.datetime64("NaT"), dtype="datetime64[D]"
    )
    days_since_known = np.full(n, np.nan, dtype=float)
    days_since_biological = np.full(n, np.nan, dtype=float)
    next_onset = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    next_onset_lead = np.full(n, np.nan, dtype=float)
    next_future_onset = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    next_future_onset_lead = np.full(n, np.nan, dtype=float)
    obs_1_14 = np.zeros(n, dtype=np.int32)
    obs_15_30 = np.zeros(n, dtype=np.int32)
    obs_31_60 = np.zeros(n, dtype=np.int32)
    obs_61_90 = np.zeros(n, dtype=np.int32)

    for lake, group in candidate.groupby("lake_id", sort=False):
        positions = group.index.to_numpy(dtype=np.int64)
        origins = _date_to_day(group["origin_date"])

        obs_dates = observation_lookup.get(lake, np.array([], dtype="datetime64[D]"))
        if len(obs_dates):
            obs_1_14[positions] = count_interval(obs_dates, origins, 0, 14)
            obs_15_30[positions] = count_interval(obs_dates, origins, 14, 30)
            obs_31_60[positions] = count_interval(obs_dates, origins, 30, 60)
            obs_61_90[positions] = count_interval(obs_dates, origins, 60, 90)

        severe_dates = severe_sample_lookup.get(
            lake, np.array([], dtype="datetime64[D]")
        )
        availability_dates = severe_available_lookup.get(
            lake, np.array([], dtype="datetime64[D]")
        )
        if len(severe_dates):
            if len(availability_dates) != len(severe_dates):
                raise EndpointError(
                    f"Severe availability mismatch for lake {lake}."
                )
            known_pos = np.searchsorted(
                availability_dates, origins, side="left"
            ) - 1
            has_known = known_pos >= 0
            if has_known.any():
                target = positions[has_known]
                samples = severe_dates[known_pos[has_known]]
                available = availability_dates[known_pos[has_known]]
                last_known_sample[target] = samples
                last_known_available[target] = available
                days_since_known[target] = (
                    origins[has_known] - samples
                ).astype("timedelta64[D]").astype(float)

            biological_pos = np.searchsorted(
                severe_dates, origins, side="left"
            ) - 1
            has_biological = biological_pos >= 0
            if has_biological.any():
                target = positions[has_biological]
                days_since_biological[target] = (
                    origins[has_biological]
                    - severe_dates[biological_pos[has_biological]]
                ).astype("timedelta64[D]").astype(float)

        episode_dates = episode_lookup.get(
            lake, np.array([], dtype="datetime64[D]")
        )
        if len(episode_dates):
            stage_pos = np.searchsorted(episode_dates, origins, side="left")
            has_stage = stage_pos < len(episode_dates)
            if has_stage.any():
                target = positions[has_stage]
                dates = episode_dates[stage_pos[has_stage]]
                next_onset[target] = dates
                next_onset_lead[target] = (
                    dates - origins[has_stage]
                ).astype("timedelta64[D]").astype(float)

            future_pos = np.searchsorted(episode_dates, origins, side="right")
            has_future = future_pos < len(episode_dates)
            if has_future.any():
                target = positions[has_future]
                dates = episode_dates[future_pos[has_future]]
                next_future_onset[target] = dates
                next_future_onset_lead[target] = (
                    dates - origins[has_future]
                ).astype("timedelta64[D]").astype(float)

    candidate["last_known_severe_sample_date"] = pd.to_datetime(last_known_sample)
    candidate["last_known_severe_available_date"] = pd.to_datetime(
        last_known_available
    )
    candidate["days_since_last_known_severe_sample"] = days_since_known
    candidate["days_since_last_biological_severe_sample"] = days_since_biological
    candidate["future_chla_observation_days_1_14"] = obs_1_14
    candidate["future_chla_observation_days_15_30"] = obs_15_30
    candidate["future_chla_observation_days_31_60"] = obs_31_60
    candidate["future_chla_observation_days_61_90"] = obs_61_90
    candidate["future_chla_observation_days_1_30"] = obs_1_14 + obs_15_30
    candidate["future_chla_observation_days_1_60"] = (
        obs_1_14 + obs_15_30 + obs_31_60
    )
    candidate["future_chla_observed_all_stage_intervals"] = (
        (obs_1_14 > 0) & (obs_15_30 > 0) & (obs_31_60 > 0)
    )
    candidate["next_incident_onset_date_including_origin"] = pd.to_datetime(
        next_onset
    )
    candidate["days_to_next_incident_onset_including_origin"] = next_onset_lead
    candidate["next_future_incident_onset_date"] = pd.to_datetime(
        next_future_onset
    )
    candidate["days_to_next_future_incident_onset"] = next_future_onset_lead

    known_recent = np.isfinite(days_since_known) & (
        days_since_known <= PRIMARY_WASHOUT_DAYS
    )
    biological_recent_past = np.isfinite(days_since_biological) & (
        days_since_biological <= PRIMARY_WASHOUT_DAYS
    )
    candidate["known_recent_severe_exclusion"] = known_recent
    candidate["retrospective_recent_biological_severe"] = biological_recent_past
    candidate["unavailable_active_episode_boundary"] = (
        biological_recent_past & ~known_recent
    )
    candidate["eligible_common_development_candidate"] = (
        candidate["full_stage_window_within_development_cutoff"]
        & ~candidate["known_recent_severe_exclusion"]
    )
    candidate["eligible_rolling_evaluation_candidate"] = (
        candidate["eligible_common_development_candidate"]
        & candidate["origin_year"].isin(ROLLING_TARGET_YEARS)
    )

    candidate["same_day_incident_onset"] = (
        pd.to_numeric(candidate["days_to_next_incident_onset_including_origin"], errors="coerce")
        .eq(0).fillna(False)
    )
    return candidate
