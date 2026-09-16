from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class IncidentEvaluationContractError(RuntimeError):
    """Raised when incident-event evaluation inputs violate the frozen contract."""


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise IncidentEvaluationContractError(f"{name} is missing columns: {missing}")


def build_episode_starts(
    severe_events: pd.DataFrame,
    *,
    washout_days: int,
    lake_column: str = "lake_id",
    date_column: str = "event_date",
) -> pd.DataFrame:
    """Collapse severe observation dates into incident episode starts."""

    if washout_days < 1:
        raise IncidentEvaluationContractError("washout_days must be positive.")
    _require_columns(severe_events, [lake_column, date_column], name="severe_events")
    events = severe_events[[lake_column, date_column]].copy()
    if events[lake_column].isna().any():
        raise IncidentEvaluationContractError("Severe events contain missing lake identifiers.")
    events[lake_column] = events[lake_column].astype(str)
    events[date_column] = pd.to_datetime(events[date_column], errors="raise")
    events = events.drop_duplicates().sort_values(
        [lake_column, date_column], kind="mergesort"
    )
    events["previous_severe_gap_days"] = (
        events.groupby(lake_column, sort=False)[date_column].diff().dt.days
    )
    starts = events.loc[
        events["previous_severe_gap_days"].isna()
        | events["previous_severe_gap_days"].gt(washout_days)
    ].copy()
    serialized = (
        starts[lake_column]
        + "|"
        + starts[date_column].dt.strftime("%Y-%m-%d")
        + f"|washout{washout_days}"
    )
    starts["episode_id"] = serialized.map(
        lambda value: "episode__"
        + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    )
    starts["washout_days"] = int(washout_days)
    return starts.reset_index(drop=True)


def build_domain_episode_starts(
    severe_history: pd.DataFrame,
    domain_events: pd.DataFrame,
    *,
    washout_days: int,
    lake_column: str = "lake_id",
    date_column: str = "event_date",
) -> pd.DataFrame:
    """Build episode starts from full history, then retain a domain's events."""

    _require_columns(domain_events, [lake_column, date_column], name="domain_events")
    starts = build_episode_starts(
        severe_history,
        washout_days=washout_days,
        lake_column=lake_column,
        date_column=date_column,
    )
    domain = domain_events[[lake_column, date_column]].drop_duplicates().copy()
    domain[lake_column] = domain[lake_column].astype(str)
    domain[date_column] = pd.to_datetime(domain[date_column], errors="raise")
    return starts.merge(
        domain,
        on=[lake_column, date_column],
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)


def filter_incident_candidates(
    candidates: pd.DataFrame,
    severe_events: pd.DataFrame,
    *,
    washout_days: int,
    current_value_column: str = "period_chla_max",
    severe_threshold: float = 75.0,
    lake_column: str = "lake_id",
    origin_column: str = "origin_date",
    episode_date_column: str = "event_date",
) -> pd.DataFrame:
    """Remove origins at or within the same severe-chlorophyll episode.

    ``time_since_last_high`` is intentionally not accepted here: that feature is
    defined against the historical >20 ug/L threshold, whereas incident episodes
    use the >=75 ug/L primary endpoint. Prior severe history is reconstructed from
    endpoint event dates so the washout and endpoint share one threshold.
    """

    _require_columns(
        candidates,
        [lake_column, origin_column, current_value_column],
        name="candidates",
    )
    _require_columns(
        severe_events,
        [lake_column, episode_date_column],
        name="severe_events",
    )
    if washout_days < 1:
        raise IncidentEvaluationContractError("washout_days must be positive.")

    work = candidates.copy().reset_index(drop=True)
    work[lake_column] = work[lake_column].astype(str)
    work[origin_column] = pd.to_datetime(work[origin_column], errors="raise")
    event = severe_events[[lake_column, episode_date_column]].drop_duplicates().copy()
    event[lake_column] = event[lake_column].astype(str)
    event[episode_date_column] = pd.to_datetime(
        event[episode_date_column], errors="raise"
    )
    event_lookup = {
        lake_id: np.sort(group[episode_date_column].to_numpy(dtype="datetime64[ns]"))
        for lake_id, group in event.groupby(lake_column, sort=False)
    }

    days_since_prior_severe = np.full(len(work), np.nan, dtype=float)
    for lake_id, group in work.groupby(lake_column, sort=False):
        severe_dates = event_lookup.get(lake_id)
        if severe_dates is None or len(severe_dates) == 0:
            continue
        origins = group[origin_column].to_numpy(dtype="datetime64[ns]")
        prior_positions = np.searchsorted(severe_dates, origins, side="left") - 1
        has_prior = prior_positions >= 0
        if has_prior.any():
            group_positions = group.index.to_numpy(dtype=np.int64)
            days_since_prior_severe[group_positions[has_prior]] = (
                origins[has_prior] - severe_dates[prior_positions[has_prior]]
            ) / np.timedelta64(1, "D")

    work["days_since_prior_severe_event"] = days_since_prior_severe
    current = pd.to_numeric(work[current_value_column], errors="coerce")
    current_high = current.ge(severe_threshold).fillna(False)
    prior_gap = work["days_since_prior_severe_event"]
    eligible = ~current_high & (prior_gap.isna() | prior_gap.gt(washout_days))
    return work.loc[eligible].copy().reset_index(drop=True)


def link_candidates_to_episodes(
    candidates: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    horizon_days: int = 30,
    candidate_key: str = "candidate_row_key",
    lake_column: str = "lake_id",
    origin_column: str = "origin_date",
    episode_date_column: str = "event_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link eligible pre-origin rows to future episode starts within the horizon."""

    if horizon_days < 1:
        raise IncidentEvaluationContractError("horizon_days must be positive.")
    _require_columns(
        candidates,
        [candidate_key, lake_column, origin_column],
        name="candidates",
    )
    _require_columns(
        episodes,
        ["episode_id", lake_column, episode_date_column],
        name="episodes",
    )
    if candidates[candidate_key].duplicated().any():
        raise IncidentEvaluationContractError("Candidate keys must be unique.")
    candidate = candidates[[candidate_key, lake_column, origin_column]].copy()
    candidate[lake_column] = candidate[lake_column].astype(str)
    candidate[origin_column] = pd.to_datetime(candidate[origin_column], errors="raise")
    event = episodes[["episode_id", lake_column, episode_date_column]].copy()
    event[lake_column] = event[lake_column].astype(str)
    event[episode_date_column] = pd.to_datetime(
        event[episode_date_column], errors="raise"
    )
    links = candidate.merge(event, on=lake_column, how="inner", validate="many_to_many")
    links["lead_days"] = (
        links[episode_date_column] - links[origin_column]
    ).dt.days
    links = links.loc[links["lead_days"].between(1, horizon_days)].copy()
    links = links.drop_duplicates([candidate_key, "episode_id"])
    universe = (
        event.loc[event["episode_id"].isin(links["episode_id"])]
        .drop_duplicates("episode_id")
        .reset_index(drop=True)
    )
    return links.reset_index(drop=True), universe


def build_fixed_capacity_queue(
    candidates: pd.DataFrame,
    *,
    score_column: str,
    policy_id: str,
    capacity_fraction: float = 0.10,
    candidate_key: str = "candidate_row_key",
    origin_column: str = "origin_date",
) -> pd.DataFrame:
    """Rank every origin separately and select exactly ceil(capacity * rows)."""

    _require_columns(
        candidates,
        [candidate_key, origin_column, score_column],
        name="candidates",
    )
    if not 0.0 < capacity_fraction <= 1.0:
        raise IncidentEvaluationContractError(
            "capacity_fraction must lie in (0, 1]."
        )
    if candidates[candidate_key].duplicated().any():
        raise IncidentEvaluationContractError("Candidate keys must be unique.")
    work = candidates[[candidate_key, origin_column, score_column]].copy()
    work[origin_column] = pd.to_datetime(work[origin_column], errors="raise")
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    if not np.isfinite(work[score_column].to_numpy(dtype=float)).all():
        raise IncidentEvaluationContractError(f"Invalid scores in {score_column}.")
    work = work.sort_values(
        [origin_column, score_column, candidate_key],
        ascending=[True, False, True],
        kind="mergesort",
    )
    work["policy_rank"] = work.groupby(origin_column, sort=False).cumcount() + 1
    size = work.groupby(origin_column, sort=False)[candidate_key].transform("size")
    work["origin_candidate_rows"] = size.astype(int)
    work["origin_budget_rows"] = np.ceil(size * capacity_fraction).astype(int)
    work["selected"] = work["policy_rank"].le(work["origin_budget_rows"])
    work["policy_id"] = str(policy_id)
    work["policy_score"] = work[score_column].astype(float)
    return work.reset_index(drop=True)


def build_protected_lifecycle_queue(
    candidates: pd.DataFrame,
    *,
    flat_score_column: str,
    lifecycle_score_column: str,
    policy_id: str,
    flat_protection_fraction: float = 0.50,
    capacity_fraction: float = 0.10,
    candidate_key: str = "candidate_row_key",
    lake_column: str = "lake_id",
    origin_column: str = "origin_date",
) -> pd.DataFrame:
    """Protect flat-model slots, then fill the remaining budget by lifecycle score.

    The route is applied independently at every origin date.  It therefore preserves
    the same review capacity as every comparator while allowing lifecycle-stage
    information to change only the unprotected half of the queue.
    """

    _require_columns(
        candidates,
        [
            candidate_key,
            lake_column,
            origin_column,
            flat_score_column,
            lifecycle_score_column,
        ],
        name="candidates",
    )
    if not 0.0 <= flat_protection_fraction <= 1.0:
        raise IncidentEvaluationContractError(
            "flat_protection_fraction must lie in [0, 1]."
        )
    if not 0.0 < capacity_fraction <= 1.0:
        raise IncidentEvaluationContractError(
            "capacity_fraction must lie in (0, 1]."
        )
    if candidates[candidate_key].duplicated().any():
        raise IncidentEvaluationContractError("Candidate keys must be unique.")

    columns = [
        candidate_key,
        lake_column,
        origin_column,
        flat_score_column,
        lifecycle_score_column,
    ]
    work = candidates[columns].copy()
    work[lake_column] = work[lake_column].astype(str)
    work[origin_column] = pd.to_datetime(work[origin_column], errors="raise")
    for score_column in (flat_score_column, lifecycle_score_column):
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        if not np.isfinite(work[score_column].to_numpy(dtype=float)).all():
            raise IncidentEvaluationContractError(
                f"Invalid scores in {score_column}."
            )

    queues = []
    for _, origin_rows in work.groupby(origin_column, sort=True):
        origin_count = int(len(origin_rows))
        budget = max(1, int(math.ceil(origin_count * capacity_fraction)))
        protected_slots = min(
            budget,
            int(math.ceil(budget * flat_protection_fraction)),
        )
        flat_order = origin_rows.sort_values(
            [flat_score_column, candidate_key],
            ascending=[False, True],
            kind="mergesort",
        )
        protected = flat_order.head(protected_slots).copy()
        protected_keys = set(protected[candidate_key])
        supplement = origin_rows.loc[
            ~origin_rows[candidate_key].isin(protected_keys)
        ].sort_values(
            [lifecycle_score_column, candidate_key],
            ascending=[False, True],
            kind="mergesort",
        )
        protected["route_stage"] = "flat_protected"
        supplement["route_stage"] = "lifecycle_supplement"
        ordered = pd.concat([protected, supplement], ignore_index=True)
        ordered["policy_rank"] = np.arange(1, origin_count + 1, dtype=np.int64)
        ordered["origin_candidate_rows"] = origin_count
        ordered["origin_budget_rows"] = budget
        ordered["selected"] = ordered["policy_rank"].le(budget)
        ordered["policy_id"] = str(policy_id)
        ordered["policy_score"] = ordered[lifecycle_score_column].astype(float)
        ordered["route_flat_protection_fraction"] = float(
            flat_protection_fraction
        )
        ordered["route_flat_protected_slots"] = protected_slots
        ordered["route_lifecycle_supplement_slots"] = budget - protected_slots
        queues.append(ordered)

    if not queues:
        raise IncidentEvaluationContractError("Lifecycle route queue is empty.")
    queue = pd.concat(queues, ignore_index=True)
    expected = np.ceil(
        queue.groupby(origin_column, sort=False)[candidate_key].transform("size")
        * capacity_fraction
    ).astype(int)
    selected = queue.groupby(origin_column, sort=False)["selected"].transform("sum")
    if not selected.eq(expected).all():
        raise IncidentEvaluationContractError(
            "Lifecycle route failed to preserve the fixed capacity."
        )
    return queue.reset_index(drop=True)


def capture_episodes(
    queue: pd.DataFrame,
    links: pd.DataFrame,
    *,
    candidate_key: str = "candidate_row_key",
) -> pd.DataFrame:
    """Retain the earliest selected warning for every captured episode."""

    _require_columns(queue, [candidate_key, "policy_id", "selected"], name="queue")
    _require_columns(
        links,
        [candidate_key, "episode_id", "lake_id", "event_date", "lead_days"],
        name="links",
    )
    selected = queue.loc[queue["selected"], [candidate_key, "policy_id"]]
    captured = links.merge(selected, on=candidate_key, how="inner", validate="many_to_one")
    if captured.empty:
        return captured
    captured = captured.sort_values(
        ["policy_id", "episode_id", "lead_days", candidate_key],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    return captured.drop_duplicates(["policy_id", "episode_id"]).reset_index(drop=True)


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def _tail_probabilities(values: np.ndarray) -> tuple[float, float]:
    draws = np.asarray(values, dtype=float)
    one_sided = (float(np.count_nonzero(draws <= 0.0)) + 1.0) / (len(draws) + 1.0)
    return one_sided, min(1.0, 2.0 * one_sided)


def reranked_lake_cluster_bootstrap(
    candidates: pd.DataFrame,
    links: pd.DataFrame,
    episode_universe: pd.DataFrame,
    *,
    policy_scores: Mapping[str, str],
    candidate_policy: str,
    baseline_policies: Sequence[str],
    reps: int,
    seed: int,
    capacity_fraction: float = 0.10,
    early_threshold_days: int = 28,
    candidate_key: str = "candidate_row_key",
    lake_column: str = "lake_id",
    origin_column: str = "origin_date",
) -> pd.DataFrame:
    """Lake bootstrap that rebuilds every origin-specific review queue per draw."""

    if reps < 1000:
        raise IncidentEvaluationContractError(
            "At least 1000 bootstrap replicates are required."
        )
    if candidate_policy not in policy_scores:
        raise IncidentEvaluationContractError("candidate_policy has no score column.")
    missing_baselines = [p for p in baseline_policies if p not in policy_scores]
    if missing_baselines:
        raise IncidentEvaluationContractError(
            f"Baseline policies have no score columns: {missing_baselines}"
        )
    _require_columns(
        candidates,
        [candidate_key, lake_column, origin_column, *policy_scores.values()],
        name="candidates",
    )
    _require_columns(
        links,
        [candidate_key, "episode_id", lake_column, "lead_days"],
        name="links",
    )
    _require_columns(
        episode_universe,
        ["episode_id", lake_column],
        name="episode_universe",
    )
    if not 0.0 < capacity_fraction <= 1.0:
        raise IncidentEvaluationContractError(
            "capacity_fraction must lie in (0, 1]."
        )
    work = candidates.reset_index(drop=True).copy()
    if work[candidate_key].duplicated().any():
        raise IncidentEvaluationContractError("Candidate keys must be unique.")
    work[lake_column] = work[lake_column].astype(str)
    work[origin_column] = pd.to_datetime(work[origin_column], errors="raise")
    for score_column in policy_scores.values():
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        if not np.isfinite(work[score_column].to_numpy(dtype=float)).all():
            raise IncidentEvaluationContractError(
                f"Invalid scores in {score_column}."
            )

    lakes = pd.Index(sorted(work[lake_column].unique()))
    if len(lakes) < 2:
        raise IncidentEvaluationContractError("At least two candidate lakes are required.")
    lake_index = pd.Series(np.arange(len(lakes), dtype=np.int32), index=lakes)
    row_lake = work[lake_column].map(lake_index).to_numpy(dtype=np.int32)
    key_to_row = pd.Series(work.index.to_numpy(dtype=np.int32), index=work[candidate_key])

    event = episode_universe[["episode_id", lake_column]].drop_duplicates("episode_id").copy()
    event[lake_column] = event[lake_column].astype(str)
    if not event[lake_column].isin(lakes).all():
        raise IncidentEvaluationContractError(
            "Episode universe contains lakes outside the candidate domain."
        )
    event = event.reset_index(drop=True)
    event_index = pd.Series(event.index.to_numpy(dtype=np.int32), index=event["episode_id"])
    event_lake = event[lake_column].map(lake_index).to_numpy(dtype=np.int32)

    link = links[[candidate_key, "episode_id", "lead_days"]].drop_duplicates(
        [candidate_key, "episode_id"]
    )
    if not link[candidate_key].isin(key_to_row.index).all():
        raise IncidentEvaluationContractError("Episode links contain unknown candidates.")
    if not link["episode_id"].isin(event_index.index).all():
        raise IncidentEvaluationContractError("Episode links contain unknown episodes.")
    link_row = link[candidate_key].map(key_to_row).to_numpy(dtype=np.int32)
    link_event = link["episode_id"].map(event_index).to_numpy(dtype=np.int32)
    link_early = pd.to_numeric(link["lead_days"], errors="raise").to_numpy(float) >= float(
        early_threshold_days
    )

    origin_groups = [
        group.index.to_numpy(dtype=np.int32)
        for _, group in work.groupby(origin_column, sort=True)
    ]
    row_origin = np.empty(len(work), dtype=np.int32)
    for origin_index, rows in enumerate(origin_groups):
        row_origin[rows] = origin_index
    links_by_origin: list[np.ndarray] = []
    for origin_index in range(len(origin_groups)):
        links_by_origin.append(np.flatnonzero(row_origin[link_row] == origin_index))

    sorted_rows: dict[str, list[np.ndarray]] = {}
    for policy_id, score_column in policy_scores.items():
        policy_orders = []
        for rows in origin_groups:
            part = work.iloc[rows]
            order = part.sort_values(
                [score_column, candidate_key],
                ascending=[False, True],
                kind="mergesort",
            ).index.to_numpy(dtype=np.int32)
            policy_orders.append(order)
        sorted_rows[policy_id] = policy_orders

    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        len(lakes),
        np.full(len(lakes), 1.0 / len(lakes)),
        size=reps,
    )
    total_events = multiplicity[:, event_lake].sum(axis=1).astype(float)
    if (total_events <= 0).any():
        raise IncidentEvaluationContractError(
            "A bootstrap draw contains no incident episodes."
        )

    capture_draws: dict[str, np.ndarray] = {}
    early_draws: dict[str, np.ndarray] = {}
    selected_totals: dict[str, np.ndarray] = {}
    point_capture: dict[str, int] = {}
    point_early: dict[str, int] = {}
    for policy_id, policy_orders in sorted_rows.items():
        point_queue = build_fixed_capacity_queue(
            work,
            score_column=policy_scores[policy_id],
            policy_id=policy_id,
            capacity_fraction=capacity_fraction,
            candidate_key=candidate_key,
            origin_column=origin_column,
        )
        point_captures = capture_episodes(
            point_queue,
            links,
            candidate_key=candidate_key,
        )
        point_capture[policy_id] = int(len(point_captures))
        point_early[policy_id] = int(
            point_captures["lead_days"].ge(early_threshold_days).sum()
        )
        event_selected = np.zeros((reps, len(event)), dtype=np.uint16)
        early_selected = np.zeros((reps, len(event)), dtype=np.uint16)
        policy_selected_total = np.zeros(reps, dtype=np.int64)
        for origin_index, order in enumerate(policy_orders):
            copies = multiplicity[:, row_lake[order]]
            origin_total = copies.sum(axis=1)
            budget = np.ceil(origin_total * capacity_fraction).astype(np.int64)
            cumulative = np.cumsum(copies, axis=1)
            selected = np.clip(
                budget[:, None] - (cumulative - copies),
                0,
                copies,
            ).astype(np.uint16)
            policy_selected_total += selected.sum(axis=1, dtype=np.int64)
            inverse = np.empty(len(work), dtype=np.int32)
            inverse.fill(-1)
            inverse[order] = np.arange(len(order), dtype=np.int32)
            for link_index in links_by_origin[origin_index]:
                position = inverse[link_row[link_index]]
                values = selected[:, position]
                event_id = link_event[link_index]
                event_selected[:, event_id] = np.maximum(
                    event_selected[:, event_id], values
                )
                if link_early[link_index]:
                    early_selected[:, event_id] = np.maximum(
                        early_selected[:, event_id], values
                    )
        capture_draws[policy_id] = event_selected.sum(axis=1, dtype=np.int64)
        early_draws[policy_id] = early_selected.sum(axis=1, dtype=np.int64)
        selected_totals[policy_id] = policy_selected_total

    candidate_selected = selected_totals[candidate_policy]
    rows: list[dict[str, object]] = []
    for baseline_policy in baseline_policies:
        capacity_equal = np.array_equal(
            candidate_selected, selected_totals[baseline_policy]
        )
        if not capacity_equal:
            raise IncidentEvaluationContractError(
                f"Bootstrap capacity drifted for {baseline_policy}."
            )
        metrics = {
            "captured_episode_count_difference": (
                float(point_capture[candidate_policy] - point_capture[baseline_policy]),
                capture_draws[candidate_policy] - capture_draws[baseline_policy],
            ),
            "episode_capture_rate_difference": (
                float(
                    point_capture[candidate_policy] / len(event)
                    - point_capture[baseline_policy] / len(event)
                ),
                capture_draws[candidate_policy] / total_events
                - capture_draws[baseline_policy] / total_events,
            ),
            f"captured_at_least_{early_threshold_days}d_count_difference": (
                float(point_early[candidate_policy] - point_early[baseline_policy]),
                early_draws[candidate_policy] - early_draws[baseline_policy],
            ),
            f"capture_rate_at_least_{early_threshold_days}d_difference": (
                float(
                    point_early[candidate_policy] / len(event)
                    - point_early[baseline_policy] / len(event)
                ),
                early_draws[candidate_policy] / total_events
                - early_draws[baseline_policy] / total_events,
            ),
        }
        for metric, (estimate, draws) in metrics.items():
            low, high = _percentile_interval(draws)
            one_sided, two_sided = _tail_probabilities(draws)
            rows.append(
                {
                    "candidate_policy": candidate_policy,
                    "baseline_policy": baseline_policy,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_one_sided_p": one_sided,
                    "bootstrap_two_sided_p": two_sided,
                    "bootstrap_reps": int(reps),
                    "bootstrap_seed": int(seed),
                    "cluster_key": lake_column,
                    "queue_rebuilt_each_replicate": True,
                    "equal_capacity_all_replicates": capacity_equal,
                }
            )
    return pd.DataFrame(rows)


def _weighted_rank_cache(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache the stable score order and tie groups for frequency-weighted ranks."""

    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise IncidentEvaluationContractError(
            "Weighted percentile ranks require a non-empty one-dimensional score array."
        )
    if not np.isfinite(scores).all():
        raise IncidentEvaluationContractError(
            "Weighted percentile rank scores must be finite."
        )
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    starts = np.r_[
        0,
        np.flatnonzero(ordered[1:] != ordered[:-1]) + 1,
    ].astype(np.int32)
    lengths = np.diff(np.r_[starts, len(order)]).astype(np.int32)
    return order.astype(np.int32), starts, lengths


def _frequency_weights(values: np.ndarray, *, expected_columns: int) -> np.ndarray:
    weights = np.asarray(values)
    if weights.ndim == 0 or weights.shape[-1] != expected_columns:
        raise IncidentEvaluationContractError(
            "Frequency weights and percentile-rank scores have different lengths."
        )
    numeric = np.asarray(weights, dtype=float)
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0).any()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise IncidentEvaluationContractError(
            "Percentile-rank frequency weights must be finite nonnegative integers."
        )
    return numeric.astype(np.int64)


def weighted_average_percentile_rank(
    scores: Sequence[float] | np.ndarray,
    frequency_weights: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return pandas-compatible average percentile ranks under frequency weights.

    A weight of ``m`` is interpreted as ``m`` replicated copies of that row.  With
    unit weights this exactly matches ``Series.rank(method="average", pct=True)``.
    Rows with zero weight receive a rank for bookkeeping, but do not affect any
    positive-weight row or consume queue capacity.
    """

    cache = _weighted_rank_cache(np.asarray(scores, dtype=float))
    order, starts, lengths = cache
    weights = _frequency_weights(
        np.asarray(frequency_weights), expected_columns=len(order)
    )
    if weights.ndim != 1:
        raise IncidentEvaluationContractError(
            "weighted_average_percentile_rank requires one-dimensional weights."
        )
    ordered_weights = weights[order]
    total = int(ordered_weights.sum())
    if total <= 0:
        return np.zeros(len(order), dtype=float)
    group_weights = np.add.reduceat(ordered_weights, starts)
    below = np.cumsum(group_weights, dtype=np.int64) - group_weights
    group_rank = (below + (group_weights + 1.0) / 2.0) / float(total)
    result = np.empty(len(order), dtype=float)
    result[order] = np.repeat(group_rank, lengths)
    return result


def _weighted_average_percentile_rank_matrix(
    cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    frequency_weights: np.ndarray,
) -> np.ndarray:
    """Vectorized frequency-weighted average percentile ranks for many draws."""

    order, starts, lengths = cache
    weights = _frequency_weights(
        frequency_weights, expected_columns=len(order)
    )
    if weights.ndim != 2:
        raise IncidentEvaluationContractError(
            "Bootstrap percentile ranks require a two-dimensional weight matrix."
        )
    ordered_weights = weights[:, order]
    totals = ordered_weights.sum(axis=1, dtype=np.int64)
    group_weights = np.add.reduceat(ordered_weights, starts, axis=1)
    below = np.cumsum(group_weights, axis=1, dtype=np.int64) - group_weights
    safe_totals = np.where(totals > 0, totals, 1)
    group_rank = (
        below + (group_weights + 1.0) / 2.0
    ) / safe_totals[:, None]
    group_rank[totals <= 0] = 0.0
    ordered_rank = group_rank[:, np.repeat(np.arange(len(starts)), lengths)]
    result = np.empty_like(ordered_rank, dtype=float)
    result[:, order] = ordered_rank
    return result


def weighted_rank_fusion_lake_cluster_bootstrap(
    candidates: pd.DataFrame,
    links: pd.DataFrame,
    episode_universe: pd.DataFrame,
    *,
    rank_score_weights: Mapping[str, float],
    candidate_policy: str,
    baseline_scores: Mapping[str, str],
    reps: int,
    seed: int,
    capacity_fraction: float = 0.10,
    early_threshold_days: int = 28,
    chunk_size: int = 256,
    candidate_key: str = "candidate_row_key",
    lake_column: str = "lake_id",
    origin_column: str = "origin_date",
) -> pd.DataFrame:
    """Strict lake bootstrap for a weighted within-origin percentile-rank fusion.

    Lake resampling changes each origin's empirical candidate distribution.  This
    routine therefore recomputes frequency-weighted percentile ranks for every
    fusion component, re-fuses them, and rebuilds the fixed-capacity queue inside
    every draw.  Baselines are raw-score policies whose queues are also rebuilt.
    """

    if reps < 1000:
        raise IncidentEvaluationContractError(
            "At least 1000 bootstrap replicates are required."
        )
    if chunk_size < 1:
        raise IncidentEvaluationContractError("chunk_size must be positive.")
    if not 0.0 < capacity_fraction <= 1.0:
        raise IncidentEvaluationContractError(
            "capacity_fraction must lie in (0, 1]."
        )
    if not rank_score_weights:
        raise IncidentEvaluationContractError(
            "At least one rank-fusion component is required."
        )
    if not baseline_scores:
        raise IncidentEvaluationContractError(
            "At least one raw-score baseline policy is required."
        )
    if candidate_policy in baseline_scores:
        raise IncidentEvaluationContractError(
            "The rank-fusion candidate policy cannot also be a baseline policy."
        )
    fusion_weights = {
        str(column): float(weight) for column, weight in rank_score_weights.items()
    }
    if (
        not all(np.isfinite(weight) and weight >= 0.0 for weight in fusion_weights.values())
        or not np.isclose(sum(fusion_weights.values()), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise IncidentEvaluationContractError(
            "Rank-fusion weights must be finite, nonnegative, and sum to one."
        )

    score_columns = [*fusion_weights, *baseline_scores.values()]
    _require_columns(
        candidates,
        [candidate_key, lake_column, origin_column, *score_columns],
        name="candidates",
    )
    _require_columns(
        links,
        [candidate_key, "episode_id", lake_column, "event_date", "lead_days"],
        name="links",
    )
    _require_columns(
        episode_universe,
        ["episode_id", lake_column],
        name="episode_universe",
    )

    work = candidates.reset_index(drop=True).copy()
    if work[candidate_key].isna().any() or work[candidate_key].duplicated().any():
        raise IncidentEvaluationContractError(
            "Rank-fusion bootstrap candidate keys must be nonmissing and unique."
        )
    work[candidate_key] = work[candidate_key].astype(str)
    work[lake_column] = work[lake_column].astype(str)
    work[origin_column] = pd.to_datetime(work[origin_column], errors="raise")
    for score_column in dict.fromkeys(score_columns):
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        if not np.isfinite(work[score_column].to_numpy(dtype=float)).all():
            raise IncidentEvaluationContractError(
                f"Invalid bootstrap score in {score_column}."
            )

    lakes = pd.Index(sorted(work[lake_column].unique()))
    if len(lakes) < 2:
        raise IncidentEvaluationContractError(
            "At least two candidate lakes are required."
        )
    lake_index = pd.Series(np.arange(len(lakes), dtype=np.int32), index=lakes)
    row_lake = work[lake_column].map(lake_index).to_numpy(dtype=np.int32)
    key_to_row = pd.Series(
        work.index.to_numpy(dtype=np.int32), index=work[candidate_key]
    )

    event = (
        episode_universe[["episode_id", lake_column]]
        .drop_duplicates("episode_id")
        .copy()
    )
    if event.empty:
        raise IncidentEvaluationContractError(
            "The bootstrap episode universe is empty."
        )
    event["episode_id"] = event["episode_id"].astype(str)
    event[lake_column] = event[lake_column].astype(str)
    if not event[lake_column].isin(lakes).all():
        raise IncidentEvaluationContractError(
            "Episode universe contains lakes outside the candidate domain."
        )
    event = event.reset_index(drop=True)
    event_index = pd.Series(
        event.index.to_numpy(dtype=np.int32), index=event["episode_id"]
    )
    event_lake = event[lake_column].map(lake_index).to_numpy(dtype=np.int32)

    capture_links = links.copy()
    capture_links[candidate_key] = capture_links[candidate_key].astype(str)
    capture_links["episode_id"] = capture_links["episode_id"].astype(str)
    link = capture_links[[candidate_key, "episode_id", "lead_days"]].copy()
    link = link.drop_duplicates([candidate_key, "episode_id"])
    if not link[candidate_key].isin(key_to_row.index).all():
        raise IncidentEvaluationContractError(
            "Episode links contain unknown candidates."
        )
    if not link["episode_id"].isin(event_index.index).all():
        raise IncidentEvaluationContractError(
            "Episode links contain unknown episodes."
        )
    link_row = link[candidate_key].map(key_to_row).to_numpy(dtype=np.int32)
    link_event = link["episode_id"].map(event_index).to_numpy(dtype=np.int32)
    link_early = (
        pd.to_numeric(link["lead_days"], errors="raise").to_numpy(float)
        >= float(early_threshold_days)
    )

    origin_groups = [
        group.index.to_numpy(dtype=np.int32)
        for _, group in work.groupby(origin_column, sort=True)
    ]
    row_origin = np.empty(len(work), dtype=np.int32)
    row_local = np.empty(len(work), dtype=np.int32)
    for origin_id, rows in enumerate(origin_groups):
        row_origin[rows] = origin_id
        row_local[rows] = np.arange(len(rows), dtype=np.int32)

    origin_data: list[dict[str, object]] = []
    point_score = np.zeros(len(work), dtype=float)
    for origin_id, rows in enumerate(origin_groups):
        local_links = np.flatnonzero(row_origin[link_row] == origin_id)
        keys = work.iloc[rows][candidate_key].to_numpy(dtype=str)
        key_order = np.argsort(keys, kind="mergesort").astype(np.int32)
        rank_caches = {
            column: _weighted_rank_cache(
                work.iloc[rows][column].to_numpy(dtype=float)
            )
            for column in fusion_weights
        }
        unit_weights = np.ones(len(rows), dtype=np.int64)
        fused = np.zeros(len(rows), dtype=float)
        for column, weight in fusion_weights.items():
            fused += weight * weighted_average_percentile_rank(
                work.iloc[rows][column].to_numpy(dtype=float),
                unit_weights,
            )
        point_score[rows] = fused
        baseline_orders = {}
        for policy, score_column in baseline_scores.items():
            key_space_order = np.argsort(
                -work.iloc[rows[key_order]][score_column].to_numpy(dtype=float),
                kind="stable",
            )
            baseline_orders[policy] = key_order[key_space_order]
        origin_data.append(
            {
                "lake": row_lake[rows],
                "key_order": key_order,
                "rank_caches": rank_caches,
                "baseline_orders": baseline_orders,
                "link_local_row": row_local[link_row[local_links]],
                "link_event": link_event[local_links],
                "link_early": link_early[local_links],
            }
        )

    point_column = "__weighted_rank_fusion_point_score__"
    if point_column in work:
        raise IncidentEvaluationContractError(
            f"Reserved bootstrap column already exists: {point_column}"
        )
    work[point_column] = point_score
    point_capture: dict[str, int] = {}
    point_early: dict[str, int] = {}
    candidate_queue = build_fixed_capacity_queue(
        work,
        score_column=point_column,
        policy_id=candidate_policy,
        capacity_fraction=capacity_fraction,
        candidate_key=candidate_key,
        origin_column=origin_column,
    )
    candidate_captures = capture_episodes(
        candidate_queue, capture_links, candidate_key=candidate_key
    )
    point_capture[candidate_policy] = int(len(candidate_captures))
    point_early[candidate_policy] = int(
        candidate_captures["lead_days"].ge(early_threshold_days).sum()
    )
    for policy, score_column in baseline_scores.items():
        queue = build_fixed_capacity_queue(
            work,
            score_column=score_column,
            policy_id=policy,
            capacity_fraction=capacity_fraction,
            candidate_key=candidate_key,
            origin_column=origin_column,
        )
        captured = capture_episodes(
            queue, capture_links, candidate_key=candidate_key
        )
        point_capture[policy] = int(len(captured))
        point_early[policy] = int(
            captured["lead_days"].ge(early_threshold_days).sum()
        )

    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        len(lakes),
        np.full(len(lakes), 1.0 / len(lakes)),
        size=reps,
    )
    total_events = multiplicity[:, event_lake].sum(axis=1).astype(float)
    if (total_events <= 0).any():
        raise IncidentEvaluationContractError(
            "A bootstrap draw contains no incident episodes."
        )

    policy_ids = [candidate_policy, *baseline_scores]
    capture_draws = {
        policy: np.zeros(reps, dtype=np.int64) for policy in policy_ids
    }
    early_draws = {
        policy: np.zeros(reps, dtype=np.int64) for policy in policy_ids
    }
    selected_totals = {
        policy: np.zeros(reps, dtype=np.int64) for policy in policy_ids
    }
    for chunk_start in range(0, reps, chunk_size):
        chunk_stop = min(reps, chunk_start + chunk_size)
        draw_slice = slice(chunk_start, chunk_stop)
        draw_multiplicity = multiplicity[draw_slice]
        chunk_reps = chunk_stop - chunk_start
        event_selected = {
            policy: np.zeros((chunk_reps, len(event)), dtype=np.int32)
            for policy in policy_ids
        }
        early_selected = {
            policy: np.zeros((chunk_reps, len(event)), dtype=np.int32)
            for policy in policy_ids
        }
        for data in origin_data:
            lake_positions = np.asarray(data["lake"], dtype=np.int32)
            copies = draw_multiplicity[:, lake_positions]
            origin_total = copies.sum(axis=1, dtype=np.int64)
            budget = np.ceil(origin_total * capacity_fraction).astype(np.int64)

            fusion = np.zeros_like(copies, dtype=float)
            rank_caches = data["rank_caches"]
            if not isinstance(rank_caches, dict):
                raise AssertionError("Internal rank cache type drifted.")
            for column, weight in fusion_weights.items():
                fusion += weight * _weighted_average_percentile_rank_matrix(
                    rank_caches[column], copies
                )
            key_order = np.asarray(data["key_order"], dtype=np.int32)
            order_in_key_space = np.argsort(
                -fusion[:, key_order], axis=1, kind="stable"
            )
            order = key_order[order_in_key_space]
            ordered_copies = np.take_along_axis(copies, order, axis=1)
            cumulative = np.cumsum(ordered_copies, axis=1, dtype=np.int64)
            selected_ordered = np.clip(
                budget[:, None] - (cumulative - ordered_copies),
                0,
                ordered_copies,
            ).astype(np.int32)
            candidate_selected = np.zeros_like(selected_ordered, dtype=np.int32)
            np.put_along_axis(
                candidate_selected, order, selected_ordered, axis=1
            )
            selected_totals[candidate_policy][draw_slice] += candidate_selected.sum(
                axis=1, dtype=np.int64
            )

            selected_by_policy = {candidate_policy: candidate_selected}
            baseline_orders = data["baseline_orders"]
            if not isinstance(baseline_orders, dict):
                raise AssertionError("Internal baseline order type drifted.")
            for policy in baseline_scores:
                baseline_order = np.asarray(
                    baseline_orders[policy], dtype=np.int32
                )
                baseline_copies = copies[:, baseline_order]
                baseline_cumulative = np.cumsum(
                    baseline_copies, axis=1, dtype=np.int64
                )
                baseline_selected_ordered = np.clip(
                    budget[:, None] - (baseline_cumulative - baseline_copies),
                    0,
                    baseline_copies,
                ).astype(np.int32)
                baseline_selected = np.zeros_like(
                    baseline_selected_ordered, dtype=np.int32
                )
                np.put_along_axis(
                    baseline_selected,
                    baseline_order[None, :],
                    baseline_selected_ordered,
                    axis=1,
                )
                selected_by_policy[policy] = baseline_selected
                selected_totals[policy][draw_slice] += baseline_selected.sum(
                    axis=1, dtype=np.int64
                )

            link_local_row = np.asarray(data["link_local_row"], dtype=np.int32)
            link_events = np.asarray(data["link_event"], dtype=np.int32)
            link_early_values = np.asarray(data["link_early"], dtype=bool)
            for policy, selected in selected_by_policy.items():
                values = selected[:, link_local_row]
                for local_link in range(values.shape[1]):
                    event_id = int(link_events[local_link])
                    event_selected[policy][:, event_id] = np.maximum(
                        event_selected[policy][:, event_id],
                        values[:, local_link],
                    )
                    if bool(link_early_values[local_link]):
                        early_selected[policy][:, event_id] = np.maximum(
                            early_selected[policy][:, event_id],
                            values[:, local_link],
                        )

        for policy in policy_ids:
            capture_draws[policy][draw_slice] = event_selected[policy].sum(
                axis=1, dtype=np.int64
            )
            early_draws[policy][draw_slice] = early_selected[policy].sum(
                axis=1, dtype=np.int64
            )

    component_json = json.dumps(
        fusion_weights, sort_keys=True, separators=(",", ":")
    )
    rows: list[dict[str, object]] = []
    for baseline_policy in baseline_scores:
        capacity_equal = np.array_equal(
            selected_totals[candidate_policy], selected_totals[baseline_policy]
        )
        if not capacity_equal:
            raise IncidentEvaluationContractError(
                f"Bootstrap capacity drifted for {baseline_policy}."
            )
        metrics = {
            "captured_episode_count_difference": (
                float(point_capture[candidate_policy] - point_capture[baseline_policy]),
                capture_draws[candidate_policy] - capture_draws[baseline_policy],
            ),
            "episode_capture_rate_difference": (
                float(
                    point_capture[candidate_policy] / len(event)
                    - point_capture[baseline_policy] / len(event)
                ),
                capture_draws[candidate_policy] / total_events
                - capture_draws[baseline_policy] / total_events,
            ),
            f"captured_at_least_{early_threshold_days}d_count_difference": (
                float(point_early[candidate_policy] - point_early[baseline_policy]),
                early_draws[candidate_policy] - early_draws[baseline_policy],
            ),
            f"capture_rate_at_least_{early_threshold_days}d_difference": (
                float(
                    point_early[candidate_policy] / len(event)
                    - point_early[baseline_policy] / len(event)
                ),
                early_draws[candidate_policy] / total_events
                - early_draws[baseline_policy] / total_events,
            ),
        }
        for metric, (estimate, draws) in metrics.items():
            low, high = _percentile_interval(draws)
            one_sided, two_sided = _tail_probabilities(draws)
            rows.append(
                {
                    "candidate_policy": candidate_policy,
                    "baseline_policy": baseline_policy,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_one_sided_p": one_sided,
                    "bootstrap_two_sided_p": two_sided,
                    "bootstrap_reps": int(reps),
                    "bootstrap_seed": int(seed),
                    "cluster_key": lake_column,
                    "bootstrap_method": (
                        "multinomial_lake_cluster_within_draw_weighted_"
                        "percentile_rank_fusion"
                    ),
                    "rank_fusion_components": component_json,
                    "within_draw_percentiles_recomputed": True,
                    "rank_fusion_recomputed_each_replicate": True,
                    "queue_rebuilt_each_replicate": True,
                    "equal_capacity_all_replicates": capacity_equal,
                }
            )
    return pd.DataFrame(rows)


def protected_route_lake_cluster_bootstrap(
    candidates: pd.DataFrame,
    links: pd.DataFrame,
    episode_universe: pd.DataFrame,
    *,
    flat_score_column: str,
    lifecycle_score_column: str,
    route_policy: str,
    baseline_scores: Mapping[str, str],
    reps: int,
    seed: int,
    flat_protection_fraction: float = 0.50,
    capacity_fraction: float = 0.10,
    early_threshold_days: int = 28,
    candidate_key: str = "candidate_row_key",
    lake_column: str = "lake_id",
    origin_column: str = "origin_date",
) -> pd.DataFrame:
    """Lake bootstrap for a protected-flat lifecycle routing policy.

    Lakes are resampled with replacement.  Every draw then reconstructs each
    origin-specific review budget, the protected flat allocation, and the
    lifecycle supplement allocation before event capture is measured.
    """

    if reps < 1000:
        raise IncidentEvaluationContractError(
            "At least 1000 bootstrap replicates are required."
        )
    if not baseline_scores:
        raise IncidentEvaluationContractError(
            "At least one baseline policy is required."
        )
    if route_policy in baseline_scores:
        raise IncidentEvaluationContractError(
            "The route policy cannot also be a baseline policy."
        )
    if not 0.0 <= flat_protection_fraction <= 1.0:
        raise IncidentEvaluationContractError(
            "flat_protection_fraction must lie in [0, 1]."
        )
    if not 0.0 < capacity_fraction <= 1.0:
        raise IncidentEvaluationContractError(
            "capacity_fraction must lie in (0, 1]."
        )

    score_columns = [
        flat_score_column,
        lifecycle_score_column,
        *baseline_scores.values(),
    ]
    _require_columns(
        candidates,
        [candidate_key, lake_column, origin_column, *score_columns],
        name="candidates",
    )
    _require_columns(
        links,
        [candidate_key, "episode_id", lake_column, "lead_days"],
        name="links",
    )
    _require_columns(
        episode_universe,
        ["episode_id", lake_column],
        name="episode_universe",
    )

    work = candidates.reset_index(drop=True).copy()
    if work[candidate_key].duplicated().any():
        raise IncidentEvaluationContractError("Candidate keys must be unique.")
    work[lake_column] = work[lake_column].astype(str)
    work[origin_column] = pd.to_datetime(work[origin_column], errors="raise")
    for score_column in dict.fromkeys(score_columns):
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        if not np.isfinite(work[score_column].to_numpy(dtype=float)).all():
            raise IncidentEvaluationContractError(
                f"Invalid scores in {score_column}."
            )

    lakes = pd.Index(sorted(work[lake_column].unique()))
    if len(lakes) < 2:
        raise IncidentEvaluationContractError(
            "At least two candidate lakes are required."
        )
    lake_index = pd.Series(np.arange(len(lakes), dtype=np.int32), index=lakes)
    row_lake = work[lake_column].map(lake_index).to_numpy(dtype=np.int32)
    key_to_row = pd.Series(
        work.index.to_numpy(dtype=np.int32), index=work[candidate_key]
    )

    event = (
        episode_universe[["episode_id", lake_column]]
        .drop_duplicates("episode_id")
        .copy()
    )
    event[lake_column] = event[lake_column].astype(str)
    if not event[lake_column].isin(lakes).all():
        raise IncidentEvaluationContractError(
            "Episode universe contains lakes outside the candidate domain."
        )
    event = event.reset_index(drop=True)
    event_index = pd.Series(
        event.index.to_numpy(dtype=np.int32), index=event["episode_id"]
    )
    event_lake = event[lake_column].map(lake_index).to_numpy(dtype=np.int32)

    link = links[[candidate_key, "episode_id", "lead_days"]].drop_duplicates(
        [candidate_key, "episode_id"]
    )
    if not link[candidate_key].isin(key_to_row.index).all():
        raise IncidentEvaluationContractError(
            "Episode links contain unknown candidates."
        )
    if not link["episode_id"].isin(event_index.index).all():
        raise IncidentEvaluationContractError(
            "Episode links contain unknown episodes."
        )
    link_row = link[candidate_key].map(key_to_row).to_numpy(dtype=np.int32)
    link_event = link["episode_id"].map(event_index).to_numpy(dtype=np.int32)
    link_early = (
        pd.to_numeric(link["lead_days"], errors="raise").to_numpy(float)
        >= float(early_threshold_days)
    )

    origin_groups = [
        group.index.to_numpy(dtype=np.int32)
        for _, group in work.groupby(origin_column, sort=True)
    ]
    row_origin = np.empty(len(work), dtype=np.int32)
    for origin_index, rows in enumerate(origin_groups):
        row_origin[rows] = origin_index
    links_by_origin = [
        np.flatnonzero(row_origin[link_row] == origin_index)
        for origin_index in range(len(origin_groups))
    ]

    def ordered_rows(rows: np.ndarray, score_column: str) -> np.ndarray:
        return (
            work.iloc[rows]
            .sort_values(
                [score_column, candidate_key],
                ascending=[False, True],
                kind="mergesort",
            )
            .index.to_numpy(dtype=np.int32)
        )

    flat_orders = [ordered_rows(rows, flat_score_column) for rows in origin_groups]
    lifecycle_orders = [
        ordered_rows(rows, lifecycle_score_column) for rows in origin_groups
    ]
    baseline_orders = {
        policy: [ordered_rows(rows, score) for rows in origin_groups]
        for policy, score in baseline_scores.items()
    }

    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        len(lakes),
        np.full(len(lakes), 1.0 / len(lakes)),
        size=reps,
    )
    total_events = multiplicity[:, event_lake].sum(axis=1).astype(float)
    if (total_events <= 0).any():
        raise IncidentEvaluationContractError(
            "A bootstrap draw contains no incident episodes."
        )

    policy_ids = [route_policy, *baseline_scores]
    event_selected = {
        policy: np.zeros((reps, len(event)), dtype=np.uint16)
        for policy in policy_ids
    }
    early_selected = {
        policy: np.zeros((reps, len(event)), dtype=np.uint16)
        for policy in policy_ids
    }
    selected_totals = {
        policy: np.zeros(reps, dtype=np.int64) for policy in policy_ids
    }

    for origin_index, rows in enumerate(origin_groups):
        flat_order = flat_orders[origin_index]
        lifecycle_order = lifecycle_orders[origin_index]
        copies_flat = multiplicity[:, row_lake[flat_order]]
        origin_total = copies_flat.sum(axis=1)
        budget = np.ceil(origin_total * capacity_fraction).astype(np.int64)
        protected_budget = np.ceil(
            budget * flat_protection_fraction
        ).astype(np.int64)
        flat_cumulative = np.cumsum(copies_flat, axis=1)
        selected_flat = np.clip(
            protected_budget[:, None] - (flat_cumulative - copies_flat),
            0,
            copies_flat,
        ).astype(np.uint16)

        flat_inverse = np.empty(len(work), dtype=np.int32)
        flat_inverse.fill(-1)
        flat_inverse[flat_order] = np.arange(len(flat_order), dtype=np.int32)
        lifecycle_inverse = np.empty(len(work), dtype=np.int32)
        lifecycle_inverse.fill(-1)
        lifecycle_inverse[lifecycle_order] = np.arange(
            len(lifecycle_order), dtype=np.int32
        )
        copies_lifecycle = multiplicity[:, row_lake[lifecycle_order]]
        selected_flat_lifecycle_order = selected_flat[
            :, flat_inverse[lifecycle_order]
        ]
        remaining = copies_lifecycle - selected_flat_lifecycle_order
        supplement_budget = budget - selected_flat.sum(axis=1, dtype=np.int64)
        remaining_cumulative = np.cumsum(remaining, axis=1)
        selected_supplement = np.clip(
            supplement_budget[:, None] - (remaining_cumulative - remaining),
            0,
            remaining,
        ).astype(np.uint16)
        selected_totals[route_policy] += selected_flat.sum(
            axis=1, dtype=np.int64
        ) + selected_supplement.sum(axis=1, dtype=np.int64)

        baseline_selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for policy, orders in baseline_orders.items():
            order = orders[origin_index]
            copies = multiplicity[:, row_lake[order]]
            cumulative = np.cumsum(copies, axis=1)
            selected = np.clip(
                budget[:, None] - (cumulative - copies),
                0,
                copies,
            ).astype(np.uint16)
            inverse = np.empty(len(work), dtype=np.int32)
            inverse.fill(-1)
            inverse[order] = np.arange(len(order), dtype=np.int32)
            baseline_selected[policy] = (selected, inverse)
            selected_totals[policy] += selected.sum(axis=1, dtype=np.int64)

        for link_index in links_by_origin[origin_index]:
            row = link_row[link_index]
            route_values = (
                selected_flat[:, flat_inverse[row]]
                + selected_supplement[:, lifecycle_inverse[row]]
            )
            event_id = link_event[link_index]
            event_selected[route_policy][:, event_id] = np.maximum(
                event_selected[route_policy][:, event_id], route_values
            )
            if link_early[link_index]:
                early_selected[route_policy][:, event_id] = np.maximum(
                    early_selected[route_policy][:, event_id], route_values
                )
            for policy, (selected, inverse) in baseline_selected.items():
                values = selected[:, inverse[row]]
                event_selected[policy][:, event_id] = np.maximum(
                    event_selected[policy][:, event_id], values
                )
                if link_early[link_index]:
                    early_selected[policy][:, event_id] = np.maximum(
                        early_selected[policy][:, event_id], values
                    )

    route_queue = build_protected_lifecycle_queue(
        work,
        flat_score_column=flat_score_column,
        lifecycle_score_column=lifecycle_score_column,
        policy_id=route_policy,
        flat_protection_fraction=flat_protection_fraction,
        capacity_fraction=capacity_fraction,
        candidate_key=candidate_key,
        lake_column=lake_column,
        origin_column=origin_column,
    )
    route_captures = capture_episodes(
        route_queue, links, candidate_key=candidate_key
    )
    point_capture = {route_policy: int(len(route_captures))}
    point_early = {
        route_policy: int(
            route_captures["lead_days"].ge(early_threshold_days).sum()
        )
    }
    for policy, score_column in baseline_scores.items():
        queue = build_fixed_capacity_queue(
            work,
            score_column=score_column,
            policy_id=policy,
            capacity_fraction=capacity_fraction,
            candidate_key=candidate_key,
            origin_column=origin_column,
        )
        captures = capture_episodes(queue, links, candidate_key=candidate_key)
        point_capture[policy] = int(len(captures))
        point_early[policy] = int(
            captures["lead_days"].ge(early_threshold_days).sum()
        )

    capture_draws = {
        policy: values.sum(axis=1, dtype=np.int64)
        for policy, values in event_selected.items()
    }
    early_draws = {
        policy: values.sum(axis=1, dtype=np.int64)
        for policy, values in early_selected.items()
    }
    rows: list[dict[str, object]] = []
    for baseline_policy in baseline_scores:
        capacity_equal = np.array_equal(
            selected_totals[route_policy], selected_totals[baseline_policy]
        )
        if not capacity_equal:
            raise IncidentEvaluationContractError(
                f"Bootstrap capacity drifted for {baseline_policy}."
            )
        metrics = {
            "captured_episode_count_difference": (
                float(point_capture[route_policy] - point_capture[baseline_policy]),
                capture_draws[route_policy] - capture_draws[baseline_policy],
            ),
            "episode_capture_rate_difference": (
                float(
                    point_capture[route_policy] / len(event)
                    - point_capture[baseline_policy] / len(event)
                ),
                capture_draws[route_policy] / total_events
                - capture_draws[baseline_policy] / total_events,
            ),
            f"captured_at_least_{early_threshold_days}d_count_difference": (
                float(point_early[route_policy] - point_early[baseline_policy]),
                early_draws[route_policy] - early_draws[baseline_policy],
            ),
            f"capture_rate_at_least_{early_threshold_days}d_difference": (
                float(
                    point_early[route_policy] / len(event)
                    - point_early[baseline_policy] / len(event)
                ),
                early_draws[route_policy] / total_events
                - early_draws[baseline_policy] / total_events,
            ),
        }
        for metric, (estimate, draws) in metrics.items():
            low, high = _percentile_interval(draws)
            one_sided, two_sided = _tail_probabilities(draws)
            rows.append(
                {
                    "candidate_policy": route_policy,
                    "baseline_policy": baseline_policy,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_one_sided_p": one_sided,
                    "bootstrap_two_sided_p": two_sided,
                    "bootstrap_reps": int(reps),
                    "bootstrap_seed": int(seed),
                    "cluster_key": lake_column,
                    "queue_rebuilt_each_replicate": True,
                    "route_rebuilt_each_replicate": True,
                    "equal_capacity_all_replicates": capacity_equal,
                    "flat_protection_fraction": float(
                        flat_protection_fraction
                    ),
                }
            )
    return pd.DataFrame(rows)
