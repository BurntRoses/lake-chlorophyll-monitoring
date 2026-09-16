# Data dictionary

## Identifiers and units

| Field | Definition |
|---|---|
| `policy` or `policy_id` | Strategy identifier in `policy_registry.json` |
| `candidate_row_key` | Fixed lake–prediction-date key used to break ranking ties |
| `lake_id` | Lake identifier; read as a string |
| `origin_date` | Prediction date; 130 dates at 14-day intervals |
| `episode_id` | Event identifier; figure-table `event_id` has a display prefix with the same identifier suffix |
| `endpoint` | `primary_31_60` or `secondary_1_60`, inclusive lead-time windows in days |
| `capacity_percent` | Selection capacity as a percentage, e.g. 10 |
| `capacity_fraction` | Selection capacity as a fraction, e.g. 0.10 |
| `captured_events` | Events covered by at least one eligible selection in the specified lead-time window |
| `event_n` or `all_events_denominator` | Event denominator, including events without valid prediction dates |
| `coverage_percent` | Event coverage expressed as a percentage |
| `capture_rate` or `event_coverage` | Usually a proportion between 0 and 1; use the table's units |
| `difference_pp`, `ci_low_pp`, `ci_high_pp` | Coverage difference and 95% confidence bounds, in percentage points; first strategy minus comparator |
| `selected_slots` | Sum of lake–prediction-date selections, including repeat selections |
| `analysis_role` | Prespecified, post hoc or component comparison identity |

Chlorophyll-a is measured in micrograms per litre. A high-value day has a lake–day median of at least 75 micrograms per litre. High-value days separated by no more than 60 days belong to the same event. The first high-value day is the observed event date.

## Bootstrap and observation support

`bootstrap_draw` ranges from 0 to 4999. `event_copies` is the denominator after resampling lakes; strategy-specific `*_captured` fields count covered event copies. Candidate records and events share each lake's multiplicity. At the capacity boundary, only the available number of lake copies is selected. Event coverage uses the maximum selected-copy count over valid prediction dates.

`stratum` is `overall`, `low`, `medium`, `high` or `zero`. `observation_day_n` counts retained 2021–2025 field-observation days. Low, medium and high strata contain 1–8, 9–16 and 17–90 days, respectively. `NA` for the zero-observation stratum means coverage cannot be estimated. Strata retain the global ranking lists and resampling population.

## Historical risk and list dynamics

`historical_risk` is the fraction of retained pre-evaluation days reaching the high-value threshold. `history_day_n` and `high_chla_day_n` count all retained historical days and high-value days. `latest_available_date` uses the study convention of seven days after sampling.

`selection_n` counts how often a lake was selected across 130 dates. `jaccard_overlap` is the intersection divided by the union of consecutive lists from the same strategy; `jaccard_turnover` is one minus that value. `unique_selected_lakes` counts lakes selected at least once. Repeat-selection medians and quartiles use ever-selected lakes as their denominator.

## File-level metadata

`table_catalog.csv` records CSV and Parquet locations, dimensions and fields. `result_column_dictionary.csv` records result-field data types. `figure_source_map.csv` uses the final manuscript numbering. Full model feature definitions are in `04_analysis_data/model_training_inputs/feature_contract/`; coefficients and fitted feature order are in `05_models/`.

Historical training-product references retain source basenames when the original path is outside this distribution. Numerical input checksums in `provenance/evaluation_inputs.json` identify the exact files used by the evaluation.
