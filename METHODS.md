# Study methods

## Study population and prediction dates

The external evaluation includes 2,844 candidate lakes and 130 biweekly prediction dates during 2021–2025. Models were developed using data through 2020. The candidate set contains 369,720 lake–prediction-date records, of which 365,726 are eligible for selection. On each date, every strategy uses the same eligible set and selects the highest-ranked 10%, rounded upward. This gives 270–285 selected lakes per date and 36,631 selections per strategy over the evaluation period.

## Observed events

Field chlorophyll-a observations were standardized to micrograms per liter and aggregated to lake–day medians. A high-value day has chlorophyll-a of at least 75 micrograms per liter. High-value days separated by at most 60 days form one event; a gap longer than 60 days starts another event. The first observed high-value day is the event date. The external set contains 696 observed events across 355 lakes in 13 states.

## Monitoring strategies

The multi-horizon strategy estimates cumulative 14-, 30- and 60-day risks using regularized logistic regression, positive-slope Platt calibration and monotonic Euclidean projection. Monitoring lists are ranked by the projected 60-day risk. The direct 60-day strategy uses an independently specified feature set and regularized logistic model and ranks by its raw 60-day probability. Ties are resolved by ascending `candidate_row_key`.

The primary comparison is between these complete strategies. The historical-risk rule and the same-predictor 60-day base model have distinct identities in `07_metadata/policy_registry.json`. Historical risk is the proportion of retained pre-2021 observation days with chlorophyll-a at or above 75 micrograms per liter; availability is required before the first prediction date under the study's seven-day availability convention.

## Frozen external evaluation and outcome linkage

The external evaluation uses stored prediction scores and frozen monitoring queues generated without external event outcomes. In the release archive, these include `external_outcome_blind_scores.parquet` and `frozen_external_queues.parquet` for the multi-horizon strategy, with corresponding outcome-blind scores and frozen queues for the direct 60-day strategy. The primary capacity and endpoint definitions are also recorded in the released protocol, input-freeze and evaluation-contract files.

During evaluation, `06_code/analysis/evaluate_strategies.py` hashes the consumed inputs and verifies them against `07_metadata/provenance/evaluation_inputs.json` before calculating event coverage. The script then links frozen queues to external event outcomes and checks that reconstructed 10% selections match the frozen queues exactly. This separation is the basis for describing the primary 10% comparison and the two lead-time windows as prespecified within the study workflow. Historical risk, list dynamics, observation-support strata and the full descriptive capacity curve are labeled post hoc; the same-predictor 60-day analysis is a component comparison.

## Endpoints and uncertainty

An event is covered if its lake is selected on at least one eligible prediction date 31–60 days before the observed event date. This is the primary endpoint. The secondary endpoint uses 1–60 days and follows the primary comparison in a fixed sequence. Both endpoints use all 696 events as the denominator, including events without a valid prediction date.

Confidence intervals use 5,000 lake-level bootstrap replicates with seed 20260804. Sampling is from all 2,844 candidate lakes. Candidate records and events share each sampled lake's multiplicity. Lists are rebuilt for every prediction date in every replicate. At the capacity boundary, only the available number of lake copies is selected. An event's covered-copy count is the maximum selected-copy count across eligible dates. Coverage differences are the first strategy minus the comparator, in percentage points; the interval uses the 2.5th and 97.5th bootstrap percentiles.

## Additional comparisons

Historical risk, list dynamics, observation-support strata and the descriptive capacity curve are post hoc analyses. The same-predictor 60-day comparison is a component comparison. These roles are recorded in `07_metadata/analysis_plan.json`.

Observation-support strata use retained 2021–2025 field-observation days: low, 1–8 days; medium, 9–16 days; high, 17–90 days. Lakes with no observations remain in the candidate lists and resampling population. Their event coverage is not estimable. List dynamics include the number of lakes ever selected, repeat-selection frequency and the Jaccard overlap between consecutive lists.

## Computational scope

`06_code/run_pipeline.py` reproduces event coverage, bootstrap comparisons, result tables and figures from the included predictions and observations. `05_models/README.md` describes the fitted artifacts. Model fitting and external scoring are not executed by this entry point.

Machine-readable analysis definitions are in `07_metadata/analysis_plan.json`. The primary protocol and frozen evaluation files are distributed in the `04_analysis_data/primary_external_evaluation/` tree of the analysis-data release. Numerical data and their units are indexed in `07_metadata/table_catalog.csv` and `07_metadata/data_dictionary.md`. See `04_analysis_data/README.md` for restoring the complete release-only analysis tree.