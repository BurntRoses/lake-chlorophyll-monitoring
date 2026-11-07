# Variables, identifiers and missing values

`file_catalog.csv` lists every data/result/table file, its row count, schema and
producing code. `table_schemas.csv` records actual stored columns and types.
`config/features.csv` defines the analysis feature list and source family;
the code defines each calculation. Figure source dictionaries define plotted
fields, units, denominators and uncertainty. 

## Observation and predictor fields

Field observations retain source identifiers, date, units, censoring and quality
information. `water_quality_raw.parquet` contains standardized source observations. `chla`, Chl-a values and historical concentration
thresholds use μg/L, numerically identical to mg/m³. Raw water-quality unit fields
must be read per row for other analytes.

`field_` features describe available field Chl-a history; `cyan_` describes satellite
algal observations; `wq_` denotes other water-quality variables; `met_` denotes
meteorology. `_lagN_` summaries use a look-back window of N days. Mean, median,
minimum, maximum and percentiles retain the source unit; counts use observations
or days as indicated by the suffix. Slopes use source units per day. A missing
latest observation is represented by NaN; days-since fields can use −1 when no
prior observation exists. Missing flags are indicators, not measured zeros.

Meteorological source units are defined in
`src/biomass/meteorology_features.py`: temperature in °C; precipitation in mm/day;
wind components and speed in m/s; shortwave radiation in J/m²/day; pressure in Pa.
Sums accumulate the source quantity over the stated window. `valid_days` counts
available daily values. Log-transformed/interacted features are defined in
`src/biomass/expression.py` and `src/biomass/seasonal.py`.

## Outcomes, candidates and events

`candidate_row_key` identifies a lake considered on an `origin_date`. Candidate
eligibility and label observability are separate: unknown outcomes remain in the
ranking queue. Binary outcome fields `y_LO_HI` use inclusive day bounds; −1 means
unknown. Event IDs, onset dates, grouping gaps and history cutoffs follow
`src/biomass/endpoints.py` and the regional builders.

Score tables contain finite model scores. Coverage-correction scores are ranking
values, not calibrated probabilities. A method column in a selected-candidate or
event-coverage table is a boolean indicator. An event is covered if its lake is
selected at least once within an eligible lead-time opportunity. Each independent
event is counted once, even if multiple rounds select its lake.

## Summary statistics

Coverage is a percent; method differences are percentage points. Capacity is the
eligible-lake proportion selected per round, rounded up to an integer quota.
Bootstrap intervals use whole-lake resampling with family-specific levels as
documented in Supplementary Methods S5. `bootstrap_draws.parquet` stores replicate
coverage fractions; formatted interval/coverage CSVs usually express percentages.
Inspect the specific table fields and figure dictionary before conversion.

## Formats

CSV uses UTF-8, headers, comma delimiters and decimal points; empty cells represent
missing entries. Parquet retains typed nulls and timestamps; floating-point NaN
also denotes missing observations or unavailable predictors. Identifiers must be
read as strings. Source ZIP/XLSX files retain provider-specific structures.
