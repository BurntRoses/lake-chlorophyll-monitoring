# Ranking lakes for high-chlorophyll events under limited monitoring capacity

Code and data for capacity-limited lake monitoring with single-window and
multi-window event-coverage rankers. The US evaluation includes 2,844 lakes and
130 fortnightly rounds during 2021–2025. The primary lead window is 31–60 days,
with high-chlorophyll events defined at Chl-a ≥75 μg/L. Regional analyses also
evaluate lake-specific historical thresholds.

## Installation

Use Python 3.12. Analysis dependencies are pinned in `requirements.txt`.

```bash
python -m pip install -e .
```

## Data and reproduction

Download `lake-biomass-coverage.zip` from the
[latest release](https://github.com/BurntRoses/lake-chlorophyll-monitoring/releases/latest)
and extract it. The archive includes processed inputs, fitted models, predictions,
event records, results and figures. The Git repository contains the analysis and
plotting code, configuration, figure source tables and documentation.

From the extracted archive:

```bash
python scripts/verify_files.py
python scripts/reproduce.py
```

The default run replays 256 US prediction rows per year and recalculates US and
regional event coverage. To replay all 365,726 US rows:

```bash
python scripts/reproduce.py --full --output reproduced
```

Outputs are written to `reproduced/`. Prediction replay and event evaluation run
on CPU without network access. See [reproducibility.md](docs/reproducibility.md)
for source rebuilding, model fitting and figure rendering.

## Repository structure

| Directory | Contents |
|---|---|
| `src/biomass` | Observation and feature builders, model fitting, ranking and evaluation |
| `config` | Input paths, feature definitions, parameters, seeds and event rules |
| `data/derived` | Observation histories, features, candidates and events |
| `data/processed` | Model inputs and cohort membership |
| `models` | Fitted experts and coverage models |
| `results` | Scores, selections, event coverage, bootstrap draws and summaries |
| `figures` | Figures 1–4, S1–S6 and the graphical abstract, with plotting sources |
| `tables` | Tables 1–2 and S1–S15 |
| `docs` | Data sources, file catalog, schemas and variable definitions |
| `scripts` | Reproduction and file verification |

Large analysis files are distributed in the release archive. Original bulk
downloads are obtained from the providers listed in [data_sources.csv](docs/data_sources.csv)
and placed under `data/raw/` when rebuilding source inputs.

## Variables

`lake_id` identifies a lake within its source system; `candidate_row_key`
identifies a lake–decision-date record; `episode_id` identifies an independent
event. Unknown outcome labels use −1 and remain eligible for ranking. Missing
features use NaN or Parquet null. Chl-a is in μg/L, numerically identical to mg/m³.
Coverage is the percentage of independent events with at least one selected
opportunity in the lead window. Coverage differences use percentage points.
See [variable_conventions.md](docs/variable_conventions.md).

## Citation and licenses

The project is archived at [Zenodo](https://doi.org/10.5281/zenodo.22809123).
Citation metadata are in `CITATION.cff`; cite the version used in your analysis.

Code is licensed under MIT. Author-generated data and results are licensed under
CC BY 4.0. Third-party materials retain their source terms; see
`THIRD_PARTY_NOTICES.md` and `docs/data_sources.csv`.
