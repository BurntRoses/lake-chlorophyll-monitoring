# Lake chlorophyll-a monitoring priorities

Reproducibility materials for evaluating lake-monitoring risk rankings under fixed sampling capacity.

The study compares a multi-horizon strategy with a direct 60-day strategy across 2,844 candidate lakes and 130 biweekly prediction dates during 2021–2025. At 10% monitoring capacity, the strategies covered 263 and 207 of 696 observed high chlorophyll-a events in the primary 31–60-day window, a difference of 8.05 percentage points (95% CI, 4.36–12.22). Both strategies used 36,631 lake–prediction-date selections.

Version 1.0.1 updates archival and citation metadata; research code, numerical results and released study data are unchanged from v1.0.0.

## Start here

- [`METHODS.md`](METHODS.md): study design, endpoints, bootstrap and computational scope.
- [`01_results/RESULTS.md`](01_results/RESULTS.md): key endpoint and comparator results.
- [`07_metadata/analysis_plan.json`](07_metadata/analysis_plan.json): machine-readable analysis roles and endpoint definitions.
- [`07_metadata/data_dictionary.md`](07_metadata/data_dictionary.md): identifiers, units and result fields.
- [`07_metadata/data_sources.md`](07_metadata/data_sources.md): data provenance, provider citations and attribution.
- [`04_analysis_data/README.md`](04_analysis_data/README.md): where the release-only analysis data live and how to restore them.

## Repository and release layout

The default branch contains code, documentation, figures, numerical figure sources and manuscript tables. Large binary analysis inputs and most fitted model artifacts are kept out of Git history and distributed as a fixed release asset.

| Location | Contents | Availability |
|---|---|---|
| `01_results/` | Main findings and endpoint comparisons | default branch |
| `02_figures/` | Five main and eleven supplementary figures in PDF, SVG and PNG | default branch |
| `03_figure_source_data/` | Numerical source tables for every figure | default branch |
| `04_analysis_data/` | Analysis inputs, stored predictions, event outcomes, evaluation tables and sensitivity outputs | full contents in release v1.0.0; README on default branch |
| `05_models/` | Model documentation plus fitted model artifacts and coefficients | documentation/index on default branch; full artifacts in release v1.0.0 |
| `06_code/` | Evaluation, figure generation, model-support modules and software environment | default branch |
| `07_metadata/` | Analysis definitions, data sources, provenance, dictionaries and indexes | default branch |
| `08_manuscript_tables/` | Editable CSV sources for Tables 1–3 and S1–S2 | default branch |

The complete distribution is available from [GitHub release v1.0.0](https://github.com/BurntRoses/lake-chlorophyll-monitoring/releases/tag/v1.0.0):

- `Code_and_Figure_Data_v1.0.0.zip`: code, figures, source tables and study documentation.
- `Analysis_Data_and_Models_v1.0.0.zip`: analysis inputs, stored predictions, event observations, evaluation tables and fitted model artifacts.
- `SHA256SUMS.txt`: archive checksums.

The Zenodo v1.0.1 record archives the citable code, figure-source data and documentation; the complete large analysis distribution remains attached to GitHub release v1.0.0.

## Analysis freeze and outcome linkage

The released evaluation separates prediction generation from external outcome evaluation. The analysis archive contains outcome-blind prediction scores and frozen monitoring queues for both learned strategies, together with the external event-outcome tables, protocol, input-freeze metadata and evaluation contract. The evaluation script consumes these frozen inputs, verifies their SHA-256 identities against `07_metadata/provenance/evaluation_inputs.json`, and then links the queues to event outcomes to calculate coverage and bootstrap uncertainty.

The primary comparison, 10% capacity and the 31–60-day primary / 1–60-day secondary windows are represented in `07_metadata/analysis_plan.json`; historical risk, list dynamics, observation-support strata and the descriptive capacity curve are explicitly labeled post hoc. The same-predictor 60-day analysis is labeled as a component comparison.

This repository documents an outcome-blind/frozen evaluation workflow; it is not presented as an external preregistration registry.

## Reproduce the reported evaluation and figures

1. Clone or download the repository.
2. Download both v1.0.0 release archives and `SHA256SUMS.txt`.
3. Extract the archives into the same parent location. They expand into the `lake-chlorophyll-monitoring/` tree.
4. For an existing GitHub checkout, copy the extracted `04_analysis_data/`, additional `05_models/` contents and `MANIFEST_DATA.sha256` into the repository root.
5. Verify the supplied archive checksums.

Use Python 3.12 and run from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r 06_code/environment/requirements.txt
python 06_code/run_pipeline.py --stage verify
python 06_code/run_pipeline.py --stage all
```

On Windows, activate with `.venv\Scripts\activate`. A Conda environment is also provided in `06_code/environment/environment.yml`.

`verify` checks stored results and input integrity. `all` rebuilds the event-coverage evaluation, result indexes and all 16 figures, then verifies the outputs. Individual stages are `evaluate`, `index`, `figures` and `verify`. The bootstrap uses 5,000 replicates with seed 20260804.

The executable workflow starts from stored prediction scores. Complete model training and external scoring are outside the released executable workflow; fitted artifacts and model-support code are provided for inspection.

## Data provenance

Field observations originate from USGS Water Data for the Nation and the Water Quality Portal. Lake identifiers and attributes draw on LAGOS-US and AquaSat; meteorological predictors draw on ERA5; satellite predictors draw on EPA's Cyanobacteria Assessment Network (CyAN) Sentinel-3 OLCI products. Provider citations, access links and attribution requirements are documented in [`07_metadata/data_sources.md`](07_metadata/data_sources.md).

`MANIFEST_CODE.sha256` identifies files on the code distribution. `MANIFEST_DATA.sha256` is supplied with the restored analysis-data distribution. The file identities used by the external evaluation are recorded in `07_metadata/provenance/evaluation_inputs.json`.

## Authors and citation

Siran Luo and Jibiao Zhang  
Department of Environmental Science and Engineering, Fudan University, Shanghai, China

- Siran Luo: https://orcid.org/0009-0004-7469-2414
- Jibiao Zhang: https://orcid.org/0000-0003-2734-6477

Luo, S., Zhang, J., 2026. *Lake chlorophyll-a monitoring priorities: code and data for multi-horizon risk ranking* (Version 1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.22809124

Machine-readable citation metadata are provided in `CITATION.cff`.

## License

Original software and configuration files are licensed under the MIT License. Original derived data, result tables, model parameters, figures and study documentation are licensed under CC BY 4.0. Third-party data and geographic resources retain their original terms and attribution requirements; see [`LICENSES/README.md`](LICENSES/README.md) and [`07_metadata/data_sources.md`](07_metadata/data_sources.md).