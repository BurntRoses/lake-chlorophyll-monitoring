# Lake chlorophyll-a monitoring priorities

Code and data for evaluating lake-monitoring risk rankings under fixed sampling capacity. Version 1.0.1 contains the 2021–2025 external evaluation code, numerical source tables, documentation and reproducible figures. Complete analysis data and fitted model artifacts are distributed with the linked version 1.0.0 release.

The study compares a multi-horizon strategy with a direct 60-day strategy across 2,844 lakes and 130 biweekly prediction dates. At 10% monitoring capacity, the strategies cover 263 and 207 of 696 observed high chlorophyll-a events in the primary 31–60-day window. The difference is 8.05 percentage points (95% confidence interval, 4.36–12.22). Both strategies use 36,631 lake–prediction-date selections. [Results](01_results/RESULTS.md) also reports the secondary endpoint and historical-risk comparisons.

Version 1.0.1 updates archival and citation metadata; research code, data and results are unchanged from version 1.0.0.

## Reproduce the results

Download the two complementary archives and their checksums from [GitHub release v1.0.0](https://github.com/BurntRoses/lake-chlorophyll-monitoring/releases/tag/v1.0.0):

- `Code_and_Figure_Data_v1.0.0.zip`: code, figures, source tables and study documentation.
- `Analysis_Data_and_Models_v1.0.0.zip`: analysis inputs, stored predictions, event observations, evaluation tables and model artifacts.

Extract both archives into the same directory. They expand into `lake-chlorophyll-monitoring/`. For a GitHub checkout, copy `04_analysis_data/`, `05_models/` and `MANIFEST_DATA.sha256` from the extracted data archive into the repository root.

Use Python 3.12 and run these commands from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r 06_code/environment/requirements.txt
python 06_code/run_pipeline.py --stage verify
python 06_code/run_pipeline.py --stage all
```

On Windows, activate the environment with `.venv\Scripts\activate`. A Conda environment is also provided in `06_code/environment/environment.yml`.

`verify` checks stored results and input integrity. `all` rebuilds the evaluation, result indexes and all 16 figures, then verifies them. Individual stages are `evaluate`, `index`, `figures` and `verify`. The bootstrap uses 5,000 replicates and seed 20260804. Arial is used for the figure typography.

The executable workflow starts from stored predictions. Model-support code and fitted artifacts are provided for inspection; raw-source acquisition, complete model training and external scoring are outside this workflow. See [model documentation](05_models/README.md) for the correspondence between artifacts and strategies.

## Study materials

| Directory | Contents |
|---|---|
| `01_results/` | Main findings and endpoint comparisons |
| `02_figures/` | Five main and eleven supplementary figures in PDF, SVG and PNG |
| `03_figure_source_data/` | Numerical source tables for each figure |
| `04_analysis_data/` | Source extracts, predictors, predictions, observed events and evaluation results |
| `05_models/` | Model artifacts, coefficients, feature order and model index |
| `06_code/` | Evaluation, figure generation, model-support modules and software environment |
| `07_metadata/` | Analysis definitions, variables, data sources and file indexes |
| `08_manuscript_tables/` | Editable CSV sources for Tables 1–3 and S1–S2 |

[Methods](METHODS.md) describes the study design and calculation of event coverage. The [data dictionary](07_metadata/data_dictionary.md) defines identifiers, units and result fields. The [figure source map](07_metadata/figure_source_map.csv) connects each figure to its numerical data, and the [table catalog](07_metadata/table_catalog.csv) records file dimensions and variables.

## Data provenance

Field observations originate from USGS and the Water Quality Portal. Lake identifiers and attributes draw on LAGOS and AquaSat; meteorological predictors draw on ERA5; satellite predictors draw on EPA CyAN. [Data sources](07_metadata/data_sources.md) provides provider links and references. Third-party materials retain their source terms.

`MANIFEST_CODE.sha256` and `MANIFEST_DATA.sha256` identify the files in the two distributions. The input hashes in `07_metadata/provenance/evaluation_inputs.json` identify the data used by the evaluation.

## Authors and citation

Siran Luo and Jibiao Zhang. Department of Environmental Science and Engineering, Fudan University, Shanghai, China.

Siran Luo: https://orcid.org/0009-0004-7469-2414.

Jibiao Zhang: https://orcid.org/0000-0003-2734-6477.

Luo, S., and Zhang, J. (2026). *Lake chlorophyll-a monitoring priorities: code and data for multi-horizon risk ranking* (Version 1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.22809124

The DOI identifies the archived code, figure source data and documentation. Complete analysis inputs, stored predictions and model artifacts are available in [GitHub release v1.0.0](https://github.com/BurntRoses/lake-chlorophyll-monitoring/releases/tag/v1.0.0).

Citation metadata are provided in `CITATION.cff`.

## License

Original code is licensed under MIT. Original derived data, model parameters, figures and documentation are licensed under CC BY 4.0. Third-party materials retain their provider terms. See `LICENSES/README.md` for the scope of each license.
