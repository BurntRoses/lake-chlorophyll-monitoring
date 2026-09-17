# Analysis data and model artifacts

This directory is intentionally represented by documentation on the default branch. The complete analysis inputs, stored predictions, observed events, evaluation tables and fitted model artifacts are distributed in the `Analysis_Data_and_Models_v1.0.0.zip` asset attached to GitHub release `v1.0.0`.

## Why the data are release-only

The analysis archive is substantially larger than the normal source repository and includes binary Parquet files and fitted model artifacts. Keeping those files in a release asset avoids duplicating large binaries in Git history while preserving a fixed, citable distribution.

## Restoring the full study tree

1. Download `Analysis_Data_and_Models_v1.0.0.zip` from release `v1.0.0`.
2. Extract the archive. It expands into the repository root structure.
3. Copy the extracted `04_analysis_data/`, the additional contents of `05_models/`, and `MANIFEST_DATA.sha256` into a checkout of this repository.
4. Verify the supplied SHA-256 checksums before running the pipeline.

After restoration, the relevant subdirectories include:

- `primary_external_evaluation/`: candidate features, frozen outcome-blind prediction scores and queues, event outcomes, the protocol, input freeze metadata and formal evaluation outputs.
- `strategy_evaluation/`: regenerated event-coverage, capacity-response, list-dynamics and historical-risk summaries.
- `sensitivity_analyses/`: spatial, temporal, capacity and component comparisons used in the manuscript and Supplementary Material.
- `input_data/`: source extracts and cleaned tables used by the released evaluation.

The executable workflow in `06_code/run_pipeline.py` reproduces the reported evaluation and figures from the released stored predictions and observations. Complete model training and external scoring are outside the released executable workflow; fitted artifacts and model-support code are included for inspection.

See `METHODS.md`, `07_metadata/analysis_plan.json`, `07_metadata/data_dictionary.md`, `07_metadata/data_sources.md` and `07_metadata/provenance/evaluation_inputs.json` for study definitions and provenance.