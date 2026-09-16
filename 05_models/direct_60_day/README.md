# Direct 60-day strategy

The development models are in `04_analysis_data/model_training_inputs/direct_60_day_training/models/`. Each target year (2016–2020) has five outer lake folds. `Mbio.joblib` is the direct 60-day model; `M0.joblib` and `Menv.joblib` are the development reference models. `model_index.csv` lists these artifacts and checksums.

The external comparison uses the `direct_60_day` score in `04_analysis_data/primary_external_evaluation/direct_60_day_scores/external_direct_60_day_outcome_blind_scores.parquet`. Its selected lists are stored alongside that file. A complete external scoring driver and a verified mapping from external scores to serialized direct 60-day weights are not included. Reproduction of the reported external evaluation starts from the stored scores.
