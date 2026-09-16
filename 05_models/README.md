# Models and strategy identities

`multi_horizon/` contains the five outer-fold model bundles, coefficients and development predictions for the multi-horizon strategy. `direct_60_day/` indexes the direct 60-day development models and describes the external-score archive. The same-predictor 60-day component comparison is in `04_analysis_data/sensitivity_analyses/same_feature_60_day_control/`.

The historical-risk rule has no fitted model. Its per-lake scores are in `04_analysis_data/strategy_evaluation/historical_risk_lake_scores.csv`.

Serialized model weights retain their original bytes. Model-support modules are in `06_code/python_package/lake_lag_warning/`. The executable reproduction entry point uses stored prediction scores. Original source filenames in model metadata identify historical training artifacts; they are not executable paths in this distribution.
