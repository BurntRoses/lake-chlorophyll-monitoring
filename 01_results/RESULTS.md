# Event coverage at 10% monitoring capacity

| Window | Strategy | Covered events | Coverage |
|---|---|---:|---:|
| 31–60 days | Multi-horizon | 263/696 | 37.79% |
| 1–60 days | Multi-horizon | 443/696 | 63.65% |
| 31–60 days | Direct 60-day | 207/696 | 29.74% |
| 1–60 days | Direct 60-day | 290/696 | 41.67% |
| 31–60 days | Historical risk | 277/696 | 39.80% |
| 1–60 days | Historical risk | 351/696 | 50.43% |

| Window | Comparison | Difference (pp) | 95% confidence interval (pp) | Design |
|---|---|---:|---:|---|
| 31–60 days | Multi-horizon minus Direct 60-day | 8.05 | 4.36 to 12.22 | Prespecified |
| 1–60 days | Multi-horizon minus Direct 60-day | 21.98 | 15.26 to 28.46 | Prespecified |
| 31–60 days | Multi-horizon minus Historical risk | -2.01 | -6.94 to 3.31 | Post hoc |
| 31–60 days | Direct 60-day minus Historical risk | -10.06 | -15.53 to -4.78 | Post hoc |
| 1–60 days | Multi-horizon minus Historical risk | 13.22 | 7.45 to 18.87 | Post hoc |
| 1–60 days | Direct 60-day minus Historical risk | -8.76 | -17.36 to -0.14 | Post hoc |

All strategies used 36,631 lake–prediction-date slots. The 696-event denominator includes events without valid prediction dates. Confidence intervals use 5,000 lake-level bootstrap replicates with lists rebuilt from all 2,844 candidate lakes.

Full evaluation tables: `04_analysis_data/strategy_evaluation/`. Figure sources: `07_metadata/figure_source_map.csv`.
