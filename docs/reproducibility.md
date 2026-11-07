# Reproducing the research

## Environments

Prediction replay was tested with Python 3.12.14 and the versions in
`requirements.txt`. Serialized models require compatible versions of NumPy
(2.2.6), pandas (2.2.3), PyArrow (19.0.1), SciPy (1.15.3), scikit-learn (1.6.1)
and XGBoost (3.0.2). Package details are in `environment.json`. Original training
hardware and the CUDA build are unavailable.

Install with `python -m pip install -e .`. File verification uses only the standard
library. `scripts/reproduce.py --full` replays all saved US models and independently
recomputes US and regional event coverage. It checks 20 regional score panels and
600 method/window groups, as well as the 10 US method/window groups. Existing
bootstrap draws and published interval tables are included; this check does not
retrain models or regenerate the bootstrap draws.

## Source observations and features

The public package is self-contained for prediction replay, event evaluation and
figure rendering. To rebuild features from original sources, add the matching
source snapshots to the locations in `config/datasets.json` and the source
registry. A local source snapshot may also be used.

```bash
python -m biomass build-observations
python -m biomass build-domain
python -m biomass build-features --kind all
python -m biomass build-development --output .
python -m biomass build-evaluation --year 2021 --output .
```

Repeat the last command for 2022–2025. The cross-region builders are
`build-cross-region-observations` and `build-cross-region`. These builders replace
the generated input tables at the configured paths: work in a separate extracted
copy when rebuilding them.

`download-lagos --output data/raw` downloads and harmonizes EDI package 1439.5.
Other source products must be downloaded from the providers listed in
`data_sources.csv`; the analysis uses the specified extracts.

## Refit models

```bash
python -m biomass train --output reproduced/models/us --device cpu
python -m biomass score --model-root reproduced/models/us --output reproduced/scores/us
python -m biomass evaluate --predictions-root reproduced/scores/us --output reproduced/evaluation/us --bootstrap 5000
```

US training supports CPU or CUDA. The cross-region fitting routines use CUDA:
`train-cross-region`, `adapt-china` and `train-relative-events`. They write to
their configured model/result directories, so run them in a separate extracted
copy. Seeds, training cutoffs, region membership and fixed parameters are in
`config/`. Refit scores can differ by numerical backend; released models and scores
are the reference for reproducing the reported figures.

## Figures

Use a separate plotting environment and install `requirements-figures.txt`.
Figure rendering uses Times New Roman regular, bold and italic.
Package versions are in `environment_figures.json`; font binaries are obtained
separately. For GA on a non-Windows machine, set `BIOMASS_FONT_DIR` to a directory
containing `times.ttf`, `timesbd.ttf` and `timesi.ttf`.

Run, for example:

```bash
python figures/main/Figure3/source/Figure3.py
python figures/supplementary/FigureS1/source/FigureS1.py
python figures/graphical_abstract/source/draw_ga.py
```

Main/supplementary figure exports go next to `source/`; layout-check outputs go
under `diagnostics/` (or `--qa-dir`). GA writes beside its script. Source tables
and figure-panel roles are listed in `figure_source_index.csv` and per-figure data
dictionaries. The plotting scripts read the recorded estimates and confidence intervals.
