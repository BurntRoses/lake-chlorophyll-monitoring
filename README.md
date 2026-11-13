# Capacity-limited lake monitoring

Data and code for *Ranking lakes for high-chlorophyll events under limited monitoring capacity* by Siran Luo, Xingchen Liu, Lin Zhao, Yihan Wu, Yining Zhang and Jibiao Zhang.

The study evaluates independent high-chlorophyll event coverage under recurring sampling quotas, using US lake records, history-aware scheduling and cross-region absolute and lake-relative targets.

## Data

Download [analysis-data.zip](https://github.com/BurntRoses/lake-chlorophyll-monitoring/releases/download/research-materials/analysis-data.zip) and [SHA256SUMS.txt](https://github.com/BurntRoses/lake-chlorophyll-monitoring/releases/download/research-materials/SHA256SUMS.txt) from [Research materials](https://github.com/BurntRoses/lake-chlorophyll-monitoring/releases/tag/research-materials). Extract the archive into the repository root. It supplies `data/derived`, `data/processed`, `data/figures`, `models` and `results`.

Original datasets and attribution are listed in `docs/data_sources.csv`, `THIRD_PARTY_NOTICES.md` and `LICENSE-DATA`. Original source downloads remain with their providers.

## Reproduce

Use Python 3.12 and Times New Roman for figure rendering.

```sh
python -m pip install -e .
python -m pip install -r requirements-figures.txt
python scripts/reproduce.py --output ../lake-monitoring-reproduced
python scripts/render_figures.py
```

`reproduce.py` checks saved-model predictions and recomputes US and regional event coverage. Add `--full` to replay all US predictions. `python -m biomass --help` lists data preparation, training and evaluation commands. Recalculation outputs should use a separate directory.

## Files

| Path | Contents |
|---|---|
| `src/biomass` | Data preparation, models, scoring, scheduling and evaluation |
| `config` | Model and evaluation settings |
| `scripts` | Reproduction, stage replay and figure rendering |
| `tables` | Table 1 and Tables S1–S16 as CSV |
| `figures` | Main and supplementary figures, captions and graphical abstract |
| `docs` | Sources, variable definitions, environment and figure statistics |

Code is MIT-licensed. Data and third-party materials retain the terms in `LICENSE-DATA` and `THIRD_PARTY_NOTICES.md`. `manifest.csv` records file sizes and SHA-256 checksums for this source tree.
