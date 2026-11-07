# Figures

Install `requirements-figures.txt` in a separate Python environment. Rendering
uses Times New Roman regular, bold and italic. For the graphical abstract on
non-Windows systems, set `BIOMASS_FONT_DIR` to the directory containing
`times.ttf`, `timesbd.ttf` and `timesi.ttf`.

Run a figure from the repository root:

```bash
python figures/main/Figure1/source/Figure1.py
python figures/main/Figure2/source/Figure2.py
python figures/main/Figure3/source/Figure3.py
python figures/main/Figure4/source/Figure4.py
python figures/supplementary/FigureS1/source/FigureS1.py
python figures/supplementary/FigureS2/source/FigureS2.py
python figures/supplementary/FigureS3/source/FigureS3.py
python figures/supplementary/FigureS4/source/FigureS4.py
python figures/supplementary/FigureS5/source/FigureS5.py
python figures/supplementary/FigureS6/source/FigureS6.py
python figures/graphical_abstract/source/draw_ga.py
```

Main and supplementary figures are written next to their `source/` directory;
the graphical abstract is written beside `draw_ga.py`. Each source directory
contains its input tables, data dictionary and local dependencies. The graphical
abstract embeds its numerical and geographic inputs. Captions are stored with
the rendered figures. Layout diagnostics are written under `diagnostics/` or
the directory selected by `--qa-dir`.

Input tables and panel roles are indexed in
[`figure_source_index.csv`](../docs/figure_source_index.csv).
