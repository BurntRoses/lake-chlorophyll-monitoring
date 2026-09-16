#!/usr/bin/env python3
"""Reproduce lake-monitoring evaluation and manuscript figures from stored predictions."""
from pathlib import Path
import argparse, json, shutil, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parents[1]

def run(relative,*args):
 subprocess.run([sys.executable,'-B',str(ROOT/relative),*map(str,args)],cwd=ROOT,check=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--stage',choices=['evaluate','figures','index','verify','all'],default='all')
 a=ap.parse_args()
 if a.stage in ['all','evaluate']:
  run('06_code/analysis/evaluate_strategies.py')
 if a.stage in ['all','evaluate','index']:
  run('06_code/analysis/build_results_index.py')
 if a.stage in ['all','figures']:
  with tempfile.TemporaryDirectory(prefix='lake-study-figures-') as temp:
   dest=Path(temp)
   run('06_code/figure_generation/render_final_figures.py','--output-root',dest)
   for f in dest.glob('Figure_*.*'):
    if f.suffix in ['.png','.svg','.pdf']:
     shutil.copy2(f,ROOT/'02_figures'/('supplementary' if f.stem.startswith('Figure_S') else 'main')/f.name)
   audit=ROOT/'07_metadata/provenance/figures';audit.mkdir(parents=True,exist_ok=True)
   shutil.copy2(dest/'QA/render_audit.json',audit/'study_figure_audit.json')
  run('06_code/figure_generation/render_strategy_figures.py')
 if a.stage in ['all','verify']:
  run('06_code/analysis/verify_release.py')

if __name__=='__main__':main()
