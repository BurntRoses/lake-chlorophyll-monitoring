"""Render the graphical abstract, main figures and supplementary figures."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts' / 'figures'
FIGURES = {'GA': SCRIPTS / 'GA.py'}
FIGURES.update({f'Figure{i}': SCRIPTS / f'Figure{i}' / f'Figure{i}.py' for i in range(1, 5)})
FIGURES.update({f'FigureS{i}': SCRIPTS / f'FigureS{i}.py' for i in range(1, 8)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figures', nargs='*', help='Names such as FigureS4; omit to render all figures')
    args = parser.parse_args()
    names = args.figures or list(FIGURES)
    unknown = set(names) - FIGURES.keys()
    if unknown:
        parser.error('Unknown figures: ' + ', '.join(sorted(unknown)))
    for name in names:
        print(f'Rendering {name}', flush=True)
        subprocess.run([sys.executable, '-B', str(FIGURES[name])], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
