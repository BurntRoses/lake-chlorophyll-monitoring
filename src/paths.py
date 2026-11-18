"""Locations of research inputs and outputs."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW_DATA = DATA / 'raw'
CONFIG = ROOT / 'config'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'


def use_replay_root(root):
    global DATA, MODELS, RESULTS
    root = root.resolve()
    DATA = root / 'data'
    MODELS = root / 'models'
    RESULTS = root / 'results'
