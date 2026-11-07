"""Repository paths."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
CONFIG = ROOT / 'config'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
