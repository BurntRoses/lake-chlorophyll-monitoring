"""Named scientific inputs, resolved once from the repository root."""
from pathlib import Path
import pandas as pd
from .paths import ROOT,CONFIG
from .io import read
def input_paths():
    return {name:str(ROOT/path) for name,path in read(CONFIG/'datasets.json').items()}
