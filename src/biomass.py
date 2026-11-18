"""Keep biomass.* imports available for saved models with a flat source tree."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]
