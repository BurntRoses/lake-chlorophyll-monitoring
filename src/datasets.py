"""Named scientific inputs."""
from pathlib import Path
from .paths import CONFIG, DATA, RAW_DATA
from .io import read


def input_paths():
    paths = {}
    for name, value in read(CONFIG/'datasets.json').items():
        path = Path(value).relative_to('data')
        paths[name] = str(RAW_DATA / path.relative_to('raw') if path.parts[0] == 'raw' else DATA / path)
    return paths
