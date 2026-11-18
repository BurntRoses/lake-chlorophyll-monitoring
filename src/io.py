"""Read research configuration and write scientific products."""
from pathlib import Path
import hashlib
import json
import numpy as np
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _json(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=_json)+'\n',encoding='utf8')
def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
