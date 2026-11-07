"""Verify files against the SHA-256 manifest, using only the Python standard library."""
from pathlib import Path
import hashlib
import csv
import argparse

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-raw',action='store_true',help='Also verify the optional raw-data snapshot')
    args=parser.parse_args()
    with (ROOT/'manifest.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    if args.include_raw:
        with (ROOT/'docs/raw_snapshot_manifest.csv').open(encoding='utf-8',newline='') as f:
            rows+=list(csv.DictReader(f))
    failures=[]
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file() or p.stat().st_size!=int(row['bytes']) or digest(p)!=row['sha256']:
            failures.append(row['path'])
    print(f'Checked {len(rows)} files; mismatches: {len(failures)}.')
    for p in failures:print(p)
    raise SystemExit(bool(failures))


if __name__=='__main__':main()
