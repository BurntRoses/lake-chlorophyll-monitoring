"""Check saved-model predictions and independently recalculate event coverage."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Replay all US rows instead of 256 per year')
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced')
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT/'results' == out or (ROOT/'results') in out.parents:
        parser.error('Choose an output outside the supplied results directory')
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from biomass.prediction import replay
    from biomass.evaluation import evaluate
    from biomass.cross_region_evaluation import evaluate_frame

    checks = {}
    with threadpool_limits(limits=2):
        replay(None if args.full else 256)
        checks['US_saved_model_predictions'] = 'all rows' if args.full else '256 rows per year'
        evaluate(output=out/'us')
        checks['US_event_coverage'] = True
        panel = pd.read_csv(ROOT/'data/processed/china_adaptation/lake_panel.csv')
        held_in = set(panel.loc[panel.historical_panel, 'lake_id'])
        jobs = []
        for name in ['china_2020_2022','china_2023','oceania','new_zealand']:
            jobs.append((ROOT/f'results/external_absolute/{name}',
                         ROOT/f'data/processed/external_absolute/{name}/episodes.parquet',False))
        for name in ['china_2020_2022','china_2023']:
            for strategy in ['source','adapted']:
                jobs.append((ROOT/f'results/china_adaptation/{name}/{strategy}',
                             ROOT/f'data/processed/external_absolute/{name}/episodes.parquet',True))
        for regime in ['source','china_adapted']:
            names=['china_2020_2022','china_2023','oceania','new_zealand'] if regime=='source' else ['china_2020_2022','china_2023']
            for name in names:
                for strategy in ['absolute75_trained','q90_trained']:
                    jobs.append((ROOT/f'results/external_relative/{regime}/{name}/{strategy}',
                                 ROOT/f'data/processed/external_relative/{name}/episodes.parquet',regime=='china_adapted'))
        for folder, event_path, holdout in jobs:
            frame=pd.read_parquet(folder/'predictions.parquet')
            events=pd.read_parquet(event_path)
            if holdout:
                events=events[~events.lake_id.isin(held_in)].reset_index(drop=True)
            name=folder.relative_to(ROOT/'results').as_posix()
            target=out/name
            actual=pd.DataFrame(evaluate_frame(frame,events,target,reps=0))
            saved=pd.read_csv(folder/'event_coverage.csv')
            keys=['endpoint','method']
            actual=actual.set_index(keys).sort_index()
            saved=saved.set_index(keys).sort_index()
            if not actual.index.equals(saved.index):
                raise AssertionError(f'Coverage groups differ: {name}')
            for column in ['captured','events','capture_percent']:
                if not np.allclose(actual[column],saved[column],rtol=1e-12,atol=1e-12,equal_nan=True):
                    raise AssertionError(f'{name}: {column} differs')
            checks[name]={'groups':len(actual),'event_coverage_matches':True}
            print('MATCH',name,len(actual),'method/window groups',flush=True)
    (out/'reproduction_summary.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print('Prediction replay and all event-coverage comparisons passed.',flush=True)


if __name__=='__main__':
    main()
