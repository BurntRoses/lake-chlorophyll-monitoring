"""Replay saved ranking trees and rebuild the Figure S3 event diagnostics."""
from pathlib import Path
import argparse
import gc
import json
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH / 'src'))
from biomass.prediction import evaluation_inputs
from biomass.experts import transform_batched
from biomass.evaluation import selected_mask, opportunity_index

def rebuild_scores(cache_path):
    spec=json.loads((RESEARCH/'config/monitoring_models.json').read_text())
    configs={s['name']:s for s in spec['corrections']}
    expected=pd.read_parquet(RESEARCH/'results/us/main/predictions.parquet').set_index('candidate_row_key')
    base=pd.read_parquet(RESEARCH/'results/us/analysis/base_predictions.parquet').set_index('candidate_row_key')
    expr=joblib.load(RESEARCH/'models/us/experts/seasonal/common_expression.joblib')
    models={m:joblib.load(RESEARCH/f'models/us/coverage/{m}.joblib') for m in ['single_window','multi_window']}
    parts=[];replay_checks=[]
    for year in range(2021,2026):
        print('Reading evaluation features',year,flush=True)
        frame=evaluation_inputs(year)
        raw=transform_batched(expr,frame)
        assert raw.shape[1]+4==models['multi_window']['feature_count']
        out_columns={c:frame[c].to_numpy() for c in ['candidate_row_key','lake_id','origin_date']}
        for m in ['single_window','multi_window']:
            keys=configs[m]['expert_keys'];values=[]
            for key in keys:
                col='matched_xgboost' if key=='xgb4__y_31_60' else key
                values.append(base.loc[frame.candidate_row_key,col].to_numpy())
            logits=logit(np.clip(np.column_stack(values).astype(float),1e-6,1-1e-6))
            features=np.column_stack([raw,logits,np.zeros(len(raw))]).astype(np.float32)
            matrix=xgb.DMatrix(features,nthread=2)
            z=logits[:,0].copy();out_columns[m+'__0']=z.copy()
            payload=models[m]
            for i,(tree,step) in enumerate(zip(payload['trees'],payload['steps']),1):
                tree.set_params(device='cpu')
                delta=tree.get_booster().predict(matrix,output_margin=True)
                z+=step*np.clip(delta,-payload['parameters']['max_direction'],payload['parameters']['max_direction'])
                out_columns[m+'__'+str(i)]=z.copy()
            target=expected.loc[frame.candidate_row_key,m].to_numpy()
            error=float(np.max(np.abs(z-target)))
            print('Score audit',year,m,'maximum error',error,flush=True)
            assert error<1e-7,(year,m,error)
            for cap in [.05,.10]:
                assert np.array_equal(selected_mask(frame,z,cap),selected_mask(frame,target,cap)),(year,m,cap)
            replay_checks.append({'year':year,'method':m,'rows':len(frame),'max_score_error':error,
                                  'identical_final_selection_at_5_and_10_pct':True})
            del matrix,features
        parts.append(pd.DataFrame(out_columns))
        del frame,raw;gc.collect()
        print('Completed tree-prefix scores',year,flush=True)
    scores=pd.concat(parts,ignore_index=True).sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)
    assert len(scores)==365726
    scores.to_parquet(cache_path,index=False)
    return cache_path

def prepare_data(score_path, output):
    output.mkdir(parents=True, exist_ok=True)
    score = pd.read_parquet(score_path)
    assert len(score) == 365726 and score.candidate_row_key.is_unique
    score['lake_id'] = score.lake_id.astype(str)
    assert score[['origin_date', 'lake_id']].equals(
        score[['origin_date', 'lake_id']].sort_values(
            ['origin_date', 'lake_id'], kind='stable').reset_index(drop=True))
    assert np.array_equal(score.single_window__0, score.multi_window__0)
    episodes = pd.read_parquet(RESEARCH / 'data/processed/us_evaluation/retrospective_episodes.parquet')
    episodes['lake_id'] = episodes.lake_id.astype(str)
    ledger, ri, ei = opportunity_index(score, episodes)
    assert len(ledger) == 593 and np.unique(ei).size == 593
    dates = score.origin_date.to_numpy()
    lakes = score.lake_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1]
    ends = np.r_[starts[1:], len(score)]
    assert len(starts) == 130

    released = pd.read_parquet(RESEARCH / 'results/us/main/predictions.parquet')
    released = released.set_index('candidate_row_key').loc[score.candidate_row_key]
    saved_events = pd.read_csv(RESEARCH / 'results/us/main/evaluation/31_60_event_coverage.csv')
    saved_events = saved_events.set_index('episode_id').loc[ledger.episode_id]
    arrays = {}
    stage_rows = []
    checks = []
    for method in ['single_window', 'multi_window']:
        assert np.array_equal(score[f'{method}__60'].to_numpy(), released[method].to_numpy())
        method_arrays = {c: np.empty((593, 61), dtype=np.float64) for c in [5, 10]}
        for stage in range(61):
            values = score[f'{method}__{stage}'].to_numpy()
            assert np.isfinite(values).all()
            order = np.lexsort((lakes, -values, dates))
            ranks = np.empty(len(score), dtype=np.int32)
            for start, end in zip(starts, ends):
                ranks[order[start:end]] = np.arange(1, end - start + 1)
            for capacity in [5, 10]:
                quotas = np.empty(len(score), dtype=np.int32)
                for start, end in zip(starts, ends):
                    quotas[start:end] = int(np.ceil(capacity / 100 * (end - start)))
                ratio = ranks / quotas
                best = np.full(593, np.inf)
                np.minimum.at(best, ei, ratio[ri])
                assert np.isfinite(best).all() and (best > 0).all()
                method_arrays[capacity][:, stage] = best
                captured = int((best <= 1).sum())
                if stage in [0, 60]:
                    selected = selected_mask(score, values, capacity / 100)
                    assert np.array_equal(selected, ranks <= quotas)
                    hits = np.zeros(593, dtype=bool)
                    np.logical_or.at(hits, ei, selected[ri])
                    assert np.array_equal(hits, best <= 1)
                    if stage == 60 and capacity == 10:
                        assert np.array_equal(hits, saved_events[method].to_numpy(dtype=bool))
                stage_rows.append(dict(method=method, stage=stage, capacity_pct=capacity,
                                       captured=captured, events=593, coverage_pct=100 * captured / 593))
        for capacity, arr in method_arrays.items():
            arrays[f'{method}_{capacity}'] = arr
            checks.append(dict(method=method, capacity_pct=capacity,
                               initial_covered=int((arr[:, 0] <= 1).sum()),
                               final_covered=int((arr[:, -1] <= 1).sum())))
        print(method, checks[-2:], flush=True)

    initial = arrays['single_window_10'][:, 0]
    assert np.array_equal(initial, arrays['multi_window_10'][:, 0])
    event_order = np.argsort(initial, kind='stable')
    ledger = ledger.iloc[event_order].reset_index(drop=True)
    ledger['event_order'] = np.arange(1, 594)
    for key, arr in arrays.items():
        arrays[key] = arr[event_order]
    assert (np.diff(arrays['single_window_10'][:, 0]) >= 0).all()
    np.savez_compressed(output / 'event_priorities.npz', **arrays)
    ledger.to_csv(output / 'event_order.csv', index=False)
    pd.DataFrame(stage_rows).to_csv(output / 'stage_coverage.csv', index=False)

    turnover = []
    event_states = []
    for method in ['single_window', 'multi_window']:
        for capacity in [5, 10]:
            arr = arrays[f'{method}_{capacity}']
            base = arr[:, 0] <= 1
            for stage in range(61):
                current = arr[:, stage] <= 1
                gained = int((~base & current).sum())
                lost = int((base & ~current).sum())
                retained = int((base & current).sum())
                neither = int((~base & ~current).sum())
                assert gained + lost + retained + neither == 593
                assert base.sum() + gained - lost == current.sum()
                turnover.append(dict(method=method, capacity_pct=capacity, stage=stage,
                                     base_covered=int(base.sum()), covered=int(current.sum()),
                                     gained=gained, lost=lost, retained=retained,
                                     neither=neither, net_change=gained-lost))
            state = ledger.copy()
            state['method'] = method
            state['capacity_pct'] = capacity
            state['initial_priority'] = arr[:, 0]
            state['final_priority'] = arr[:, -1]
            state['initial_covered'] = base
            state['final_covered'] = arr[:, -1] <= 1
            state['change'] = np.select(
                [~base & state.final_covered, base & ~state.final_covered,
                 base & state.final_covered], ['gained', 'lost', 'retained'], default='neither')
            event_states.append(state)
    turn = pd.DataFrame(turnover)
    turn.to_csv(output / 'coverage_turnover.csv', index=False)
    pd.concat(event_states, ignore_index=True).to_csv(output / 'event_final_states.csv', index=False)
    pair_rows = []
    for capacity in [5, 10]:
        single = arrays[f'single_window_{capacity}'][:, -1] <= 1
        multi = arrays[f'multi_window_{capacity}'][:, -1] <= 1
        pair_rows.append(dict(capacity_pct=capacity, both=int((single & multi).sum()),
                              single_only=int((single & ~multi).sum()),
                              multi_only=int((~single & multi).sum()),
                              neither=int((~single & ~multi).sum())))
    pd.DataFrame(pair_rows).to_csv(output / 'final_method_pairing.csv', index=False)
    print("Stage diagnostics rebuilt from saved scores.", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-scores', action='store_true', help='Recompute every prefix from saved trees')
    parser.add_argument('--output', type=Path, default=RESEARCH / 'data/figures/FigureS3')
    args = parser.parse_args()
    cache = RESEARCH / 'results/us/analysis/stage_scores.parquet'
    if args.recompute_scores or not cache.exists():
        rebuild_scores(cache)
    prepare_data(cache, args.output)


if __name__ == '__main__':
    main()
