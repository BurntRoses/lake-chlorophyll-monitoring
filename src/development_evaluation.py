"""Chronological development and stability evaluation of the selected US models."""
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import CONFIG, DATA, RESULTS
from .io import read, write
from .evaluation import selected_mask, opportunity_index, pooled_capture

OUT = RESULTS / 'us/development'


def assess(predictions_root=None):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(DATA / 'processed/us_training/common_development_table.parquet',
        columns=['candidate_row_key','lake_id','origin_date','origin_year','outer_lake_fold','eligible_common_development_candidate'])
    meta['lake_id'] = meta.lake_id.astype(str)
    episodes = pd.read_parquet(DATA / 'processed/us_training/development_episodes.parquet')
    episodes['lake_id'] = episodes.lake_id.astype(str)
    chosen = read(CONFIG / 'monitoring_models.json')['corrections']
    from .monitoring_schedule import source_experts
    experts = source_experts(predictions_root)
    stages = [('discovery',2016,2018),('stability',2019,2020)]
    metrics, annual, ledgers, exchanges = [], [], [], []
    for stage, first, last in stages:
        specs = chosen
        frame = meta[meta.origin_year.between(first,last) & meta.eligible_common_development_candidate]
        frame = frame.sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)
        path = OUT / (stage+'_predictions.parquet')
        if predictions_root is not None:
            scores = frame[['candidate_row_key']].copy()
            for s in specs:
                paths = [Path(predictions_root)/'fits'/str(y)/str(f)/s['name']/'predictions.parquet'
                         for y in range(first,last+1) for f in range(5)]
                values = pd.concat([pd.read_parquet(p)[['candidate_row_key','score']] for p in paths])
                scores = scores.merge(values.rename(columns={'score':s['name']}),on='candidate_row_key',validate='one_to_one')
            scores.to_parquet(path,index=False)
        scores = pd.read_parquet(path)
        if not scores.candidate_row_key.is_unique or set(scores.candidate_row_key) != set(frame.candidate_row_key):
            raise ValueError('US source prediction population differs: '+stage)
        frame = frame.merge(scores,on='candidate_row_key',validate='one_to_one')
        frame = frame.merge(experts,on='candidate_row_key',validate='one_to_one')
        cutoff = pd.Timestamp(last+1,1,1)
        events = episodes[episodes.event_date.add(pd.Timedelta(days=7)).lt(cutoff)]
        for s in specs:
            values = frame[s['name']].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError('Incomplete source prediction: '+s['name'])
            metrics.append(dict(stage=stage,name=s['name'],family=s['family'],
                **pooled_capture(frame.assign(score=values),events)))
        policies = {s['policy_name']:s['name'] for s in chosen}
        policies.update({k:k for k in ['xgboost','elastic_net','matched_xgboost']})
        for endpoint, (lo,hi) in read(CONFIG/'analysis.json')['endpoints'].items():
            ledger, ri, ei = opportunity_index(frame,events,lo,hi)
            for method, column in policies.items():
                mask = selected_mask(frame,frame[column])
                hits = np.zeros(len(ledger),bool)
                np.logical_or.at(hits,ei,mask[ri])
                ledger[method] = hits
                for year in ['ALL',*sorted(ledger.event_date.dt.year.unique())]:
                    use = np.ones(len(ledger),bool) if year == 'ALL' else ledger.event_date.dt.year.eq(year).to_numpy()
                    annual.append(dict(stage=stage,endpoint=endpoint,year=str(year),method=method,
                        captured=int(hits[use].sum()),events=int(use.sum()),capture_percent=100*hits[use].mean()))
            for control in [k for k in policies if k!='multi_window']:
                a,b = ledger.multi_window.to_numpy(),ledger[control].to_numpy()
                exchanges.append(dict(stage=stage,endpoint=endpoint,comparator=control,events=len(ledger),
                    multi_only=int((a & ~b).sum()),control_only=int((~a & b).sum()),gain_pp=100*(a.mean()-b.mean())))
            ledgers.append(ledger.assign(stage=stage,endpoint=endpoint))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT/'configuration_metrics.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_event_coverage.csv',index=False)
    pd.DataFrame(exchanges).to_csv(OUT/'event_exchanges.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'event_captures.parquet',index=False)
    print('US_DEVELOPMENT_EVALUATION_COMPLETE',flush=True)



if __name__ == '__main__':
    assess()
