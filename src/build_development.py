"""Build lake-level training inputs and event labels."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .paths import ROOT,CONFIG,DATA
from .datasets import input_paths
BINS = ((1, 14), (15, 30), (31, 60))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    (output / 'config').mkdir(parents=True, exist_ok=True)
    config = json.loads((CONFIG/'study.json').read_text(encoding='utf-8'))
    paths = input_paths()
    candidate_asset='development_candidates'
    frame=pd.read_parquet(paths[candidate_asset])
    expected_rows = len(frame)
    obs = pd.read_parquet(paths['development_chla_observation_ledger'])
    episodes = pd.read_parquet(paths['primary_incident_episode_ledger'])
    for t in (frame, obs, episodes):
        t['lake_id'] = t.lake_id.astype(str)
    if frame.duplicated(['lake_id','origin_date']).any() or frame.candidate_row_key.duplicated().any():
        raise RuntimeError('Material data problem: duplicate candidate primary keys')
    if obs.duplicated(['lake_id','sample_date']).any() or episodes.duplicated(['lake_id','event_date']).any():
        raise RuntimeError('Material data problem: duplicate observation/episode dates')
    if not np.array_equal(obs.primary_severe.to_numpy(), (obs.value_median >= config['chla_threshold_ugL']).to_numpy()):
        raise RuntimeError('Material data problem: severe threshold inconsistency')
    severe = obs[obs.primary_severe].sort_values(['lake_id','sample_date'])
    gap = severe.groupby('lake_id').sample_date.diff().dt.days
    expected = severe[gap.isna() | gap.gt(config['washout_days'])][['lake_id','sample_date']]
    if set(map(tuple, expected.to_numpy())) != set(map(tuple, episodes[['lake_id','event_date']].to_numpy())):
        raise RuntimeError('Material data problem: event ledger conflicts with observed threshold/washout')
    frame = frame.sort_values(['lake_id','origin_date']).reset_index(drop=True)
    frame['next_episode_id'] = pd.Series(pd.NA, index=frame.index, dtype='string')
    frame['next_event_date'] = pd.NaT
    obs_groups = dict(tuple(obs.groupby('lake_id',sort=False)))
    ep_groups = dict(tuple(episodes.groupby('lake_id',sort=False)))
    for lake, ix in frame.groupby('lake_id',sort=False).groups.items():
        origins = frame.loc[ix,'origin_date'].to_numpy(dtype='datetime64[ns]')
        if lake in ep_groups:
            g = ep_groups[lake].sort_values('event_date')
            dates = g.event_date.to_numpy(dtype='datetime64[ns]')
            j = np.searchsorted(dates,origins,side='right')
            valid = j<len(g)
            frame.loc[np.asarray(ix)[valid],'next_event_date'] = dates[j[valid]]
            frame.loc[np.asarray(ix)[valid],'next_episode_id'] = g.episode_id.to_numpy()[j[valid]]
        g = obs_groups.get(lake)
        dates = np.sort(g.sample_date.to_numpy(dtype='datetime64[ns]')) if g is not None else np.array([],dtype='datetime64[ns]')
        for lo,hi in BINS:
            counts = np.searchsorted(dates,origins+np.timedelta64(hi,'D'),side='right')-np.searchsorted(dates,origins+np.timedelta64(lo,'D'),side='left')
            frame.loc[ix,f'observation_support_{lo}_{hi}'] = counts
    delta = (frame.next_event_date-frame.origin_date).dt.days
    blocked = frame.unavailable_active_episode_boundary | frame.same_day_incident_onset | ~frame.eligible_common_development_candidate
    event_known = delta.between(1,60) & ~blocked
    for lo,hi in BINS:
        y = np.full(len(frame),-1,dtype=np.int8)
        supported = frame[f'observation_support_{lo}_{hi}'].gt(0) & ~blocked
        y[supported | event_known] = 0
        positive = delta.between(lo,hi) & ~blocked
        y[positive] = 1
        frame[f'y_{lo}_{hi}'] = y
    frame['common_trainable'] = frame.y_31_60.ge(0) & ~blocked
    result_delay = (obs.assumed_result_available_date - obs.sample_date).dt.days
    maximum_result_delay = int(result_delay.max())
    frame['label_mature_date'] = frame.origin_date + pd.Timedelta(days=60 + maximum_result_delay)
    # Validate observation support against the candidate ledger.
    for lo,hi in BINS:
        recorded_count=f'future_chla_observation_days_{lo}_{hi}'
        if not np.array_equal(frame[f'observation_support_{lo}_{hi}'].to_numpy(),frame[recorded_count].to_numpy()):
            raise RuntimeError(f'Material data problem: observation support conflicts in {lo}-{hi}')
    pool = pd.read_csv(CONFIG/'features.csv').feature_name.tolist()
    feature_records=[]
    for asset in ['development_static_features','development_water_quality_features','development_biomass_features','development_meteorology_features']:
        schema=pq.ParquetFile(paths[asset]).schema_arrow
        names=[x for x in pool if x in schema.names]
        f=pd.read_parquet(paths[asset],columns=['lake_id','origin_date']+names)
        f['lake_id']=f.lake_id.astype(str)
        if f.duplicated(['lake_id','origin_date']).any():
            raise RuntimeError(f'Material data problem: duplicate feature keys in {asset}')
        frame=frame.merge(f,on=['lake_id','origin_date'],how='left',validate='one_to_one')
        if len(frame)!=expected_rows:
            raise RuntimeError('Feature join changed common row count')
        for name in names:
            if name!='ecoregion': frame[name]=pd.to_numeric(frame[name],errors='coerce').astype('float32')
            feature_records.append({'feature_name':name,'source':asset,'role':'base' if asset=='development_static_features' else 'dynamic'})
        print('joined',asset,len(names),flush=True)
    # Freeze each row's history at the beginning of its year; never use later observations.
    frame=frame.copy()
    frame['fixed_history_risk']=np.nan
    frame['fixed_history_observation_count']=0.0
    frame['fixed_history_severe_count']=0.0
    hist=[]
    for year in sorted(frame.origin_year.unique()):
        previous=obs[obs.assumed_result_available_date < pd.Timestamp(year=int(year),month=1,day=1)]
        h=previous.groupby('lake_id').agg(n=('primary_severe','size'),severe=('primary_severe','sum'))
        h['risk']=h.severe/h.n
        ix=frame.origin_year.eq(year)
        frame.loc[ix,'fixed_history_risk']=frame.loc[ix,'lake_id'].map(h.risk).to_numpy(dtype=float)
        frame.loc[ix,'fixed_history_observation_count']=frame.loc[ix,'lake_id'].map(h.n).fillna(0)
        frame.loc[ix,'fixed_history_severe_count']=frame.loc[ix,'lake_id'].map(h.severe).fillna(0)
        ht=h.reset_index();ht['evaluation_year']=int(year);hist.append(ht)
    for name in ['fixed_history_risk','fixed_history_observation_count','fixed_history_severe_count']:
        frame[name]=frame[name].astype('float32')
        feature_records.append({'feature_name':name,'source':'observations_available_before_origin_year','role':'base'})
    frame['days_since_last_known_severe_sample']=frame.days_since_last_known_severe_sample.astype('float32')
    feature_records.append({'feature_name':'days_since_last_known_severe_sample','source':'candidate_ledger_available_history','role':'base'})
    dest=output/'data/processed/us_training';dest.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(dest/'common_development_table.parquet',index=False)
    episodes.to_parquet(dest/'development_episodes.parquet',index=False)
    pd.concat(hist,ignore_index=True).to_parquet(dest/'fixed_history_by_year.parquet',index=False)
    pd.DataFrame(feature_records).to_csv(output/'config/features.csv',index=False,encoding='utf-8-sig')
    counts=[]
    for year,g in frame.groupby('origin_year'):
        record={'year':int(year),'candidates':len(g),'eligible':int(g.eligible_common_development_candidate.sum()),'common_trainable':int(g.common_trainable.sum())}
        for lo,hi in BINS:
            record.update({f'{lo}_{hi}_{label}':int((g[f'y_{lo}_{hi}']==value).sum()) for value,label in [(1,'positive'),(0,'negative'),(-1,'unknown')]})
        record['distinct_positive31_60_episodes']=int(g.loc[g.y_31_60.eq(1),'next_episode_id'].nunique())
        counts.append(record)
    out=output/'data/processed/us_training';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(counts).to_csv(out/'training_sample_counts.csv',index=False,encoding='utf-8-sig')
    cells=[]
    for year in config['development_years']:
        for fold in range(config['lake_folds']):
            train=frame.common_trainable & frame.origin_year.lt(year) & frame.outer_lake_fold.ne(fold) & frame.label_mature_date.lt(pd.Timestamp(year=year,month=1,day=1))
            val=frame.origin_year.eq(year) & frame.outer_lake_fold.eq(fold) & frame.eligible_common_development_candidate
            cells.append({'year':year,'fold':fold,'train_rows':int(train.sum()),'train_positive':int(frame.loc[train,'y_31_60'].eq(1).sum()),'train_lakes':int(frame.loc[train,'lake_id'].nunique()),'validation_candidates':int(val.sum()),'validation_lakes':int(frame.loc[val,'lake_id'].nunique()),'validation_positive31_60_episodes':int(frame.loc[val & frame.y_31_60.eq(1),'next_episode_id'].nunique())})
    pd.DataFrame(cells).to_csv(out/'development_cell_counts.csv',index=False,encoding='utf-8-sig')
    note={'rows':len(frame),'common_trainable_rows':int(frame.common_trainable.sum()),'feature_pool':len(feature_records),'source_event_count':len(episodes),'label_definition':'next recorded independent incident; observed window required for absence; unsampled windows remain unknown; exact incident yields exclusive category','historical_feature_cutoff':'assumed_result_available_date before January 1 of each origin year','maturity':f'origin + 60 days + {maximum_result_delay} days recorded result-availability delay strictly before evaluation January 1','evaluation_candidates':'all origin-available eligible candidates, including unknown outcomes','blocked_labels':'same-day incident or unavailable active episode; no use of these flags to remove evaluation candidates','candidate_asset':candidate_asset,'candidate_domain':'first result available strictly before origin, persistent to common cutoff'}
    print(json.dumps(note,ensure_ascii=False),flush=True)
    print(pd.DataFrame(counts).to_string(index=False),flush=True)

if __name__=='__main__':main()
