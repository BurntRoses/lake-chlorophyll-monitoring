"""Complete capacity queues, independent-event capture and paired uncertainty."""
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from .paths import DATA,RESULTS,CONFIG
from .io import read,write


def selected_mask(frame,score,capacity=.1):
    score=np.asarray(score,float)
    if not np.isfinite(score).all():raise ValueError('Nonfinite queue score')
    order=np.lexsort((frame.lake_id.to_numpy(),-score,frame.origin_date.to_numpy()))
    origins=frame.origin_date.to_numpy()[order]
    starts=np.r_[0,np.flatnonzero(origins[1:]!=origins[:-1])+1]
    ends=np.r_[starts[1:],len(order)]
    mask=np.zeros(len(frame),bool)
    for start,end in zip(starts,ends):mask[order[start:start+int(np.ceil(capacity*(end-start)))]]=True
    return mask


def opportunity_index(frame,episodes,lo=31,hi=60):
    rows=[];ids=[];event_rows=[]
    lookup={lake:g for lake,g in frame.groupby('lake_id',sort=False)}
    for e in episodes.itertuples(index=False):
        g=lookup.get(e.lake_id)
        if g is None:continue
        delta=(pd.Timestamp(e.event_date)-g.origin_date).dt.days.to_numpy()
        positions=g.index[(delta>=lo)&(delta<=hi)].to_numpy()
        if not len(positions):continue
        index=len(event_rows);event_rows.append((e.episode_id,e.lake_id,e.event_date))
        rows.extend(positions);ids.extend([index]*len(positions))
    ledger=pd.DataFrame(event_rows,columns=['episode_id','lake_id','event_date'])
    ledger['event_date']=pd.to_datetime(ledger.event_date)
    return ledger,np.asarray(rows,np.int64),np.asarray(ids,np.int64)


def pooled_capture(frame,events,maturity=None):
    frame=frame.sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)
    if maturity is not None:events=events[events.event_date.add(pd.Timedelta(days=7)).lt(maturity)]
    mask=selected_mask(frame,frame.score)
    values={};annual=[]
    for name,lower in [('primary',31),('secondary',1)]:
        ledger,ri,ei=opportunity_index(frame,events,lower,60)
        hits=np.zeros(len(ledger),bool);np.logical_or.at(hits,ei,mask[ri])
        values[name]=float(hits.mean()*100)
        if name=='primary':
            for year in sorted(ledger.event_date.dt.year.unique()):annual.append(float(hits[ledger.event_date.dt.year.eq(year)].mean()*100))
    values['worst_year']=min(annual)
    return values


@njit
def select_copies(multiplicity,row_lake,starts,ends,orders,capacity):
    selected=np.zeros((len(orders),len(row_lake)),np.int32)
    for group in range(len(starts)):
        start,end=starts[group],ends[group]
        total=0
        for row in range(start,end):total+=multiplicity[row_lake[row]]
        quota=int(np.ceil(capacity*total))
        for method in range(len(orders)):
            used=0
            for offset in range(start,end):
                row=orders[method,offset];copies=multiplicity[row_lake[row]]
                selected[method,row]=min(copies,max(0,quota-used));used+=copies
    return selected


@njit
def event_counts(selected,ri,ei,event_lake,multiplicity):
    denominator=0
    for lake in event_lake:denominator+=multiplicity[lake]
    numerator=np.zeros(len(selected),np.int64)
    for method in range(len(selected)):
        hits=np.zeros(len(event_lake),np.int32)
        for j in range(len(ri)):
            hits[ei[j]]=max(hits[ei[j]],selected[method,ri[j]])
        numerator[method]=hits.sum()
    return numerator,denominator


def evaluate(output=None,reps=0,predictions_root=None):
    family='US'
    source=Path(predictions_root) if predictions_root is not None else RESULTS/'us'
    frame=pd.read_parquet(source/'main/predictions.parquet')
    frame['lake_id']=frame.lake_id.astype(str)
    frame=frame.sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)
    episodes=pd.read_parquet(DATA/'processed/us_evaluation/retrospective_episodes.parquet')
    episodes['lake_id']=episodes.lake_id.astype(str)
    spec=read(CONFIG/'monitoring_models.json')
    policies={p['name']:p['column'] for p in spec['policies'] if p['name'] in ['elastic_net','xgboost','single_window','same_window_control','multi_window']}
    lakes=np.sort(frame.lake_id.unique());row_lake=pd.Index(lakes).get_indexer(frame.lake_id).astype(np.int32)
    dates=pd.factorize(frame.origin_date,sort=True)[0]
    starts=np.r_[0,np.flatnonzero(dates[1:]!=dates[:-1])+1].astype(np.int64)
    ends=np.r_[starts[1:],len(frame)].astype(np.int64)
    orders=np.array([np.lexsort((frame.lake_id.to_numpy(),-frame[column].to_numpy(),dates)) for column in policies.values()],np.int64)
    ones=np.ones(len(lakes),np.int64)
    selected=select_copies(ones,row_lake,starts,ends,orders,spec['capacity'])
    points=[];endpoints={}
    for endpoint,lo,expected_events in [('31_60',31,593),('1_60',1,681)]:
        ledger,ri,ei=opportunity_index(frame,episodes,lo,60)
        if len(ledger)!=expected_events:raise ValueError('Event population differs')
        event_lake=pd.Index(lakes).get_indexer(ledger.lake_id).astype(np.int32)
        endpoints[endpoint]=(ri,ei,event_lake)
        numerator,denominator=event_counts(selected,ri,ei,event_lake,ones)
        saved=(pd.read_csv(RESULTS/'us/main/evaluation'/(endpoint+'_event_coverage.csv')).set_index('episode_id')
               if predictions_root is None else None)
        captures=ledger.copy()
        for method_index,method in enumerate(policies):
            hits=np.zeros(len(ledger),bool);np.logical_or.at(hits,ei,selected[method_index,ri]>0)
            if saved is not None and set(ledger.loc[hits,'episode_id'])!=set(saved.index[saved[method].eq(True)]):raise ValueError('Event capture set differs: '+method)
            captures[method]=hits
            points.append(dict(endpoint=endpoint,method=method,captured=int(numerator[method_index]),events=denominator,capture_percent=100*numerator[method_index]/denominator))
            print(f'{family} {endpoint} {method}: {numerator[method_index]}/{denominator}',flush=True)
        if output is not None:
            output.mkdir(parents=True,exist_ok=True)
            captures.to_csv(output/(endpoint+'_event_coverage.csv'),index=False)
    if output is not None:
        output.mkdir(parents=True,exist_ok=True);pd.DataFrame(points).to_csv(output/'event_coverage.csv',index=False)
    if reps:
        if output is None:raise ValueError('Bootstrap requires an output directory')
        rng=np.random.default_rng(20261065);records=[]
        for rep in range(reps):
            multiplicity=np.bincount(rng.integers(0,len(lakes),len(lakes)),minlength=len(lakes)).astype(np.int64)
            chosen=select_copies(multiplicity,row_lake,starts,ends,orders,spec['capacity'])
            record={'replicate':rep}
            for endpoint,(ri,ei,el) in endpoints.items():
                numerator,denominator=event_counts(chosen,ri,ei,el,multiplicity)
                for k,method in enumerate(policies):record[endpoint+'__'+method]=float(numerator[k]/denominator) if denominator else np.nan
            records.append(record)
            if (rep+1)%250==0:print('Bootstrap',family,rep+1,'/',reps,flush=True)
        draws=pd.DataFrame(records);draws.to_parquet(output/'bootstrap_draws.parquet',index=False)
        contrasts=[]
        for endpoint in endpoints:
            for control in ['elastic_net','xgboost','single_window','same_window_control']:
                differences=100*(draws[endpoint+'__multi_window']-draws[endpoint+'__'+control])
                lower,upper=np.quantile(differences,[.0125,.9875])
                point={r['method']:r['capture_percent'] for r in points if r['endpoint']==endpoint}
                contrasts.append(dict(endpoint=endpoint,proposed='multi_window',comparator=control,
                    comparison_family='standard_controls' if control in ['elastic_net','xgboost'] else 'window_contribution',
                    gain_pp=point['multi_window']-point[control],lower97_5=float(lower),upper97_5=float(upper),
                    usable_replicates=int(differences.notna().sum())))
        pd.DataFrame(contrasts).to_csv(output/'method_comparisons.csv',index=False)
    return points
