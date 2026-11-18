"""Repeated-visit scheduling selected from chronological US development predictions."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .paths import CONFIG, DATA, RESULTS
from .io import read, write
from .evaluation import opportunity_index, selected_mask

EXPERT_COLUMNS = ['matched_xgboost','xgb4__y_1_14','xgb4__y_15_30',
                  'xgb4__long_seed1','xgb4__long_seed2','elastic_net','xgboost']
SOURCE = RESULTS/'us/development'


def source_experts(predictions_root=None):
    path = SOURCE/'expert_predictions.parquet'
    if predictions_root is not None or not path.exists():
        from .training import development, fit_experts
        frame, _ = development()
        output = Path(predictions_root) if predictions_root is not None else SOURCE
        parts = []
        for year in range(2016,2021):
            for fold in range(5):
                frames = []
                for variant, keys in [('original',['xgb2__y_31_60']),
                    ('seasonal',['xgb4__y_31_60','xgb4__y_1_14','xgb4__y_15_30',
                                 'xgb4__long_seed1','xgb4__long_seed2','lr4'])]:
                    values = fit_experts(frame,year,(fold,),variant,keys,
                        output/'crossfit'/variant/str(year)/str(fold))
                    frames.append(values[['candidate_row_key',*keys]])
                parts.append(frames[0].merge(frames[1],on='candidate_row_key',validate='one_to_one'))
        values = pd.concat(parts,ignore_index=True).rename(columns={
            'xgb2__y_31_60':'xgboost','lr4':'elastic_net','xgb4__y_31_60':'matched_xgboost'})
        values.to_parquet(path,index=False)
    frame = pd.read_parquet(path)
    if not frame.candidate_row_key.is_unique or not np.isfinite(frame[EXPERT_COLUMNS]).all().all():
        raise ValueError('Incomplete current source expert scores')
    return frame


def source_frame():
    spec = read(CONFIG/'monitoring_models.json')
    frame = pd.read_parquet(DATA/'processed/us_training/common_development_table.parquet',
        columns=['candidate_row_key','lake_id','origin_date','origin_year',
                 'eligible_common_development_candidate'])
    frame['lake_id'] = frame.lake_id.astype(str)
    frame = frame[frame.eligible_common_development_candidate & frame.origin_year.between(2016,2020)]
    joint = pd.concat([pd.read_parquet(SOURCE/(s+'_predictions.parquet'))
                      for s in ['discovery','stability']],ignore_index=True)
    joint = joint[['candidate_row_key',*[s['name'] for s in spec['corrections']]]]
    joint = joint.rename(columns={s['name']:s['policy_name'] for s in spec['corrections']})
    frame = frame.merge(joint,on='candidate_row_key',validate='one_to_one')
    frame = frame.merge(source_experts(),on='candidate_row_key',validate='one_to_one')
    return frame.sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)


def stage_opportunities(frame, spec):
    events = pd.read_parquet(DATA/'processed/us_training/development_episodes.parquet')
    events['lake_id'] = events.lake_id.astype(str)
    groups = {}
    # A continuous allocation history is used; event evaluation has stage-specific maturity.
    for stage, first, last in [('discovery',2016,2018),('stability',2019,2020)]:
        eligible = events[events.event_date.dt.year.between(first,last) &
                          events.event_date.add(pd.Timedelta(days=7)).lt(pd.Timestamp(last+1,1,1))]
        # Reference candidate opportunities are restricted to the stage, matching source assessment.
        for endpoint, (lo,hi) in spec['endpoints'].items():
            stage_frame = frame[frame.origin_year.between(first,last)]
            ledger,ri,ei = opportunity_index(stage_frame,eligible,lo,hi)
            groups[(stage,endpoint)] = (ledger,ri,ei)
    return groups


def summarize(mask, opportunities):
    from .us_analysis import event_hits
    rows = []
    for (stage,endpoint),(ledger,ri,ei) in opportunities.items():
        hits = event_hits(mask,ri,ei,len(ledger))
        for year in ['ALL', *sorted(ledger.event_date.dt.year.unique())]:
            use = np.ones(len(ledger),bool) if year=='ALL' else ledger.event_date.dt.year.eq(year).to_numpy()
            rows.append(dict(stage=stage,endpoint=endpoint,year=str(year),events=int(use.sum()),
                captured=int(hits[use].sum()),capture_percent=100*hits[use].mean()))
    return rows


def selection_key(rows, name):
    g = [r for r in rows if r['stage']=='discovery']
    primary = next(r['capture_percent'] for r in g if r['endpoint']=='31_60' and r['year']=='ALL')
    worst = min(r['capture_percent'] for r in g if r['endpoint']=='31_60' and r['year']!='ALL')
    secondary = next(r['capture_percent'] for r in g if r['endpoint']=='1_60' and r['year']=='ALL')
    return (-primary,-worst,-secondary,name)




def run_schedule(evaluation=None):
    from .us_analysis import OUT, event_hits
    from .us_structure import repeat_adjusted_selection
    started = time.perf_counter()
    spec = read(CONFIG/'analysis.json')
    frame = source_frame()
    ops = stage_opportunities(frame,spec)
    methods = ['xgboost','elastic_net','matched_xgboost','single_window','same_window_control','multi_window']
    searches, selected, results = [], {}, []
    for method in methods:
        trials = []
        for power in [0.,*spec['diagnostics']['repeat_penalty_powers']]:
            mask = repeat_adjusted_selection(frame,method,.1,power,spec['diagnostics']['repeat_interval_days'])
            rows = summarize(mask,ops)
            name = 'repeat_'+format(power,'.2f')
            searches.extend(dict(experiment='repeat_scheduling',method=method,parameter=power,**r) for r in rows)
            trials.append((selection_key(rows,name),power,rows))
        _,power,rows = min(trials)
        selected['repeat_'+method] = dict(experiment='repeat_scheduling',method=method,power=power)
        results.extend(dict(experiment='repeat_scheduling',method=method,**r) for r in rows)
        print('US_SOURCE_REPEAT_SELECTED',method,power,flush=True)
    pd.DataFrame(searches).to_csv(SOURCE/'schedule_development.csv',index=False)
    pd.DataFrame(results).to_csv(SOURCE/'schedule_validation.csv',index=False)
    write(SOURCE/'selected_schedule.json',dict(selection_period='2016-2018',stability_period='2019-2020',
        model_family='joint',policies=selected))
    if evaluation is None:
        return
    target, target_ops = evaluation
    rows, predictions = [], target[['candidate_row_key']].copy()
    for name,s in selected.items():
        mask = repeat_adjusted_selection(target,s['method'],.1,s['power'],spec['diagnostics']['repeat_interval_days'])
        predictions[name+'__selected'] = mask
        for endpoint in spec['endpoints']:
            ledger,ri,ei = target_ops[('absolute75',endpoint)]
            hits = event_hits(mask,ri,ei,len(ledger))
            for year in ['ALL',*sorted(ledger.event_date.dt.year.unique())]:
                use = np.ones(len(ledger),bool) if year=='ALL' else ledger.event_date.dt.year.eq(year).to_numpy()
                rows.append(dict(policy=name,experiment=s['experiment'],endpoint=endpoint,year=str(year),
                    events=int(use.sum()),captured=int(hits[use].sum()),capture_percent=100*hits[use].mean()))
    pd.DataFrame(rows).to_csv(OUT/'scheduled_event_coverage.csv',index=False)
    predictions.to_parquet(OUT/'scheduled_selections.parquet',index=False)

