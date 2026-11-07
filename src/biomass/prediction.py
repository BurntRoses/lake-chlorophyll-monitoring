"""Predict with the released experts and coverage corrections."""
from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from .paths import CONFIG, DATA, MODELS, RESULTS
from .io import read, digest
from . import seasonal
from .experts import transform_batched
from .coverage import FirstOrderCoverageBooster, infer_cpu, head_features


US_MODELS = MODELS / 'us'
MODEL_CONFIG = {'joint':'monitoring_models.json','primary_only':'ablation_models.json'}
MODEL_FOLDER = {'joint':'coverage','primary_only':'ablations'}


def verify_models(model_root=US_MODELS):
    model_root=Path(model_root)
    for name, expected in read(model_root/'checksums.json').items():
        if digest(model_root/name) != expected:
            raise ValueError('Model checksum differs: '+name)


def predict_experts(frame, directory, keys):
    expression=joblib.load(directory/'common_expression.joblib')
    matrix=transform_batched(expression,frame)
    values={}
    data=xgb.DMatrix(matrix,nthread=2)
    for key in keys:
        if (directory/(key+'.joblib')).exists():
            model=joblib.load(directory/(key+'.joblib'))
            if not model.converged: raise ValueError('Elastic-net KKT convergence failed')
            values[key]=model.predict_proba(matrix)[:,1]
        else:
            model=xgb.Booster();model.load_model(directory/(key+'.ubj'))
            values[key]=model.predict(data)
        if not np.isfinite(values[key]).all():raise ValueError('Nonfinite expert: '+key)
    return matrix,values


def predict(frame, family='joint', model_root=US_MODELS):
    spec=read(CONFIG/MODEL_CONFIG[family])
    identities=[c for c in ['candidate_row_key','lake_id','origin_date','origin_year'] if c in frame]
    result=frame[identities].copy()
    for variant in sorted({x['variant'] for x in spec['corrections']+spec['standard_controls']}):
        corrections=[x for x in spec['corrections'] if x['variant']==variant]
        controls=[x for x in spec['standard_controls'] if x['variant']==variant]
        keys=sorted({k for x in corrections for k in x['expert_keys']}|{x['expert_key'] for x in controls})
        matrix,values=predict_experts(frame,model_root/'experts'/variant,keys)
        for control in controls:result[control['name']]=values[control['expert_key']]
        for correction in corrections:
            logits=logit(np.clip(np.column_stack([values[k] for k in correction['expert_keys']]).astype(float),1e-6,1-1e-6))
            features=head_features(matrix,logits) if family=='joint' else np.column_stack([matrix,logits]).astype(np.float32)
            name=correction['policy_name']
            model=FirstOrderCoverageBooster.load(model_root/MODEL_FOLDER[family]/(name+'.joblib'))
            result[name]=infer_cpu(model,features,logits[:,0])
        del matrix,values
        gc.collect()
    return result


def evaluation_inputs(year, rows=None):
    fields=pd.read_csv(CONFIG/'features.csv').feature_name.tolist()
    frame=pd.read_parquet(DATA/f'processed/us_evaluation/common_retrospective_table_{year}.parquet',
        columns=['candidate_row_key','lake_id','origin_date','origin_year',*fields],
        filters=[('eligible_external_candidate','==',True)])
    if rows is not None:frame=frame.head(rows)
    frame=frame.reset_index(drop=True);frame['lake_id']=frame.lake_id.astype(str)
    history=pd.read_parquet(DATA/'derived/retrospective_chla_history.parquet',
        columns=['lake_id','sample_date','value_median','field_available_date'])
    if not history.field_available_date.eq(history.sample_date+pd.Timedelta(days=7)).all():
        raise ValueError('Retrospective observation availability differs from seven days')
    history=history.rename(columns={'sample_date':'date','value_median':'chla'})
    history=history[np.isfinite(history.chla)&history.chla.ge(0)][['lake_id','date','chla']]
    history=history.sort_values(['lake_id','date'],kind='stable').reset_index(drop=True)
    history['lake_id']=history.lake_id.astype(str)
    groups=dict(tuple(history.groupby('lake_id',sort=False)))
    values=np.full((len(frame),len(seasonal.FEATURES)),np.nan,np.float32)
    for lake,group in frame.groupby('lake_id',sort=False):
        prior=groups.get(lake)
        days=np.array([],dtype='datetime64[D]') if prior is None else prior.date.to_numpy(dtype='datetime64[D]')
        chla=np.array([],float) if prior is None else prior.chla.to_numpy(float)
        values[group.index]=seasonal.lake_features(group.origin_date.to_numpy(dtype='datetime64[D]'),days,chla)
    return pd.concat([frame,pd.DataFrame(values,columns=seasonal.FEATURES,index=frame.index)],axis=1)


def replay(rows=None, years=range(2021,2026)):
    verify_models()
    expected=pd.read_parquet(RESULTS/'us/main/predictions.parquet').set_index('candidate_row_key')
    ablations=pd.read_parquet(RESULTS/'us/ablations/predictions.parquet').set_index('candidate_row_key')
    for year in years:
        frame=evaluation_inputs(year,rows)
        actual=predict(frame)
        columns=['xgboost','elastic_net','single_window','same_window_control','multi_window']
        delta=actual[columns].to_numpy()-expected.loc[actual.candidate_row_key,columns].to_numpy()
        error=float(np.max(np.abs(delta)))
        if not np.isfinite(delta).all() or error>1e-7:raise ValueError(f'US {year}: score error {error}')
        control=predict(frame,'primary_only')
        observed=control[['same_window_control','multi_window']].to_numpy()
        saved=ablations.loc[control.candidate_row_key,['ablation_same_window','ablation_multi_window']].to_numpy()
        ablation_error=float(np.max(np.abs(observed-saved)))
        if not np.isfinite(observed).all() or ablation_error>1e-7:raise ValueError(f'US ablation {year}: score error {ablation_error}')
        print(f'US {year}: {len(frame)} rows; maximum score error {max(error,ablation_error):.3g}',flush=True)


def score(model_root,output,years=range(2021,2026),rows=None):
    model_root=Path(model_root).resolve();output=Path(output).resolve()
    verify_models(model_root)
    main,ablations=[],[]
    for year in years:
        frame=evaluation_inputs(year,rows)
        main.append(predict(frame,model_root=model_root))
        a=predict(frame,'primary_only',model_root)
        ablations.append(a[['candidate_row_key','same_window_control','multi_window']].rename(columns={
            'same_window_control':'ablation_same_window','multi_window':'ablation_multi_window'}))
        print('Scored US',year,len(frame),flush=True)
    for name,parts in [('main',main),('ablations',ablations)]:
        folder=output/name;folder.mkdir(parents=True,exist_ok=True)
        pd.concat(parts,ignore_index=True).to_parquet(folder/'predictions.parquet',index=False)

