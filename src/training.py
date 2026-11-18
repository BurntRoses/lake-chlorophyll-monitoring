"""Chronological and lake-excluded expert fitting and coverage learning."""
from pathlib import Path
import gc
import itertools
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from .paths import CONFIG, DATA, MODELS
from .io import read,write,digest
from . import experts, seasonal
from .coverage import (HistogramCoverageBooster, LogCoverageContext, JointContext,
                       window_context, pooled_features, head_features, infer_cpu)

IDENTITY=['candidate_row_key','lake_id','origin_date']
PRIMITIVES=['xgb4__y_31_60','xgb4__y_1_14','xgb4__y_15_30','xgb4__long_seed1','xgb4__long_seed2']


def development():
    frame=pd.read_parquet(DATA/'processed/us_training/common_development_table.parquet')
    keys=pd.read_parquet(DATA/'derived/development_candidates.parquet',columns=['candidate_row_key'])
    if not frame.candidate_row_key.equals(keys.candidate_row_key):raise ValueError('Seasonal feature identities differ')
    values=np.load(DATA/'processed/us_training/seasonal_history.npy',mmap_mode='r')
    if values.shape != (len(frame),len(seasonal.FEATURES)):raise ValueError('Seasonal array shape differs')
    frame=pd.concat([frame,pd.DataFrame(values,columns=seasonal.FEATURES)],axis=1)
    frame['lake_id']=frame.lake_id.astype(str)
    events=pd.read_parquet(DATA/'processed/us_training/development_episodes.parquet')
    events['lake_id']=events.lake_id.astype(str)
    return frame,events


def fit_experts(frame,year,excluded,variant,keys,output):
    """Train before January 1 and exclude every lake in the requested folds."""
    cutoff=pd.Timestamp(year,1,1)
    mask=(frame.origin_year.lt(year) & ~frame.outer_lake_fold.isin(excluded) &
          frame.common_trainable & frame.label_mature_date.lt(cutoff))
    validation=frame.origin_year.eq(year) & frame.outer_lake_fold.isin(excluded) & frame.eligible_common_development_candidate
    train=frame.loc[mask].reset_index(drop=True);val=frame.loc[validation].reset_index(drop=True)
    if train.empty or val.empty or set(train.lake_id)&set(val.lake_id):raise ValueError('Invalid lake/time split')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'predictions.parquet'
    if path.exists():
        result=pd.read_parquet(path)
        if not result.candidate_row_key.equals(val.candidate_row_key):raise ValueError('Auxiliary prediction identities differ')
        if set(keys)<=set(result):return result
    spec,config=experts.feature_setup(variant)
    expression_path=output/'common_expression.joblib'
    if expression_path.exists():expression=joblib.load(expression_path)
    else:
        expression=experts.CommonExpression(spec,config).fit(train)
        joblib.dump(expression,expression_path)
    X=experts.transform_batched(expression,train);V=experts.transform_batched(expression,val)
    task_map={x[0]:x for x in experts.tasks()}
    result=val[IDENTITY].copy()
    for key in keys:result[key]=experts.fit_task(train,X,V,task_map[key],output)
    result.to_parquet(path,index=False)
    write(output/'split.json',dict(year=year,excluded_folds=list(excluded),variant=variant,
        training_rows=len(train),validation_rows=len(val),last_label_maturity=str(train.label_mature_date.max())))
    return result


def context_predictions(frame,year,fold,variant,keys,output):
    """Each correction row is scored by earlier-year experts excluding its lake fold.

    For outer validation, both its fold and the correction row's fold are excluded.
    """
    parts=[]
    for prior_year in range(2013,year):
        for row_fold in range(5):
            if row_fold==fold:continue
            excluded=tuple(sorted({row_fold} if fold is None else {fold,row_fold}))
            path=Path(output)/variant/str(prior_year)/('_'.join(map(str,excluded)))
            prediction=fit_experts(frame,prior_year,excluded,variant,keys,path)
            expected=frame.loc[frame.origin_year.eq(prior_year)&frame.outer_lake_fold.eq(row_fold)&
                               frame.eligible_common_development_candidate,'candidate_row_key']
            parts.append(prediction[prediction.candidate_row_key.isin(expected)])
    result=pd.concat(parts,ignore_index=True)
    if not result.candidate_row_key.is_unique:raise ValueError('Repeated nested expert predictions')
    return result.set_index('candidate_row_key')


def fit_correction(frame,events,raw,predictions,spec,family,cutoff,path,device='cuda'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    values=predictions[spec['expert_keys']].to_numpy(float)
    logits=logit(np.clip(values,1e-6,1-1e-6))
    if family=='joint':
        contexts=[window_context(frame,events,cutoff,label,logits[:,head])[0]
                  for head,label in enumerate(spec['target_windows'])]
        context=JointContext(contexts,spec['auxiliary_weight'])
        features=pooled_features(raw,logits);base=logits.T.reshape(-1)
    else:
        # Primary-window coverage uses supported incident opportunities from mature labels.
        labels=np.where(frame.common_trainable & frame.label_mature_date.lt(cutoff)&frame.y_31_60.ge(0),
                        frame.y_31_60.to_numpy(float),np.nan)
        ri=np.flatnonzero(labels==1.)
        ids=frame.iloc[ri].next_episode_id
        event_index,_=pd.factorize(ids,sort=False)
        context=LogCoverageContext(pd.factorize(frame.origin_date,sort=False)[0],ri,event_index,
                                   labels,logits[:,0],**spec['context_config'])
        features=np.column_stack([raw,logits]).astype(np.float32);base=logits[:,0]
    model=HistogramCoverageBooster(**spec['learner_config'],device=device).fit(features,base,context)
    model.save(path)
    return model


def refit(output,device='cuda'):
    output=Path(output).resolve()
    if output==(MODELS/'us').resolve():raise ValueError('Choose a separate output directory for fitting')
    frame,events=development();cutoff=pd.Timestamp('2021-01-01')
    train=frame[frame.common_trainable & frame.label_mature_date.lt(cutoff)&frame.origin_date.lt(cutoff)].reset_index(drop=True)
    context=frame[frame.origin_year.ge(2013)&frame.origin_date.lt(cutoff)&frame.eligible_common_development_candidate]
    context=context.sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)
    if len(train)!=136122 or len(context)!=573461:raise ValueError('Final training cohorts differ')
    configurations={'joint':read(CONFIG/'monitoring_models.json'),'primary_only':read(CONFIG/'ablation_models.json')}
    for variant in ['original','seasonal']:
        specs=[s for config in configurations.values() for s in config['corrections'] if s['variant']==variant]
        keys=sorted({k for s in specs for k in s['expert_keys']}|{s['expert_key'] for config in configurations.values() for s in config['standard_controls'] if s['variant']==variant})
        directory=output/'experts'/variant;directory.mkdir(parents=True,exist_ok=True)
        spec,config=experts.feature_setup(variant)
        expression=experts.CommonExpression(spec,config).fit(train)
        joblib.dump(expression,directory/'common_expression.joblib')
        X=experts.transform_batched(expression,train);tasks={t[0]:t for t in experts.tasks()}
        for key in keys:experts.fit_task(train,X,X[:8],tasks[key],directory)
        del X;gc.collect()
        if not specs:continue
        probabilities=context_predictions(frame,2021,None,variant,keys,output/'crossfit')
        probabilities=probabilities.loc[context.candidate_row_key].reset_index()
        raw=experts.transform_batched(expression,context)
        for family,config in configurations.items():
            folder=output/('coverage' if family=='joint' else 'ablations')
            for correction in config['corrections']:
                if correction['variant']!=variant:continue
                name=correction['policy_name']
                fit_correction(context,events,raw,probabilities,correction,family,cutoff,folder/(name+'.joblib'),device)
                print('Fitted coverage model',family,name,flush=True)
    train[['candidate_row_key']].to_parquet(output/'training_keys.parquet',index=False)
    context[['candidate_row_key']].to_parquet(output/'context_keys.parquet',index=False)
    write(output/'checksums.json',{p.relative_to(output).as_posix():digest(p) for directory in ['experts','coverage','ablations'] for p in (output/directory).rglob('*') if p.suffix in ['.ubj','.joblib']})


def validate(output,device='cuda',years=range(2016,2021),folds=range(5)):
    frame,events=development();output=Path(output)
    specs=read(CONFIG/'monitoring_models.json')['corrections']
    for year in years:
        for fold in folds:
            cutoff=pd.Timestamp(year,1,1)
            train=frame[frame.origin_year.lt(year)&frame.outer_lake_fold.ne(fold)&frame.common_trainable&frame.label_mature_date.lt(cutoff)]
            context=frame[frame.origin_year.between(2013,year-1)&frame.outer_lake_fold.ne(fold)&frame.eligible_common_development_candidate]
            context=context.sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)
            val=frame[frame.origin_year.eq(year)&frame.outer_lake_fold.eq(fold)&frame.eligible_common_development_candidate].sort_values(['origin_date','lake_id'],kind='stable').reset_index(drop=True)
            keys=sorted({k for s in specs for k in s['expert_keys']})
            feature_spec,config=experts.feature_setup('seasonal')
            expression=experts.CommonExpression(feature_spec,config).fit(train)
            X=experts.transform_batched(expression,context);V=experts.transform_batched(expression,val)
            inner=context_predictions(frame,year,fold,'seasonal',keys,output/'crossfit').loc[context.candidate_row_key].reset_index()
            outer=fit_experts(frame,year,(fold,),'seasonal',keys,output/'crossfit/seasonal'/str(year)/str(fold)).set_index('candidate_row_key').loc[val.candidate_row_key].reset_index()
            for spec in specs:
                folder=output/'fits'/str(year)/str(fold)/spec['policy_name']
                model=fit_correction(context,events,X,inner,spec,'joint',cutoff,folder/'model.joblib',device)
                logits=logit(np.clip(outer[spec['expert_keys']].to_numpy(float),1e-6,1-1e-6))
                prediction=val[IDENTITY].copy();prediction['score']=infer_cpu(model,head_features(V,logits),logits[:,0])
                prediction.to_parquet(folder/'predictions.parquet',index=False)
                print('US validation',year,fold,spec['policy_name'],flush=True)
