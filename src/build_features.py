"""Build each development feature table directly from source observations."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from .datasets import input_paths
from .paths import CONFIG
from . import biomass_features,meteorology_features

def default_array(name,size):
    if name.endswith('_date'):return np.full(size,np.datetime64('NaT'),dtype='datetime64[ns]')
    if name.endswith(('_source_record_days','_valid_value_days','_positive_days')):return np.zeros(size,np.float32)
    if '_days_since_' in name:return np.full(size,-1,np.float32)
    return np.full(size,np.nan,np.float32)

def source_features(work,names,sources):
    values={name:default_array(name,len(work)) for name in names}
    for prefix,history,available,observed,value,quality,censor,sites,threshold in sources:
        history['lake_id']=history.lake_id.astype(str)
        groups=dict(tuple(history.groupby('lake_id',sort=False)))
        wanted=[name for name in names if name.startswith(prefix+'_')]
        for index,(lake,group) in enumerate(work.groupby('lake_id',sort=False)):
            h=groups.get(lake)
            if h is None:continue
            computed=biomass_features.compute_source_features(source=prefix,origins=group.origin_date.to_numpy(),history=h,
                availability_col=available,observation_col=observed,value_col=value,quality_col=quality,
                censor_col=censor,site_count_col=sites,positive_threshold=threshold,windows=(14,30,60,90))
            for name in wanted:values[name][group.index.to_numpy()]=computed[name]
            for name,dates in computed.items():
                if name.endswith(('_availability_date','_observation_date')) and (pd.to_datetime(dates)>=group.origin_date.to_numpy()).any():
                    raise ValueError('Feature reads an unavailable source: '+name)
            if (index+1)%500==0:print('Feature lakes',prefix,index+1,flush=True)
        del groups;gc.collect()
    return pd.concat([work,pd.DataFrame(values)],axis=1)

def build(kind):
    paths=input_paths();spec=pd.read_csv(CONFIG/'features.csv')
    work=pd.read_parquet(paths['development_candidates'],columns=['lake_id','origin_date'])
    work['lake_id']=work.lake_id.astype(str);work=work.reset_index(drop=True)
    asset='development_'+kind+'_features';names=spec.loc[spec.source.eq(asset),'feature_name'].tolist()
    if kind=='static':
        attributes=pd.read_parquet(paths['lake_attributes']);attributes['lake_id']=attributes.lake_id.astype(str)
        result=work.merge(attributes,on='lake_id',validate='many_to_one')
        result['origin_dayofyear_sin']=np.sin(2*np.pi*result.origin_date.dt.dayofyear/366)
        result['origin_dayofyear_cos']=np.cos(2*np.pi*result.origin_date.dt.dayofyear/366)
        result=result[['lake_id','origin_date',*names]]
    elif kind in ['water_quality','biomass']:
        history=pd.read_parquet(paths['development_versioned_wq_history'])
        history['available_date']=history.sample_date+pd.Timedelta(days=7)
        if kind=='water_quality':
            sources=[('wq_'+v,history[history.variable.eq(v)].copy(),'available_date','sample_date','value_median',
                     'record_quality_fraction','censored_fraction_among_numeric_valid','site_count',0)
                     for v in ['tn','tp','do','ph','conductivity','turbidity','secchi_depth']]
        else:
            field=history[history.variable.eq('chla')].copy()
            satellite=pd.read_parquet(paths['development_satellite_raster_history'])
            satellite['nasa_cyan_week_end']=pd.to_datetime(satellite.nasa_cyan_week_end)
            satellite['available_date']=satellite.nasa_cyan_week_end+pd.Timedelta(days=2)
            satellite['quality']=satellite.nasa_cyan_valid.astype(float)
            satellite['value']=pd.to_numeric(satellite.nasa_cyan_ci).where(satellite.nasa_cyan_valid.eq(1))
            sources=[('field_chla',field,'available_date','sample_date','value_median','record_quality_fraction',
                      'censored_fraction_among_numeric_valid','site_count',25),
                     ('satellite_cyan',satellite,'available_date','nasa_cyan_week_end','value','quality',None,None,0)]
        result=source_features(work,names,sources)
    else:
        mapping=pd.read_parquet(paths['development_era5_grid_mapping'],columns=['lake_id','era5_grid_id']);mapping['lake_id']=mapping.lake_id.astype(str)
        joined=work.merge(mapping,on='lake_id',how='left',validate='many_to_one')
        if joined.era5_grid_id.isna().any():raise ValueError('A lake has no ERA5 grid')
        allocated=meteorology_features.allocate_outputs(len(work));values={name:allocated[name] for name in names}
        del allocated
        groups={str(g):x.index.to_numpy() for g,x in joined.groupby('era5_grid_id',sort=False)}
        columns=['era5_grid_id','date']+sorted({s['source'] for s in meteorology_features.VARIABLE_SPECS.values()})
        seen=set()
        for grid,daily in meteorology_features.iter_contiguous_grid_frames(Path(paths['meteorology_daily']),set(groups),columns):
            indices=groups[grid];unique=joined.iloc[indices].origin_date.drop_duplicates().sort_values()
            computed=meteorology_features.compute_group_features(unique.to_numpy(),daily)
            positions=pd.Index(unique).get_indexer(joined.iloc[indices].origin_date)
            for name in names:values[name][indices]=np.asarray(computed[name])[positions]
            seen.add(grid)
        if seen!=set(groups):raise ValueError('Missing ERA5 grids')
        result=pd.concat([work,pd.DataFrame(values)],axis=1)
    result.to_parquet(paths[asset],index=False)
    print('Feature table',kind,len(result),len(names),flush=True)

def observations():
    from .water_quality import RAW_COLUMNS,prepare_raw,aggregate_site_date,aggregate_lake_date
    from .endpoints import load_chla_observations,build_episode_starts
    paths=input_paths()
    raw=pd.read_parquet(paths['water_quality_raw'],columns=RAW_COLUMNS)
    lake=aggregate_lake_date(aggregate_site_date(prepare_raw(raw)))
    lake.to_parquet(paths['development_versioned_wq_history'],index=False)
    chla,_=load_chla_observations(Path(paths['development_versioned_wq_history']))
    events=build_episode_starts(chla,value_column='value_median',aggregation='lake_day_median_primary',washout_days=60)
    chla.to_parquet(paths['development_chla_observation_ledger'],index=False)
    events.to_parquet(paths['primary_incident_episode_ledger'],index=False)
    print('Lake-day observations',len(lake),'independent events',len(events),flush=True)

def seasonal_history():
    from . import seasonal
    from .paths import DATA
    paths=input_paths();meta=pd.read_parquet(paths['development_candidates'],columns=['candidate_row_key','lake_id','origin_date'])
    obs=seasonal.observations(Path(paths['development_chla_observation_ledger']))
    groups=dict(tuple(obs.groupby('lake_id',sort=False)))
    matrix=np.full((len(meta),len(seasonal.FEATURES)),np.nan,np.float32)
    for lake,group in meta.groupby('lake_id',sort=False):
        source=groups.get(str(lake))
        days=np.array([],dtype='datetime64[D]') if source is None else source.date.to_numpy(dtype='datetime64[D]')
        values=np.array([],float) if source is None else source.chla.to_numpy(float)
        matrix[group.index]=seasonal.lake_features(group.origin_date.to_numpy(dtype='datetime64[D]'),days,values)
    target=DATA/'processed/us_training/seasonal_history.npy'
    target.parent.mkdir(parents=True,exist_ok=True)
    np.save(target,matrix)
