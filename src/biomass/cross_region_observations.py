"""Read fixed satellite ZIP and LAWA workbook into lake-day observations."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import openpyxl
from .paths import DATA, CONFIG

RAW = DATA / 'raw/cross_region/lawa_nz/raw/lawa-lake-quality-dataset__16dec2025.xlsx'
STOP = pd.Timestamp('2025-01-01')

def field_observations():
    workbook = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    iterator = workbook['Monitoring Dataset (2004-24)'].iter_rows(values_only=True)
    ix = {v: i for i, v in enumerate(next(iterator))}
    rows = []; reasons = {}
    for row in iterator:
        if row[ix['Indicator']] != 'Chl-a':
            continue
        date = pd.Timestamp(row[ix['SampleDateTime']])
        if pd.isna(date) or date >= STOP:
            continue
        if str(row[ix['Units']]) != 'mg/m3':
            raise RuntimeError('Unexpected field Chl-a unit')
        quality = str(row[ix['QCNEMSEquivalent']]); q = pd.to_numeric(quality, errors='coerce')
        unknown = quality in ['NA', 'None', 'nan', '']
        if not unknown and not (500 <= q < 700):
            reasons['known_compromised_or_invalid'] = reasons.get('known_compromised_or_invalid', 0) + 1
            continue
        if str(row[ix['Symbol']]) not in ['NA', 'None', 'False', 'false', '', 'nan']:
            reasons['censored_or_qualified'] = reasons.get('censored_or_qualified', 0) + 1
            continue
        value = pd.to_numeric(row[ix['Value']], errors='coerce')
        if pd.isna(value) or not np.isfinite(value) or value < 0:
            reasons['missing_or_invalid_value'] = reasons.get('missing_or_invalid_value', 0) + 1
            continue
        lid = str(row[ix['LFENZID']]).removesuffix('.0')
        rows.append({'lake_id': 'LAWA_' + lid, 'station_id': str(row[ix['LawaSiteID']]),
            'date': date.normalize(), 'chla': float(value), 'lat': float(row[ix['Latitude']]),
            'lon': float(row[ix['Longitude']]), 'quality_unknown': unknown})
    workbook.close()
    samples = pd.DataFrame(rows)
    station = samples.groupby(['lake_id', 'station_id', 'date'], as_index=False).agg(
        chla=('chla', 'median'), lat=('lat', 'median'), lon=('lon', 'median'),
        quality_unknown=('quality_unknown', 'any'), samples=('chla', 'size'))
    result = station.groupby(['lake_id', 'date'], as_index=False).agg(chla=('chla', 'median'),
        lat=('lat', 'median'), lon=('lon', 'median'), samples=('samples', 'sum'),
        stations=('station_id', 'nunique'), quality_unknown=('quality_unknown', 'any'))
    result = result.sort_values(['lake_id', 'date']).reset_index(drop=True)
    return result


def satellite_observations():
    base=DATA/'raw/cross_region/cci_v2';meta=pd.read_csv(base/'lake_description.csv');lookup=meta.set_index('hylak_id');parts=[]
    z=zipfile.ZipFile(base/'daily_lakes.zip');keep=['date','chla_mean_mean','chla_mean_coverage_ratio','lake_surface_water_temperature_mean','turbidity_mean_mean']
    geography=json.loads((CONFIG/'geography.json').read_text(encoding='utf-8'))
    for j,name in enumerate(z.namelist()):
        lid=int(Path(name).stem);m=lookup.loc[lid]
        if pd.isna(m.country):continue
        d=pd.read_csv(z.open(name),usecols=keep)
        # Coverage is expressed in percent by the source dataset.
        # Full-series temporal QA and long-term mean columns are never used.
        d=d[d.chla_mean_coverage_ratio.ge(10)&d.chla_mean_mean.ge(0)].copy()
        if not len(d):continue
        d=d.rename(columns={'chla_mean_mean':'chla','lake_surface_water_temperature_mean':'temperature','turbidity_mean_mean':'turbidity'})
        d['date']=pd.to_datetime(d.date,errors='raise');d['lake_id']='CCI_'+str(lid);d['country']=str(m.country);d['continent']=geography.get(str(m.country))
        d['lake_name']='HydroLAKES_'+str(lid);d['lat']=m.central_lat_wgs84_dd;d['lon']=m.central_lon_wgs84_dd
        d['area']=np.nan;d['depth']=np.nan;d['tp']=np.nan;d['tn']=np.nan;d['known_surface_depth']=True
        parts.append(d.drop(columns='chla_mean_coverage_ratio'))
        if (j+1)%1000==0:print('CCI_LAKES_READ',j+1,flush=True)
    obs=pd.concat(parts,ignore_index=True).sort_values(['lake_id','date']).reset_index(drop=True)
    assert not obs.duplicated(['lake_id','date']).any()
    obs['continent']=obs.country.map(geography)
    if obs.continent.isna().any():raise ValueError('Unknown country allocation')
    return obs


def build_observations():
    folder = DATA / 'derived'
    folder.mkdir(parents=True, exist_ok=True)
    for name, builder in [('satellite', satellite_observations), ('field', field_observations)]:
        frame = builder()
        frame.to_parquet(folder / ('cross_region_' + name + '_observations.parquet'), index=False)
        print('REGIONAL_OBSERVATIONS', name, len(frame), frame.lake_id.nunique(), flush=True)
