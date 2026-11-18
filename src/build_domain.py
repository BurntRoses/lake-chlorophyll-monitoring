"""Construct the complete prior-available development candidate domain."""
import numpy as np
import pandas as pd
from .paths import DATA,CONFIG
from .datasets import input_paths
from .io import read
from .endpoints import build_candidate_ledger
def main():
    paths=input_paths()
    observations=pd.read_parquet(paths['development_chla_observation_ledger'])
    episodes=pd.read_parquet(paths['primary_incident_episode_ledger'])
    attributes=pd.read_parquet(paths['lake_attributes'])
    for frame in [observations,episodes,attributes]:frame['lake_id']=frame.lake_id.astype(str)
    first=observations.groupby('lake_id').assumed_result_available_date.min()
    schedule=read(CONFIG/'schedule.json')
    dates=pd.date_range(schedule['first_origin'],schedule['last_origin'],freq=str(schedule['spacing_days'])+'D')
    risk=pd.concat([pd.DataFrame({'lake_id':lake,'origin_date':dates[dates>first.loc[lake]]})
                    for lake in sorted(attributes.lake_id)],ignore_index=True)
    candidate=build_candidate_ledger(risk,observations,episodes)
    candidate['origin_date']=candidate.origin_date.astype('datetime64[us]')
    candidate.to_parquet(paths['development_candidates'],index=False)
    print('Development candidates:',len(candidate),'eligible:',int(candidate.eligible_common_development_candidate.sum()),flush=True)
