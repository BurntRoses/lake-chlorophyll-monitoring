#!/usr/bin/env python3
"""Evaluate monitoring strategies from committed scores and event outcomes."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / '04_analysis_data/primary_external_evaluation'
Q = 'multi_horizon'
D = 'direct_60_day'
H = 'historical_risk'
POLICIES = (Q, D, H)
SCORES = {Q: 'multi_horizon_score', D: D, H: H}
ENDPOINTS = {'primary_31_60': (31, 60), 'secondary_1_60': (1, 60)}
SEED, REPS, CHUNK = 20260804, 5000, 128
INPUTS = {
 'candidate_ledger': EXT/'candidate_features/external_candidate_ledger.parquet',
 'stage_scores': EXT/'multi_horizon_scores/external_outcome_blind_scores.parquet',
 'stage_queues': EXT/'multi_horizon_scores/frozen_external_queues.parquet',
 'direct_scores': EXT/'direct_60_day_scores/external_direct_60_day_outcome_blind_scores.parquet',
 'direct_queues': EXT/'direct_60_day_scores/frozen_external_direct_60_day_queues.parquet',
 'outcome_csv': EXT/'event_outcomes/external_incident_events.csv',
 'lake_day': EXT/'event_outcomes/combined_lake_day_2005_2025.parquet',
 'history': EXT/'candidate_features/external_predictor_chla_history.parquet',
 'protocol': EXT/'protocol/protocol.json',
 'freeze': EXT/'input_freeze/freeze.json',
 'evaluation_contract': EXT/'evaluation/evaluation_contract.json',
 'formal_points': EXT/'evaluation/point_estimates.csv',
 'formal_summary': EXT/'evaluation/endpoint_summary.csv',
 'formal_draws': EXT/'evaluation/bootstrap_draws.parquet',
 'old_capacity': ROOT/'04_analysis_data/sensitivity_analyses/selection_capacity/external_capacity_curve_summary.csv',
 'event_roles': ROOT/'03_figure_source_data/Figure_3/event_roles.csv',
 'overlap': ROOT/'03_figure_source_data/Figure_4/forecast_origin_overlap.csv',
 'lake_support': ROOT/'03_figure_source_data/Figure_1/lake_support.csv',
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20), b''): h.update(b)
 return h.hexdigest()

def dump(path, value):
 Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'04_analysis_data/strategy_evaluation');args=parser.parse_args()
 out=args.output.resolve();res=out;prov=ROOT/'07_metadata/provenance/evaluation'
 res.mkdir(parents=True,exist_ok=True);prov.mkdir(parents=True,exist_ok=True)
 started=time.monotonic()
 # Hash all consumed inputs before computation and verify the canonical data receipt.
 existing={str(p.relative_to(ROOT)):sha(p) for p in INPUTS.values()}
 manifest={k:{'relative_path':str(p.relative_to(ROOT)),'sha256':sha(p)} for k,p in INPUTS.items()}
 dump(prov/'input_manifest.json',manifest)
 receipt=json.loads((ROOT/'07_metadata/provenance/evaluation_inputs.json').read_text())
 audit=[]
 for k in ('candidate_ledger','stage_scores','stage_queues','direct_scores','direct_queues','outcome_csv','lake_day','protocol'):
  assert sha(INPUTS[k])==receipt[k]['sha256'],k
  audit.append({'check':'input_hash_'+k,'passed':True,'detail':receipt[k]['sha256']})
 plan=json.loads((ROOT/'07_metadata/analysis_plan.json').read_text())
 dump(prov/'executed_plan.json',plan)
 keys=['candidate_row_key','lake_id','origin_date']
 frame=pd.read_parquet(INPUTS['candidate_ledger'])
 frame['lake_id']=frame.lake_id.astype(str);frame['origin_date']=pd.to_datetime(frame.origin_date).dt.normalize()
 for policy,key,col in [(Q,'stage_scores','queue_selected__multi_horizon'),(D,'direct_scores','queue_selected__direct_60_day')]:
  s=pd.read_parquet(INPUTS[key],columns=keys+['eligible_external_candidate',SCORES[policy],col])
  s['lake_id']=s.lake_id.astype(str);s['origin_date']=pd.to_datetime(s.origin_date).dt.normalize()
  s=s.rename(columns={'eligible_external_candidate':'eligible_score',col:'frozen__'+policy})
  assert len(s)==len(frame)==369720 and not s.candidate_row_key.duplicated().any()
  frame=frame.merge(s,on=keys,validate='one_to_one',how='left')
  assert frame.eligible_score.equals(frame.eligible_external_candidate)
  frame=frame.drop(columns='eligible_score')
 assert not frame[keys].isna().any().any() and not frame.duplicated(['lake_id','origin_date']).any()
 assert frame.eligible_external_candidate.sum()==365726
 assert frame.lake_id.nunique()==2844 and frame.origin_date.nunique()==130
 assert np.isfinite(frame[[SCORES[Q],SCORES[D]]].to_numpy(float)).all()
 lakes=pd.Index(sorted(frame.lake_id.unique()));lakeidx=pd.Series(np.arange(len(lakes)),index=lakes)
 events=pd.read_csv(INPUTS['outcome_csv'],dtype={'episode_id':str,'lake_id':str})
 events['event_date']=pd.to_datetime(events.event_date).dt.normalize()
 events=events.sort_values('episode_id',kind='mergesort').reset_index(drop=True)
 assert len(events)==696 and events.lake_id.nunique()==355 and not events.episode_id.duplicated().any()
 assert events.lake_id.isin(lakes).all()
 event_lake=events.lake_id.map(lakeidx).to_numpy(int)
 # Static historical risk rule, using only already available pre-2021 days.
 hist=pd.read_parquet(INPUTS['history']);hist['lake_id']=hist.lake_id.astype(str)
 for c in ['sample_date','field_available_date']:hist[c]=pd.to_datetime(hist[c]).dt.normalize()
 assert not hist.duplicated(['lake_id','sample_date']).any()
 first=frame.origin_date.min()
 pre=hist.loc[(hist.sample_date<pd.Timestamp('2021-01-01'))&(hist.field_available_date<first)&hist.lake_id.isin(lakes)].copy()
 assert np.isfinite(pre.value_median.to_numpy(float)).all()
 pre['high']=pre.value_median.ge(75).astype(int)
 base=pre.groupby('lake_id').agg(history_day_n=('high','size'),high_chla_day_n=('high','sum'),latest_sample_date=('sample_date','max'),latest_available_date=('field_available_date','max')).reindex(lakes)
 base[['history_day_n','high_chla_day_n']]=base[['history_day_n','high_chla_day_n']].fillna(0).astype(int)
 base[H]=(base.high_chla_day_n/base.history_day_n.replace(0,np.nan)).fillna(0)
 base.index.name='lake_id';base=base.reset_index()
 assert (base.latest_available_date.dropna()<first).all()
 base.to_csv(res/'historical_risk_lake_scores.csv',index=False)
 frame[H]=frame.lake_id.map(base.set_index('lake_id')[H]);assert frame[H].notna().all()
 dump(prov/'baseline_availability_audit.json',{'rule':'sample_date < 2021-01-01 and field_available_date < first_origin; count(value_median >=75) / count(retained lake-days)', 'first_origin':str(first.date()),'history_rows':len(pre),'history_lakes':pre.lake_id.nunique(),'zero_history_candidate_lakes':int(base.history_day_n.eq(0).sum()),'latest_sample_date':str(pre.sample_date.max().date()),'latest_available_date':str(pre.field_available_date.max().date()),'post2020_history_rows_used':0,'future_information_used':False,'availability_semantics':'Frozen sample_date + 7 days convention; not verified provider publication timestamps','no_parameter_fitting':True})
 # Monitoring-support strata defined without inspecting captures, across monitored candidate lakes.
 obs=pd.read_parquet(INPUTS['lake_day']);obs['lake_id']=obs.lake_id.astype(str);obs['date']=pd.to_datetime(obs.date).dt.normalize()
 obs=obs.loc[obs.date.between('2021-01-01','2025-12-31')&obs.lake_id.isin(lakes)]
 assert not obs.duplicated(['lake_id','date']).any()
 support=obs.groupby('lake_id').size().reindex(lakes,fill_value=0).rename('observation_day_n').to_frame()
 cuts=support.loc[support.observation_day_n.gt(0),'observation_day_n'].quantile([1/3,2/3]).to_numpy()
 assert cuts[1]>cuts[0] and support.observation_day_n.gt(0).sum()==1679
 support['stratum']=np.select([support.observation_day_n.eq(0),support.observation_day_n.le(cuts[0]),support.observation_day_n.le(cuts[1])],['zero','low','medium'],default='high')
 support.index.name='lake_id';support=support.reset_index()
 stratum_map=support.set_index('lake_id').stratum
 events['support_stratum']=events.lake_id.map(stratum_map)
 support['event_n']=support.lake_id.map(events.groupby('lake_id').size()).fillna(0).astype(int)
 support.to_csv(res/'monitoring_support_lakes.csv',index=False)
 dump(prov/'monitoring_strata_definition.json',{'support_variable':'retained 2021-2025 field chlorophyll-a lake-days','tertile_domain':'1679 monitored candidate lakes, including lakes without events','cutpoints':cuts.tolist(),'quantile_method':'linear','interval_rule':'low: 0<n<=cut1; medium: cut1<n<=cut2; high: n>cut2; zero separate','ties_kept_together':True,'strata_fixed_across_bootstrap':True,'queues_restricted_to_strata':False,'observation_days':len(obs)})
 # Prepare legal links and full score orders; the original eligible universe is never filtered by support.
 records=[];selection_parts=[];origin_audit=[]
 captures={};capacity_rows=[];slots={}
 cap_grid=sorted(set(list(range(1,31))+[7.5]))
 for origin,g in frame.loc[frame.eligible_external_candidate].groupby('origin_date',sort=True):
  g=g.reset_index(drop=True);n=len(g)
  key_order=np.argsort(g.candidate_row_key.to_numpy(str),kind='mergesort')
  orders={p:key_order[np.argsort(-g[SCORES[p]].to_numpy(float)[key_order],kind='stable')] for p in POLICIES}
  lead=(events.event_date-origin).dt.days.to_numpy()
  legal=(lead>=1)&(lead<=60)&events.lake_id.isin(g.lake_id)
  ids=np.flatnonzero(legal);local=events.loc[legal,'lake_id'].map(pd.Series(np.arange(n),index=g.lake_id)).to_numpy(int)
  record={'origin':origin,'n':n,'lake':g.lake_id.map(lakeidx).to_numpy(int),'orders':orders,'event_ids':ids,'local_rows':local,'primary':lead[legal]>=31}
  records.append(record)
  k=max(1,int(np.ceil(.1*n)))
  for p in POLICIES:
   selected=np.zeros(n,dtype=bool);selected[orders[p][:k]]=True
   if p in (Q,D):assert np.array_equal(selected,g['frozen__'+p].to_numpy(bool)),(p,origin)
   sel=g.loc[selected,keys+[SCORES[p]]].copy().rename(columns={SCORES[p]:'score'})
   sel['policy']=p;sel['capacity']=k;selection_parts.append(sel)
  origin_audit.append({'origin_date':origin,'eligible_candidates':n,'capacity_10pct':k,'both_formal_queues_exact':True})
 assert len(records)==130
 selection=pd.concat(selection_parts,ignore_index=True)
 selection.to_parquet(res/'selection_records_10pct.parquet',index=False)
 pd.DataFrame(origin_audit).to_csv(res/'origin_capacity_audit.csv',index=False)
 for p,key,srcname in [(Q,'stage_queues','multi_horizon'),(D,'direct_queues',D)]:
  frozen=pd.read_parquet(INPUTS[key]);frozen=frozen.loc[frozen.policy.eq(srcname)]
  assert set(frozen.candidate_row_key)==set(selection.loc[selection.policy.eq(p),'candidate_row_key'])
 assert selection.groupby('policy').size().eq(36631).all()
 for c in cap_grid:
  for p in POLICIES:
   cap={e:np.zeros(len(events),dtype=bool) for e in ENDPOINTS};total=0
   for r in records:
    k=max(1,int(np.ceil((c/100)*r['n'])));total+=k
    selected=np.zeros(r['n'],dtype=bool);selected[r['orders'][p][:k]]=True
    for e in ENDPOINTS:
     mask=r['primary'] if e=='primary_31_60' else np.ones(len(r['event_ids']),bool)
     ids=r['event_ids'][mask];cap[e][ids]|=selected[r['local_rows'][mask]]
   for e,v in cap.items():
    captures[(c,p,e)]=v;capacity_rows.append({'capacity_percent':c,'capacity_fraction':c/100,'endpoint':e,'policy':p,'event_n':len(events),'captured_events':int(v.sum()),'coverage_percent':100*v.mean(),'selected_slots':total,'analysis_role':'formal_frozen_reference' if c==10 and p in (Q,D) else 'post_hoc_descriptive'})
 capacity=pd.DataFrame(capacity_rows)
 for p in POLICIES:
  for e in ENDPOINTS:
   z=capacity.loc[capacity.policy.eq(p)&capacity.endpoint.eq(e)].sort_values('capacity_percent')
   assert z.captured_events.diff().dropna().ge(0).all()
 formal=pd.read_csv(INPUTS['formal_points'])
 for x in formal.itertuples():assert int(captures[(10,x.policy,x.endpoint)].sum())==x.captured_events
 role=pd.read_csv(INPUTS['event_roles']).rename(columns={'event_id':'episode_id'})
 # Public Source Data renamed the literal prefix episode__ to events__; the 24-character identity is unchanged.
 role['episode_id']=role.episode_id.str.replace('events__','episode__',regex=False)
 assert set(role.episode_id)==set(events.episode_id)
 role=role.set_index('episode_id').reindex(events.episode_id)
 for p,prefix in [(Q,'multi_horizon'),(D,'direct_60_day')]:
  for e,suffix in [('primary_31_60','primary'),('secondary_1_60','secondary')]:
   assert np.array_equal(role[prefix+'_'+suffix].to_numpy(bool),captures[(10,p,e)])
 for e in ENDPOINTS:
  for p in POLICIES:events['captured__'+e+'__'+p]=captures[(10,p,e)]
 events.to_csv(res/'event_capture_10pct.csv',index=False)
 diffs=[];old=pd.read_csv(INPUTS['old_capacity'])
 for c in cap_grid:
  for e in ENDPOINTS:
   q=captures[(c,Q,e)].sum();d=captures[(c,D,e)].sum()
   row={'capacity_percent':c,'endpoint':e,'event_n':len(events),'multi_horizon_events':int(q),'direct_60_day_events':int(d),'difference_pp':100*(q-d)/len(events),'ci_low_pp':np.nan,'ci_high_pp':np.nan,'ci_source':'not_computed_descriptive_point'}
   if e=='primary_31_60' and c in [10,15,20,30]:
    reference=old.loc[np.isclose(old.capacity_fraction,c/100)].iloc[0]
    assert q==reference.multi_horizon_captured_events and d==reference.direct_60_day_captured_events
    row.update(ci_low_pp=reference.ci_low_pp,ci_high_pp=reference.ci_high_pp,ci_source='existing_formal_10pct' if c==10 else 'existing_capacity_sensitivity')
   diffs.append(row)
 capacity.to_csv(res/'capacity_coverage.csv',index=False)
 pd.DataFrame(diffs).to_csv(res/'capacity_differences.csv',index=False)
 # Adjacent Jaccard overlap, not between-policy same-origin overlap.
 turnover=[];lake_repeats=[];summ=[]
 for p in (Q,D):
  s=selection.loc[selection.policy.eq(p)]
  groups=[(t,set(g.lake_id)) for t,g in s.groupby('origin_date',sort=True)]
  for (t0,a),(t1,b) in zip(groups[:-1],groups[1:]):
   assert (t1-t0).days==14
   j=len(a&b)/len(a|b)
   turnover.append({'policy':p,'previous_origin':t0,'current_origin':t1,'gap_days':14,'previous_capacity':len(a),'current_capacity':len(b),'shared_lakes':len(a&b),'union_lakes':len(a|b),'jaccard_overlap':j,'jaccard_turnover':1-j})
  counts=s.groupby('lake_id').size().reindex(lakes,fill_value=0)
  positive=counts[counts>0]
  lake_repeats.extend({'policy':p,'lake_id':l,'selection_n':int(v),'ever_selected':bool(v)} for l,v in counts.items())
  j=np.array([x['jaccard_overlap'] for x in turnover if x['policy']==p])
  summ.append({'policy':p,'candidate_lake_n':len(lakes),'origin_n':130,'adjacent_pair_n':129,'total_selection_slots':len(s),'unique_selected_lakes':len(positive),'unique_lake_reach_percent':len(positive)/len(lakes)*100,'repeat_selection_median':positive.median(),'repeat_selection_lower_quartile':positive.quantile(.25),'repeat_selection_upper_quartile':positive.quantile(.75),'repeat_selection_mean':positive.mean(),'jaccard_mean':j.mean(),'jaccard_median':np.median(j),'jaccard_lower_quartile':np.quantile(j,.25),'jaccard_upper_quartile':np.quantile(j,.75),'jaccard_turnover_mean':1-j.mean()})
 turnover=pd.DataFrame(turnover);turnover.to_csv(res/'turnover_adjacent_origins.csv',index=False)
 pd.DataFrame(lake_repeats).to_csv(res/'turnover_lake_selection_counts.csv',index=False)
 pd.DataFrame(summ).to_csv(res/'turnover_summary.csv',index=False)
 old_support=pd.read_csv(INPUTS['lake_support'],dtype={'lake_id':str}).set_index('lake_id')
 for p,c in [(Q,'multi_horizon_ever_selected'),(D,'direct_60_day_ever_selected')]:
  new=pd.DataFrame(lake_repeats).query('policy == @p').set_index('lake_id').ever_selected
  assert new.reindex(old_support.index).equals(old_support[c].rename('ever_selected'))
 print('Point estimates and frozen queue/event-role reproduction PASS',flush=True)
 # Shared original bootstrap: all 2844 lakes, same multiplicities for candidates/events.
 rng=np.random.default_rng(SEED);draw_rows=[];canonical=[]
 strata=['overall','zero','low','medium','high']
 masks={s:np.ones(len(events),bool) if s=='overall' else events.support_stratum.eq(s).to_numpy() for s in strata}
 for start in range(0,REPS,CHUNK):
  m=min(CHUNK,REPS-start);mult=rng.multinomial(len(lakes),np.full(len(lakes),1/len(lakes)),size=m).astype(np.int32)
  cap={p:{e:np.zeros((m,len(events)),np.int32) for e in ENDPOINTS} for p in POLICIES}
  for r in records:
   if not len(r['event_ids']):continue
   copies=mult[:,r['lake']];budget=np.maximum(1,np.ceil(.1*copies.sum(axis=1,dtype=np.int64)).astype(np.int64))
   for p in POLICIES:
    order=r['orders'][p];ordered=copies[:,order];before=np.cumsum(ordered,axis=1,dtype=np.int64)-ordered
    inverse=np.empty(r['n'],int);inverse[order]=np.arange(r['n']);event_rank=inverse[r['local_rows']]
    # Equal to the existing selector's partial boundary-copy selection; never booleanize copies.
    val=np.clip(budget[:,None]-before[:,event_rank],0,copies[:,r['local_rows']]).astype(np.int32)
    for e in ENDPOINTS:
     mask=r['primary'] if e=='primary_31_60' else np.ones(len(r['event_ids']),bool)
     ids=r['event_ids'][mask];cap[p][e][:,ids]=np.maximum(cap[p][e][:,ids],val[:,mask])
  denominators=mult[:,event_lake]
  for s in strata:
   mask=masks[s]
   if not mask.any():continue
   den=denominators[:,mask].sum(axis=1)
   for e in ENDPOINTS:
    counts={p:cap[p][e][:,mask].sum(axis=1) for p in POLICIES}
    for i in range(m):
     row={'bootstrap_draw':start+i,'stratum':s,'endpoint':e,'event_copies':int(den[i]),'multi_horizon_captured':int(counts[Q][i]),'direct_60_day_captured':int(counts[D][i]),'historical_risk_captured':int(counts[H][i])}
     row['difference_pp']=100*(counts[Q][i]/den[i]-counts[D][i]/den[i]) if den[i] else np.nan
     draw_rows.append(row)
  if start%1024==0:print(f'Bootstrap {start+m}/{REPS}',flush=True)
 draws=pd.DataFrame(draw_rows)
 frozen=pd.read_parquet(INPUTS['formal_draws'])
 for e in ENDPOINTS:
  part=draws.loc[draws.stratum.eq('overall')&draws.endpoint.eq(e)].sort_values('bootstrap_draw')
  assert np.array_equal(part.event_copies,frozen.total_event_copies)
  for p,c in [(Q,'multi_horizon_captured'),(D,'direct_60_day_captured')]:assert np.array_equal(part[c],frozen[f'captured__{e}__{p}'])
  assert np.allclose(part.difference_pp/100,frozen['capture_rate_difference__'+e],rtol=0,atol=1e-15)
 draws.to_parquet(res/'bootstrap_10pct_draws.parquet',index=False)
 # Support summary; zero support has no observable events and hence no coverage estimate.
 strata_rows=[]
 for s in strata:
  sp=support if s=='overall' else support.loc[support.stratum.eq(s)]
  mask=masks[s];n=int(mask.sum())
  for e in ENDPOINTS:
   q=int(captures[(10,Q,e)][mask].sum());d=int(captures[(10,D,e)][mask].sum())
   vals=draws.loc[draws.stratum.eq(s)&draws.endpoint.eq(e),'difference_pp'].dropna()
   lo,hi=np.quantile(vals,[.025,.975]) if len(vals) else [np.nan,np.nan]
   strata_rows.append({'stratum':s,'endpoint':e,'candidate_lake_n':len(sp),'event_lake_n':int(sp.event_n.gt(0).sum()),'event_n':n,'observation_day_min':int(sp.observation_day_n.min()),'observation_day_max':int(sp.observation_day_n.max()),'multi_horizon_captured':q,'direct_60_day_captured':d,'multi_horizon_coverage_percent':100*q/n if n else np.nan,'direct_60_day_coverage_percent':100*d/n if n else np.nan,'difference_pp':100*(q-d)/n if n else np.nan,'net_covered_events':q-d,'contribution_to_overall_difference_pp':100*(q-d)/len(events),'ci_low_pp':lo,'ci_high_pp':hi,'valid_bootstrap_replicates':len(vals),'analysis_role':'formal_frozen_reference' if s=='overall' else 'post_hoc_support_stratification'})
 pd.DataFrame(strata_rows).to_csv(res/'monitoring_support_strata_results.csv',index=False)
 base_results=[]
 for e in ENDPOINTS:
  part=draws.loc[draws.stratum.eq('overall')&draws.endpoint.eq(e)]
  for p,c in [(Q,'multi_horizon_captured'),(D,'direct_60_day_captured')]:
   delta=100*(part[c]-part.historical_risk_captured)/part.event_copies
   lo,hi=np.quantile(delta,[.025,.975]);pc=int(captures[(10,p,e)].sum());hc=int(captures[(10,H,e)].sum())
   base_results.append({'endpoint':e,'policy':p,'baseline_policy':H,'capacity_percent':10,'event_n':len(events),'policy_captured':pc,'historical_risk_captured':hc,'policy_coverage_percent':100*pc/len(events),'historical_risk_coverage_percent':100*hc/len(events),'difference_pp':100*(pc-hc)/len(events),'ci_low_pp':lo,'ci_high_pp':hi,'bootstrap_replicates':REPS,'bootstrap_seed':SEED,'analysis_role':'post_hoc_simple_baseline_comparison'})
 pd.DataFrame(base_results).to_csv(res/'historical_risk_comparisons.csv',index=False)
 # Existing evidence only, not a new analysis direction.
 dev=ROOT/'04_analysis_data/development_evaluation/fixed_capacity_efficiency'
 pm=pd.read_csv(dev/'row_probability_metrics.csv');pm=pm.loc[pm['group'].eq('all_2017_2020')&pm.policy.isin(['multi_horizon',D])]
 pm.to_csv(res/'development_probability_metrics.csv',index=False)
 efficiency=pd.read_csv(dev/'policy_efficiency.csv');efficiency.loc[efficiency.policy.isin(['multi_horizon',D]),['policy','selected_rows','captured_events_1_60','captured_events_31_60']].to_csv(res/'development_event_coverage.csv',index=False)
 for rel,h in existing.items():assert sha(ROOT/rel)==h,('input modified',rel)
 audit.extend([{'check':x,'passed':True,'detail':d} for x,d in [
 ('all_consumed_inputs_unchanged',str(len(existing))),('candidate_contract','369720 rows / 365726 eligible / 2844 lakes / 130 origins'),('exact_formal_queue_records','Both stored queues and score selection flags reproduced'),('formal_event_roles','All four event capture vectors exactly equal Figure 1 source data'),('formal_bootstrap_all_draws','5000 denominators and both policy counts identical for both endpoints'),('existing_capacity_points','10,15,20,30 percent exact'),('ever_selected_lakes','Exact match to Figure 4 source data'),('capacity_coverage_monotonicity','All three policies, both endpoints'),('baseline_temporal_availability','No post2020 observations used')]])
 pd.DataFrame(audit).to_csv(prov/'validation_checks.csv',index=False)
 dump(prov/'run_summary.json',{'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pyarrow.__version__,'seed':SEED,'bootstrap_replicates':REPS,'capacity_percent_grid':cap_grid,'consumed_input_count':len(existing),'all_consumed_inputs_unchanged':True,'formal_bootstrap_reproduced_draw_by_draw':True})
 print('All analyses and validation PASS',flush=True)

if __name__=='__main__':main()
