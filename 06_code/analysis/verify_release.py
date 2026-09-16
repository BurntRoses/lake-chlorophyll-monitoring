"""Verify strategy identities, independently reconstruct event coverage and check release assets."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import ast, json, re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'04_analysis_data/strategy_evaluation'

def digest(p):
 h=sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def main():
 checks=[]
 def check(name,condition,detail=''):
  assert bool(condition),(name,detail)
  checks.append({'check':name,'status':'PASS','detail':str(detail)})
 policies={x['policy_id'] for x in json.loads((ROOT/'07_metadata/policy_registry.json').read_text())}
 check('four_distinct_strategy_identities',len(policies)==4)
 points=pd.read_csv(RES/'point_estimates.csv')
 check('three_strategies_two_endpoints',len(points)==6)
 selection=pd.read_parquet(RES/'selection_records_10pct.parquet')
 check('matched_36631_slots',selection.groupby('policy').size().eq(36631).all())
 events=pd.read_csv(RES/'event_capture_10pct.csv',dtype={'lake_id':str,'episode_id':str})
 check('complete_event_denominator',len(events)==696 and events.episode_id.nunique()==696)
 events.event_date=pd.to_datetime(events.event_date);selection.origin_date=pd.to_datetime(selection.origin_date);selection.lake_id=selection.lake_id.astype(str)
 for policy in sorted(selection.policy.unique()):
  s=selection.loc[selection.policy.eq(policy)]
  joined=events[['episode_id','lake_id','event_date']].merge(s[['lake_id','origin_date']],on='lake_id',how='left')
  lead=(joined.event_date-joined.origin_date).dt.days
  for endpoint,low in [('primary_31_60',31),('secondary_1_60',1)]:
   ids=set(joined.loc[lead.between(low,60),'episode_id'])
   actual=events.episode_id.isin(ids).to_numpy()
   check(f'independent_event_capture_{policy}_{endpoint}',np.array_equal(actual,events['captured__'+endpoint+'__'+policy].to_numpy(bool)))
   point=points.loc[points.policy.eq(policy)&points.endpoint.eq(endpoint)].iloc[0]
   check(f'point_table_{policy}_{endpoint}',int(actual.sum())==point.captured_events)
 draws=pd.read_parquet(RES/'bootstrap_10pct_draws.parquet')
 original=pd.read_parquet(ROOT/'04_analysis_data/primary_external_evaluation/evaluation/bootstrap_draws.parquet')
 for endpoint in ['primary_31_60','secondary_1_60']:
  g=draws.loc[draws.stratum.eq('overall')&draws.endpoint.eq(endpoint)].sort_values('bootstrap_draw')
  check(f'5000_denominators_{endpoint}',len(g)==5000 and np.array_equal(g.event_copies,original.total_event_copies))
  for policy in ['multi_horizon','direct_60_day']:
   check(f'5000_counts_{endpoint}_{policy}',np.array_equal(g[policy+'_captured'],original['captured__'+endpoint+'__'+policy]))
  hist=pd.read_csv(RES/'historical_risk_comparisons.csv')
  for x in hist.loc[hist.endpoint.eq(endpoint)].itertuples():
   delta=100*(g[x.policy+'_captured']-g.historical_risk_captured)/g.event_copies
   check(f'historical_interval_{endpoint}_{x.policy}',np.allclose(np.quantile(delta,[.025,.975]),[x.ci_low_pp,x.ci_high_pp],atol=1e-12,rtol=0))
 support=pd.read_csv(RES/'monitoring_support_strata_results.csv')
 z=support.loc[support.stratum.eq('zero')]
 check('zero_support_is_not_zero_coverage',len(z)==2 and z.event_n.eq(0).all() and z.difference_pp.isna().all() and z.multi_horizon_coverage_percent.isna().all())
 cap=pd.read_csv(RES/'capacity_coverage.csv')
 check('31_capacities_two_endpoints_three_policies',len(cap)==186 and cap.capacity_percent.nunique()==31)
 for (policy,endpoint),g in cap.groupby(['policy','endpoint']):
  check(f'monotone_coverage_{policy}_{endpoint}',g.sort_values('capacity_percent').captured_events.diff().dropna().ge(0).all())
 comp=pd.read_csv(RES/'component_control.csv').iloc[0]
 check('base_model_distinct_from_direct_strategy',comp.matched_60_day_base_captured_events==274 and set(points.loc[points.policy.eq('direct_60_day'),'captured_events'])=={207,290})
 for x in pd.read_csv(RES/'turnover_summary.csv').itertuples():
  s=selection.loc[selection.policy.eq(x.policy)]
  groups=[set(g.lake_id) for _,g in s.groupby('origin_date',sort=True)]
  values=[len(a&b)/len(a|b) for a,b in zip(groups,groups[1:])]
  check('turnover_'+x.policy,len(values)==129 and s.lake_id.nunique()==x.unique_selected_lakes and np.isclose(np.mean(values),x.jaccard_mean))
 figure_ids=[*(f'Figure_{i}' for i in range(1,6)),*(f'Figure_S{i}' for i in range(1,12))]
 fmap=pd.read_csv(ROOT/'07_metadata/figure_source_map.csv')
 check('continuous_figure_numbering',set(fmap.figure)==set(figure_ids))
 for fid in figure_ids:
  for ext in ['png','svg','pdf']:
   p=ROOT/'02_figures'/('supplementary' if '_S' in fid else 'main')/(fid+'.'+ext)
   check('asset_'+fid+'_'+ext,p.is_file() and p.stat().st_size>1000)
 for x in fmap.itertuples():check('source_'+x.figure+'_'+Path(x.source_data_path).name,(ROOT/x.source_data_path).is_file())
 for p in (ROOT/'06_code').rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 check('python_syntax',True)
 forbidden=re.compile(r'q(?:14|30|60)|direct[-_]y(?:60)|champion|addition(?:al)_analys|Figure_A[123]',re.I)
 problems=[]
 for p in ROOT.rglob('*'):
  if not p.is_file() or any(x.startswith('.') or x == '__pycache__' for x in p.relative_to(ROOT).parts):continue
  if forbidden.search(p.name):problems.append(str(p.relative_to(ROOT)))
  if p.resolve()==Path(__file__).resolve():continue
  if p.suffix.lower()=='.svg':
   visible=' '.join(''.join(t.itertext()) for t in ET.parse(p).iter() if t.tag.endswith('}text'))
   if forbidden.search(visible):problems.append(str(p.relative_to(ROOT)))
  elif p.suffix.lower() in ['.csv','.json','.md','.py','.yaml','.yml','.toml']:
   if forbidden.search(p.read_text()):problems.append(str(p.relative_to(ROOT)))
  elif p.suffix=='.parquet':
   if any(forbidden.search(n) for n in pq.read_schema(p).names):problems.append(str(p.relative_to(ROOT)))
 check('terminology_consistency',not problems,problems)
 check('research_only_outputs',not list(ROOT.rglob('*.docx')) and not (ROOT/'.build').exists())
 receipt=json.loads((ROOT/'07_metadata/provenance/evaluation_inputs.json').read_text())
 for key,item in receipt.items():check('committed_input_'+key,digest(ROOT/item['relative_path'])==item['sha256'])
 out=ROOT/'07_metadata/provenance/release_validation.json'
 out.write_text(json.dumps({'status':'PASS','checks':checks,'check_count':len(checks),'scope':'Evaluation from stored predictions, unified identities, source paths, original bootstrap counts and research figure assets.'},indent=2,ensure_ascii=False)+'\n')
 print(f'Release validation PASS: {len(checks)} checks')

if __name__=='__main__':main()
