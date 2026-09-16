"""Build English result tables and manuscript figure and data indexes."""
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'04_analysis_data/strategy_evaluation'
META=ROOT/'07_metadata'

def main():
    registry=pd.DataFrame(json.loads((META/'policy_registry.json').read_text()))
    labels=registry.set_index('policy_id').name_en
    cap=pd.read_csv(RES/'capacity_coverage.csv')
    points=cap.loc[cap.capacity_percent.eq(10)].copy()
    points['name_en']=points.policy.map(labels)
    points=points[['endpoint','policy','name_en','captured_events','event_n','coverage_percent','selected_slots','capacity_percent','analysis_role']]
    points.to_csv(RES/'point_estimates.csv',index=False)
    points.to_csv(META/'key_results.csv',index=False)
    formal=pd.read_csv(ROOT/'04_analysis_data/primary_external_evaluation/evaluation/endpoint_summary.csv')
    rows=[]
    for x in formal.itertuples():
        rows.append({'endpoint':x.endpoint,'policy':'multi_horizon','comparator':'direct_60_day','event_n':696,'captured_events':x.candidate_captured,'comparator_captured_events':x.baseline_captured,'difference_pp':x.point_difference*100,'ci_low_pp':x.percentile_ci_low*100,'ci_high_pp':x.percentile_ci_high*100,'analysis_role':'prespecified_comparison','bootstrap_replicates':5000})
    for x in pd.read_csv(RES/'historical_risk_comparisons.csv').itertuples():
        rows.append({'endpoint':x.endpoint,'policy':x.policy,'comparator':'historical_risk','event_n':696,'captured_events':x.policy_captured,'comparator_captured_events':x.historical_risk_captured,'difference_pp':x.difference_pp,'ci_low_pp':x.ci_low_pp,'ci_high_pp':x.ci_high_pp,'analysis_role':'post_hoc_comparison','bootstrap_replicates':5000})
    comparisons=pd.DataFrame(rows)
    comparisons['policy_name_en']=comparisons.policy.map(labels)
    comparisons['comparator_name_en']=comparisons.comparator.map(labels)
    comparisons.to_csv(RES/'endpoint_comparisons.csv',index=False)
    pd.read_csv(ROOT/'03_figure_source_data/Figure_S8/target_identity.csv').to_csv(RES/'component_control.csv',index=False)
    registry.to_csv(META/'policy_registry.csv',index=False)
    topics={
        'Figure_1':'Candidate lakes, observations and events',
        'Figure_2':'Fixed-capacity evaluation and lake bootstrap',
        'Figure_3':'Advance event coverage',
        'Figure_4':'Risk ranks and selected lists',
        'Figure_5':'Capacity response and historical risk',
        'Figure_S1':'Event construction and distribution',
        'Figure_S2':'Observation availability and valid prediction dates',
        'Figure_S3':'Development probability metrics and coverage',
        'Figure_S4':'Lists across prediction dates',
        'Figure_S5':'Evaluation-scope sensitivity',
        'Figure_S6':'Joint bootstrap distributions',
        'Figure_S7':'Event dates and separation thresholds',
        'Figure_S8':'Same-predictor 60-day base model',
        'Figure_S9':'List updates and cumulative lake reach',
        'Figure_S10':'Observation-support strata',
        'Figure_S11':'Capacity intervals and equal-lake weighting'}
    sources=[]
    for fid,topic in topics.items():
        for p in sorted((ROOT/'03_figure_source_data'/fid).glob('*.csv')):
            sources.append({'figure':fid,'topic':topic,'figure_path':f'02_figures/{"supplementary" if "_S" in fid else "main"}/{fid}.pdf','source_data_path':p.relative_to(ROOT).as_posix(),'rows':len(pd.read_csv(p))})
    pd.DataFrame(sources).to_csv(META/'figure_source_map.csv',index=False)
    lines=['# Event coverage at 10% monitoring capacity','','| Window | Strategy | Covered events | Coverage |','|---|---|---:|---:|']
    for x in points.itertuples():
        window='31–60 days' if x.endpoint=='primary_31_60' else '1–60 days'
        lines.append(f'| {window} | {x.name_en} | {x.captured_events}/{x.event_n} | {x.coverage_percent:.2f}% |')
    lines+=['','| Window | Comparison | Difference (pp) | 95% confidence interval (pp) | Design |','|---|---|---:|---:|---|']
    for x in comparisons.itertuples():
        window='31–60 days' if x.endpoint=='primary_31_60' else '1–60 days'
        design='Prespecified' if x.analysis_role=='prespecified_comparison' else 'Post hoc'
        lines.append(f'| {window} | {x.policy_name_en} minus {x.comparator_name_en} | {x.difference_pp:.2f} | {x.ci_low_pp:.2f} to {x.ci_high_pp:.2f} | {design} |')
    lines+=['','All strategies used 36,631 lake–prediction-date slots. The 696-event denominator includes events without valid prediction dates. Confidence intervals use 5,000 lake-level bootstrap replicates with lists rebuilt from all 2,844 candidate lakes.','', 'Full evaluation tables: `04_analysis_data/strategy_evaluation/`. Figure sources: `07_metadata/figure_source_map.csv`.']
    (ROOT/'01_results/RESULTS.md').write_text('\n'.join(lines)+'\n')
    tables=[];columns=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(x.startswith('.') or x == '__pycache__' for x in path.relative_to(ROOT).parts) or path.suffix not in ['.csv','.parquet'] or path.name in ['table_catalog.csv','result_column_dictionary.csv']:continue
        if path.relative_to(ROOT).as_posix().startswith(('07_metadata/provenance/evaluation/', '07_metadata/provenance/figures/')):continue
        if path.suffix=='.parquet':
            pf=pq.ParquetFile(path);n=pf.metadata.num_rows;names=pf.schema_arrow.names;types=[str(f.type) for f in pf.schema_arrow]
        else:
            try:df=pd.read_csv(path);n=len(df);names=list(df.columns);types=[str(t) for t in df.dtypes]
            except pd.errors.EmptyDataError:n=0;names=[];types=[]
        rel=path.relative_to(ROOT).as_posix()
        role='figure_source_data' if rel.startswith('03_') else 'strategy_result' if '/strategy_evaluation/' in rel else 'model' if rel.startswith('05_') else 'metadata' if rel.startswith('07_') else 'research_data'
        tables.append({'path':rel,'role':role,'rows':n,'column_count':len(names),'columns':' | '.join(names)})
        if role=='strategy_result':columns.extend({'path':rel,'column':name,'dtype':dtype,'definition_reference':'07_metadata/data_dictionary.md'} for name,dtype in zip(names,types))
    pd.DataFrame(tables).to_csv(META/'table_catalog.csv',index=False)
    pd.DataFrame(columns).to_csv(META/'result_column_dictionary.csv',index=False)
    print('English result tables and indexes for 16 manuscript figures written.')

if __name__=='__main__':main()
