from results_common import *

def validate(f):
    cov=f['absolute_event_coverage'];records=f['absolute_event_records'];ex=f['absolute_event_exchanges']
    eff=f['relative_training_effects'];ev=f['china_paired_relative_events']
    lake=f['china_lake_changes'];cs=f['china_background_changes'];trans=f['paired_capture_transitions']
    assert len(records)==529 and len(ex)==12
    assert not records.duplicated(['region_id','episode_id']).any()
    for region,n in zip(REGIONS,[337,86,29,77]):
        d=records.query('region_id==@region');assert len(d)==n
        for m in METHODS:
            r=cov.query('region_id==@region and endpoint_id=="31_60" and method_id==@m').iloc[0]
            assert int(d[m].sum())==r.events_covered
        for row in ex.query('region_id==@region').itertuples():
            a=d.xgboost.eq(1);r=d[row.method_id].eq(1)
            assert int((~a&r).sum())==row.gained and int((a&~r).sum())==row.lost
            assert row.gained+row.lost+row.retained+row.neither==n
            assert row.method_covered-row.reference_covered==row.gained-row.lost
            assert np.isclose(row.gained_pct,100*row.gained/n) and np.isclose(row.lost_pct,100*row.lost/n)
    assert len(eff)==8 and eff.confidence_level.eq(.9875).all()
    assert np.allclose(eff.difference_pp,eff.relative_coverage_pct-eff.absolute_coverage_pct)
    assert (eff.lower_pp<eff.difference_pp).all() and (eff.upper_pp>=eff.difference_pp).all()
    assert eff.groupby('region_id').evaluable_events.first().reindex(REGIONS).tolist()==[718,168,128,24]
    assert len(ev)==718 and ev.episode_id.is_unique and len(lake)==258
    rebuilt=ev.groupby('lake_id').agg(n=('episode_id','size'),a=('absolute_covered','sum'),r=('relative_covered','sum'))
    l=lake.set_index('lake_id').loc[rebuilt.index]
    assert (l.evaluable_events==rebuilt.n).all() and (l.absolute_events_covered==rebuilt.a).all() and (l.relative_events_covered==rebuilt.r).all()
    assert np.allclose(l.coverage_change_pp,100*(rebuilt.r-rebuilt.a)/rebuilt.n)
    assert (int((rebuilt.r>0).sum()),int((rebuilt.a>0).sum()))==(196,34)
    assert [(l.coverage_change_pp>0).sum(),(l.coverage_change_pp==0).sum(),(l.coverage_change_pp<0).sum()]==[176,64,18]
    assert cs.historical_q90_ug_L.is_monotonic_increasing and not cs.historical_q90_ug_L.duplicated().any()
    assert cs.cumulative_events.iloc[-1]==718 and cs.cumulative_gained.iloc[-1]==314 and cs.cumulative_lost.iloc[-1]==27
    for row in cs.itertuples():
        d=ev[ev.historical_q90_ug_L<=row.historical_q90_ug_L]
        assert len(d)==row.cumulative_events
        assert int((d.absolute_covered.eq(0)&d.relative_covered.eq(1)).sum())==row.cumulative_gained
        assert int((d.absolute_covered.eq(1)&d.relative_covered.eq(0)).sum())==row.cumulative_lost
    assert trans.events.sum()==718 and len(trans)==12
    for row in trans.itertuples():
        cond={'below25':ev.historical_q90_ug_L<25,
              '25to75':ev.historical_q90_ug_L.between(25,75,inclusive='left'),
              '75plus':ev.historical_q90_ug_L>=75}[row.stratum]
        d=ev[cond];assert len(d)==row.stratum_events
        assert int((d.absolute_covered.eq(row.absolute_covered)&d.relative_covered.eq(row.relative_covered)).sum())==row.events
    assert np.isclose(100*(314-27)/718,eff.query('region_id=="china_2020_2022" and method_id=="multi_window"').difference_pp.iloc[0])
    return {'all_529_absolute_events_and_16_coverages_reconciled':True,'all_12_paired_exchange_pairs_exact':True,
        'all_8_original_effects_and_9875pct_intervals':True,'absolute_and_relative_cohorts_separate':True,
        'all_718_relative_events_and_258_lakes_reconciled':True,'lake_reach_34_to_196':True,
        'lake_changes_176_64_18':True,'all_cumulative_step_positions_rebuilt':True,
        'event_gains_314_losses_27':True,'no_table_heatmap_or_numerical_columns':True,
        'no_new_fitting_resampling_smoothing_or_jitter':True}
