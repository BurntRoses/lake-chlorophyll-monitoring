"""Validate Figure 2 event records and paired coverage summaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ['xgboost', 'elastic_net', 'single_window', 'multi_window']

def read_and_validate():
    dictionary = json.loads((ROOT/'data_dictionary.json').read_text(encoding='utf-8'))
    hashes = {}
    frames = {}
    for entry in dictionary['data_files']:
        p = ROOT/entry['file']
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == entry['sha256'], f'Checksum mismatch: {p.name}'
        df = pd.read_csv(p, dtype={'episode_id': str, 'lake_id': str})
        assert len(df) == entry['rows'], p.name
        hashes[p.name] = actual
        frames[p.stem] = df

    fingerprint = frames['primary_event_fingerprint']
    assert not fingerprint.duplicated(['episode_id', 'method_id']).any()
    assert set(fingerprint.covered) == {0, 1}
    reference = fingerprint[fingerprint.method_id == METHODS[0]].sort_values('event_order')
    assert len(reference) == 593 and reference.lake_id.nunique() == 354
    assert reference.event_order.tolist() == list(range(1, 594))
    assert reference.event_date.is_monotonic_increasing
    expected_counts = [344, 332, 461, 463]
    binary = []
    keys = ['episode_id', 'lake_id', 'event_date', 'event_year', 'event_order']
    for method, count in zip(METHODS, expected_counts):
        sub = fingerprint[fingerprint.method_id == method].sort_values('event_order')
        assert sub[keys].reset_index(drop=True).equals(reference[keys].reset_index(drop=True))
        assert int(sub.covered.sum()) == count
        binary.append(sub.covered.to_numpy())
    binary = np.asarray(binary)
    s, m = binary[2].astype(bool), binary[3].astype(bool)
    overlap = frames['event_overlap']
    observed = dict(zip(overlap.category, overlap['count']))
    derived = {'Both': int((s & m).sum()), 'Multi-window only': int((~s & m).sum()),
               'Single-window only': int((s & ~m).sum()), 'Neither': int((~s & ~m).sum())}
    assert derived == observed == {'Both': 446, 'Multi-window only': 17,
                                  'Single-window only': 15, 'Neither': 115}
    assert overlap.total.eq(593).all() and sum(derived.values()) == 593
    assert np.allclose(overlap.percent, overlap['count']/593*100)

    earliest = frames['earliest_selection_events']
    accumulation = frames['coverage_accumulation']
    assert not earliest.duplicated(['episode_id', 'method_id']).any()
    assert not accumulation.duplicated(['method_id', 'lead_days']).any()
    cohort_ids = set(earliest[earliest.method_id == METHODS[0]].episode_id)
    for method, end_count in zip(METHODS, [449, 431, 597, 590]):
        events = earliest[earliest.method_id == method]
        assert len(events) == 681 and set(events.episode_id) == cohort_ids
        assert events.first_lead_days.dropna().between(1, 60).all()
        assert events.first_lead_days.notna().sum() == end_count
        assert events.loc[events.first_lead_days.isna(), 'earliest_category'].eq('Uncovered').all()
        series = accumulation[accumulation.method_id == method].sort_values('lead_days', ascending=False)
        assert series.lead_days.tolist() == list(range(60, 0, -1))
        assert series.cohort_events.eq(681).all()
        counts = np.array([(events.first_lead_days >= day).sum() for day in series.lead_days])
        assert np.array_equal(counts, series.events_selected_by_lead.to_numpy())
        assert np.allclose(counts/681*100, series.cumulative_coverage_pct)
        assert np.all(np.diff(counts) >= 0)
        assert int(series.loc[series.lead_days == 31, 'events_selected_by_lead'].iloc[0]) == expected_counts[METHODS.index(method)]

    differences = frames['paired_differences']
    assert differences.endpoint_id.eq('31_60').all()
    assert differences.proposed_id.eq('multi_window').all()
    assert differences.bootstrap_replicates.eq(5000).all()
    assert differences.confidence_level.eq(.975).all()
    assert differences.groupby('comparison_family').size().tolist() == [2, 2]
    assert (differences.lower_pp < differences.difference_pp).all()
    assert (differences.upper_pp > differences.difference_pp).all()
    for method, count in zip(METHODS[:3], expected_counts[:3]):
        r = differences[differences.comparator_id == method].iloc[0]
        assert np.isclose(r.difference_pp, (463-count)/593*100)
    paired = frames['primary_paired_events'].set_index('episode_id')
    assert len(paired) == 593 and set(paired.index) == set(reference.episode_id)
    assert not paired.index.duplicated().any()
    for method, indicators in zip(METHODS, binary):
        assert np.array_equal(paired.loc[reference.episode_id, method].astype(int), indicators)
    assert int(paired.same_window_control.sum()) == 459
    for method in METHODS[:3]+['same_window_control']:
        r = differences[differences.comparator_id == method].iloc[0]
        gained = int((paired.multi_window & ~paired[method]).sum())
        lost = int((~paired.multi_window & paired[method]).sum())
        assert np.isclose((gained-lost)/593*100, r.difference_pp)
    return frames, reference, binary, hashes
