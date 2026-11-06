import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import json
from chile_priority_attributes import prepare_chile_numba_inputs_from_rankings
from gale_shapley import gale_shapley_per_school_numba_wrapper

RANKINGS_CSV = 'experimental_output/chile_res_logs/20261012_123851/chilean_experiment_K=6_M=15_iter=13_opt=15_lr=0.05_maxp=10_loss=allTopP+unm_seed=40_20261012_123851_synthetic_rankings.csv'
CAPACITY_XLSX = 'sample-data/data/chilean_data_processed/school_capacity_with_region_and_province.xlsx'
PRIORITY_CONFIG = 'sample-data/data/chilean_data_processed/chile_priority_config.json'

DISTRICT = 'Antofagasta'

df = pd.read_csv(RANKINGS_CSV)
df = df[df['district'] == DISTRICT].reset_index(drop=True)
print(f"Restricted to district={DISTRICT!r}: {len(df)} students")
choice_cols = [c for c in df.columns if c.startswith('choice_')]
truncated_rankings = []
for _, row in df.iterrows():
    ranking = [str(row[c]) for c in choice_cols if pd.notna(row[c]) and str(row[c]) != '']
    truncated_rankings.append(ranking)
print(f"Loaded {len(truncated_rankings)} students' rankings (top-10 truncated), "
      f"avg length={np.mean([len(r) for r in truncated_rankings]):.2f}")

with open(PRIORITY_CONFIG) as f:
    priority_config = json.load(f)
region_overrides = priority_config.get('region_overrides', {})
first_region = next(iter(region_overrides.values()))
fracs = first_region.get('student_attribute_fractions', {})
tiers = first_region.get('priority_tiers', [])
calibration = {
    'priority_student_student_rate': fracs.get('disadvantaged', 0.6889),
    'priority_sibling_student_rate': next((t['fraction_eligible'] for t in tiers if t['group'] == 'sibling'), 0.1539),
    'priority_parent_civil_servant_student_rate': next((t['fraction_eligible'] for t in tiers if t['group'] == 'working_parent'), 0.0066),
    'priority_ex_student_student_rate': next((t['fraction_eligible'] for t in tiers if t['group'] == 'returning_student'), 0.0521),
    'priority_already_registered_student_rate': fracs.get('already_registered', 0.1398),
}
print("Calibration used:", calibration)

prepared = prepare_chile_numba_inputs_from_rankings(
    truncated_rankings=truncated_rankings,
    capacity_rows=CAPACITY_XLSX,
    seed=40,
    calibration=calibration,
)

print(f"\nn students: {len(prepared['student_ids'])}, n schools: {len(prepared['school_ids'])}")
print(f"application_table rows (after capacity filter): {len(prepared['application_table'])}")
total_pref_rows = sum(len(r) for r in truncated_rankings)
print(f"total preference rows submitted: {total_pref_rows}")
print(f"fraction of preference rows surviving capacity filter: {len(prepared['application_table']) / total_pref_rows:.4f}")

print("\npriority_tier distribution in application_table:")
print(prepared['application_table']['priority_tier'].value_counts().sort_index())

matches = gale_shapley_per_school_numba_wrapper(
    prepared['student_rankings'],
    prepared['school_priority_scores'],
    prepared['school_capacities'],
)
matches = np.asarray(matches)
n_matched = (matches >= 0).sum()
print(f"\nn matched: {n_matched} / {len(matches)} ({100*n_matched/len(matches):.1f}%)")

rank_of_match = []
for i, m in enumerate(matches):
    if m >= 0:
        try:
            r = prepared['student_rankings'][i].index(int(m)) + 1
        except (ValueError, IndexError):
            r = None
        rank_of_match.append(r)
import collections
print("rank-of-match distribution (None = matched to something outside their own ranking list):")
print(collections.Counter(rank_of_match))
