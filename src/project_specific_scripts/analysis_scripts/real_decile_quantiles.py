import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/project_specific_scripts')
import numpy as np
import pandas as pd

from file_config import DATA_GENERATION_SEED
from chile_priority_attributes import _prepare_school_capacity_table
from chilean_real_welfare_comparison import load_df, build_applications_long, run_matching
from chile_lottery_decile_stb_mtb import compute_student_records

INDIVIDUAL = 'sample-data/data/chilean_data_processed/individual_level_preferences_and_result.xlsx'
CAPACITY = 'sample-data/data/chilean_data_processed/school_capacity_with_region_and_santiago_macrozones.xlsx'

rng = np.random.default_rng(DATA_GENERATION_SEED)

indv_df = load_df(INDIVIDUAL)
capacity_df = load_df(CAPACITY)
school_table = _prepare_school_capacity_table(capacity_df)

applications_long = build_applications_long(indv_df)
list_lengths = applications_long.groupby('mrun')['preference_number'].max().rename('list_length')
print(f"n students: {len(list_lengths)}")
print(list_lengths.describe())
print(f"pct with list_length < 5: {100*(list_lengths < 5).mean():.2f}")
print(f"pct with list_length == 5: {100*(list_lengths == 5).mean():.2f}")
print(f"pct with list_length > 5: {100*(list_lengths > 5).mean():.2f}")

student_ids, rankings, matches, lottery_df = run_matching(
    applications_long, school_table, rng, student_lottery=None, return_lottery_df=True,
)
df = compute_student_records(student_ids, rankings, matches, lottery_df, 'min_topk', topk=5)

# df's index values are the original positions in student_ids (compute_student_records
# does dropna without resetting the index), so recover mrun via that index, not position.
df['mrun'] = np.asarray(student_ids)[df.index.to_numpy()]
df = df.merge(list_lengths.reset_index(), on='mrun', how='left')

print("\n=== REAL empirical quantiles of decile_val (min of up to top-5 lottery draws) ===")
qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
real_q = df['decile_val'].quantile(qs)
print(real_q)

print("\n=== for comparison: idealized min-of-5-uniforms quantiles ===")
for q in qs:
    x = 1 - (1 - q) ** (1/5) if q < 1.0 else 1.0
    print(f"  {q}: {x:.4f}")

print("\n=== list length distribution WITHIN each real decile bucket ===")
df['decile'] = pd.qcut(df['decile_val'], 10, labels=range(1, 11))
print(df.groupby('decile')['list_length'].describe()[['mean', '50%', 'min', 'max']])

print("\n=== fraction of students in each decile with list_length < 5 (i.e. min-of-<5, not min-of-5) ===")
print(df.groupby('decile')['list_length'].apply(lambda s: (s < 5).mean()))
