"""
Chile analogue of nyc_district_heterogeneity.py: checks whether a given
province/region's applicants apply to many different programs and few apply
locally, using the REAL individual-level preference data (Chile, unlike NYC,
has true per-student rank lists, so this is computed per-student and then
averaged -- not an instance-weighted population proxy).

For each student: list_length, number of distinct programs ranked, and the
share of their list that is "local" (school's own Provincia == student's
residential Provincia). These are then averaged within province, and
separately rolled up (student-count-weighted) to the 16 real Chile regions.

Usage:
    python3 src/project_specific_scripts/chile_region_heterogeneity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from data_ingestion import read_data
from file_config import (
    CHILEAN_DATA_DIR,
    CHILEAN_INDV_PREF_PROVINCE_SANTIAGOZONES_FILEPATH,
    CHILEAN_SCHOOL_CAPACITY_BY_REGION_PROVINCE_SANTIAGOZONES_FILEPATH,
)
from constants import CHILE_PROVINCE_TO_REGION_MAPPING


def main():
    pref_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_INDV_PREF_PROVINCE_SANTIAGOZONES_FILEPATH}")
    school_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_SCHOOL_CAPACITY_BY_REGION_PROVINCE_SANTIAGOZONES_FILEPATH}")

    school_provincia = school_df.drop_duplicates(['rbd', 'program_code']).set_index(['rbd', 'program_code'])['Provincia']
    pref_df = pref_df.copy()
    pref_df['school_provincia'] = pref_df.set_index(['rbd', 'program_code']).index.map(school_provincia)
    unmatched = pref_df['school_provincia'].isna().sum()
    if unmatched:
        print(f"Dropping {unmatched} preference rows with no matching school ({unmatched/len(pref_df):.2%})")
    pref_df = pref_df.dropna(subset=['school_provincia'])
    pref_df['is_local'] = pref_df['school_provincia'] == pref_df['Provincia']

    student_stats = pref_df.groupby('mrun').agg(
        residence=('Provincia', 'first'),
        list_length=('program_code', 'count'),
        n_distinct_programs=('program_code', 'nunique'),
        local_count=('is_local', 'sum'),
    )
    student_stats['local_rate_pct'] = 100 * student_stats['local_count'] / student_stats['list_length']

    prov = student_stats.groupby('residence').agg(
        n_students=('list_length', 'size'),
        avg_list_length=('list_length', 'mean'),
        avg_n_distinct_programs=('n_distinct_programs', 'mean'),
        avg_local_rate_pct=('local_rate_pct', 'mean'),
    ).reset_index().rename(columns={'residence': 'province'})

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 140)

    print("\n── Province-level: avg local rate per student, ascending ──")
    print(prov.sort_values('avg_local_rate_pct')
          [['province', 'avg_local_rate_pct', 'avg_n_distinct_programs', 'avg_list_length', 'n_students']]
          .to_string(index=False))

    print("\n── Province-level: avg distinct programs per student, descending ──")
    print(prov.sort_values('avg_n_distinct_programs', ascending=False)
          [['province', 'avg_n_distinct_programs', 'avg_local_rate_pct', 'n_students']]
          .to_string(index=False))

    n_prov = len(prov)
    print(f"\n  Median province avg_local_rate_pct:        {prov['avg_local_rate_pct'].median():.1f}%")
    print(f"  Median province avg_n_distinct_programs:    {prov['avg_n_distinct_programs'].median():.2f}")

    # Region-level rollup, student-count-weighted.
    prov['region'] = prov['province'].map(lambda p: CHILE_PROVINCE_TO_REGION_MAPPING.get(p, p))

    def wavg(g, col):
        return (g[col] * g['n_students']).sum() / g['n_students'].sum()

    region_rows = []
    for region, g in prov.groupby('region'):
        region_rows.append({
            'region': region,
            'n_students': int(g['n_students'].sum()),
            'avg_local_rate_pct': wavg(g, 'avg_local_rate_pct'),
            'avg_n_distinct_programs': wavg(g, 'avg_n_distinct_programs'),
        })
    region_df = pd.DataFrame(region_rows)

    print("\n── Region-level (student-weighted): avg local rate, ascending ──")
    print(region_df.sort_values('avg_local_rate_pct').to_string(index=False))

    print("\n── Region-level (student-weighted): avg distinct programs, descending ──")
    print(region_df.sort_values('avg_n_distinct_programs', ascending=False).to_string(index=False))

    n_reg = len(region_df)
    print(f"\n  ({n_prov} provinces rolled up to {n_reg} regions)")
    print(f"  Median region avg_local_rate_pct:           {region_df['avg_local_rate_pct'].median():.1f}%")
    print(f"  Median region avg_n_distinct_programs:       {region_df['avg_n_distinct_programs'].median():.2f}")


if __name__ == '__main__':
    main()
