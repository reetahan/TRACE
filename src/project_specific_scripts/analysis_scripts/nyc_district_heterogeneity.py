"""
Checks the paper claim that Districts 10 and 19 are underestimated because
their applicants "apply to many different programs and few apply locally" --
i.e. their preference pools are unusually spread-out / non-local relative to
other NYC districts. Uses only the real (observed) application data, at the
(School DBN x Residential District) granularity -- there is no per-student
rank-level real data for NYC, so these are population-level proxies:

  - locality_rate_d:      share of district d's applications that go to a
                           program whose School District == d
  - programs_per_1k_apps: distinct programs applied to per 1,000 applications
                           from d (higher = more spread out / heterogeneous)

Usage:
    python3 src/project_specific_scripts/nyc_district_heterogeneity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from data_ingestion import read_data
from file_config import POLISHED_DATA_DIR, MAIN_AGG_APP_STATS_FILEPATH

OUTLIER_DISTRICTS = {'10', '19'}


def main():
    df = read_data(f"{POLISHED_DATA_DIR}/{MAIN_AGG_APP_STATS_FILEPATH}")
    df = df[df['Residential District'] != 'Unknown'].copy()
    df['Residential District'] = df['Residential District'].astype(str)
    df['School District int'] = df['School District'].astype(float).astype(int)

    rows = []
    for d, g in df.groupby('Residential District'):
        g = g[g['Total Applicants by Residential District'] > 0]
        total_apps = g['Total Applicants by Residential District'].sum()
        if total_apps == 0:
            continue
        n_programs = g['School DBN'].nunique()
        local_apps = g.loc[g['School District int'] == int(d), 'Total Applicants by Residential District'].sum()
        rows.append({
            'district': d,
            'total_applications': total_apps,
            'n_distinct_programs': n_programs,
            'programs_per_1k_apps': 1000 * n_programs / total_apps,
            'locality_rate_pct': 100 * local_apps / total_apps,
        })

    out = pd.DataFrame(rows).sort_values('locality_rate_pct')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 140)

    print("\n── Locality rate, ascending (lowest = least local first) ──")
    print(out[['district', 'locality_rate_pct', 'programs_per_1k_apps', 'n_distinct_programs', 'total_applications']]
          .to_string(index=False))

    print("\n── Programs-per-1k-applications, descending (most spread-out first) ──")
    print(out.sort_values('programs_per_1k_apps', ascending=False)
          [['district', 'programs_per_1k_apps', 'locality_rate_pct', 'n_distinct_programs', 'total_applications']]
          .to_string(index=False))

    n = len(out)
    print(f"\n── Districts 10, 19 vs the rest (n={n} districts) ──")
    for d in sorted(OUTLIER_DISTRICTS):
        row = out[out['district'] == d]
        if row.empty:
            print(f"  District {d}: not found")
            continue
        loc_rank = int((out['locality_rate_pct'] < row['locality_rate_pct'].iloc[0]).sum()) + 1
        spread_rank = int((out['programs_per_1k_apps'] > row['programs_per_1k_apps'].iloc[0]).sum()) + 1
        print(f"  District {d}: locality={row['locality_rate_pct'].iloc[0]:.1f}% "
              f"(rank {loc_rank}/{n} lowest), "
              f"programs_per_1k_apps={row['programs_per_1k_apps'].iloc[0]:.2f} "
              f"(rank {spread_rank}/{n} highest)")

    print(f"\n  Median locality rate (all districts):        {out['locality_rate_pct'].median():.1f}%")
    print(f"  Median programs_per_1k_apps (all districts):  {out['programs_per_1k_apps'].median():.2f}")


if __name__ == '__main__':
    main()
