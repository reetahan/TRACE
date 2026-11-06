"""
Concatenates two min-list-length sweep output directories' sweep_borough.csv /
sweep_lottery.csv / sweep_summary.csv (e.g. an existing 1-10 sweep plus a new
11-15 sweep) into one combined set of CSVs, so convert_sweep_csv_to_plot.py
can be run once over the full range without re-running the 1-10 portion.

Usage:
    python3 src/project_specific_scripts/merge_minlen_sweep_csvs.py \
        --dir1 new_figs_regen_10_8 --dir2 nyc_minlen_sweep_11to15 \
        --out nyc_minlen_sweep_1to15_merged
"""
import argparse
from pathlib import Path

import pandas as pd

FILES = ['sweep_borough.csv', 'sweep_lottery.csv', 'sweep_summary.csv']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', required=True)
    parser.add_argument('--dir2', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        p1 = Path(args.dir1) / fname
        p2 = Path(args.dir2) / fname
        if not p1.exists() or not p2.exists():
            print(f"Skipping {fname}: missing in {'dir1' if not p1.exists() else 'dir2'}")
            continue
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        overlap = set(df1['list_length_min'].unique()) & set(df2['list_length_min'].unique())
        if overlap:
            raise ValueError(f"{fname}: overlapping list_length_min values between dirs: {overlap}")
        merged = pd.concat([df1, df2], ignore_index=True).sort_values('list_length_min')
        merged.to_csv(out_dir / fname, index=False)
        print(f"Saved: {out_dir / fname}  ({len(df1)} + {len(df2)} = {len(merged)} rows, "
              f"list_length_min range {merged['list_length_min'].min()}-{merged['list_length_min'].max()})")


if __name__ == '__main__':
    main()
