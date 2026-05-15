

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

NYC_COLOR   = '#2C3E7A'
CHILE_COLOR = '#C0602A'
FONT        = 9


def make_scatter(nyc_data, chile_data, obs_col, sim_col, label,
                 nyc_outlier_n, chile_outlier_n, out_path):

    nx_o = nyc_data[obs_col].values
    nx_s = nyc_data[sim_col].values
    ch_o = chile_data[obs_col].values
    ch_s = chile_data[sim_col].values

    all_v = np.concatenate([nx_o, nx_s, ch_o, ch_s])
    lo, hi = np.nanmin(all_v) - 4, np.nanmax(all_v) + 4

    #fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot([lo, hi], [lo, hi], color='#999999', lw=1, ls='--', zorder=1)

    ax.scatter(ch_o, ch_s, marker='^', s=34, color=CHILE_COLOR,
               alpha=0.70, linewidths=0, zorder=3, label='Chile (region)')
    ax.scatter(nx_o, nx_s, marker='o', s=30, color=NYC_COLOR,
               alpha=0.75, linewidths=0, zorder=3, label='NYC (district)')

    # Annotate worst NYC
    nyc_ann = nyc_data.copy()
    nyc_ann['abs_diff'] = (nyc_ann[obs_col] - nyc_ann[sim_col]).abs()
    for _, row in nyc_ann.nlargest(nyc_outlier_n, 'abs_diff').iterrows():
        ax.annotate(str(int(row['district'])), (row[obs_col], row[sim_col]),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7.5, color=NYC_COLOR, fontweight='bold')

    # Annotate worst Chile
    chile_ann = chile_data.copy()
    chile_ann['abs_diff'] = (chile_ann[obs_col] - chile_ann[sim_col]).abs()
    for _, row in chile_ann.nlargest(chile_outlier_n, 'abs_diff').iterrows():
        ax.annotate(str(row['district']).replace("Region de", "").replace("Región de", "").replace("Región", ""), (row[obs_col], row[sim_col]),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7.5, color=CHILE_COLOR, fontweight='bold')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Observed {label}', fontsize=FONT)
    ax.set_ylabel(f'Simulated {label}', fontsize=FONT)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc='upper left',
              markerscale=1.2, handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {Path(out_path).name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nyc_fit',    required=True, help='nyc_val_district_fit.csv')
    parser.add_argument('--chile_fit',  required=True, help='chile_val_district_fit.csv')
    parser.add_argument('--nyc_util',   required=True, help='nyc_val_district_util.csv')
    parser.add_argument('--chile_util', required=True, help='chile_val_district_util.csv')
    parser.add_argument('--nyc_top3_outliers',   type=int, default=4)
    parser.add_argument('--chile_top3_outliers', type=int, default=1)
    parser.add_argument('--nyc_util_outliers',   type=int, default=1)
    parser.add_argument('--chile_util_outliers', type=int, default=1)
    parser.add_argument('--out', default='.')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nyc_fit   = pd.read_csv(args.nyc_fit)
    chile_fit = pd.read_csv(args.chile_fit)
    nyc_util  = pd.read_csv(args.nyc_util)
    chile_util = pd.read_csv(args.chile_util)

    make_scatter(
        nyc_fit, chile_fit,
        obs_col='obs_top3', sim_col='sim_top3',
        label='Top-3 Match Rate (%)',
        nyc_outlier_n=args.nyc_top3_outliers,
        chile_outlier_n=args.chile_top3_outliers,
        out_path=out_dir / 'val_scatter_top3.png'
    )

    make_scatter(
        nyc_util, chile_util,
        obs_col='obs_util', sim_col='sim_util',
        label='Program Utilization (%)',
        nyc_outlier_n=args.nyc_util_outliers,
        chile_outlier_n=args.chile_util_outliers,
        out_path=out_dir / 'val_scatter_util.png'
    )