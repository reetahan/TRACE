"""
Hybrid versions of Fig 24 (STB-MTB diff bar, synthetic profiles -- recreation
of Fig 6) and Fig 25 (region cumulative buckets, synthetic profiles --
recreation of Fig 8): bar heights/means come from the single fixed-sample run
on Fig 16's saved preferences (chile_fig16_sample_individual.csv via
--individual, n_stb_runs=1), but the error bars come from the ORIGINAL
independent N=10-replicate average runs (fig6_chile_synthetic_diff_bar.csv /
fig8_chile_region_cumulative_synthetic.csv). Same idea as
hybrid_validation_bars.py, applied to these two chart types' own plotting
code (copied verbatim from chilean_real_welfare_comparison.py /
chile_region_cumulative_stb_mtb.py so the styling matches exactly).

Pure merge + replot -- no simulation, both source CSVs already exist on disk.

Usage:
    python3 src/project_specific_scripts/hybrid_chile_synthetic_figs24_25.py \
        --fig24_mean_csv fig6_chile_synthetic_diff_bar_fig16sample.csv \
        --fig24_ci_csv fig6_chile_synthetic_diff_bar.csv \
        --fig24_out fig6_chile_synthetic_diff_bar_hybrid.png \
        --fig25_mean_csv fig8_chile_region_cumulative_synthetic_fig16sample.csv \
        --fig25_ci_csv fig8_chile_region_cumulative_synthetic.csv \
        --fig25_out fig8_chile_region_cumulative_synthetic_hybrid.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chile_region_cumulative_stb_mtb import ROMAN_ORDER


def plot_fig24_hybrid(mean_csv, ci_csv, output_path):
    mean_df = pd.read_csv(mean_csv).set_index('label')
    ci_df = pd.read_csv(ci_csv).set_index('label')
    labels = list(mean_df.index)

    stb_mean = mean_df['stb_mean']
    mtb_mean = mean_df['mtb_mean']
    stb_ci = ci_df['stb_ci95']
    mtb_ci = ci_df['mtb_ci95']

    diffs = (stb_mean - mtb_mean).tolist()
    diff_ci95 = [float(np.sqrt(stb_ci[l] ** 2 + mtb_ci[l] ** 2)) for l in labels]

    dump = pd.DataFrame({
        'label': labels,
        'stb_mean': stb_mean.values, 'stb_ci95': stb_ci.reindex(labels).values,
        'mtb_mean': mtb_mean.values, 'mtb_ci95': mtb_ci.reindex(labels).values,
        'diff': diffs, 'diff_ci95': diff_ci95,
    })
    dump_path = output_path.rsplit('.', 1)[0] + '.csv'
    dump.to_csv(dump_path, index=False)
    print(f"Saved: {dump_path}")

    x = np.arange(len(labels))
    STB_COLOR = '#1a2f6e'
    MTB_COLOR = '#c1652f'

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [STB_COLOR if d >= 0 else MTB_COLOR for d in diffs]
    colors[-1] = MTB_COLOR  # W_unmatch always orange, regardless of sign
    ax.bar(x, diffs, color=colors, width=0.65, yerr=diff_ci95, capsize=5,
           error_kw={'ecolor': '#000000', 'elinewidth': 2})
    ax.axhline(0, color='black', linewidth=0.8)

    ps = [l for l in labels if l != 'Unm.']
    flip_idx = next((i for i in range(1, len(ps)) if diffs[i] < 0 <= diffs[i - 1]), None)
    if flip_idx is not None:
        ax.axvline(flip_idx - 0.5, color='gray', linestyle=':', linewidth=1)

    tick_labels = [f'$W_{{{l}}}$' if l != 'Unm.' else '$W_{\\mathrm{unmatch}}$' for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Matched Rank', fontsize=12)
    ax.set_ylabel('STB - MTB (pp)', fontsize=12)

    bar_tops = [d + e for d, e in zip(diffs, diff_ci95)] + [0]
    bar_bottoms = [d - e for d, e in zip(diffs, diff_ci95)] + [0]
    y_max, y_min = max(bar_tops), min(bar_bottoms)
    pad = 0.22 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.text(0.02, 0.97, '← STB better', transform=ax.transAxes,
            ha='left', va='top', color=STB_COLOR, fontsize=11, fontweight='bold')
    ax.text(0.98, 0.03, 'MTB better →', transform=ax.transAxes,
            ha='right', va='bottom', color=MTB_COLOR, fontsize=11, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_fig25_hybrid(mean_csv, ci_csv, output_path):
    mean_df = pd.read_csv(mean_csv)
    ci_df = pd.read_csv(ci_csv)

    regions = [r for r in ROMAN_ORDER if r in mean_df['region'].values and r in ci_df['region'].values]

    def get_buckets(df, cond):
        return df[df['condition'] == cond].set_index('region')

    stb_mean = get_buckets(mean_df, 'STB')
    mtb_mean = get_buckets(mean_df, 'MTB')
    stb_ci = get_buckets(ci_df, 'STB')
    mtb_ci = get_buckets(ci_df, 'MTB')

    dump_rows = []
    for buckets, prefix in [(stb_mean, 'STB'), (mtb_mean, 'MTB')]:
        ci_src = stb_ci if prefix == 'STB' else mtb_ci
        merged = buckets.loc[regions, ['top1', 'top2_5', 'top6p']].join(
            ci_src.loc[regions, ['top1', 'top2_5', 'top6p']], rsuffix='_ci95'
        )
        merged['condition'] = prefix
        dump_rows.append(merged.reset_index())
    dump_path = output_path.rsplit('.', 1)[0] + '.csv'
    pd.concat(dump_rows, ignore_index=True).to_csv(dump_path, index=False)
    print(f"Saved: {dump_path}")

    STB_COLORS = {'top1': '#1a2f6e', 'top2_5': '#3f6fd6', 'top6p': '#a9c0f0'}
    MTB_COLORS = {'top1': '#7a2a10', 'top2_5': '#c1652f', 'top6p': '#f0c19b'}

    x = np.arange(len(regions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 5))

    for offset, buckets, ci95_buckets, colors, prefix in [
        (-width / 2, stb_mean, stb_ci, STB_COLORS, 'STB'),
        (width / 2, mtb_mean, mtb_ci, MTB_COLORS, 'MTB'),
    ]:
        bottoms = np.zeros(len(regions))
        for seg, seg_label in [('top1', 'Top 1'), ('top2_5', 'Top 2-5'), ('top6p', 'Top 6+')]:
            vals = np.array([buckets.loc[r, seg] for r in regions])
            errs = np.array([ci95_buckets.loc[r, f'{seg}_ci95'] for r in regions])
            ax.bar(x + offset, vals, width, bottom=bottoms, color=colors[seg],
                   label=f'{prefix} - {seg_label}', yerr=errs,
                   error_kw={'ecolor': '#000000', 'elinewidth': 1.6, 'capsize': 3.5})
            bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig24_mean_csv', required=True)
    parser.add_argument('--fig24_ci_csv', required=True)
    parser.add_argument('--fig24_out', required=True)
    parser.add_argument('--fig25_mean_csv', required=True)
    parser.add_argument('--fig25_ci_csv', required=True)
    parser.add_argument('--fig25_out', required=True)
    args = parser.parse_args()

    plot_fig24_hybrid(args.fig24_mean_csv, args.fig24_ci_csv, args.fig24_out)
    plot_fig25_hybrid(args.fig25_mean_csv, args.fig25_ci_csv, args.fig25_out)


if __name__ == '__main__':
    main()
