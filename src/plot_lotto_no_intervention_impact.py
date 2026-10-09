

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FONT   = 15
LEGEND_FONT   = 12
MSIZE  = 8
LWIDTH = 2.5

DECILE_LABELS = [f'D{i}' for i in range(1, 11)]


def compute_decile_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decile in DECILE_LABELS:
        sub = df[df['lottery_decile'] == decile]
        n   = len(sub)
        if n == 0:
            continue

        matched      = sub['matched'].sum()
        pct_unmatched = 100.0 * (n - matched) / n
        top1_pct     = 100.0 * (sub['match_rank'] == 1).sum() / n
        top5_pct     = 100.0 * (sub['match_rank'] <= 5).sum() / n
        avg_rank     = sub.loc[sub['matched'], 'match_rank'].mean()

        rows.append({
            'decile':       decile,
            'pct_unmatched': pct_unmatched,
            'top1_pct':     top1_pct,
            'top5_pct':     top5_pct,
            'avg_rank':     avg_rank,
            'n':            n,
        })
    return pd.DataFrame(rows).set_index('decile')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                         help='Sweep output dir, e.g. the --output_dir passed to nyc_list_len_welfare.py.')
    args = parser.parse_args()

    run_paths = sorted(glob.glob(os.path.join(args.dir, "min_len_1/run_*/student_level.csv")))
    if not run_paths:
        run_paths = [os.path.join(args.dir, "min_len_1/student_level.csv")]
    output = os.path.join(args.dir, "min_len_1/lottery_distributions.png")

    metrics_by_run = []
    for path in run_paths:
        df = pd.read_csv(path)
        required = {'matched', 'match_rank', 'lottery_decile'}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        metrics_by_run.append(compute_decile_metrics(df))

    n_runs = len(metrics_by_run)
    print(f"  Replicates: {n_runs}")
    concat = pd.concat(metrics_by_run)
    mean = concat.groupby(level=0).mean().reindex(DECILE_LABELS)
    std  = concat.groupby(level=0).std().reindex(DECILE_LABELS)
    ci95 = 1.96 * std / np.sqrt(n_runs)

    print(mean.to_string())

    overall_match_rate = 100.0 - mean['pct_unmatched'].mean()
    d1 = mean.loc['D1', 'top5_pct']
    d10 = mean.loc['D10', 'top5_pct']
    print(f"\n── Fig 5 Summary ─────────────────────────────")
    print(f"  Top-5 rate D1:               {d1:.1f}%")
    print(f"  Top-5 rate D10:              {d10:.1f}%")
    print(f"  D1 vs D10 top-5 diff:        {d1-d10:+.1f}pp")
    print(f"  Avg rank range:              {mean['avg_rank'].min():.2f} (D1) to {mean['avg_rank'].max():.2f} (D10)")
    print(f"──────────────────────────────────────────────\n")

    x = np.arange(len(DECILE_LABELS))

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # Left axis — match rates (%)
    ax1.errorbar(x, mean['pct_unmatched'], yerr=ci95['pct_unmatched'], marker='o', linewidth=LWIDTH,
         markersize=MSIZE, color='black', label='% Unmatched',
         ecolor='#000000', elinewidth=2, capsize=5, capthick=2)
    ax1.errorbar(x, mean['top1_pct'], yerr=ci95['top1_pct'], marker='s', linewidth=LWIDTH,
            markersize=MSIZE, color='#2166ac', linestyle='--', label='Top-1 Match %',
            ecolor='#000000', elinewidth=2, capsize=5, capthick=2)
    ax1.errorbar(x, mean['top5_pct'], yerr=ci95['top5_pct'], marker='^', linewidth=LWIDTH,
            markersize=MSIZE, color='#4393c3', linestyle='--', label='Top-5 Match %',
            ecolor='#000000', elinewidth=2, capsize=5, capthick=2)

    # Right axis — average rank
    ax2.errorbar(x, mean['avg_rank'], yerr=ci95['avg_rank'], marker='D', linewidth=LWIDTH,
             markersize=MSIZE, color='#b2182b', linestyle=':',
             label='Avg Rank (matched)',
             ecolor='#000000', elinewidth=2, capsize=5, capthick=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.replace('D', '') for d in DECILE_LABELS], fontsize=FONT)
    ax1.set_ylabel('Match Rate (%)', fontsize=FONT)
    ax2.set_ylabel('Average Rank', fontsize=FONT)
    ax1.tick_params(axis='both', labelsize=FONT)
    ax2.tick_params(axis='both', labelsize=FONT)
    ax1.set_ylim(bottom=0)

    # Combined legend, single line, above the plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    fig.legend(all_lines, all_labels, fontsize=LEGEND_FONT, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), ncol=len(all_labels), frameon=True)
    fig.subplots_adjust(top=0.85)

    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == '__main__':
    main()
