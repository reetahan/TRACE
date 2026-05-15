

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
    return pd.DataFrame(rows)

def main():

    df = pd.read_csv("min_len_1/student_level.csv")
    output = "nyc_lottery_distributions.png"

    # Validate required columns
    required = {'matched', 'match_rank', 'lottery_decile'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"student_level.csv is missing columns: {missing}")

    print(f"  Students: {len(df):,}")
    print(f"  Matched:  {df['matched'].sum():,} ({100*df['matched'].mean():.1f}%)")
    print(f"  Deciles present: {sorted(df['lottery_decile'].dropna().unique())}")

    metrics = compute_decile_metrics(df)
    print(metrics.to_string(index=False))

    metrics_out = metrics.copy()
    metrics_out['pct_matched'] = 100.0 - metrics_out['pct_unmatched']
    metrics_out.to_csv(output.replace('.png', '_decile_metrics.csv'), index=False)
    print(f"Saved: {output.replace('.png', '_decile_metrics.csv')}")

    overall_match_rate = 100.0 * df['matched'].mean()
    d1 = metrics[metrics['decile'] == 'D1']['top5_pct'].values[0]
    d10 = metrics[metrics['decile'] == 'D10']['top5_pct'].values[0]
    print(f"\n── Fig 5 Summary ─────────────────────────────")
    print(f"  Overall match rate:          {overall_match_rate:.1f}%")
    print(f"  Top-5 rate D1:               {d1:.1f}%")
    print(f"  Top-5 rate D10:              {d10:.1f}%")
    print(f"  D1 vs D10 top-5 diff:        {d1-d10:+.1f}pp")
    print(f"  Avg rank range:              {metrics['avg_rank'].min():.2f} (D1) to {metrics['avg_rank'].max():.2f} (D10)")
    print(f"\n── Match Rate by Decile ───────────────────────")
    for _, row in metrics.iterrows():
        pct_matched = 100.0 - row['pct_unmatched']
        print(f"  {row['decile']}: matched={pct_matched:.1f}%  top1={row['top1_pct']:.1f}%  top5={row['top5_pct']:.1f}%  avg_rank={row['avg_rank']:.2f}")
    print(f"──────────────────────────────────────────────\n")

    x = np.arange(len(DECILE_LABELS))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    # Left axis — match rates (%)
    ax1.plot(x, metrics['pct_unmatched'], marker='o', linewidth=LWIDTH,
         markersize=MSIZE, color='black', label='% Unmatched')
    ax1.plot(x, metrics['top1_pct'], marker='s', linewidth=LWIDTH,
            markersize=MSIZE, color='#2166ac', linestyle='--', label='Top-1 Match %')
    ax1.plot(x, metrics['top5_pct'], marker='^', linewidth=LWIDTH,
            markersize=MSIZE, color='#4393c3', linestyle='--', label='Top-5 Match %')

    # Right axis — average rank
    ax2.plot(x, metrics['avg_rank'], marker='D', linewidth=LWIDTH,
             markersize=MSIZE, color='#b2182b', linestyle=':',
             label='Avg Rank (matched)')

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.replace('D', '') for d in DECILE_LABELS], fontsize=FONT)
    ax1.set_xlabel('Lottery Decile', fontsize=FONT)
    ax1.set_ylabel('Match Rate (%)', fontsize=FONT)
    ax2.set_ylabel('Average Rank', fontsize=FONT)
    ax1.tick_params(axis='both', labelsize=FONT)
    ax2.tick_params(axis='both', labelsize=FONT)
    ax1.set_ylim(bottom=0)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=LEGEND_FONT, loc='upper center', bbox_to_anchor=(0.5, 1.28),
          ncol=2, borderaxespad=0)
    

    fig.tight_layout(rect=[0, 0.05, 1, 0.85])
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()