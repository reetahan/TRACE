

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
    output = "min_len_1/lottery_distributions.png"

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

    x = np.arange(len(DECILE_LABELS))

    fig, ax1 = plt.subplots(figsize=(12, 5))
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
    ax1.set_ylabel('Match Rate (%)', fontsize=FONT)
    ax2.set_ylabel('Average Rank', fontsize=FONT)
    ax1.tick_params(axis='both', labelsize=FONT)
    ax2.tick_params(axis='both', labelsize=FONT)
    ax1.set_ylim(bottom=0)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=LEGEND_FONT, loc='center left')
    

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == '__main__':
    main()