import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

df = pd.read_csv('nyc_lottery_distributions_decile_metrics.csv')
deciles = list(range(1, 11))

UNMATCHED_COLOR = '#111111'
TOP1_COLOR      = '#1565C0'
TOP5_COLOR      = '#AD1457'
RANK_COLOR      = '#E65100'

fig, ax1 = plt.subplots(figsize=(5.0, 2.9))  # reduced height

ax2 = ax1.twinx()

l1, = ax1.plot(deciles, df['pct_unmatched'], color=UNMATCHED_COLOR,
               ls='-',  marker='o', ms=4, lw=1.4, label='Unmatched')
l2, = ax1.plot(deciles, df['top1_pct'],      color=TOP1_COLOR,
               ls='--', marker='s', ms=4, lw=1.4, label='Top-1')
l3, = ax1.plot(deciles, df['top5_pct'],      color=TOP5_COLOR,
               ls=':',  marker='^', ms=4, lw=1.4, label='Top-5')
l4, = ax2.plot(deciles, df['avg_rank'],      color=RANK_COLOR,
               ls='-.', marker='D', ms=4, lw=1.4, label='Average Rank (matched)')

ax1.set_xlabel('Lottery Decile', fontsize=10)
ax1.set_ylabel('Match Rate (%)', fontsize=10)
ax2.set_ylabel('Average Rank (matched)', fontsize=10, color=RANK_COLOR)
ax2.tick_params(axis='y', labelcolor=RANK_COLOR)
ax1.set_xticks(deciles)
ax1.tick_params(labelsize=9)
ax2.tick_params(labelsize=9)
ax1.set_ylim(-2, 105)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=9.5, frameon=False, ncol=4,
           loc='upper center', bbox_to_anchor=(0.45, 1.18),
           handlelength=1.8, handletextpad=0.4, columnspacing=1.0)

fig.tight_layout()
fig.savefig('fig3_nyc_lottery.png',
            dpi=180, bbox_inches='tight')
plt.close(fig)
print('Done fig3')
