import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import pandas as pd
import numpy as np

df = pd.read_csv('fig7_plot_data.csv')
overall = df[['list_length_min','overall_pct_matched','overall_avg_rank','overall_top3_pct']].drop_duplicates()
x_all = overall['list_length_min'].values
df['pct_unmatched'] = 100.0 - df['overall_pct_matched']
deciles = sorted(df['lottery_decile'].unique())

# Blue (low decile) -> Red (high decile) colormap
cmap = cm.get_cmap('RdYlBu_r')
DECILE_COLORS = {d: cmap((d-1)/9) for d in deciles}

LABEL_FS = 12
TICK_FS  = 11
LEG_FS   = 10

fig, axes = plt.subplots(1, 3, figsize=(11, 2.5), sharey=False)

metrics = [
    ('pct_matched', 'overall_pct_matched',  '% Matched',            axes[0]),
    ('avg_rank',    'overall_avg_rank',      'Average Rank (matched)',         axes[1]),
    ('top_p_pct',   'overall_top3_pct',      'Top-3 Match Rate (%)', axes[2]),
]

for col, ov_col, ylabel, ax in metrics:
    ax.plot(x_all, overall[ov_col].values, color='black', lw=2,
            marker='o', ms=4, zorder=5, label='Overall')
    for d in deciles:
        sub = df[df['lottery_decile'] == d].sort_values('list_length_min')
        ax.plot(sub['list_length_min'], sub[col], color=DECILE_COLORS[d],
                lw=1.2, ls=':', label=str(d))
    ax.set_xlabel('Minimum List Length', fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[1].yaxis.set_label_coords(-0.08, 0.42)
# Shared legend: Overall + deciles 1-10 in two rows
handles = [mlines.Line2D([], [], color='black', lw=2, marker='o', ms=4, label='Overall')]
for d in deciles:
    handles.append(mlines.Line2D([], [], color=DECILE_COLORS[d], lw=1.2, ls=':', label=str(d)))

fig.legend(handles=handles, fontsize=11, frameon=False, ncol=11,
           loc="upper center", bbox_to_anchor=(0.5, 1.10),
           handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

fig.tight_layout()
fig.savefig('fig7_nyc_lottery.png',
            dpi=180, bbox_inches='tight')
plt.close(fig)
print('Done fig7')
