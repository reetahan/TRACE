import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

df = pd.read_csv('fig5_plot_data.csv')
df['pct_unmatched'] = 100.0 - df['overall_pct_matched']
BOROUGH_MAP = {'K': 'Brooklyn', 'M': 'Manhattan', 'Q': 'Queens', 'R': 'Staten Island', 'X': 'Bronx'}
# Colors matching original: Manhattan=blue, Bronx=cyan, Brooklyn=green, Queens=purple, Staten Island=orange
BOROUGH_COLORS = {
    'M': '#1565C0',   # Manhattan - dark blue
    'X': '#00ACC1',   # Bronx - cyan
    'K': '#2E7D32',   # Brooklyn - green
    'Q': '#6A1B9A',   # Queens - purple
    'R': '#E65100',   # Staten Island - orange
}

overall = df[['list_length_min','overall_pct_matched','overall_avg_rank','overall_top3_pct']].drop_duplicates()
x_all = overall['list_length_min'].values

LABEL_FS = 11
TICK_FS  = 10
LEG_FS   = 10

fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharey=False)

metrics = [
    ('pct_matched',  'overall_pct_matched',  '% Matched',          axes[0]),
    ('avg_rank',     'overall_avg_rank',      'Average Rank (matched)',       axes[1]),
    ('top_p_pct',    'overall_top3_pct',      'Top-3 Match Rate (%)', axes[2]),
]

for col, ov_col, ylabel, ax in metrics:
    # Overall solid black with markers
    ax.plot(x_all, overall[ov_col].values, color='black', lw=2,
            marker='o', ms=4, zorder=5, label='Overall')
    # Boroughs as dashed
    for bcode, bname in BOROUGH_MAP.items():
        sub = df[df['borough'] == bcode].sort_values('list_length_min')
        ax.plot(sub['list_length_min'], sub[col], color=BOROUGH_COLORS[bcode],
                lw=1.4, ls='--', label=bname)
    ax.set_xlabel('Minimum List Length', fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared legend above
handles = [mlines.Line2D([], [], color='black', lw=2, marker='o', ms=4, label='Overall')]
for bcode, bname in [('M','Manhattan'),('X','Bronx'),('K','Brooklyn'),('Q','Queens'),('R','Staten Island')]:
    handles.append(mlines.Line2D([], [], color=BOROUGH_COLORS[bcode], lw=1.4, ls='--', label=bname))

fig.legend(handles=handles, fontsize=12, frameon=False, ncol=6,
           loc='upper center', bbox_to_anchor=(0.5, 1.06),
           handlelength=1.8, handletextpad=0.4, columnspacing=1.2)

fig.tight_layout()
fig.savefig('fig5_nyc_borough.png',
            dpi=180, bbox_inches='tight')
plt.close(fig)
print('Done fig5')
