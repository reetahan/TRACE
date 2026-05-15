import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BOROUGH_NAMES  = {'M': 'Manhattan', 'X': 'Bronx', 'K': 'Brooklyn',
                  'Q': 'Queens',    'R': 'Staten Island'}
BOROUGH_COLORS = {'M': "#035388", 'X': "#3fdff8", 'K': '#27ae60',
                  'Q': '#8e44ad', 'R': '#f39c12'}
DECILE_COLORS = {
    '1':  '#053061',  
    '2':  '#2166ac',
    '3':  '#4393c3',
    '4':  '#92c5de',
    '5':  '#d1e5f0',
    '6':  '#fddbc7',
    '7':  '#f4a582',
    '8':  '#d6604d',
    '9':  '#b2182b',
    '10': '#67001f',  
}


FONT   = 15
LEGEND_FONT   = 13
LWIDTH_OVERALL = 2.5
LWIDTH_GROUP   = 1.5
MSIZE  = 6

df         = pd.read_csv('sweep_summary.csv')
borough_df = pd.read_csv('sweep_borough.csv')
lottery_df = pd.read_csv('sweep_lottery.csv')
lottery_df['lottery_decile'] = lottery_df['lottery_decile'].str.replace('D', '', regex=False)

max_p     = borough_df['p'].max()
b_matched = borough_df[borough_df['p'] == max_p].copy()
b_matched['pct_matched'] = b_matched['top_p_pct']
b_stats   = borough_df[borough_df['p'] == 3].copy()

l_matched = lottery_df[lottery_df['p'] == max_p].copy()
l_matched['pct_matched'] = l_matched['top_p_pct']
l_stats   = lottery_df[lottery_df['p'] == 3].copy()

def plot_single_metric(overall_val, group_data, group_col, group_values,
                       group_names, group_colors, linestyle, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df['list_length_min'], overall_val,
            marker='o', color='black', linewidth=LWIDTH_OVERALL,
            markersize=MSIZE, label='Overall', zorder=5)

    for val in group_values:
        color = group_colors[val]
        label = group_names.get(val, val)
        g = group_data[group_data[group_col] == val].sort_values('list_length_min')
        if not g.empty:
            ax.plot(g['list_length_min'], g['top_p_pct'],
                    color=color, linewidth=LWIDTH_GROUP,
                    markersize=MSIZE, linestyle=linestyle, label=label)

    ax.relim()
    ax.autoscale_view()

    ax.set_ylabel(ylabel, fontsize=FONT)
    ax.tick_params(axis='both', labelsize=FONT)
    ax.legend(fontsize=LEGEND_FONT, loc='lower right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_sweep(overall_df, group_data_matched, group_data_stats,
               group_col, group_values, group_names, group_colors,
               linestyle, output_path, overall_vals=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if overall_vals is None:
        overall_vals = {
            'pct_matched': df['pct_matched'],
            'avg_rank':    df['avg_rank'],
            'top_p_pct':   overall_top3['top_p_pct'],
        }
    for ax, col in zip(axes, ['pct_matched', 'avg_rank', 'top_p_pct']):
        ax.plot(overall_df['list_length_min'], overall_vals[col],
                marker='o', color='black', linewidth=LWIDTH_OVERALL,
                markersize=MSIZE, label='Overall', zorder=5)

    for val in group_values:
        color = group_colors[val]
        label = group_names.get(val, val)
        gm = group_data_matched[group_data_matched[group_col] == val].sort_values('list_length_min')
        gs = group_data_stats[group_data_stats[group_col] == val].sort_values('list_length_min')
        print(f"val: {val}, matched rows: {len(gm)}, stats rows: {len(gs)}")
        if not gm.empty:
            axes[0].plot(gm['list_length_min'], gm['pct_matched'],
                         color=color, linewidth=LWIDTH_GROUP,
                         markersize=MSIZE, linestyle=linestyle, label=label)
        if not gs.empty:
            axes[1].plot(gs['list_length_min'], gs['avg_rank'],
                         color=color, linewidth=LWIDTH_GROUP,
                         markersize=MSIZE, linestyle=linestyle, label=label)
            axes[2].plot(gs['list_length_min'], gs['top_p_pct'],
                         color=color, linewidth=LWIDTH_GROUP,
                         markersize=MSIZE, linestyle=linestyle, label=label)

    for ax in axes:
        ax.relim()
        ax.autoscale_view()
        ax.set_xlabel('Minimum List Length', fontsize=FONT)
        ax.tick_params(axis='both', labelsize=FONT)

    axes[0].set_ylabel('% Matched', fontsize=FONT)
    axes[1].set_ylabel('Average Rank', fontsize=FONT)
    axes[2].set_ylabel('Top-3 Match Rate (%)', fontsize=FONT)

    handles, labels = axes[0].get_legend_handles_labels()
    n_items = 1 + len(group_values)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=min(n_items, 6), fontsize=LEGEND_FONT, borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


overall_top3 = (
    borough_df[borough_df['p'] == 3]
    .groupby('list_length_min')
    .apply(lambda g: np.average(g['top_p_pct'], weights=g['students']))
    .reset_index(name='top_p_pct')
)

overall_vals = {
    'pct_matched': df['pct_matched'],
    'avg_rank':    df['avg_rank'],
    'top_p_pct':   overall_top3['top_p_pct'],
}

# Figure 1 — borough breakdown
plot_sweep(
    overall_df=df,
    group_data_matched=b_matched,
    group_data_stats=b_stats,
    group_col='borough',
    group_values=['M', 'X', 'K', 'Q', 'R'],
    group_names=BOROUGH_NAMES,
    group_colors=BOROUGH_COLORS,
    linestyle='--',
    output_path='unmatched_avgrank_top3_min_list_length_borough.png',
    overall_vals=overall_vals
)

# Figure 2 — lottery decile breakdown
plot_sweep(
    overall_df=df,
    group_data_matched=l_matched,
    group_data_stats=l_stats,
    group_col='lottery_decile',
    group_values=[f'{i}' for i in range(1, 11)],
    group_names={f'{i}': f'{i}'.strip(' ()') 
                 for i in range(1, 11)},
    group_colors=DECILE_COLORS,
    linestyle=':',
    output_path='unmatched_avgrank_top3_min_list_length_lottery.png',
    overall_vals=overall_vals
)

# Fig 6 — borough
min_len_min = df['list_length_min'].min()
min_len_max = df['list_length_min'].max()
pct_at_min = df[df['list_length_min'] == min_len_min]['pct_matched'].values[0]
pct_at_max = df[df['list_length_min'] == min_len_max]['pct_matched'].values[0]

borough_gains = {}
for b in ['M', 'X', 'K', 'Q', 'R']:
    sub = b_matched[b_matched['borough'] == b].sort_values('list_length_min')
    if not sub.empty:
        borough_gains[b] = sub['pct_matched'].max() - sub['pct_matched'].min()

most_benefit = max(borough_gains, key=borough_gains.get)
least_benefit = min(borough_gains, key=borough_gains.get)
print(f"\n── Fig 6 Summary ─────────────────────────────")
print(f"  Overall % matched at min_len={min_len_min}: {pct_at_min:.1f}%")
print(f"  Overall % matched at min_len={min_len_max}: {pct_at_max:.1f}%")
print(f"  Borough benefiting most:     {BOROUGH_NAMES[most_benefit]} ({borough_gains[most_benefit]:+.1f}pp)")
print(f"  Borough benefiting least:    {BOROUGH_NAMES[least_benefit]} ({borough_gains[least_benefit]:+.1f}pp)")
overall_top3_baseline = overall_top3[overall_top3['list_length_min'] == min_len_min]['top_p_pct'].values[0]
overall_top3_max = overall_top3[overall_top3['list_length_min'] == min_len_max]['top_p_pct'].values[0]
print(f"  Overall top-3 rate at baseline (min_len={min_len_min}): {overall_top3_baseline:.1f}%")
print(f"  Overall top-3 rate at max (min_len={min_len_max}):      {overall_top3_max:.1f}%")
print(f"  Borough match rates at baseline (min_len={min_len_min}):")
baseline_borough = b_matched[b_matched['list_length_min'] == min_len_min]
for b in ['M', 'X', 'K', 'Q', 'R']:
    sub = baseline_borough[baseline_borough['borough'] == b]
    if not sub.empty:
        print(f"    {BOROUGH_NAMES[b]}: {sub['pct_matched'].values[0]:.1f}%")
print(f"  Borough match rates at max (min_len={min_len_max}):")
max_borough = b_matched[b_matched['list_length_min'] == min_len_max]
for b in ['M', 'X', 'K', 'Q', 'R']:
    sub = max_borough[max_borough['borough'] == b]
    if not sub.empty:
        print(f"    {BOROUGH_NAMES[b]}: {sub['pct_matched'].values[0]:.1f}%")

print(f"  Borough top-3 rates at baseline (min_len={min_len_min}):")
baseline_borough_stats = b_stats[b_stats['list_length_min'] == min_len_min]
for b in ['M', 'X', 'K', 'Q', 'R']:
    sub = baseline_borough_stats[baseline_borough_stats['borough'] == b]
    if not sub.empty:
        print(f"    {BOROUGH_NAMES[b]}: {sub['top_p_pct'].values[0]:.1f}%")

print(f"  Borough top-3 rates at max (min_len={min_len_max}):")
max_borough_stats = b_stats[b_stats['list_length_min'] == min_len_max]
for b in ['M', 'X', 'K', 'Q', 'R']:
    sub = max_borough_stats[max_borough_stats['borough'] == b]
    if not sub.empty:
        print(f"    {BOROUGH_NAMES[b]}: {sub['top_p_pct'].values[0]:.1f}%")
print(f"──────────────────────────────────────────────\n")

print(f"  Overall avg rank at baseline (min_len={min_len_min}): {df[df['list_length_min'] == min_len_min]['avg_rank'].values[0]:.2f}")
print(f"  Overall avg rank at max (min_len={min_len_max}):      {df[df['list_length_min'] == min_len_max]['avg_rank'].values[0]:.2f}")

print(f"  Borough avg rank at baseline (min_len={min_len_min}):")
for b in ['M', 'X', 'K', 'Q', 'R']:
    sub = b_stats[(b_stats['list_length_min'] == min_len_min) & (b_stats['borough'] == b)]
    if not sub.empty:
        print(f"    {BOROUGH_NAMES[b]}: {sub['avg_rank'].values[0]:.2f}")

print(f"  Borough avg rank at max (min_len={min_len_max}):")
for b in ['M', 'X', 'K', 'Q', 'R']:
    sub = b_stats[(b_stats['list_length_min'] == min_len_max) & (b_stats['borough'] == b)]
    if not sub.empty:
        print(f"    {BOROUGH_NAMES[b]}: {sub['avg_rank'].values[0]:.2f}")

# Fig 7 — lottery decile
baseline_top3 = l_stats[l_stats['list_length_min'] == min_len_min]
decile_gains = {}
for d in [f'{i}' for i in range(1, 11)]:
    sub = l_matched[l_matched['lottery_decile'] == d].sort_values('list_length_min')
    if not sub.empty:
        decile_gains[d] = sub['pct_matched'].max() - sub['pct_matched'].min()

most_benefit_d = max(decile_gains, key=decile_gains.get)
least_benefit_d = min(decile_gains, key=decile_gains.get)
top3_at_baseline = baseline_top3.groupby('lottery_decile')['top_p_pct'].first()
print(f"\n── Fig 7 Summary ─────────────────────────────")
print(f"  Top-3 rate range at baseline: {top3_at_baseline.min():.1f}% (D{top3_at_baseline.idxmin()}) to {top3_at_baseline.max():.1f}% (D{top3_at_baseline.idxmax()})")
print(f"  Decile benefiting most from longer lists: D{most_benefit_d} ({decile_gains[most_benefit_d]:+.1f}pp)")
print(f"  Decile benefiting least:                  D{least_benefit_d} ({decile_gains[least_benefit_d]:+.1f}pp)")
print(f"  Decile match rates at baseline (min_len={min_len_min}):")
baseline_lottery = l_matched[l_matched['list_length_min'] == min_len_min]
for d in [f'{i}' for i in range(1, 11)]:
    sub = baseline_lottery[baseline_lottery['lottery_decile'] == d]
    if not sub.empty:
        print(f"    D{d}: {sub['pct_matched'].values[0]:.1f}%")

print(f"  Decile match rates at max (min_len={min_len_max}):")
max_lottery = l_matched[l_matched['list_length_min'] == min_len_max]
for d in [f'{i}' for i in range(1, 11)]:
    sub = max_lottery[max_lottery['lottery_decile'] == d]
    if not sub.empty:
        print(f"    D{d}: {sub['pct_matched'].values[0]:.1f}%")

print(f"  Decile top-3 rates at baseline (min_len={min_len_min}):")
baseline_lottery_stats = l_stats[l_stats['list_length_min'] == min_len_min]
for d in [f'{i}' for i in range(1, 11)]:
    sub = baseline_lottery_stats[baseline_lottery_stats['lottery_decile'] == d]
    if not sub.empty:
        print(f"    D{d}: {sub['top_p_pct'].values[0]:.1f}%")

print(f"  Decile top-3 rates at max (min_len={min_len_max}):")
max_lottery_stats = l_stats[l_stats['list_length_min'] == min_len_max]
for d in [f'{i}' for i in range(1, 11)]:
    sub = max_lottery_stats[max_lottery_stats['lottery_decile'] == d]
    if not sub.empty:
        print(f"    D{d}: {sub['top_p_pct'].values[0]:.1f}%")
print(f"──────────────────────────────────────────────\n")

print(f"  Decile avg rank at baseline (min_len={min_len_min}):")
for d in [f'{i}' for i in range(1, 11)]:
    sub = l_stats[(l_stats['list_length_min'] == min_len_min) & (l_stats['lottery_decile'] == d)]
    if not sub.empty:
        print(f"    D{d}: {sub['avg_rank'].values[0]:.2f}")

print(f"  Decile avg rank at max (min_len={min_len_max}):")
for d in [f'{i}' for i in range(1, 11)]:
    sub = l_stats[(l_stats['list_length_min'] == min_len_max) & (l_stats['lottery_decile'] == d)]
    if not sub.empty:
        print(f"    D{d}: {sub['avg_rank'].values[0]:.2f}")

# ── Write plot-ready CSVs ─────────────────────────────────────────────────
fig6 = b_matched[['borough', 'list_length_min', 'pct_matched']].merge(
    b_stats[['borough', 'list_length_min', 'top_p_pct', 'avg_rank']],
    on=['borough', 'list_length_min'], how='outer'
).merge(
    df[['list_length_min', 'pct_matched', 'avg_rank']].rename(
        columns={'pct_matched': 'overall_pct_matched', 'avg_rank': 'overall_avg_rank'}),
    on='list_length_min', how='left'
).merge(
    overall_top3.rename(columns={'top_p_pct': 'overall_top3_pct'}),
    on='list_length_min', how='left'
)
fig6.to_csv('fig6_plot_data.csv', index=False)
print("Saved: fig6_plot_data.csv")

fig7 = l_matched[['lottery_decile', 'list_length_min', 'pct_matched']].merge(
    l_stats[['lottery_decile', 'list_length_min', 'top_p_pct', 'avg_rank']],
    on=['lottery_decile', 'list_length_min'], how='outer'
).merge(
    df[['list_length_min', 'pct_matched', 'avg_rank']].rename(
        columns={'pct_matched': 'overall_pct_matched', 'avg_rank': 'overall_avg_rank'}),
    on='list_length_min', how='left'
).merge(
    overall_top3.rename(columns={'top_p_pct': 'overall_top3_pct'}),
    on='list_length_min', how='left'
)
fig7.to_csv('fig7_plot_data.csv', index=False)
print("Saved: fig7_plot_data.csv")