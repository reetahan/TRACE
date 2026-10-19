import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

NYC_COLOR   = '#2C3E7A'
CHILE_COLOR = '#C0602A'
FONT        = 11

# Candidate label offsets (display points), tried nearest-first, for collision avoidance.
_OFFSET_CANDIDATES = [
    (5, 3), (5, -8), (-18, 3), (-18, -8),
    (12, 10), (12, -14), (-24, 10), (-24, -14),
    (20, 18), (20, -22), (-30, 18), (-30, -22),
]


def _bbox_overlaps(b1, b2, pad=2.5):
    return not (b1.x1 + pad < b2.x0 or b2.x1 + pad < b1.x0 or
                b1.y1 + pad < b2.y0 or b2.y1 + pad < b1.y0)


def place_labels_no_collision(ax, fig, entries):
    """
    entries: list of (x, y, text, color) in data coords.
    Places each label at the nearest offset (from _OFFSET_CANDIDATES) that doesn't
    overlap any already-placed label's bounding box, then draws a thin leader line
    from the point to the label so it's unambiguous which dot each label belongs to,
    even once nudged away from the default fixed offset.
    """
    renderer = fig.canvas.get_renderer()
    placed_bboxes = []

    for x, y, text, color in entries:
        chosen_offset = _OFFSET_CANDIDATES[0]
        chosen_bbox = None
        for offset in _OFFSET_CANDIDATES:
            ann = ax.annotate(text, (x, y), textcoords='offset points', xytext=offset,
                               fontsize=9, color=color, fontweight='bold', zorder=6)
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer)
            if not any(_bbox_overlaps(bbox, pb) for pb in placed_bboxes):
                chosen_offset = offset
                chosen_bbox = bbox
                break
            ann.remove()
        else:
            # every candidate collided — keep the last attempt rather than dropping the label
            chosen_bbox = bbox

        placed_bboxes.append(chosen_bbox)

        # Always draw a leader line from point to label -- in a dense cluster, proximity
        # alone doesn't tell a reader which dot a label belongs to, even when label
        # bounding boxes don't overlap each other.
        label_disp = ax.transData.transform((x, y))
        label_disp = (label_disp[0] + chosen_offset[0], label_disp[1] + chosen_offset[1])
        label_data = ax.transData.inverted().transform(label_disp)
        ax.plot([x, label_data[0]], [y, label_data[1]], color=color, lw=0.6,
                alpha=0.7, zorder=5)


def make_scatter(fig, ax, nyc_data, chile_data, obs_col, sim_col, label, panel_caption,
                 nyc_outlier_n, chile_outlier_n, font_scale=1.0):

    nx_o = nyc_data[obs_col].values
    nx_s = nyc_data[sim_col].values
    ch_o = chile_data[obs_col].values
    ch_s = chile_data[sim_col].values

    all_v = np.concatenate([nx_o, nx_s, ch_o, ch_s])
    lo, hi = np.nanmin(all_v) - 4, np.nanmax(all_v) + 4

    ax.plot([lo, hi], [lo, hi], color='#999999', lw=1, ls='--', zorder=1)

    ax.scatter(ch_o, ch_s, marker='^', s=34, color=CHILE_COLOR,
               alpha=0.70, linewidths=0, zorder=3, label='Chile (region)')
    ax.scatter(nx_o, nx_s, marker='o', s=30, color=NYC_COLOR,
               alpha=0.75, linewidths=0, zorder=3, label='NYC (district)')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    ax.set_xlabel(f'Observed {label}', fontsize=FONT * font_scale)
    ax.set_ylabel(f'Simulated {label}', fontsize=FONT * font_scale)
    ax.tick_params(labelsize=10 * font_scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Worst NYC + worst Chile, placed together so labels can't collide across groups either.
    nyc_ann = nyc_data.copy()
    nyc_ann['abs_diff'] = (nyc_ann[obs_col] - nyc_ann[sim_col]).abs()
    chile_ann = chile_data.copy()
    chile_ann['abs_diff'] = (chile_ann[obs_col] - chile_ann[sim_col]).abs()

    entries = []
    for _, row in nyc_ann.nlargest(nyc_outlier_n, 'abs_diff').iterrows():
        entries.append((row[obs_col], row[sim_col], str(int(row['district'])), NYC_COLOR))
    for _, row in chile_ann.nlargest(chile_outlier_n, 'abs_diff').iterrows():
        name = re.sub(r'^de[l]?\s+', '', str(row['district']).replace("Región", "").strip())
        entries.append((row[obs_col], row[sim_col], name, CHILE_COLOR))

    fig.canvas.draw()
    place_labels_no_collision(ax, fig, entries)

    ax.set_title(panel_caption, fontsize=12, y=-0.26)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nyc_fit',    required=True, help='nyc_val_district_fit_1.csv')
    parser.add_argument('--chile_fit',  required=True, help='chile_val_district_fit_1.csv')
    parser.add_argument('--nyc_util',   required=True, help='nyc_val_district_util_1.csv')
    parser.add_argument('--chile_util', required=True, help='chile_val_district_util_1.csv')
    parser.add_argument('--nyc_top3_outliers',   type=int, default=2)
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

    fig, (ax_top3, ax_util) = plt.subplots(1, 2, figsize=(13, 5.2),
                                            gridspec_kw={'wspace': 0.12})

    make_scatter(
        fig, ax_top3, nyc_fit, chile_fit,
        obs_col='obs_top3', sim_col='sim_top3',
        label='Top-3 Match Rate (%)',
        panel_caption=r'(a) Top-3 match rate $W_3(\mu \mid d)$.',
        nyc_outlier_n=args.nyc_top3_outliers,
        chile_outlier_n=args.chile_top3_outliers,
    )

    make_scatter(
        fig, ax_util, nyc_util, chile_util,
        obs_col='obs_util', sim_col='sim_util',
        label='Program Utilization (%)',
        panel_caption=r'(b) Program utilization $\bar{u}_d(\mu)$.',
        nyc_outlier_n=args.nyc_util_outliers,
        chile_outlier_n=args.chile_util_outliers,
    )

    handles, labels = ax_top3.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=13, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, markerscale=1.2, handletextpad=0.4,
               columnspacing=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / 'fig2_validation_scatter.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    def print_mae(label, nyc_data, chile_data, obs_col, sim_col):
        nyc_mae = (nyc_data[obs_col] - nyc_data[sim_col]).abs().mean()
        chile_mae = (chile_data[obs_col] - chile_data[sim_col]).abs().mean()
        combined = pd.concat([nyc_data[[obs_col, sim_col]], chile_data[[obs_col, sim_col]]])
        all_mae = (combined[obs_col] - combined[sim_col]).abs().mean()
        print(f"  {label}: NYC MAE={nyc_mae:.2f}pp, Chile MAE={chile_mae:.2f}pp, combined MAE={all_mae:.2f}pp")

    print("\n=== MAE (|observed - simulated|) ===")
    print_mae('Top-3 match rate', nyc_fit, chile_fit, 'obs_top3', 'sim_top3')
    print_mae('Program utilization', nyc_util, chile_util, 'obs_util', 'sim_util')

    # Separate single-panel versions of each, for standalone use.
    for ax_label, nyc_data, chile_data, obs_col, sim_col, label, nyc_n, chile_n, fname in [
        ('top3', nyc_fit, chile_fit, 'obs_top3', 'sim_top3', 'Top-3 Match Rate (%)',
         args.nyc_top3_outliers, args.chile_top3_outliers, 'val_scatter_top3.png'),
        ('util', nyc_util, chile_util, 'obs_util', 'sim_util', 'Program Utilization (%)',
         args.nyc_util_outliers, args.chile_util_outliers, 'val_scatter_util.png'),
    ]:
        fig_single, ax_single = plt.subplots(figsize=(7, 6))
        make_scatter(
            fig_single, ax_single, nyc_data, chile_data,
            obs_col=obs_col, sim_col=sim_col, label=label, panel_caption='',
            nyc_outlier_n=nyc_n, chile_outlier_n=chile_n, font_scale=1.15,
        )
        ax_single.set_title('')
        h, l = ax_single.get_legend_handles_labels()
        ax_single.legend(h, l, fontsize=14, frameon=False, loc='upper left',
                          markerscale=1.2, handletextpad=0.4)
        fig_single.tight_layout()
        single_path = out_dir / fname
        fig_single.savefig(single_path, dpi=180, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved: {single_path}")
