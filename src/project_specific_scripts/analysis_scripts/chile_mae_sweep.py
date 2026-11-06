import sys, os, re, time
sys.path.insert(0, 'src')
from validation_generation import parse_log, aggregate_to_region, aggregate_scalar_to_region
from constants import CHILE_PROVINCE_TO_REGION_MAPPING
import pandas as pd
import numpy as np

LOG_ROOT = 'experimental_output/chile_res_logs'
PROVINCE_FILE = 'sample-data/data/chilean_data_processed/matching_outcome_by_province.xlsx'
SANTIAGO_FILE = 'sample-data/data/chilean_data_processed/matching_outcome_by_subdivision_santiago_macrozones.xlsx'

province_df = pd.read_excel(PROVINCE_FILE)
province_students = {str(k): int(v) for k, v in province_df.set_index('Provincia')['n_students'].to_dict().items()}

santiago_df = pd.read_excel(SANTIAGO_FILE)
santiago_students = {str(k): int(v) for k, v in santiago_df.set_index('Provincia')['n_students'].to_dict().items()}

def parse_config(fname):
    cfg = {}
    for key in ['K', 'M', 'iter', 'opt', 'lr', 'maxp', 'seed', 'loss']:
        m = re.search(rf'{key}=([^_]+)', fname)
        if m:
            cfg[key] = m.group(1)
    return cfg

TARGETS = [
    '20261014_102833',  # current ALLTOPP run (post-fix), in progress
    '20260922_210358',  # Sept22 reference (pre-fix, no priority_config)
    '20260514_002222',  # best May run
    '20260513_172106',  # May province, non-Santiago-split
    '20260515_013745',  # May Santiago-split
    '20260512_030350',  # exception run (priority loaded, healthy)
]

results = []
for d in TARGETS:
    dirpath = os.path.join(LOG_ROOT, d)
    if not os.path.isdir(dirpath):
        print(f"MISSING: {d}")
        continue
    txts = [f for f in os.listdir(dirpath) if f.endswith('.txt')]
    if not txts:
        continue
    # for dirs with multiple logs (timestamp collisions), prefer the one matching ALLTOPP/allTopP
    fname = txts[0]
    if len(txts) > 1:
        preferred = [f for f in txts if 'allTopP' in f or ('loss=' not in f)]
        if preferred:
            fname = preferred[0]
    logpath = os.path.join(dirpath, fname)
    t0 = time.time()
    try:
        best_ll, best_block, best_util, best_mae_util, min_mae_util, best_dist_util = parse_log(logpath)
    except Exception as e:
        results.append({'dir': d, 'logfile': fname, 'error': str(e)[:100]})
        continue
    elapsed = time.time() - t0
    print(f"{d} ({fname}) parsed in {elapsed:.1f}s", flush=True)

    if not best_block:
        results.append({'dir': d, 'logfile': fname, 'error': 'no best_block'})
        continue

    keys = [k for k in best_block.keys() if k != 'Utilization']
    n_keys = len(keys)
    is_region_level = all(k.startswith('Región') for k in keys)
    has_santiago_split = any(k.startswith('Santiago_') for k in keys)

    try:
        if is_region_level:
            region_block = best_block
            region_util = best_dist_util
        else:
            students = santiago_students if has_santiago_split else province_students
            region_block = aggregate_to_region(best_block, CHILE_PROVINCE_TO_REGION_MAPPING, students)
            region_util = aggregate_scalar_to_region(best_dist_util, CHILE_PROVINCE_TO_REGION_MAPPING, students) if best_dist_util else {}
    except Exception as e:
        results.append({'dir': d, 'logfile': fname, 'error': f'aggregation failed: {e}'[:120]})
        continue

    sample_vals = next(iter(region_block.values()))
    n_stats = len(sample_vals['obs'])
    if n_stats == 4:
        metrics_here = ['top3', 'top5', 'top10', 'unmatched']
    else:
        metrics_here = [f'top{p}' for p in range(1, n_stats)] + ['unmatched']
    idx_map = {k: i for i, k in enumerate(metrics_here)}

    top3_mae = None
    if 'top3' in idx_map:
        ti = idx_map['top3']
        diffs = [abs(v['obs'][ti] - v['sim'][ti]) for v in region_block.values()]
        top3_mae = float(np.mean(diffs))

    util_mae = None
    if region_util:
        diffs_u = [abs(v['obs'] - v['sim']) for v in region_util.values()]
        util_mae = float(np.mean(diffs_u))

    cfg = parse_config(fname)
    results.append({
        'dir': d, 'logfile': fname,
        'K': cfg.get('K'), 'M': cfg.get('M'), 'iter': cfg.get('iter'), 'opt': cfg.get('opt'),
        'lr': cfg.get('lr'), 'maxp': cfg.get('maxp'), 'loss': cfg.get('loss', 'allTopP'),
        'n_districts_raw': n_keys,
        'granularity': 'region' if is_region_level else ('province_santiago_split' if has_santiago_split else 'province'),
        'n_regions_agg': len(region_block),
        'best_ll': best_ll,
        'top3_mae_region': top3_mae,
        'util_mae_region': util_mae,
    })

df = pd.DataFrame(results)
df.to_csv('chile_mae_comparison_vs_references.csv', index=False)
print()
print("=== Comparison, sorted by top3_mae_region ===")
with pd.option_context('display.max_rows', None, 'display.width', 200):
    print(df.sort_values('top3_mae_region', na_position='last').to_string(index=False))
