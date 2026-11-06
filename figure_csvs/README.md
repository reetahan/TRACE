# Figure 2-8 source CSVs

| file | figure | source |
|---|---|---|
| `fig2_nyc_fit.csv` | Fig 2a (top3 scatter, NYC side) | `experimental_output/nyc_res_logs/20260501_144617/validation_plots/nyc_val_district_fit_1.csv` |
| `fig2_nyc_util.csv` | Fig 2b (utilization scatter, NYC side) | `experimental_output/nyc_res_logs/20260501_144617/validation_plots/nyc_val_utilization_fresh_district_util_1.csv` |
| `fig2_chile_fit.csv` | Fig 2a (top3 scatter, Chile side) | `experimental_output/chile_res_logs/20261014_102833/validation_plots/chile_val_district_fit_1.csv` |
| `fig2_chile_util.csv` | Fig 2b (utilization scatter, Chile side) | `experimental_output/chile_res_logs/20261014_102833/validation_plots/chile_val_district_util_1.csv` |
| `fig3_nyc.csv` | Fig 3 (W1/W5/Wunmatch/avg-rank by lottery decile, natural list length) | `nyc_minlen_sweep_1to15_full/sweep_lottery.csv`, filtered to `list_length_min == 1` |
| `fig4_chile.csv` | Fig 4 (lottery decile, STB vs MTB, real data) | `fig4_chile_lottery_decile_real_min_k5.csv` |
| `fig5_nyc.csv` | Fig 5 (min list length sweep, by borough) | `nyc_minlen_sweep_1to15_full/sweep_borough.csv` |
| `fig6_chile.csv` | Fig 6 (STB-MTB diff bar, real data) | `fig6_chile_real_welfare_diff_bar.csv` |
| `fig7_nyc.csv` | Fig 7 (min list length sweep, by lottery decile) | `nyc_minlen_sweep_1to15_full/sweep_lottery.csv` (full, unfiltered -- same source as Fig 3, different slice) |
| `fig8_chile.csv` | Fig 8 (region cumulative, STB vs MTB, real data) | `fig8_chile_region_cumulative.csv` |

Fig 3 and Fig 7 come from the same underlying sweep (`nyc_minlen_sweep_1to15_full/sweep_lottery.csv`); Fig 3 is the `list_length_min == 1` (natural, unmodified list length) slice, Fig 7 is the full sweep across `list_length_min` 1-15.

Fig 4 has several older variant CSVs still sitting in the repo root (`fig4_chile_lottery_decile_*.csv`) from earlier iterations -- not cleaned up, `fig4_chile.csv` here is the current real-data one (`_real_min_k5` suffix).
