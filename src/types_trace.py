from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd


class Metric(Enum):
    GLOBAL_TOP_P_SWEEP              = "global_top_p_sweep"
    RANK_DISTRIBUTION               = "rank_distribution"
    RANK_STATS                      = "rank_stats"
    TOP_P_BY_CATEGORY               = "top_p_by_category"
    RANK_DISTRIBUTION_BY_CATEGORY   = "rank_distribution_by_category"
    RANK_STATS_BY_CATEGORY          = "rank_stats_by_category"
    RANK_VARIANCE_BY_CATEGORY       = "rank_variance_by_category"
    TOP_P_BY_LIST_LENGTH            = "top_p_by_list_length"
    AVG_RANK_BY_LIST_LENGTH         = "avg_rank_by_list_length"
    TOP_P_BY_PRIORITY_PERCENTILE    = "top_p_by_priority_percentile"
    TOP_P_BY_CONJUNCTION            = "top_p_by_conjunction"
    LIST_LENGTH_SWEEP               = "list_length_sweep"
    MTB_VS_STB                      = "mtb_vs_stb"


class DataKey(str, Enum):
    INDIVIDUAL       = "individual_df"
    SCHOOL           = "school_df"
    MATCH_STATS      = "match_stats_df"
    FINAL_AGGREGATES = "final_aggregates_df"


REQUIRED_COLUMNS: dict[DataKey, list[str]] = {
    DataKey.FINAL_AGGREGATES: ['school_id', 'subdivision', 'rank'],
    DataKey.MATCH_STATS:      ['subdivision', 'n_students', 'pct_unmatched'],
    DataKey.SCHOOL:           ['school_id', 'capacity'],
    DataKey.INDIVIDUAL:       ['student_id', 'school_id', 'preference_number'],
}


@dataclass
class EvaluateConfig:
    max_p: int                          = 15
    stratify_by: list[str] | None       = None
    output_dir: str | None              = None
    show_plots: bool                    = False
    conjunctions: list[list[str]]       = field(default_factory=list)
    priority_col: str | None            = None
    priority_matrix: np.ndarray | None  = None
    priority_higher_is_better: bool     = True
    n_priority_bins: int                = 10
    sweep_min_lengths: list[int]        = field(default_factory=lambda: list(range(1, 6)))
    sweep_n_jobs: int                   = 1
    n_stb_runs: int                     = 5


@dataclass
class MatchOutcomes:
    rankings_as_indices: list[np.ndarray] 
    matches_idx: np.ndarray                #  -1 = unmatched
    student_attributes: pd.DataFrame   


@dataclass
class SweepResults:
    """
    Welfare results across a minimum-list-length sweep.
    """
    results_by_min_len: dict[int, list[Any]]

    @property
    def min_lengths(self) -> list[int]:
        return sorted(self.results_by_min_len.keys())

    def __getitem__(self, min_len: int) -> list[Any]:
        return self.results_by_min_len[min_len]

    def summary(self) -> pd.DataFrame:
        rows = []
        for min_len in self.min_lengths:
            wrs = self.results_by_min_len[min_len]
            avg_ranks   = [wr.rank_stats['avg_rank']    for wr in wrs]
            pct_matched = [wr.rank_stats['pct_matched'] for wr in wrs]
            rows.append({
                'min_len':          min_len,
                'avg_rank_mean':    float(np.mean(avg_ranks)),
                'avg_rank_std':     float(np.std(avg_ranks)),
                'pct_matched_mean': float(np.mean(pct_matched)),
                'pct_matched_std':  float(np.std(pct_matched)),
                'n_runs':           len(wrs),
            })
        return pd.DataFrame(rows)