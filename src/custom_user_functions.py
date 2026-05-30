from types_trace import DataKey, MatchOutcomes, EvaluateConfig
from data_ingestion import nyc_preprocess_data, to_generic

# ── Column mappings ───────────────────────────────────────────────────────────
# Map your column names -> TRACE generic names. Applied automatically on load.
# Required generic names:
#   FINAL_AGGREGATES : school_id, subdivision, rank
#   MATCH_STATS      : subdivision, n_students, pct_unmatched, pct_top_{k}
#   SCHOOL           : school_id, capacity, utilization
#   INDIVIDUAL       : student_id, school_id, preference_number

COLUMN_MAPS = {
    DataKey.FINAL_AGGREGATES: {},
    DataKey.MATCH_STATS:      {},
    DataKey.SCHOOL:           {},
    DataKey.INDIVIDUAL:       {},
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Called by model.preprocess(). Receives DataFrames already renamed via
# COLUMN_MAPS. Must return (final_agg_df, match_stats_df, school_df) in
# TRACE generic column format. 
def preprocess_data(final_agg_df, match_stats_df, school_df, addtl_df=None):
    # Implement your preprocessing here.
    # Must return (final_agg_df, match_stats_df, school_df) in TRACE generic column format.
    # Pass this function to model.preprocess(fn=preprocess_data) in your notebook.
    raise NotImplementedError("Implement preprocess_data() in custom_user_functions.py.")

# ── Custom evaluation functions ───────────────────────────────────────────────
# Each receives (MatchOutcomes, EvaluateConfig) and returns a dict.
# Pass to: model.evaluate(custom_function_list=[my_metric])
# Access via: results.custom_results['my_metric']

# def my_metric(outcomes: MatchOutcomes, config: EvaluateConfig) -> dict:
#     return {'pct_matched': float((outcomes.matches_idx >= 0).mean()) * 100}