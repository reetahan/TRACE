# TRACE: Trade-offs in Rules, Agency, Chance, and Equity

**TRACE** is an open-source computational framework for evaluating welfare trade-offs in centralized school assignment systems. It addresses two core challenges: the scarcity of individual-level student preference data, and the difficulty of counterfactual policy analysis.

TRACE synthesizes individual-level preference profiles from aggregate matching data using mixtures of Mallows models (a family of probabilistic models for ranked preferences) and evaluates welfare across a comprehensive set of metrics. This makes it possible to study, in advance, how design changes to an assignment system (tie-breaking rules, priority structures, list length requirements) affect different groups of students differently. A design choice that reduces unmatched rates may simultaneously lower top-ranked assignment rates for specific subgroups; TRACE makes these trade-offs visible.



---

## How It Works

![TRACE Architecture](figures/trace_architecture.png)


TRACE operates in two stages.

**1. Preference Inference**

Given aggregate statistics about student applications and match outcomes, such as the fraction of students from each residential district matched to their top-3 or top-5 schools, TRACE fits a K-component Mallows mixture model via an EM algorithm.

Each component of the mixture is defined by a central ranking $\sigma_d$ (a district-level ordering of schools, initialized by revealed preference signals in the aggregate data) and a dispersion parameter $\phi_k \in (0, 1]$ (where $\phi_k$ close to 0 indicates strong concentration around the central ranking). Mixture weights are initialized uniformly.

The EM alternates between:
- **E-step**: sampling synthetic preference lists from the current model and running Deferred Acceptance to produce simulated aggregate statistics
- **M-step**: updating the dispersion parameters $\phi_k$ via coordinate descent to reduce the gap between observed and simulated statistics

Individual submitted list lengths are modeled either from the empirical distribution (when available) or from a clipped Gaussian parameterized by summary statistics. Once a list length $L_i$ is drawn, the full Mallows ranking is truncated to its top $L_i$ entries to form the submitted preference list. Fitted parameters are saved for downstream use.

**2. Counterfactual Welfare Evaluation**

Given fitted parameters, TRACE samples synthetic preference lists, assigns priority attributes and lottery tie-breaking scores, runs Deferred Acceptance under one or more policy configurations, and computes welfare metrics stratified by residential district, borough, or demographic subgroup.

To faithfully represent complex priority systems including multiple reserve pools, heterogeneous admission methods, and intersecting eligibility categories, TRACE expands each school program into a set of virtual programs, one per seat pool and admission sub-group. Student priority attributes are sampled from distributions calibrated to the observed priority configuration. Each student–program pair is then assigned a composite priority score from reserve bucket, priority tier, and lottery tie-breaker components, and Deferred Acceptance is run over virtual programs before mapping results back to parent programs for welfare evaluation.

Welfare metrics reported by TRACE include rank distributions, top-p match rates (the fraction of students matched to one of their top-p choices), unmatched rates, and school utilization, all stratified by subgroup.

---

## Quickstart

TRACE exposes a single `TRACE` class in `src/trace.py`. Three modes are supported, auto-detected from the inputs provided.

### Mode 1 — Full EM pipeline (aggregate data)

```python
from trace import TRACE
from types import EvaluateConfig

model = TRACE(
    final_aggregates_fpath  = "data/agg_app_stats.csv",
    matching_data_fpath     = "data/match_stats.csv",
    school_data_fpath       = "data/school_info.csv",
    priority_config_fpath   = "data/priority_config.json",
)
model.preprocess()   # calls preprocess_data() in data_ingestion.py
model.fit(K=6, M=15, max_iter=20, n_jobs=32, seed=40,
          list_length_params={"list_length_mode": "gaussian",
                              "list_length_mean": 7, "list_length_std": 2,
                              "list_length_min": 1})
results = model.evaluate(config=EvaluateConfig(
    max_p=10,
    stratify_by=["subdivision"],
    output_dir="output/welfare",
))
print(f"Avg rank: {results.rank_stats['avg_rank']:.3f}")
print(f"Pct matched: {results.rank_stats['pct_matched']:.1f}%")
```

### Mode 2 — Sampling from pre-fitted parameters

```python
model = TRACE(
    mallows_params_fpath  = "output/run_params.pkl",
    matching_data_fpath   = "data/match_stats.csv",
    school_data_fpath     = "data/school_info.csv",
    priority_config_fpath = "data/priority_config.json",
)
model.set_list_length_params({"list_length_mode": "gaussian",
                              "list_length_mean": 7, "list_length_std": 2,
                              "list_length_min": 1})
model.sample()
model.run_matching(seed=42)
results = model.evaluate(config=EvaluateConfig(stratify_by=["subdivision"]))
```

### Mode 3 — Raw individual preferences

```python
model = TRACE(
    individual_data_fpath = "data/individual_prefs.csv",
    school_data_fpath     = "data/school_info.csv",
    priority_config_fpath = "data/priority_config.json",
)
# individual_data_fpath: long format, one row per (student_id, school_id, preference_number)
# Priority attribute columns (e.g. priority_sibling, priority_student) and
# stratification attributes (e.g. female) are carried through automatically.
model.run_matching(
    priority_attribute_cols=["priority_sibling", "priority_student"],
    student_attribute_cols=["female"],
    seed=42,
)
results = model.evaluate(config=EvaluateConfig(stratify_by=["subdivision", "female"]))
```

---

## Adapting TRACE to a New System

**1. Implement `preprocess_data` in `data_ingestion.py`**


```python
def preprocess_data(final_agg_df, match_stats_df, school_df, addtl_df=None):
    # Transform your raw data and return three DataFrames in TRACE generic column format.
    # See schema table below. Use to_generic() to translate if starting from a
    # system-specific preprocessor that returns em.py column names.
    ...
    return final_agg_df, match_stats_df, school_df
```

Two reference implementations are provided:

| Function | System | Notes |
|---|---|---|
| `nyc_preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df)` | NYC | Splits schools into virtual programs by seat pool, computes Rank from a revealed-preference ratio, normalizes match rates from raw counts |
| `preprocess_chilean_data(indv_df, match_df, school_cap_df, is_province_level=False)` | Chile | Constructs aggregate application stats from individual-level data, computes Rank via Borda score, aggregates school capacities by program |
| `to_generic(df)` | Both | Translates em.py internal column names to TRACE generic names; call on any DataFrame returned by the above functions |

The three returned DataFrames must have these columns:

| DataFrame | Required columns |
|---|---|
| `final_agg_df` | `school_id`, `subdivision`, `rank` |
| `school_df` | `school_id`, `capacity`, `utilization` (used for fit diagnostics) |
| `match_stats_df` | `subdivision`, `n_students`, `pct_top_{k}` (one column per k, e.g. `pct_top_3`), `pct_unmatched` |


**2. Write a priority config JSON**

```json
{
  "__meta__": { "system_name": "MyCity" },
  "system_defaults": {
    "student_attribute_fractions": {
      "disadvantaged": 0.45,
      "sibling": 0.12
    },
    "priority_tiers": [
      { "group": "disadvantaged", "tier": 1, "school_dependent": false },
      { "group": "sibling",       "tier": 2, "school_dependent": true  },
      { "group": "all",           "tier": 3 }
    ]
  },
  "school_overrides": {
    "SCHOOL_ID": {
      "reserves": { "SWD": 0.20 }
    }
  }
}
```

Use `TRACE.validate_priority_config(config)` to check for schema issues before running.

If no priority config is provided, TRACE uses plain single tie-breaking Gale-Shapley.

**Built-in evaluation metrics**

Pass any subset to `evaluate(metrics=[...])`. All run by default except `LIST_LENGTH_SWEEP`.

| `Metric` | Description |
|---|---|
| `GLOBAL_TOP_P_SWEEP` | Fraction of students matched to one of their top-p choices, for p = 1 to `max_p` |
| `RANK_DISTRIBUTION` | Full histogram of match ranks across all students |
| `RANK_STATS` | Overall average rank, rank variance, and % matched |
| `TOP_P_BY_CATEGORY` | Top-p rates stratified by each column in `stratify_by` |
| `RANK_DISTRIBUTION_BY_CATEGORY` | Rank histogram per category value |
| `RANK_STATS_BY_CATEGORY` | Average rank and variance per category value |
| `RANK_VARIANCE_BY_CATEGORY` | Rank variance (social inequity) per category, as a bar chart |
| `TOP_P_BY_LIST_LENGTH` | Top-p rates broken down by submitted list length |
| `AVG_RANK_BY_LIST_LENGTH` | Average rank ± std by submitted list length |
| `TOP_P_BY_PRIORITY_PERCENTILE` | Top-p rates by priority score percentile bin (requires `priority_col` or `priority_matrix` in `EvaluateConfig`) |
| `TOP_P_BY_CONJUNCTION` | Top-p rates at intersections of multiple categories (requires `conjunctions` in `EvaluateConfig`) |
| `LIST_LENGTH_SWEEP` | Counterfactual welfare sweep over minimum list length requirements; runs `n_stb_runs` DA simulations per length value (requires prior `fit()`) |

**3. Add custom evaluation functions**

Define any additional evaluation functions in `custom_user_functions.py` and pass them to `evaluate()`:

```python
from custom_user_functions import my_equity_metric, my_utilization_metric

results = model.evaluate(
    custom_function_list=[my_equity_metric, my_utilization_metric],
    config=EvaluateConfig(stratify_by=["subdivision"]),
)
```

Each function receives `(MatchOutcomes, EvaluateConfig)` and should return a results dict. Custom functions run alongside the built-in metric suite.

---

## Core Files

| File | Role |
|---|---|
| `src/trace.py` | **TRACE API** — main entry point for all three modes |
| `src/types.py` | Shared types: `DataKey`, `Metric`, `EvaluateConfig`, `MatchOutcomes`, `SweepResults` |
| `src/data_ingestion.py` | Data loading and preprocessing; implement `preprocess_data()` here |
| `src/priority_attributes.py` | Generic priority attribute sampling and composite rank matrix construction |
| `src/mallows.py` | Mallows model: sampling rankings and computing likelihoods |
| `src/gale_shapley.py` | Student-proposing Deferred Acceptance with configurable priority rules |
| `src/em.py` | EM algorithm: fits a Mallows mixture from aggregate statistics |
| `src/welfare.py` | Computes welfare metrics from simulated match outcomes |
| `src/list_length.py` | List length sampling utilities |
| `src/util.py` | Shared utilities |
| `src/constants.py` | Geographic and demographic mappings |
| `src/driver.py` | Top-level experiment driver |

---

## System-Specific Files

These files implement the empirical applications from the paper and serve as reference implementations for applying TRACE to a new system.

| File | Role |
|---|---|
| `src/nyc_experiment_driver.py` | End-to-end NYC inference run |
| `src/chilean_experiment_driver.py` | End-to-end Chile inference run |
| `src/nyc_list_len_welfare.py` | Sweeps minimum list length requirements for NYC |
| `src/nyc_priority_attributes.py` | NYC-specific priority matching (virtual programs, borough tiers) |
| `src/chile_priority_attributes.py` | Chile-specific priority matching |
| `src/chilean_real_welfare_comparison.py` | Observed vs. synthetic welfare comparison for Chile |
| `src/plot_lottery_pure_chance.py` | Pure-lottery welfare benchmark |
| `src/DataGeneration/NYC/` | R scripts for processing raw NYC DOE data |
| `src/DataGeneration/Chile/` | R scripts for processing raw Chilean SAE data |

---


## Data Requirements

TRACE requires aggregate matching statistics organized by student subgroup (e.g., residential district or region). For each subgroup:

- Total number of applicants
- The fraction of applicants who applied to each program, or equivalently the fraction whose top-1, top-2, ..., top-p choice was a given program (used to initialize central rankings $\sigma_d$)
- The fraction matched to their top-1, top-2, ..., top-p choice (the target moments for EM calibration)
- Program-level seat capacities and observed fill rates

These subdivision-level targets ensure the model is calibrated to subgroup-specific outcomes rather than population-level averages alone.

A **priority configuration** (JSON) specifies the school-side priority system: priority tiers, reserve pools and their seat fractions, eligibility criteria, and admission method classifications. See `sample-data/data/nyc_priority_config.json` for a worked example covering NYC's unscreened, screened, EdOpt, and zoned program types.

When priority attribute distributions are needed for counterfactual experiments (e.g., shares of students with sibling priority or disability status), these can be supplied as marginal distributions; TRACE samples individual attributes consistent with them.

The `sample-data/` directory contains processed data for the NYC and Chile applications and serves as a reference for the expected data format.

---

## Dependencies

Python 3.8+ with `numpy`, `pandas`, `scipy`, and `matplotlib`. The data preprocessing pipelines use R.

---

## True Recovered Parameters

| Parameter | Definition / Initialization | Estimated Value | Weight $w_k$ |
|---|---|---|---|
| $\sigma_d$ | Sorted by $\text{Ratio}_{d,j} = T_{d,j}^2 / A_{d,j}$ (descending) | See examples below | --- |
| $\phi_k$ | $\phi_k \sim \max(0.5, \min(\text{Beta}(6,1), 0.99))$ | See estimated values below | $1/6$ (uniform) |
| | **Estimated dispersion parameters** | | |
| $\phi_1$ | | 0.847 | $1/6$ |
| $\phi_2$ | | 0.906 | $1/6$ |
| $\phi_3$ | | 0.935 | $1/6$ |
| $\phi_4$ | | 0.961 | $1/6$ |
| $\phi_5$ | | 0.984 | $1/6$ |
| $\phi_6$ | | 0.985 | $1/6$ |
| | **Example central rankings** | | |
| $\sigma_1$ (District 1) | Bard High School Early College (Prog. 1) | 1st | --- |
| | New Explorations into Science, Technology and Math HS (Prog. 1) | 2nd | --- |
| | East Side Community School (Prog. 1) | 3rd | --- |
| $\sigma_{25}$ (District 25) | Townsend High School (Prog. 1) | 1st | --- |
| | Francis Lewis High School (Prog. 2) | 2nd | --- |
| | Bayside High School (Prog. 5) | 3rd | --- |
| | **Model fit** | | |
| Log-likelihood | | −13,344.96 | |
| MAE, top-3 match rate | | 6.6 pp | |
| MAE, program utilization | | 5.4% | |


---

## Version Notes

This is TRACE v1.0. The current release requires users to preprocess their data into a fixed schema before passing it to the API (see `preprocess_data()` in `data_ingestion.py`). A future release may include automated ingestion that handles a wider variety of raw input formats without requiring manual schema alignment.