import numpy as np

def return_nyc_list_params_per_district(df, match_stats_df, std=2, list_length_min=1):
    """
    Compute per-district mean list length as total_applications / total_students,
    using the existing aggregated data already in df and match_stats_df.
    """
    total_apps_by_district = (
        df.groupby('Residential District')['Total Applicants by Residential District']
        .sum()
    )
    total_students_by_district = (
        match_stats_df.set_index('Residential District')['Total Applicants']
    )
    mean_per_district = (total_apps_by_district / total_students_by_district).to_dict()
    mean_per_district = {str(k): float(v) for k, v in mean_per_district.items()}

    return {
        "list_length_mode": "gaussian_per_district",
        "list_length_mean_per_district": mean_per_district,
        "list_length_std": std,
        "list_length_min": list_length_min,
    }

def sample_truncated_normal_lengths(
    n_students,
    mean=10,
    std=2,
    min_len=1,
    max_len=None,
    rng=None
):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.normal(loc=mean, scale=std, size=n_students)
    lengths = np.rint(x).astype(int)
    lengths = np.clip(lengths, min_len, max_len) if max_len is not None else np.clip(lengths, min_len)
    return lengths


def sample_empirical_lengths(n_students, empirical_probs, rng):
    lengths = np.array(list(empirical_probs.keys()), dtype=int)
    probs   = np.array(list(empirical_probs.values()), dtype=float)
    probs  /= probs.sum()  
    return rng.choice(lengths, size=n_students, p=probs).astype(int)

def return_nyc_list_params(std=None, list_length_min=None):
    res = {
        "list_length_mode": "gaussian",
        "list_length_mean": 7,
        "list_length_std": 2,
        "list_length_min": 1,
    }
    if(std is not None):
        res["list_length_std"] = std
    if(list_length_min is not None):
        res["list_length_min"] = list_length_min
    return res

def return_chilean_list_params(indv_df):
    list_lengths = indv_df.groupby('mrun')['preference_number'].max()
    counts = list_lengths.value_counts().sort_index()
    empirical_probs = (counts / counts.sum()).to_dict()
    return {
        "list_length_mode": "empirical",
        "list_length_empirical_probs": empirical_probs,
    }

def return_list_params():
    pass