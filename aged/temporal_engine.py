"""
Module 3: Temporal Environment & AGED Engine
Computes chronological prior review environments and Aspect-Global Evaluative Dissonance (AGED).
Optimized for O(N) linear time execution using cumulative running statistics.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from aged.config import DEFAULT_ASPECT_TAXONOMY


def normalize_rating_to_sentiment_scale(rating: float) -> float:
    """
    Normalizes 1-5 star ratings to [-1.0, 1.0] scale to match sentiment polarity:
    1 star -> -1.0
    2 stars -> -0.5
    3 stars ->  0.0
    4 stars -> +0.5
    5 stars -> +1.0
    """
    return float((rating - 3.0) / 2.0)


def compute_prior_environment_and_aged(
    df: pd.DataFrame,
    taxonomy: Optional[Dict[str, List[str]]] = None,
    min_prior_reviews: int = 1
) -> pd.DataFrame:
    """
    Constructs the prior review environment for each focal review and calculates
    Aspect-Prior Mismatch and Aspect-Global Evaluative Dissonance (AGED).

    Directly implements the paper's pipeline:
    - Prior Aspect Environment Aggregation (Time t)
    - Focal Review Aspect Evaluation
    - Dissonance / Mismatch Calculation: |Current Aspect Eval - Prior Aspect Env|
    - AGED (Aspect vs Global Star Rating Discrepancy)
    """
    aspects = list(taxonomy.keys()) if taxonomy else list(DEFAULT_ASPECT_TAXONOMY.keys())
    out_df = df.sort_values(by=["product_id", "date", "review_id"]).copy()

    # Pre-allocate output lists for fast row-by-row assembly
    prior_counts = []
    prior_rating_means = []
    prior_rating_vars = []

    # Aspect prior values: prior_env_<aspect> and prior_vol_<aspect>
    prior_aspect_envs = {a: [] for a in aspects}
    prior_aspect_vols = {a: [] for a in aspects}

    # Aspect mismatch values: mismatch_<aspect> and abs_dissonance_<aspect>
    mismatch_signed = {a: [] for a in aspects}
    dissonance_abs = {a: [] for a in aspects}

    mean_env_dissonances = []
    aged_scores = []
    aged_signed_scores = []
    rating_normalized_list = []

    # Process grouped by product_id in O(N) linear time per product
    for product_id, group in out_df.groupby("product_id", sort=False):
        # Running accumulators for overall star rating
        cum_count = 0
        cum_rating_sum = 0.0
        cum_rating_sq_sum = 0.0

        # Running accumulators for each aspect
        aspect_cum_count = {a: 0 for a in aspects}
        aspect_cum_sum = {a: 0.0 for a in aspects}

        for _, row in group.iterrows():
            focal_rating = float(row["rating"])
            norm_rating = normalize_rating_to_sentiment_scale(focal_rating)
            rating_normalized_list.append(norm_rating)

            # 1. Compute PRIOR environment before this review (time t)
            if cum_count >= min_prior_reviews:
                p_mean = cum_rating_sum / cum_count
                # Sample variance formula
                p_var = (cum_rating_sq_sum - (cum_rating_sum ** 2) / cum_count) / max(1, cum_count - 1)
                p_var = max(0.0, float(p_var))
            else:
                p_mean = np.nan
                p_var = np.nan

            prior_counts.append(cum_count)
            prior_rating_means.append(p_mean)
            prior_rating_vars.append(p_var)

            # Prior aspect environment for each aspect
            row_aspect_dissonances = []

            for a in aspects:
                a_count = aspect_cum_count[a]
                if a_count >= 1:
                    p_aspect_mean = aspect_cum_sum[a] / a_count
                else:
                    p_aspect_mean = np.nan

                prior_aspect_envs[a].append(p_aspect_mean)
                prior_aspect_vols[a].append(a_count)

                # Focal review's evaluation for aspect a
                focal_val = row.get(f"focal_{a}", np.nan)
                if pd.notna(focal_val) and pd.notna(p_aspect_mean):
                    diff = float(focal_val - p_aspect_mean)
                    abs_diff = abs(diff)
                    mismatch_signed[a].append(diff)
                    dissonance_abs[a].append(abs_diff)
                    row_aspect_dissonances.append(abs_diff)
                else:
                    mismatch_signed[a].append(np.nan)
                    dissonance_abs[a].append(np.nan)

            # Mean Aspect-Prior Environmental Dissonance across mentioned aspects
            if row_aspect_dissonances:
                mean_env_dissonances.append(float(np.mean(row_aspect_dissonances)))
            else:
                mean_env_dissonances.append(np.nan)

            # AGED: Discrepancy between Focal Aspect Experience and Global Star Rating
            focal_aspect_mean = row.get("mean_aspect_sentiment", np.nan)
            if pd.notna(focal_aspect_mean):
                aged_signed = norm_rating - float(focal_aspect_mean)
                aged_val = abs(aged_signed)
            else:
                aged_signed = np.nan
                aged_val = np.nan

            aged_scores.append(aged_val)
            aged_signed_scores.append(aged_signed)

            # 2. UPDATE cumulative accumulators AFTER recording the prior state
            cum_count += 1
            cum_rating_sum += focal_rating
            cum_rating_sq_sum += (focal_rating ** 2)

            for a in aspects:
                focal_val = row.get(f"focal_{a}", np.nan)
                if pd.notna(focal_val):
                    aspect_cum_count[a] += 1
                    aspect_cum_sum[a] += float(focal_val)

    # Attach computed variables to DataFrame
    out_df["rating_normalized"] = rating_normalized_list
    out_df["prior_review_count"] = prior_counts
    out_df["prior_rating_mean"] = prior_rating_means
    out_df["prior_rating_variance"] = prior_rating_vars
    out_df["mean_env_dissonance"] = mean_env_dissonances
    out_df["aged_dissonance"] = aged_scores
    out_df["aged_signed_gap"] = aged_signed_scores

    for a in aspects:
        out_df[f"prior_env_{a}"] = prior_aspect_envs[a]
        out_df[f"prior_vol_{a}"] = prior_aspect_vols[a]
        out_df[f"mismatch_{a}"] = mismatch_signed[a]
        out_df[f"dissonance_{a}"] = dissonance_abs[a]

    # Create composite expectation mismatch index
    out_df["has_prior_history"] = out_df["prior_review_count"] >= min_prior_reviews

    return out_df
