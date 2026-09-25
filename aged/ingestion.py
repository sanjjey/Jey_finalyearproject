"""
Module 1: Ingestion & Preprocessing
Handles parsing, schema normalization, chronological sorting, and sentence segmentation.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import nltk
from datetime import datetime

# Column mappings to accommodate standard datasets (Amazon, Yelp, Kaggle, etc.)
STANDARD_COL_MAP = {
    "review_id": ["review_id", "id", "reviewid", "asin_review_id", "review_idx"],
    "product_id": ["product_id", "asin", "item_id", "product", "parent_asin"],
    "date": ["date", "timestamp", "review_date", "time", "review_time"],
    "rating": ["rating", "overall", "stars", "star_rating", "score"],
    "text": ["text", "review_text", "body", "review", "content", "comments"]
}


def _match_column(df_columns: List[str], candidate_keys: List[str]) -> Optional[str]:
    """Finds the best matching column name case-insensitively."""
    lower_cols = {col.lower().strip(): col for col in df_columns}
    for cand in candidate_keys:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and normalizes dataframe column names to canonical schema:
    [review_id, product_id, date, rating, text]
    """
    rename_dict = {}
    for canonical_name, candidates in STANDARD_COL_MAP.items():
        matched = _match_column(df.columns.tolist(), candidates)
        if matched:
            rename_dict[matched] = canonical_name
        else:
            if canonical_name == "review_id":
                # Auto-generate review IDs if missing
                df["review_id"] = [f"rev_{i}" for i in range(len(df))]
                rename_dict["review_id"] = "review_id"
            elif canonical_name in ["product_id", "date", "rating", "text"]:
                raise ValueError(
                    f"Required column for '{canonical_name}' could not be identified in columns: {list(df.columns)}"
                )

    normalized_df = df.rename(columns=rename_dict).copy()
    return normalized_df[list(STANDARD_COL_MAP.keys())]


def segment_sentences(text: str) -> List[str]:
    """Splits a review text into grammatical sentences."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        sentences = nltk.sent_tokenize(text.strip())
    except Exception:
        # Fallback if punkt tokenizer encounters unhandled patterns
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def load_and_preprocess(filepath_or_df: Any) -> pd.DataFrame:
    """
    Loads, cleans, chronologically sorts, and segments raw review data.
    Ensures linear O(N) downstream processing by calculating review sequence order.
    """
    if isinstance(filepath_or_df, str):
        if filepath_or_df.endswith(".csv"):
            df = pd.read_csv(filepath_or_df)
        elif filepath_or_df.endswith(".json") or filepath_or_df.endswith(".jsonl"):
            df = pd.read_json(filepath_or_df, lines=filepath_or_df.endswith(".jsonl"))
        elif filepath_or_df.endswith(".parquet"):
            df = pd.read_parquet(filepath_or_df)
        else:
            df = pd.read_csv(filepath_or_df, sep=None, engine="python")
    elif isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df.copy()
    else:
        raise ValueError("Unsupported data input type. Expected file path or pandas DataFrame.")

    # 1. Normalize schema
    df = normalize_schema(df)

    # 2. Clean types and handle missing values
    df = df.dropna(subset=["product_id", "rating", "text"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(float)
    df["text"] = df["text"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["review_id"] = df["review_id"].astype(str)

    # 3. Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # If some dates fail to parse, fill with sequential pseudo-dates
    if df["date"].isna().any():
        df["date"] = df["date"].fillna(pd.Timestamp("2024-01-01"))

    # 4. Chronological sort per product
    df = df.sort_values(by=["product_id", "date", "review_id"]).reset_index(drop=True)

    # 5. Compute sequential review order (1, 2, ... N) per product
    df["review_order"] = df.groupby("product_id").cumcount() + 1

    # 6. Sentence segmentation
    df["sentences"] = df["text"].apply(segment_sentences)
    df["sentence_count"] = df["sentences"].apply(len)
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))

    return df
