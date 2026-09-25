"""
Command-line Execution Pipeline for AGED Framework
Usage:
    python run_pipeline.py --input data/sample_smartphones.csv --output data/aged_results.parquet
"""

import argparse
import sys
import time
import pandas as pd
from aged.ingestion import load_and_preprocess
from aged.absa import AspectSentimentAnalyzer
from aged.temporal_engine import compute_prior_environment_and_aged
from aged.econometrics import EconometricEngine


def run_aged_pipeline(input_path: str, output_path: str = "data/aged_processed.csv"):
    start_time = time.time()
    print("=" * 70)
    print("   AGED FRAMEWORK: ASPECT-GLOBAL EVALUATIVE DISSONANCE PIPELINE   ")
    print("=" * 70)

    # 1. Ingestion
    print(f"\n[Phase 1] Ingesting and preprocessing reviews from: {input_path}")
    raw_df = load_and_preprocess(input_path)
    print(f" -> Loaded {len(raw_df)} valid reviews across {raw_df['product_id'].nunique()} products.")
    print(f" -> Sentence segmentation generated {raw_df['sentence_count'].sum()} total sentences.")

    # 2. ABSA
    print("\n[Phase 2] Running Aspect Extraction and Sentiment Analysis (ABSA)...")
    analyzer = AspectSentimentAnalyzer()
    absa_df = analyzer.process_dataframe(raw_df)
    print(" -> ABSA completed. Aspect focal sentiment columns extracted.")

    # 3. Temporal Environment & Dissonance
    print("\n[Phase 3] Computing O(N) Chronological Prior Environment and AGED metrics...")
    aged_df = compute_prior_environment_and_aged(absa_df)
    print(f" -> Prior environments computed. Reviews with history: {(aged_df['prior_review_count'] >= 1).sum()}")
    print(f" -> Mean AGED dissonance score: {aged_df['aged_dissonance'].mean():.4f}")
    print(f" -> Mean Environmental Expectation Dissonance: {aged_df['mean_env_dissonance'].dropna().mean():.4f}")

    # Save output
    if output_path.endswith(".parquet"):
        # Drop complex nested list columns for parquet or serialize
        save_df = aged_df.drop(columns=["sentences", "sentence_evaluations"])
        save_df.to_parquet(output_path, index=False)
    else:
        save_df = aged_df.drop(columns=["sentences", "sentence_evaluations"])
        save_df.to_csv(output_path, index=False)
    print(f" -> Processed panel dataset exported to: {output_path}")

    # 4. Econometrics
    print("\n[Phase 4] Estimating Econometric Models (OLS, Moderation, Logit)...")
    econ = EconometricEngine(aged_df)
    results = econ.run_all_models()

    for m_key in ["model_1", "model_2", "model_3"]:
        m_info = results[m_key]
        if "error" in m_info:
            print(f"\n[!] {m_key.upper()} Error: {m_info['error']}")
            continue
        print("\n" + "-" * 60)
        print(f" {m_info['model_name']} ")
        print(f" Formula: {m_info['formula']}")
        print(f" R-squared: {m_info['r_squared']:.4f} | Adj R-squared: {m_info['adj_r_squared']:.4f} | N = {m_info['nobs']}")
        print("-" * 60)
        print(m_info["summary_table"].to_string(index=False))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"   PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS   ")
    print("=" * 70)
    return aged_df, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AGED Analysis Pipeline")
    parser.add_argument("--input", default="data/sample_smartphones.csv", help="Input reviews file (CSV/JSON)")
    parser.add_argument("--output", default="data/aged_processed.csv", help="Output destination file")
    args = parser.parse_args()

    run_aged_pipeline(args.input, args.output)
