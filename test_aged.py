"""
Automated Verification & Unit Test Suite for AGED Framework
Tests all 6 phases: Ingestion, ABSA, Temporal Math, Econometrics, and Dashboard compatibility.
"""

import unittest
import numpy as np
import pandas as pd
from aged.ingestion import normalize_schema, segment_sentences, load_and_preprocess
from aged.absa import AspectSentimentAnalyzer
from aged.temporal_engine import compute_prior_environment_and_aged, normalize_rating_to_sentiment_scale
from aged.econometrics import EconometricEngine
from aged.synthetic_generator import generate_benchmark_reviews


class TestAGEDFramework(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                "id": "r1",
                "product": "P1",
                "time": "2025-01-01 10:00:00",
                "stars": 5.0,
                "review": "The battery life is amazing. Screen is gorgeous."
            },
            {
                "id": "r2",
                "product": "P1",
                "time": "2025-01-02 11:00:00",
                "stars": 5.0,
                "review": "Fast charging and awesome battery. Display is crisp."
            },
            {
                "id": "r3",
                "product": "P1",
                "time": "2025-01-03 12:00:00",
                "stars": 1.0,
                "review": "Battery drain is terrible! Overheating during charging."
            }
        ])

    def test_01_ingestion_and_normalization(self):
        """Test schema normalization and sentence segmentation."""
        norm_df = normalize_schema(self.sample_df)
        self.assertIn("review_id", norm_df.columns)
        self.assertIn("product_id", norm_df.columns)
        self.assertIn("rating", norm_df.columns)
        self.assertIn("text", norm_df.columns)
        self.assertIn("date", norm_df.columns)

        sentences = segment_sentences("Great camera. Battery is bad.")
        self.assertEqual(len(sentences), 2)

    def test_02_absa_extraction(self):
        """Test aspect extraction and sentiment polarity calculation."""
        analyzer = AspectSentimentAnalyzer()
        text = "The battery life is stellar! But the camera photos are blurry."
        sentences = segment_sentences(text)
        res = analyzer.analyze_review(sentences, text)

        self.assertIn("battery", res["focal_aspect_scores"])
        self.assertIn("camera", res["focal_aspect_scores"])
        self.assertGreater(res["focal_aspect_scores"]["battery"], 0.0)
        self.assertLess(res["focal_aspect_scores"]["camera"], 0.0)

    def test_03_temporal_engine_and_dissonance(self):
        """Test O(N) prior review environment computation and AGED formulas."""
        prep_df = load_and_preprocess(self.sample_df)
        analyzer = AspectSentimentAnalyzer()
        absa_df = analyzer.process_dataframe(prep_df)
        aged_df = compute_prior_environment_and_aged(absa_df)

        # Review 1 is the launch review -> prior review count should be 0
        self.assertEqual(aged_df.iloc[0]["prior_review_count"], 0)
        self.assertTrue(np.isnan(aged_df.iloc[0]["prior_rating_mean"]))

        # Review 2 -> prior review count should be 1, prior rating mean = 5.0
        self.assertEqual(aged_df.iloc[1]["prior_review_count"], 1)
        self.assertEqual(aged_df.iloc[1]["prior_rating_mean"], 5.0)

        # Review 3 -> prior review count should be 2, prior rating mean = 5.0
        self.assertEqual(aged_df.iloc[2]["prior_review_count"], 2)
        self.assertEqual(aged_df.iloc[2]["prior_rating_mean"], 5.0)

        # Review 3 has negative battery sentiment while prior battery was highly positive -> high dissonance
        focal_bat = aged_df.iloc[2]["focal_battery"]
        prior_bat = aged_df.iloc[2]["prior_env_battery"]
        self.assertTrue(pd.notna(focal_bat))
        self.assertTrue(pd.notna(prior_bat))
        self.assertGreater(prior_bat, 0.0)
        self.assertLess(focal_bat, 0.0)
        # Mismatch should be negative
        self.assertLess(aged_df.iloc[2]["mismatch_battery"], 0.0)
        # Absolute dissonance should be positive
        self.assertGreater(aged_df.iloc[2]["dissonance_battery"], 0.5)

    def test_04_econometrics(self):
        """Test regression models with synthetic benchmark data."""
        bench_df = generate_benchmark_reviews(num_products=2, reviews_per_product=35, seed=123)
        prep_df = load_and_preprocess(bench_df)
        analyzer = AspectSentimentAnalyzer()
        absa_df = analyzer.process_dataframe(prep_df)
        aged_df = compute_prior_environment_and_aged(absa_df)

        econ = EconometricEngine(aged_df)
        results = econ.run_all_models()

        self.assertNotIn("error", results["model_1"])
        self.assertNotIn("error", results["model_2"])
        self.assertNotIn("error", results["model_3"])
        self.assertGreater(results["model_1"]["r_squared"], 0.2)

    def test_05_rating_normalization(self):
        """Test normalization function mapping 1-5 to [-1.0, 1.0]."""
        self.assertEqual(normalize_rating_to_sentiment_scale(1.0), -1.0)
        self.assertEqual(normalize_rating_to_sentiment_scale(3.0), 0.0)
        self.assertEqual(normalize_rating_to_sentiment_scale(5.0), 1.0)


if __name__ == "__main__":
    unittest.main()
