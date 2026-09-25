"""
Module 2: Aspect Extraction & Sentiment Analysis (ABSA)
Identifies product aspects from sentences and computes fine-grained aspect sentiments.
"""

from typing import Dict, List, Tuple, Any, Optional
import re
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from aged.config import DEFAULT_ASPECT_TAXONOMY


class AspectSentimentAnalyzer:
    """
    Lightweight, deterministic, and modular ABSA engine.
    Combines rule-based taxonomy matching with VADER sentiment intensity analysis.
    """

    def __init__(self, taxonomy: Optional[Dict[str, List[str]]] = None):
        self.taxonomy = taxonomy or DEFAULT_ASPECT_TAXONOMY
        self._compiled_patterns = self._compile_taxonomy(self.taxonomy)
        self.vader = SentimentIntensityAnalyzer()
        self._add_domain_lexicon()

    def _compile_taxonomy(self, taxonomy: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
        """Pre-compiles word-boundary regex patterns for fast matching."""
        compiled = {}
        for aspect, keywords in taxonomy.items():
            patterns = []
            for kw in keywords:
                # Use regex word boundaries (\b)
                escaped = re.escape(kw)
                patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
            compiled[aspect] = patterns
        return compiled

    def _add_domain_lexicon(self):
        """Augments VADER with electronics & e-commerce domain-specific polarity."""
        domain_updates = {
            "stellar": 2.8,
            "flawless": 2.8,
            "exceptional": 2.6,
            "blazing": 2.8,
            "beast": 2.6,
            "crisp": 2.2,
            "snappy": 2.5,
            "fluid": 2.2,
            "punchy": 2.0,
            "sturdy": 2.3,
            "premium": 2.7,
            "gorgeous": 2.6,
            "lag": -2.5,
            "laggy": -2.8,
            "throttle": -2.0,
            "throttling": -2.2,
            "drain": -2.2,
            "draining": -2.5,
            "overheating": -3.0,
            "bloatware": -2.4,
            "stutter": -2.0,
            "subpar": -2.4,
            "flimsy": -2.6,
            "buggy": -2.7,
            "glitch": -2.2,
            "glitchy": -2.5,
            "horrific": -2.8,
            "appalling": -2.8,
            "blurry": -2.0
        }
        self.vader.lexicon.update(domain_updates)

    def extract_aspects_from_sentence(self, sentence: str) -> List[str]:
        """Identifies which aspects are referenced in a sentence."""
        matched_aspects = []
        for aspect, patterns in self._compiled_patterns.items():
            for pat in patterns:
                if pat.search(sentence):
                    matched_aspects.append(aspect)
                    break
        return matched_aspects

    def score_sentence_sentiment(self, sentence: str) -> float:
        """
        Returns compound sentiment score in [-1.0, 1.0].
        -1.0 = extremely negative, 0.0 = neutral, +1.0 = extremely positive.
        """
        scores = self.vader.polarity_scores(sentence)
        return float(scores["compound"])

    def analyze_review(self, sentences: List[str], full_text: str) -> Dict[str, Any]:
        """
        Extracts sentence-level aspects and aggregates to review-level focal aspect sentiments.
        """
        sentence_evaluations = []
        aspect_scores_accumulator: Dict[str, List[float]] = {k: [] for k in self.taxonomy.keys()}

        for s_idx, sentence in enumerate(sentences):
            aspects_found = self.extract_aspects_from_sentence(sentence)
            sentiment = self.score_sentence_sentiment(sentence)

            sentence_evaluations.append({
                "sentence_idx": s_idx,
                "text": sentence,
                "aspects": aspects_found,
                "sentiment": sentiment
            })

            for asp in aspects_found:
                aspect_scores_accumulator[asp].append(sentiment)

        # Review-level focal aspect sentiment (None if aspect not mentioned)
        focal_aspect_scores = {}
        for asp, scores in aspect_scores_accumulator.items():
            if len(scores) > 0:
                focal_aspect_scores[asp] = float(np.mean(scores))
            else:
                focal_aspect_scores[asp] = None

        # Overall text sentiment
        overall_text_sentiment = self.score_sentence_sentiment(full_text)

        # Average of all mentioned aspects
        mentioned_scores = [v for v in focal_aspect_scores.values() if v is not None]
        mean_aspect_sentiment = float(np.mean(mentioned_scores)) if mentioned_scores else overall_text_sentiment

        return {
            "sentence_evaluations": sentence_evaluations,
            "focal_aspect_scores": focal_aspect_scores,
            "overall_text_sentiment": overall_text_sentiment,
            "mean_aspect_sentiment": mean_aspect_sentiment,
            "aspects_mentioned_count": len(mentioned_scores)
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies ABSA to entire DataFrame.
        Adds focal aspect sentiment columns for each aspect in taxonomy.
        """
        out_df = df.copy()

        # Storage for results
        focal_records = []
        overall_sentiments = []
        mean_aspect_sentiments = []
        mentioned_counts = []
        sentence_evals_list = []

        for _, row in out_df.iterrows():
            res = self.analyze_review(row["sentences"], row["text"])
            focal_records.append(res["focal_aspect_scores"])
            overall_sentiments.append(res["overall_text_sentiment"])
            mean_aspect_sentiments.append(res["mean_aspect_sentiment"])
            mentioned_counts.append(res["aspects_mentioned_count"])
            sentence_evals_list.append(res["sentence_evaluations"])

        # Add overall sentiment metrics
        out_df["overall_text_sentiment"] = overall_sentiments
        out_df["mean_aspect_sentiment"] = mean_aspect_sentiments
        out_df["aspects_mentioned_count"] = mentioned_counts
        out_df["sentence_evaluations"] = sentence_evals_list

        # Add individual aspect focal columns: focal_sentiment_<aspect>
        focal_df = pd.DataFrame(focal_records, index=out_df.index)
        for aspect in self.taxonomy.keys():
            col_name = f"focal_{aspect}"
            out_df[col_name] = focal_df[aspect]

        return out_df
