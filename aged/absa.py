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
            "stunning": 2.8,
            "vibrant": 2.5,
            "sharp": 2.3,
            "durable": 2.4,
            "reliable": 2.5,
            "solid": 2.2,
            "responsive": 2.4,
            "bargain": 2.4,
            "lag": -2.5,
            "laggy": -2.8,
            "lags": -2.5,
            "sluggish": -2.6,
            "slow": -2.0,
            "throttle": -2.0,
            "throttling": -2.2,
            "drain": -2.2,
            "drains": -2.2,
            "drained": -2.2,
            "draining": -2.5,
            "dies": -2.5,
            "died": -2.8,
            "overheating": -3.0,
            "overheats": -2.8,
            "bloatware": -2.4,
            "stutter": -2.0,
            "stutters": -2.0,
            "subpar": -2.4,
            "flimsy": -2.6,
            "buggy": -2.7,
            "glitch": -2.2,
            "glitches": -2.3,
            "glitchy": -2.5,
            "horrific": -2.8,
            "appalling": -2.8,
            "blurry": -2.2,
            "grainy": -2.0,
            "scratches": -2.0,
            "scratched": -2.0,
            "overpriced": -2.5,
            "disappointing": -2.4,
            "disappointed": -2.4,
            "useless": -2.9,
            "defect": -2.6,
            "defective": -2.8
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
        good_aspects = []
        bad_aspects = []
        goods_count = 0
        bads_count = 0

        for asp, scores in aspect_scores_accumulator.items():
            if len(scores) > 0:
                mean_asp_score = float(np.mean(scores))
                focal_aspect_scores[asp] = mean_asp_score
                # Classify aspect polarity as good (> 0.05) or bad (< -0.05)
                if mean_asp_score >= 0.05:
                    good_aspects.append(asp)
                    goods_count += len([s for s in scores if s >= 0.05])
                elif mean_asp_score <= -0.05:
                    bad_aspects.append(asp)
                    bads_count += len([s for s in scores if s <= -0.05])
            else:
                focal_aspect_scores[asp] = None

        # Overall text sentiment
        overall_text_sentiment = self.score_sentence_sentiment(full_text)

        # Average of all mentioned aspects
        mentioned_scores = [v for v in focal_aspect_scores.values() if v is not None]
        mean_aspect_sentiment = float(np.mean(mentioned_scores)) if mentioned_scores else overall_text_sentiment

        total_valence_mentions = goods_count + bads_count
        valence_ratio = float(goods_count / total_valence_mentions) if total_valence_mentions > 0 else 0.5
        net_aspect_balance = goods_count - bads_count
        good_bad_pattern = f"{len(good_aspects)} Goods, {len(bad_aspects)} Bads"

        return {
            "sentence_evaluations": sentence_evaluations,
            "focal_aspect_scores": focal_aspect_scores,
            "overall_text_sentiment": overall_text_sentiment,
            "mean_aspect_sentiment": mean_aspect_sentiment,
            "aspects_mentioned_count": len(mentioned_scores),
            "goods_count": goods_count,
            "bads_count": bads_count,
            "num_good_aspects": len(good_aspects),
            "num_bad_aspects": len(bad_aspects),
            "good_aspects": good_aspects,
            "bad_aspects": bad_aspects,
            "net_aspect_balance": net_aspect_balance,
            "aspect_valence_ratio": valence_ratio,
            "good_bad_pattern": good_bad_pattern
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies ABSA to entire DataFrame.
        Adds focal aspect sentiment columns for each aspect in taxonomy and good/bad counts.
        """
        out_df = df.copy()

        # Storage for results
        focal_records = []
        overall_sentiments = []
        mean_aspect_sentiments = []
        mentioned_counts = []
        sentence_evals_list = []
        goods_counts = []
        bads_counts = []
        num_good_aspects_list = []
        num_bad_aspects_list = []
        good_aspects_str_list = []
        bad_aspects_str_list = []
        net_balances = []
        valence_ratios = []
        patterns = []

        for _, row in out_df.iterrows():
            res = self.analyze_review(row["sentences"], row["text"])
            focal_records.append(res["focal_aspect_scores"])
            overall_sentiments.append(res["overall_text_sentiment"])
            mean_aspect_sentiments.append(res["mean_aspect_sentiment"])
            mentioned_counts.append(res["aspects_mentioned_count"])
            sentence_evals_list.append(res["sentence_evaluations"])
            goods_counts.append(res["goods_count"])
            bads_counts.append(res["bads_count"])
            num_good_aspects_list.append(res["num_good_aspects"])
            num_bad_aspects_list.append(res["num_bad_aspects"])
            good_aspects_str_list.append(", ".join(res["good_aspects"]) if res["good_aspects"] else "None")
            bad_aspects_str_list.append(", ".join(res["bad_aspects"]) if res["bad_aspects"] else "None")
            net_balances.append(res["net_aspect_balance"])
            valence_ratios.append(res["aspect_valence_ratio"])
            patterns.append(res["good_bad_pattern"])

        # Add overall sentiment & polarity metrics
        out_df["overall_text_sentiment"] = overall_sentiments
        out_df["mean_aspect_sentiment"] = mean_aspect_sentiments
        out_df["aspects_mentioned_count"] = mentioned_counts
        out_df["goods_count"] = goods_counts
        out_df["bads_count"] = bads_counts
        out_df["num_good_aspects"] = num_good_aspects_list
        out_df["num_bad_aspects"] = num_bad_aspects_list
        out_df["good_aspects"] = good_aspects_str_list
        out_df["bad_aspects"] = bad_aspects_str_list
        out_df["net_aspect_balance"] = net_balances
        out_df["aspect_valence_ratio"] = valence_ratios
        out_df["good_bad_pattern"] = patterns
        out_df["sentence_evaluations"] = sentence_evals_list

        # Archetype labeling if rating column exists
        if "rating" in out_df.columns:
            archetypes = []
            for _, r in out_df.iterrows():
                rt = float(r["rating"])
                g = int(r["num_good_aspects"])
                b = int(r["num_bad_aspects"])
                if rt >= 4.0 and b >= 1:
                    archetypes.append("Forgiving / Minor Flaw Tolerated")
                elif rt == 3.0 and g >= 1 and b >= 1:
                    archetypes.append("Balanced Ambivalence")
                elif rt == 1.0 and g >= 1:
                    archetypes.append("Catastrophic Dealbreaker")
                elif rt <= 2.0 and b >= g and b >= 1:
                    archetypes.append("Loss Aversion / Negativity Dominance")
                elif rt >= 4.0 and b == 0 and g >= 1:
                    archetypes.append("Consistently Satisfied")
                elif rt <= 2.0 and g == 0 and b >= 1:
                    archetypes.append("Consistently Dissatisfied")
                else:
                    archetypes.append("Standard / Unaligned")
            out_df["evaluative_archetype"] = archetypes

        # Add individual aspect focal columns: focal_<aspect>
        focal_df = pd.DataFrame(focal_records, index=out_df.index)
        for aspect in self.taxonomy.keys():
            col_name = f"focal_{aspect}"
            out_df[col_name] = focal_df[aspect]

        return out_df
