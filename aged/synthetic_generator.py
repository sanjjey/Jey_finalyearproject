"""
Module 5: Synthetic Benchmark Data Generator
Generates realistic longitudinal e-commerce review streams with known AGED dissonance dynamics
for demonstration, simulation, unit testing, and model benchmarking.
"""

from typing import List, Dict
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


SAMPLE_ASPECT_SNIPPETS = {
    "battery": {
        "positive": [
            "The battery life is stellar, easily lasting two full days on a single charge.",
            "Charging is blazing fast with 65W, 0 to 80% in 25 minutes.",
            "Excellent battery backup, 8 hours of screen-on-time easily."
        ],
        "negative": [
            "Battery drain is horrific, it loses 20% in an hour on standby.",
            "The charger gets alarmingly warm and the phone dies by afternoon.",
            "Severe battery draining issue after the latest firmware patch."
        ]
    },
    "camera": {
        "positive": [
            "The camera captures crisp details and punchy natural colors.",
            "Night mode is magical, crisp photos even in dim lighting.",
            "Portrait lens creates a gorgeous creamy bokeh effect."
        ],
        "negative": [
            "Camera photos look washed out and blurry in low light.",
            "The shutter lag makes it impossible to capture moving kids or pets.",
            "Video recording stutters and the autofocus hunts constantly."
        ]
    },
    "display": {
        "positive": [
            "The 120Hz AMOLED display is silky smooth and vibrant.",
            "Outdoor brightness is incredible, easily readable in direct sunlight.",
            "Bezels are razor-thin with beautiful HDR contrast."
        ],
        "negative": [
            "Display has an annoying green tint at low brightness levels.",
            "Screen scratches way too easily and refresh rate drops to 60Hz randomly.",
            "Washed out colors and poor viewing angles."
        ]
    },
    "performance": {
        "positive": [
            "Performance is blazing fast with zero lag during heavy gaming.",
            "Snappy multitasking, apps stay open in RAM with no stutters.",
            "Fluid UI animations, the chipset stays completely cool."
        ],
        "negative": [
            "Noticeable lag when switching apps and intense thermal throttling.",
            "The phone overheats during basic navigation and gaming drops fps.",
            "Terrible stuttering in the UI, feels slow and unoptimized."
        ]
    },
    "price_value": {
        "positive": [
            "Unbelievable value for money at this price point.",
            "Affordable budget device with flagship-grade features.",
            "Worth every single penny spent on this purchase."
        ],
        "negative": [
            "Massively overpriced for what it offers.",
            "Cheap plastic feel does not justify this premium cost.",
            "Poor value proposition compared to competing devices."
        ]
    }
}


def generate_benchmark_reviews(
    num_products: int = 3,
    reviews_per_product: int = 60,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic longitudinal review dataset demonstrating AGED.
    Simulates early adopter praise followed by mid-cycle expectation disconfirmation.
    """
    random.seed(seed)
    np.random.seed(seed)

    products = [
        {"id": "PROD-ALPHA-100", "name": "Flagship Pro Phone"},
        {"id": "PROD-BETA-200", "name": "Budget Value Neo"},
        {"id": "PROD-GAMMA-300", "name": "Gaming Ultra Max"}
    ][:num_products]

    records = []
    base_date = datetime(2025, 1, 15)

    for p in products:
        product_id = p["id"]
        current_date = base_date

        for idx in range(1, reviews_per_product + 1):
            review_id = f"rev_{product_id}_{idx:03d}"
            current_date += timedelta(hours=random.randint(6, 48))

            # Early reviews (1-20) are high praise (bandwagon/honeymoon period)
            # Later reviews (21-60) encounter aspect degradation (e.g., battery/heating)
            is_early = idx <= 20

            sentences = []
            aspect_evals = {}

            # Pick 2 to 3 aspects to discuss in this review
            chosen_aspects = random.sample(list(SAMPLE_ASPECT_SNIPPETS.keys()), k=random.randint(2, 3))

            for asp in chosen_aspects:
                if is_early:
                    # 85% chance of positive
                    pol = "positive" if random.random() < 0.85 else "negative"
                else:
                    # In later lifecycle, 50% chance of negative due to wear/expectations
                    pol = "negative" if random.random() < 0.55 else "positive"

                snippet = random.choice(SAMPLE_ASPECT_SNIPPETS[asp][pol])
                sentences.append(snippet)
                aspect_evals[asp] = 1 if pol == "positive" else -1

            # Star rating calculation influenced by aspect sentiments + expectation gap
            pos_count = sum(1 for v in aspect_evals.values() if v > 0)
            neg_count = sum(1 for v in aspect_evals.values() if v < 0)

            if is_early:
                # Early reviews give 4 or 5 stars
                rating = 5 if pos_count >= neg_count else 4
            else:
                # Later reviews: if there's a negative aspect when prior ratings were 4.8,
                # the expectation dissonance causes sharp rating drop (e.g., 1 or 2 stars)
                if neg_count > 0:
                    rating = random.choice([1, 2, 3])
                else:
                    rating = random.choice([4, 5])

            review_text = " ".join(sentences)

            records.append({
                "review_id": review_id,
                "product_id": product_id,
                "date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "rating": float(rating),
                "text": review_text
            })

    return pd.DataFrame(records)
