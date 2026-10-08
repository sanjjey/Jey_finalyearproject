"""
AGED Framework: FastAPI Backend
Provides high-performance REST APIs for Aspect-Global Evaluative Dissonance analytics,
Kaggle Amazon Cell Phone review datasets, and machine learning models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aged.ingestion import load_and_preprocess, segment_sentences
from aged.absa import AspectSentimentAnalyzer
from aged.temporal_engine import compute_prior_environment_and_aged
from aged.econometrics import EconometricEngine
from aged.synthetic_generator import generate_benchmark_reviews
from aged.config import DEFAULT_ASPECT_TAXONOMY

app = FastAPI(
    title="AGED Framework API",
    description="Backend API for Aspect-Global Evaluative Dissonance & Amazon Review Analytics",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches
analyzer = AspectSentimentAnalyzer()
MODEL_PATH = "data/amazon_aged_model.joblib"
KAGGLE_PATH = "data/kaggle_amazon_processed.csv"
BENCHMARK_PATH = "data/sample_smartphones.csv"

model_bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def load_dataset(dataset_type: str = "kaggle") -> pd.DataFrame:
    if dataset_type == "kaggle":
        if os.path.exists(KAGGLE_PATH):
            df = pd.read_csv(KAGGLE_PATH)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        raise HTTPException(status_code=404, detail="Kaggle processed dataset not found")
    else:
        if os.path.exists(BENCHMARK_PATH):
            df = pd.read_csv(BENCHMARK_PATH)
        else:
            df = generate_benchmark_reviews(num_products=3, reviews_per_product=60)
        # Process benchmark through pipeline
        raw = load_and_preprocess(df)
        absa = analyzer.process_dataframe(raw)
        return compute_prior_environment_and_aged(absa, min_prior_reviews=1)


class ReviewAnalyzeRequest(BaseModel):
    text: str = Field(..., description="Review text to analyze")
    actual_rating: float = Field(default=5.0, ge=1.0, le=5.0, description="User-awarded star rating")
    prior_rating_mean: float = Field(default=3.8, ge=1.0, le=5.0, description="Community prior rating baseline")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_bundle is not None,
        "kaggle_available": os.path.exists(KAGGLE_PATH)
    }


@app.get("/api/meta")
def get_metadata():
    k_df = load_dataset("kaggle")
    products = []
    if "product_name" in k_df.columns:
        prods = k_df[["product_id", "product_name"]].drop_duplicates()
        for _, r in prods.iterrows():
            products.append({"id": str(r["product_id"]), "name": str(r["product_name"])})
    else:
        for pid in k_df["product_id"].unique():
            products.append({"id": str(pid), "name": str(pid)})

    metrics = model_bundle["metrics"] if model_bundle else {}
    return {
        "aspect_taxonomy": list(DEFAULT_ASPECT_TAXONOMY.keys()),
        "products": products,
        "ml_metrics": {
            "r2": metrics.get("regressor_r2", 0.47),
            "mae": metrics.get("regressor_mae", 0.83),
            "loss_aversion_ratio": metrics.get("loss_aversion_ratio", 41.8),
            "feature_importances": metrics.get("feature_importances", {})
        }
    }


@app.get("/api/presets")
def get_presets():
    return [
        {
            "id": "scenario-1",
            "title": "5★ with 2 Goods, 1 Bad",
            "subtitle": "Minor Flaw Tolerated (Positive AGED)",
            "text": "The camera photo quality is stunning and the screen display is gorgeous, though battery drains a bit fast.",
            "actual_rating": 5.0,
            "prior_rating_mean": 4.1
        },
        {
            "id": "scenario-2",
            "title": "3★ with 2 Goods, 2 Bads",
            "subtitle": "Balanced Ambivalence",
            "text": "Screen display is vibrant and build quality is sturdy, but software is laggy with bloatware and price is overpriced.",
            "actual_rating": 3.0,
            "prior_rating_mean": 3.5
        },
        {
            "id": "scenario-3",
            "title": "1★ with 0 Goods, 2 Bads",
            "subtitle": "Catastrophic Dealbreaker",
            "text": "Overheating terribly and battery drain is horrific. Absolute waste of money.",
            "actual_rating": 1.0,
            "prior_rating_mean": 3.8
        },
        {
            "id": "scenario-4",
            "title": "2★ with 1 Good, 2 Bads",
            "subtitle": "Loss Aversion / Negativity Dominance",
            "text": "The build design feels premium, but camera photos are blurry and performance suffers from constant lag.",
            "actual_rating": 2.0,
            "prior_rating_mean": 3.6
        }
    ]


@app.get("/api/dashboard-data")
def get_dashboard_data(
    dataset: str = Query("kaggle", pattern="^(kaggle|benchmark)$"),
    product_id: str = Query("ALL")
):
    df = load_dataset(dataset)
    if product_id != "ALL":
        df = df[df["product_id"] == product_id].copy()

    if len(df) == 0:
        raise HTTPException(status_code=404, detail="No reviews found for given filter")

    # 1. KPIs
    g_mean = float(df["num_good_aspects"].mean()) if "num_good_aspects" in df.columns else 0.0
    b_mean = float(df["num_bad_aspects"].mean()) if "num_bad_aspects" in df.columns else 0.0
    kpis = {
        "total_reviews": int(len(df)),
        "mean_rating": round(float(df["rating"].mean()), 2),
        "mean_goods": round(g_mean, 2),
        "mean_bads": round(b_mean, 2),
        "mean_aged_dissonance": round(float(df["aged_dissonance"].dropna().mean()), 3),
        "loss_aversion_ratio": round(float(model_bundle["metrics"].get("loss_aversion_ratio", 2.8) if model_bundle else 2.5), 1)
    }

    # 2. Empirical Matrix Heatmap (Goods vs Bads vs Mean Rating)
    matrix = []
    if "num_good_aspects" in df.columns and "num_bad_aspects" in df.columns:
        grouped = df.groupby(["num_good_aspects", "num_bad_aspects"]).agg(
            mean_rating=("rating", "mean"),
            count=("rating", "count")
        ).reset_index()
        for _, r in grouped.iterrows():
            matrix.append({
                "goods": int(r["num_good_aspects"]),
                "bads": int(r["num_bad_aspects"]),
                "mean_rating": round(float(r["mean_rating"]), 2),
                "count": int(r["count"])
            })

    # 3. Archetype Distribution
    archetypes = []
    if "evaluative_archetype" in df.columns:
        arch_counts = df["evaluative_archetype"].value_counts().reset_index()
        arch_counts.columns = ["name", "count"]
        for _, r in arch_counts.iterrows():
            archetypes.append({"name": str(r["name"]), "count": int(r["count"])})

    # 4. Net Aspect Balance Curve
    balance_curve = []
    if "net_aspect_balance" in df.columns:
        b_grouped = df.groupby("net_aspect_balance").agg(
            mean_rating=("rating", "mean"),
            count=("rating", "count")
        ).reset_index()
        b_grouped = b_grouped[b_grouped["count"] >= 3]
        for _, r in b_grouped.iterrows():
            balance_curve.append({
                "net_balance": int(r["net_aspect_balance"]),
                "mean_rating": round(float(r["mean_rating"]), 2),
                "count": int(r["count"])
            })

    # 5. Longitudinal Series (sample up to 150 points for chart clarity)
    series_df = df.sort_values(by="review_order")
    if len(series_df) > 150:
        step = len(series_df) // 150
        series_df = series_df.iloc[::step]

    trajectory = []
    for _, r in series_df.iterrows():
        trajectory.append({
            "order": int(r["review_order"]),
            "rating": float(r["rating"]),
            "prior_rating_mean": round(float(r["prior_rating_mean"]), 2) if pd.notna(r["prior_rating_mean"]) else None,
            "sentiment": round(float(r["mean_aspect_sentiment"]), 2) if pd.notna(r["mean_aspect_sentiment"]) else None
        })

    return {
        "kpis": kpis,
        "empirical_matrix": matrix,
        "archetypes": archetypes,
        "balance_curve": balance_curve,
        "trajectory": trajectory
    }


@app.get("/api/econometrics")
def get_econometrics(dataset: str = Query("kaggle", pattern="^(kaggle|benchmark)$")):
    df = load_dataset(dataset)
    engine = EconometricEngine(df)
    results = engine.run_all_models()

    cleaned = {}
    for m_key in ["model_1", "model_2", "model_3", "model_4", "model_5"]:
        m = results.get(m_key, {})
        if "error" in m:
            cleaned[m_key] = {"error": m["error"]}
        else:
            table = []
            if "summary_table" in m and isinstance(m["summary_table"], pd.DataFrame):
                table = m["summary_table"].to_dict(orient="records")
            cleaned[m_key] = {
                "model_name": m.get("model_name"),
                "formula": m.get("formula"),
                "r_squared": m.get("r_squared"),
                "adj_r_squared": m.get("adj_r_squared"),
                "pseudo_r2_type": m.get("pseudo_r2_type"),
                "loss_aversion_ratio": m.get("loss_aversion_ratio"),
                "nobs": m.get("nobs"),
                "aic": m.get("aic"),
                "summary_table": table
            }
    return cleaned


@app.get("/api/reviews")
def get_reviews(
    dataset: str = Query("kaggle", pattern="^(kaggle|benchmark)$"),
    product_id: str = Query("ALL"),
    limit: int = Query(50, le=200)
):
    df = load_dataset(dataset)
    if product_id != "ALL":
        df = df[df["product_id"] == product_id]

    sub = df.head(limit)
    reviews = []
    for _, r in sub.iterrows():
        reviews.append({
            "review_id": str(r["review_id"]),
            "product_id": str(r["product_id"]),
            "product_name": str(r.get("product_name", r["product_id"])),
            "rating": float(r["rating"]),
            "review_order": int(r["review_order"]),
            "text": str(r["text"]),
            "goods_count": int(r.get("goods_count", 0)),
            "bads_count": int(r.get("bads_count", 0)),
            "good_aspects": str(r.get("good_aspects", "None")),
            "bad_aspects": str(r.get("bad_aspects", "None")),
            "aged_dissonance": round(float(r["aged_dissonance"]), 3) if pd.notna(r.get("aged_dissonance")) else None,
            "prior_rating_mean": round(float(r["prior_rating_mean"]), 2) if pd.notna(r.get("prior_rating_mean")) else None
        })
    return reviews


@app.post("/api/analyze")
def analyze_review(req: ReviewAnalyzeRequest):
    if not model_bundle:
        raise HTTPException(status_code=500, detail="ML model bundle not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    sentences = segment_sentences(text)
    res = analyzer.analyze_review(sentences, text)

    feats = pd.DataFrame([{col: 0.0 for col in model_bundle['feature_names']}])
    feats['num_good_aspects'] = float(res['num_good_aspects'])
    feats['num_bad_aspects'] = float(res['num_bad_aspects'])
    feats['goods_count'] = float(res['goods_count'])
    feats['bads_count'] = float(res['bads_count'])
    feats['net_aspect_balance'] = float(res['net_aspect_balance'])
    feats['aspect_valence_ratio'] = float(res['aspect_valence_ratio'])
    feats['goods_times_bads'] = float(res['num_good_aspects'] * res['num_bad_aspects'])
    feats['bads_squared'] = float(res['num_bad_aspects'] ** 2)
    feats['has_both_goods_and_bads'] = 1.0 if (res['num_good_aspects'] > 0 and res['num_bad_aspects'] > 0) else 0.0
    feats['only_bads'] = 1.0 if (res['num_good_aspects'] == 0 and res['num_bad_aspects'] > 0) else 0.0
    feats['only_goods'] = 1.0 if (res['num_good_aspects'] > 0 and res['num_bad_aspects'] == 0) else 0.0

    for g in res['good_aspects']:
        if f'good_{g}' in feats.columns:
            feats[f'good_{g}'] = 1.0
    for b in res['bad_aspects']:
        if f'bad_{b}' in feats.columns:
            feats[f'bad_{b}'] = 1.0

    feats['overall_text_sentiment'] = float(res['overall_text_sentiment'])
    feats['mean_aspect_sentiment'] = float(res['mean_aspect_sentiment'])
    feats['prior_rating_mean'] = float(req.prior_rating_mean)

    pred_stars = float(np.clip(model_bundle['regressor'].predict(feats)[0], 1.0, 5.0))
    probs = model_bundle['classifier'].predict_proba(feats)[0]
    classes = model_bundle['classifier'].classes_
    prob_dist = {int(c): round(float(p), 3) for c, p in zip(classes, probs)}

    gap = req.actual_rating - pred_stars
    abs_gap = abs(gap)

    if abs_gap <= 0.6:
        status = "aligned"
        title = "Evaluatively Aligned"
        desc = "The star rating accurately mirrors the empirical balance of positive and negative aspects."
    elif gap > 0.6:
        status = "positive_dissonance"
        title = "Forgiving / Minor Flaw Tolerated (Positive AGED)"
        desc = f"Reviewer awarded {req.actual_rating:.0f}★ despite negative aspects ({', '.join(res['bad_aspects'])}). Flaws were deemed secondary."
    else:
        status = "negative_dissonance"
        title = "Fatal Dealbreaker / Loss Aversion (Negative AGED)"
        desc = f"Reviewer downgraded rating to {req.actual_rating:.0f}★ despite positive aspects ({', '.join(res['good_aspects'])}). Severe loss aversion."

    clauses = []
    for ev in res.get("sentence_evaluations", []):
        sentiment = float(ev["sentiment"])
        v_type = "good" if sentiment >= 0.05 else ("bad" if sentiment <= -0.05 else "neutral")
        clauses.append({
            "clause_idx": ev["sentence_idx"] + 1,
            "text": ev["text"],
            "aspects": ev["aspects"],
            "sentiment": round(sentiment, 2),
            "valence": v_type
        })

    return {
        "goods_count": res["goods_count"],
        "bads_count": res["bads_count"],
        "num_good_aspects": res["num_good_aspects"],
        "num_bad_aspects": res["num_bad_aspects"],
        "good_aspects": res["good_aspects"],
        "bad_aspects": res["bad_aspects"],
        "overall_sentiment": round(float(res["overall_text_sentiment"]), 2),
        "predicted_stars": round(pred_stars, 2),
        "actual_stars": req.actual_rating,
        "dissonance_gap": round(gap, 2),
        "diagnosis": {
            "status": status,
            "title": title,
            "description": desc
        },
        "probabilities": prob_dist,
        "clauses": clauses
    }


# Mount built React Frontend if present
if os.path.exists("frontend/dist"):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        file_path = os.path.join("frontend/dist", full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse("frontend/dist/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
