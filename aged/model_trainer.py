"""
AGED Framework: Model Training on Kaggle Amazon Cell Phone Reviews
Trains ML models that accommodate complex aspect-rating variations
(e.g., 5★ with 2 goods 1 bad; 3★ with 2 goods 2 bads; 1★ with 0 goods 2 bads; 2★ with 1 good 2 bads).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

from aged.ingestion import load_and_preprocess
from aged.absa import AspectSentimentAnalyzer
from aged.temporal_engine import compute_prior_environment_and_aged
from aged.config import DEFAULT_ASPECT_TAXONOMY


def load_kaggle_amazon_subset(
    reviews_path: str = "data/kaggle_amazon/20191226-reviews.csv",
    items_path: str = "data/kaggle_amazon/20191226-items.csv",
    top_n_products: int = 8,
    max_reviews_per_product: int = 500
) -> pd.DataFrame:
    """
    Loads top longitudinal smartphone products from Kaggle Amazon dataset.
    Merges brand and model titles for user-friendly display.
    """
    reviews = pd.read_csv(reviews_path)
    reviews = reviews.dropna(subset=["body", "rating", "asin"]).copy()
    
    # Select top products with most reviews
    top_asins = reviews["asin"].value_counts().head(top_n_products).index.tolist()
    
    selected_dfs = []
    for asin in top_asins:
        sub = reviews[reviews["asin"] == asin].sort_values(by="date")
        if len(sub) > max_reviews_per_product:
            sub = sub.head(max_reviews_per_product)
        selected_dfs.append(sub)
        
    combined = pd.concat(selected_dfs, ignore_index=True)
    
    # Merge item metadata if available
    if os.path.exists(items_path):
        items = pd.read_csv(items_path)
        items_map = items.set_index("asin")
        product_names = []
        for asin in combined["asin"]:
            if asin in items_map.index:
                brand = str(items_map.loc[asin, "brand"]) if pd.notna(items_map.loc[asin, "brand"]) else ""
                title = str(items_map.loc[asin, "title"]) if pd.notna(items_map.loc[asin, "title"]) else asin
                # Clean title
                clean_name = f"{brand} {title}".strip()
                if len(clean_name) > 40:
                    clean_name = clean_name[:37] + "..."
                product_names.append(clean_name)
            else:
                product_names.append(asin)
        combined["product_name"] = product_names
    else:
        combined["product_name"] = combined["asin"]
        
    return combined


def build_ml_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Constructs rich feature matrix designed to accommodate complex goods/bads variations.
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. Direct Counts of Goods and Bads
    features["num_good_aspects"] = df["num_good_aspects"].fillna(0).astype(float)
    features["num_bad_aspects"] = df["num_bad_aspects"].fillna(0).astype(float)
    features["goods_count"] = df["goods_count"].fillna(0).astype(float)
    features["bads_count"] = df["bads_count"].fillna(0).astype(float)
    
    # 2. Valence Ratio & Net Balance
    features["net_aspect_balance"] = df["net_aspect_balance"].fillna(0).astype(float)
    features["aspect_valence_ratio"] = df["aspect_valence_ratio"].fillna(0.5).astype(float)
    
    # 3. Non-linear Interaction & Asymmetry (Loss Aversion)
    features["goods_times_bads"] = features["num_good_aspects"] * features["num_bad_aspects"]
    features["bads_squared"] = features["num_bad_aspects"] ** 2
    features["has_both_goods_and_bads"] = ((features["num_good_aspects"] > 0) & (features["num_bad_aspects"] > 0)).astype(float)
    features["only_bads"] = ((features["num_good_aspects"] == 0) & (features["num_bad_aspects"] > 0)).astype(float)
    features["only_goods"] = ((features["num_good_aspects"] > 0) & (features["num_bad_aspects"] == 0)).astype(float)
    
    # 4. Aspect-Specific Indicators (is battery good? is camera bad?)
    for asp in DEFAULT_ASPECT_TAXONOMY.keys():
        focal_col = f"focal_{asp}"
        if focal_col in df.columns:
            val = df[focal_col].fillna(0.0)
            features[f"good_{asp}"] = (val >= 0.05).astype(float)
            features[f"bad_{asp}"] = (val <= -0.05).astype(float)
        else:
            features[f"good_{asp}"] = 0.0
            features[f"bad_{asp}"] = 0.0
            
    # 5. Continuous Sentiment & Prior Environment
    features["overall_text_sentiment"] = df["overall_text_sentiment"].fillna(0.0).astype(float)
    features["mean_aspect_sentiment"] = df["mean_aspect_sentiment"].fillna(0.0).astype(float)
    features["prior_rating_mean"] = df["prior_rating_mean"].fillna(df["rating"].mean()).astype(float)
    features["prior_rating_variance"] = df["prior_rating_variance"].fillna(0.0).astype(float)
    features["mean_env_dissonance"] = df["mean_env_dissonance"].fillna(0.0).astype(float)
    
    y = df["rating"].astype(float)
    return features, y, list(features.columns)


def train_amazon_models(
    processed_df: pd.DataFrame,
    save_path: str = "data/amazon_aged_model.joblib"
) -> Dict[str, Any]:
    """
    Trains regression and classification models to accommodate aspect goods/bads variations.
    """
    X, y, feature_names = build_ml_feature_matrix(processed_df)
    y_discrete = y.round().astype(int).clip(1, 5)
    
    X_train, X_test, y_train, y_test, yd_train, yd_test = train_test_split(
        X, y, y_discrete, test_size=0.2, random_state=42, stratify=y_discrete
    )
    
    # 1. Gradient Boosting Regressor (Captures non-linear goods/bads trade-offs)
    regressor = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=42
    )
    regressor.fit(X_train, y_train)
    y_pred_reg = regressor.predict(X_test)
    
    # 2. Linear Ridge Model (To inspect exact Loss Aversion weights beta_good vs beta_bad)
    linear_model = Ridge(alpha=1.0)
    linear_model.fit(X_train, y_train)
    coef_dict = dict(zip(feature_names, linear_model.coef_))
    
    beta_good = coef_dict.get("num_good_aspects", 0.0)
    beta_bad = coef_dict.get("num_bad_aspects", 0.0)
    # Loss aversion ratio lambda: how many times more damaging is a bad vs a good
    loss_aversion_ratio = abs(beta_bad) / max(0.01, abs(beta_good)) if abs(beta_good) > 0 else 2.0
    
    # 3. Random Forest Classifier for 1-5 Star Probabilities
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    classifier.fit(X_train, yd_train)
    yd_pred = classifier.predict(X_test)
    
    metrics = {
        "regressor_r2": float(r2_score(y_test, y_pred_reg)),
        "regressor_mae": float(mean_absolute_error(y_test, y_pred_reg)),
        "regressor_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_reg))),
        "classifier_accuracy": float(accuracy_score(yd_test, yd_pred)),
        "loss_aversion_ratio": float(loss_aversion_ratio),
        "beta_good": float(beta_good),
        "beta_bad": float(beta_bad),
        "feature_importances": dict(sorted(
            zip(feature_names, regressor.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )[:12])
    }
    
    bundle = {
        "regressor": regressor,
        "classifier": classifier,
        "linear_model": linear_model,
        "feature_names": feature_names,
        "metrics": metrics,
        "taxonomy": DEFAULT_ASPECT_TAXONOMY
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(bundle, save_path)
    print(f"Model saved to {save_path}")
    print(f"Performance: R2={metrics['regressor_r2']:.3f}, MAE={metrics['regressor_mae']:.3f}, Loss Aversion Ratio={loss_aversion_ratio:.2f}x")
    
    return bundle


def run_kaggle_pipeline(
    sample_size: int = 3500,
    output_processed_csv: str = "data/kaggle_amazon_processed.csv",
    output_model_path: str = "data/amazon_aged_model.joblib"
):
    """Full execution of Kaggle Amazon dataset ingestion, ABSA, temporal engine, and ML training."""
    print("Step 1: Loading Kaggle Amazon Review stream...")
    raw_df = load_kaggle_amazon_subset(top_n_products=8, max_reviews_per_product=500)
    
    print(f"Step 2: Loaded {len(raw_df)} reviews across {raw_df['asin'].nunique()} products. Running Ingestion...")
    prep_df = load_and_preprocess(raw_df)
    
    # Preserve product_name if present
    if "product_name" in raw_df.columns:
        prep_df["product_name"] = raw_df["product_name"].values
        
    print("Step 3: Running ABSA with Goods & Bads extraction...")
    analyzer = AspectSentimentAnalyzer()
    absa_df = analyzer.process_dataframe(prep_df)
    
    print("Step 4: Computing Temporal Prior Environments and AGED Dissonance...")
    aged_df = compute_prior_environment_and_aged(absa_df, min_prior_reviews=1)
    
    # Save processed dataset
    aged_df.to_csv(output_processed_csv, index=False)
    print(f"Processed dataset saved to {output_processed_csv}")
    
    print("Step 5: Training ML Models on Goods/Bads Variations...")
    bundle = train_amazon_models(aged_df, save_path=output_model_path)
    
    print("Pipeline completed successfully!")
    return aged_df, bundle


if __name__ == "__main__":
    run_kaggle_pipeline()
