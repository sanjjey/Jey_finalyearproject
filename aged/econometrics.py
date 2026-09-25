"""
Module 4: Econometric & Statistical Modeling Engine
Constructs regression-ready panel datasets and estimates econometric models
(OLS, Moderation Interaction models, and Ordered Logit) to test AGED hypotheses.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel


def prepare_econometric_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and formats dataset for statistical estimation:
    - Filters rows with prior review history (prior_review_count >= 1)
    - Replaces missing values and computes standardized control variables
    - Adds interaction terms (Aspect_Sentiment * Prior_Environment)
    """
    panel = df[df["has_prior_history"] == True].copy()

    # Drop any rows with NaN in key target variables
    required_cols = ["rating", "mean_aspect_sentiment", "prior_rating_mean"]
    panel = panel.dropna(subset=required_cols).copy()

    # Fallbacks for optional controls
    panel["prior_rating_variance"] = panel["prior_rating_variance"].fillna(0.0)
    panel["mean_env_dissonance"] = panel["mean_env_dissonance"].fillna(0.0)
    panel["word_count"] = panel["word_count"].fillna(panel["word_count"].median() if len(panel) > 0 else 50)
    panel["log_word_count"] = np.log1p(panel["word_count"])
    panel["log_review_order"] = np.log1p(panel["review_order"])

    # Interaction terms for Moderation Analysis
    panel["interaction_sentiment_prior"] = panel["mean_aspect_sentiment"] * panel["prior_rating_mean"]
    panel["interaction_sentiment_dissonance"] = panel["mean_aspect_sentiment"] * panel["mean_env_dissonance"]

    return panel


class EconometricEngine:
    """
    Estimates econometric models to validate the AGED theoretical framework:
    1. Direct Effects (OLS)
    2. Prior Environment Moderation (Interaction Model)
    3. Dissonance / Expectation Mismatch Model
    4. Ordered Logit Model (for discrete 1-5 star ratings)
    """

    def __init__(self, data: pd.DataFrame):
        self.raw_data = data
        self.panel_data = prepare_econometric_dataset(data)

    def fit_model_1_direct_effects(self) -> Dict[str, Any]:
        """
        Model 1: Tests Objective 1
        Rating = beta_0 + beta_1*AspectSentiment + beta_2*PriorRatingMean + Controls
        """
        if len(self.panel_data) < 10:
            return {"error": "Insufficient sample size (N < 10 with prior history)"}

        formula = "rating ~ mean_aspect_sentiment + prior_rating_mean + prior_rating_variance + log_review_order + log_word_count"
        model = smf.ols(formula=formula, data=self.panel_data).fit()

        return {
            "model_name": "Model 1: Direct Aspect & Prior Environment Effects",
            "formula": formula,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_pvalue": float(model.f_pvalue),
            "nobs": int(model.nobs),
            "summary_table": self._extract_summary(model),
            "statsmodels_result": model
        }

    def fit_model_2_moderation(self) -> Dict[str, Any]:
        """
        Model 2: Tests Objective 2 & 3 (Moderation Hypothesis)
        Rating = beta_0 + beta_1*AspectSentiment + beta_2*PriorRatingMean + beta_3*(AspectSentiment * PriorRatingMean) + Controls
        """
        if len(self.panel_data) < 10:
            return {"error": "Insufficient sample size"}

        formula = (
            "rating ~ mean_aspect_sentiment + prior_rating_mean + "
            "mean_aspect_sentiment:prior_rating_mean + prior_rating_variance + "
            "log_review_order + log_word_count"
        )
        model = smf.ols(formula=formula, data=self.panel_data).fit()

        return {
            "model_name": "Model 2: Moderation by Prior Environment",
            "formula": formula,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_pvalue": float(model.f_pvalue),
            "nobs": int(model.nobs),
            "summary_table": self._extract_summary(model),
            "statsmodels_result": model
        }

    def fit_model_3_dissonance_mismatch(self) -> Dict[str, Any]:
        """
        Model 3: Tests Objective 3 (Expectation Mismatch & Dissonance)
        Rating = beta_0 + beta_1*AspectSentiment + beta_2*PriorRatingMean + beta_3*Dissonance + beta_4*(AspectSentiment * Dissonance) + Controls
        """
        if len(self.panel_data) < 10:
            return {"error": "Insufficient sample size"}

        formula = (
            "rating ~ mean_aspect_sentiment + prior_rating_mean + "
            "mean_env_dissonance + mean_aspect_sentiment:mean_env_dissonance + "
            "prior_rating_variance + log_review_order + log_word_count"
        )
        model = smf.ols(formula=formula, data=self.panel_data).fit()

        return {
            "model_name": "Model 3: AGED Dissonance & Mismatch Dynamics",
            "formula": formula,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_pvalue": float(model.f_pvalue),
            "nobs": int(model.nobs),
            "summary_table": self._extract_summary(model),
            "statsmodels_result": model
        }

    def fit_model_4_ordered_logit(self) -> Dict[str, Any]:
        """
        Model 4: Ordered Logit Model for Discrete Star Ratings (1 to 5)
        Accounts for ordinal nature of rating choices.
        """
        if len(self.panel_data) < 20:
            return {"error": "Insufficient sample size for Ordered Logit (requires N >= 20)"}

        try:
            exog_vars = [
                "mean_aspect_sentiment",
                "prior_rating_mean",
                "mean_env_dissonance",
                "prior_rating_variance",
                "log_review_order"
            ]
            X = self.panel_data[exog_vars].copy()
            y = self.panel_data["rating"].astype(int)

            ordered_model = OrderedModel(y, X, distr="logit")
            res = ordered_model.fit(disp=False, method="bfgs", maxiter=200)

            return {
                "model_name": "Model 4: Ordered Logistic Regression",
                "aic": float(res.aic),
                "bic": float(res.bic),
                "nobs": int(res.nobs),
                "summary_table": self._extract_summary(res),
                "statsmodels_result": res
            }
        except Exception as e:
            return {"error": f"Ordered logit estimation error: {str(e)}"}

    def run_all_models(self) -> Dict[str, Any]:
        """Executes all econometric models and returns comprehensive findings."""
        return {
            "model_1": self.fit_model_1_direct_effects(),
            "model_2": self.fit_model_2_moderation(),
            "model_3": self.fit_model_3_dissonance_mismatch(),
            "model_4": self.fit_model_4_ordered_logit(),
            "panel_summary": {
                "total_reviews": len(self.raw_data),
                "panel_reviews_with_history": len(self.panel_data),
                "unique_products": self.panel_data["product_id"].nunique() if len(self.panel_data) > 0 else 0
            }
        }

    def _extract_summary(self, model_res: Any) -> pd.DataFrame:
        """Parses model parameters into a clean user-facing DataFrame."""
        params = model_res.params
        bse = model_res.bse
        tvalues = getattr(model_res, "tvalues", getattr(model_res, "zvalues", None))
        pvalues = model_res.pvalues

        records = []
        for var in params.index:
            p_val = pvalues[var]
            # Significance stars
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "." if p_val < 0.1 else ""
            records.append({
                "Variable": var,
                "Coefficient": round(float(params[var]), 4),
                "Std. Error": round(float(bse[var]), 4),
                "t / z Stat": round(float(tvalues[var]), 3) if tvalues is not None else np.nan,
                "P-value": round(float(p_val), 4),
                "Sig": stars
            })
        return pd.DataFrame(records)
