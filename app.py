"""
AGED Framework Interactive Analytics & Econometric Dashboard
Streamlit Application for Aspect-Global Evaluative Dissonance in Online Product Reviews
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from aged.ingestion import load_and_preprocess
from aged.absa import AspectSentimentAnalyzer
from aged.temporal_engine import compute_prior_environment_and_aged
from aged.econometrics import EconometricEngine
from aged.synthetic_generator import generate_benchmark_reviews
from aged.config import DEFAULT_ASPECT_TAXONOMY

st.set_page_config(
    page_title="AGED Analytics Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, academic & modern presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def process_data_pipeline(df_input: pd.DataFrame, min_priors: int = 1) -> tuple:
    """Cached pipeline execution for fast, responsive dashboard interactions."""
    raw_df = load_and_preprocess(df_input)
    analyzer = AspectSentimentAnalyzer()
    absa_df = analyzer.process_dataframe(raw_df)
    aged_df = compute_prior_environment_and_aged(absa_df, min_prior_reviews=min_priors)
    econ_engine = EconometricEngine(aged_df)
    econ_results = econ_engine.run_all_models()
    return aged_df, econ_results


def main():
    st.markdown('<div class="main-title">AGED: Aspect–Global Evaluative Dissonance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A Temporal Econometric Analysis of Consumer Experience and Prior Review Environments</div>',
        unsafe_allow_html=True
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Data & Configuration")
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Benchmark Dataset (Smartphones)", "Upload CSV / JSON Dataset"]
    )

    if data_source == "Benchmark Dataset (Smartphones)":
        if os.path.exists("data/sample_smartphones.csv"):
            input_df = pd.read_csv("data/sample_smartphones.csv")
        else:
            input_df = generate_benchmark_reviews(num_products=3, reviews_per_product=60)
        st.sidebar.success(f"Loaded benchmark dataset ({len(input_df)} reviews)")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload review file (CSV/JSON)", type=["csv", "json", "parquet"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    input_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".json"):
                    input_df = pd.read_json(uploaded_file)
                else:
                    input_df = pd.read_parquet(uploaded_file)
                st.sidebar.success(f"Uploaded {len(input_df)} rows")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                st.stop()
        else:
            st.info("👆 Please upload a review file or switch to 'Benchmark Dataset' in the sidebar.")
            st.stop()

    min_priors = st.sidebar.slider("Min Prior Reviews for Baseline", min_value=1, max_value=10, value=1)

    # Process pipeline
    with st.spinner("Analyzing Aspect Sentiments and Temporal Prior Environments..."):
        aged_df, econ_results = process_data_pipeline(input_df, min_priors)

    # Product selector
    available_products = ["ALL PRODUCTS"] + sorted(aged_df["product_id"].unique().tolist())
    selected_product = st.sidebar.selectbox("Filter Product for Visualizations:", available_products)

    if selected_product == "ALL PRODUCTS":
        view_df = aged_df
    else:
        view_df = aged_df[aged_df["product_id"] == selected_product].copy()

    # Top KPI Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Reviews", f"{len(view_df):,}")
    with c2:
        st.metric("Mean Star Rating", f"{view_df['rating'].mean():.2f} / 5.0")
    with c3:
        st.metric("Avg Aspect Sentiment", f"{view_df['mean_aspect_sentiment'].mean():+.2f}")
    with c4:
        st.metric("Mean AGED Dissonance", f"{view_df['aged_dissonance'].mean():.3f}")
    with c5:
        env_diss = view_df["mean_env_dissonance"].dropna().mean()
        st.metric("Mean Expectation Gap", f"{env_diss:.3f}" if pd.notna(env_diss) else "N/A")

    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Temporal Trajectories",
        "⚖️ AGED & Expectation Mismatch",
        "🔬 Econometric Regression Models",
        "🔍 Single Review Inspector"
    ])

    # TAB 1: Temporal Trajectories
    with tab1:
        st.subheader("Longitudinal Rating & Sentiment Evolution")
        st.caption("Investigates how focal star ratings and aspect-level experiences diverge from prior expectations over time.")

        fig_temp = go.Figure()
        # Focal rating
        fig_temp.add_trace(go.Scatter(
            x=view_df["review_order"],
            y=view_df["rating"],
            mode="markers+lines",
            name="Focal Star Rating",
            marker=dict(size=6, color="#2563EB", opacity=0.7),
            line=dict(color="#93C5FD", width=1)
        ))
        # Prior rating mean
        fig_temp.add_trace(go.Scatter(
            x=view_df["review_order"],
            y=view_df["prior_rating_mean"],
            mode="lines",
            name="Prior Rating Baseline (Expectation Frame)",
            line=dict(color="#DC2626", width=2.5, dash="dash")
        ))
        fig_temp.update_layout(
            title="Star Rating vs. Prior Review Environment Over Review Sequence",
            xaxis_title="Chronological Review Order (Sequence Position)",
            yaxis_title="Rating (1 - 5 Stars)",
            template="plotly_white",
            hovermode="x unified",
            height=420
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        # Aspect Trajectories
        st.subheader("Aspect-Specific Sentiment Dynamics Over Time")
        aspect_cols = [c for c in view_df.columns if c.startswith("focal_") and view_df[c].notna().any()]
        if aspect_cols:
            fig_aspect = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, col in enumerate(aspect_cols):
                asp_name = col.replace("focal_", "").capitalize()
                sub = view_df.dropna(subset=[col])
                fig_aspect.add_trace(go.Scatter(
                    x=sub["review_order"],
                    y=sub[col],
                    mode="markers+lines",
                    name=asp_name,
                    line=dict(width=1.5, color=colors[i % len(colors)]),
                    marker=dict(size=5)
                ))
            fig_aspect.update_layout(
                title="Aspect-Level Sentiment Scores [-1.0 (Neg) to +1.0 (Pos)]",
                xaxis_title="Chronological Review Order",
                yaxis_title="Aspect Sentiment",
                template="plotly_white",
                height=380
            )
            st.plotly_chart(fig_aspect, use_container_width=True)

    # TAB 2: AGED & Expectation Mismatch
    with tab2:
        st.subheader("Aspect-Global Evaluative Dissonance (AGED) Distribution")
        st.caption("Examines the gap between current consumer experience and prior environmental sentiment (|S_focal - S_prior|).")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(
                view_df.dropna(subset=["aged_dissonance"]),
                x="aged_dissonance",
                nbins=25,
                title="AGED Dissonance Distribution (|Normalized Rating - Aspect Sentiment|)",
                color_discrete_sequence=["#4F46E5"],
                labels={"aged_dissonance": "AGED Dissonance Score"}
            )
            fig_hist.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            sub_valid = view_df.dropna(subset=["mean_aspect_sentiment", "rating"]).copy()
            fig_scatter = px.scatter(
                sub_valid,
                x="mean_aspect_sentiment",
                y="rating",
                color="aged_dissonance",
                size="word_count",
                title="Aspect Sentiment vs. Overall Star Rating",
                color_continuous_scale="Viridis",
                labels={
                    "mean_aspect_sentiment": "Current Aspect Sentiment",
                    "rating": "Overall Star Rating",
                    "aged_dissonance": "Dissonance"
                }
            )
            fig_scatter.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Aspect Mismatch Breakdown
        st.subheader("Expectation Mismatch Breakdown by Aspect")
        mismatch_summary = []
        for asp in DEFAULT_ASPECT_TAXONOMY.keys():
            col_m = f"dissonance_{asp}"
            if col_m in view_df.columns:
                valid_vals = view_df[col_m].dropna()
                if len(valid_vals) > 0:
                    mismatch_summary.append({
                        "Aspect": asp.replace("_", " ").capitalize(),
                        "Reviews Mentioning": len(valid_vals),
                        "Mean Mismatch Dissonance": round(float(valid_vals.mean()), 3),
                        "Max Mismatch": round(float(valid_vals.max()), 3)
                    })
        if mismatch_summary:
            m_df = pd.DataFrame(mismatch_summary).sort_values(by="Mean Mismatch Dissonance", ascending=False)
            fig_bar = px.bar(
                m_df,
                x="Aspect",
                y="Mean Mismatch Dissonance",
                color="Mean Mismatch Dissonance",
                title="Average Expectation Mismatch per Product Aspect (|S_focal - S_prior|)",
                color_continuous_scale="Reds"
            )
            fig_bar.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(m_df, use_container_width=True)

    # TAB 3: Econometric Regression Models
    with tab3:
        st.subheader("Econometric Estimation & Hypothesis Validation")
        st.markdown("""
        These models directly answer the research objectives outlined in the AGED paper:
        - **Objective 1:** Effect of aspect sentiment on global ratings ($S_{focal} \\rightarrow R$).
        - **Objective 2 & 3:** Moderation by the prior review environment and expectation mismatch dissonance.
        """)

        model_choice = st.selectbox(
            "Select Econometric Model to Inspect:",
            [
                "Model 1: Direct Aspect & Prior Environment Effects (OLS)",
                "Model 2: Moderation by Prior Environment (Interaction OLS)",
                "Model 3: AGED Dissonance & Mismatch Dynamics (OLS)",
                "Model 4: Ordered Logistic Regression (Discrete 1-5 Stars)"
            ]
        )

        model_key_map = {
            "Model 1: Direct Aspect & Prior Environment Effects (OLS)": "model_1",
            "Model 2: Moderation by Prior Environment (Interaction OLS)": "model_2",
            "Model 3: AGED Dissonance & Mismatch Dynamics (OLS)": "model_3",
            "Model 4: Ordered Logistic Regression (Discrete 1-5 Stars)": "model_4"
        }
        selected_model = econ_results.get(model_key_map[model_choice], {})

        if "error" in selected_model:
            st.warning(f"Model could not be estimated: {selected_model['error']}")
        else:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Observations (N)", selected_model.get("nobs", len(aged_df)))
            with col_m2:
                r2 = selected_model.get("r_squared")
                st.metric("R-squared", f"{r2:.4f}" if r2 is not None else "N/A (Logit AIC)")
            with col_m3:
                adj_r2 = selected_model.get("adj_r_squared")
                st.metric("Adj. R-squared", f"{adj_r2:.4f}" if adj_r2 is not None else "N/A")

            st.code(f"Formula: {selected_model.get('formula', 'logit(rating) ~ exog_vars')}", language="r")

            st.subheader("Parameter Estimates & Significance")
            summary_table = selected_model.get("summary_table")
            if summary_table is not None and isinstance(summary_table, pd.DataFrame):
                st.dataframe(summary_table, use_container_width=True)
                st.caption("Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, . p < 0.1")

    # TAB 4: Single Review Inspector
    with tab4:
        st.subheader("Deep-Dive Review & Sentence Decomposition")
        st.caption("Inspect individual reviews to trace sentence-level aspect tags and prior environment calculations.")

        rev_ids = view_df["review_id"].tolist()
        chosen_rev_id = st.selectbox("Select Review ID to Inspect:", rev_ids)
        rev_row = view_df[view_df["review_id"] == chosen_rev_id].iloc[0]

        ci1, ci2, ci3, ci4 = st.columns(4)
        with ci1:
            st.metric("Star Rating", f"{rev_row['rating']} / 5.0")
        with ci2:
            st.metric("Sequence Order", f"Review #{rev_row['review_order']}")
        with ci3:
            p_mean = rev_row["prior_rating_mean"]
            st.metric("Prior Rating Mean", f"{p_mean:.2f}" if pd.notna(p_mean) else "Launch Review")
        with ci4:
            st.metric("AGED Dissonance", f"{rev_row['aged_dissonance']:.3f}" if pd.notna(rev_row['aged_dissonance']) else "N/A")

        st.markdown(f"**Full Review Text:**\n> *\"{rev_row['text']}\"*")

        st.subheader("Sentence-Level Aspect Breakdown")
        sentence_evals = rev_row.get("sentence_evaluations", [])
        if sentence_evals:
            s_rows = []
            for se in sentence_evals:
                s_rows.append({
                    "Sentence #": se["sentence_idx"] + 1,
                    "Sentence Text": se["text"],
                    "Detected Aspects": ", ".join(se["aspects"]) if se["aspects"] else "None",
                    "Sentiment Polarity": f"{se['sentiment']:+.2f}"
                })
            st.dataframe(pd.DataFrame(s_rows), use_container_width=True)
        else:
            st.info("No sentence decomposition available for this record.")


if __name__ == "__main__":
    main()
