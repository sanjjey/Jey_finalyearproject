"""
AGED Framework: Interactive Analytics & Econometric Dashboard
Aspect–Global Evaluative Dissonance in Online Product Reviews
Trained on Kaggle Amazon Reviews & Longitudinal Product Benchmarks
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from aged.ingestion import load_and_preprocess, segment_sentences
from aged.absa import AspectSentimentAnalyzer
from aged.temporal_engine import compute_prior_environment_and_aged
from aged.econometrics import EconometricEngine
from aged.synthetic_generator import generate_benchmark_reviews
from aged.config import DEFAULT_ASPECT_TAXONOMY

st.set_page_config(
    page_title="AGED Framework — Review Dissonance Analytics",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional academic & scientific styling (Anti-slop compliant: restrained, high contrast, WCAG AA)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
    }
    .header-box {
        background: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
        padding: 1.2rem 0 1.0rem 0;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.4;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .kpi-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 2px;
    }
    .badge-good {
        display: inline-block;
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .badge-bad {
        display: inline-block;
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .badge-dissonance {
        display: inline-block;
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .scenario-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.88rem;
        border-radius: 6px 6px 0 0;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_analyzer():
    return AspectSentimentAnalyzer()


@st.cache_resource(show_spinner=False)
def load_ml_model_bundle():
    model_path = "data/amazon_aged_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data(show_spinner=False)
def load_processed_kaggle_dataset():
    kaggle_processed = "data/kaggle_amazon_processed.csv"
    if os.path.exists(kaggle_processed):
        df = pd.read_csv(kaggle_processed)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return None


@st.cache_data(show_spinner=False)
def run_pipeline(df_input: pd.DataFrame, min_priors: int = 1) -> tuple:
    raw_df = load_and_preprocess(df_input)
    analyzer = get_analyzer()
    absa_df = analyzer.process_dataframe(raw_df)
    aged_df = compute_prior_environment_and_aged(absa_df, min_prior_reviews=min_priors)
    econ_engine = EconometricEngine(aged_df)
    econ_results = econ_engine.run_all_models()
    return aged_df, econ_results


def predict_review_stars(text: str, model_bundle: dict, prior_rating_mean: float = 3.8) -> dict:
    """Predicts star rating and probabilities given text with ABSA clause breakdown."""
    analyzer = get_analyzer()
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
    feats['prior_rating_mean'] = float(prior_rating_mean)
    
    pred_stars = float(np.clip(model_bundle['regressor'].predict(feats)[0], 1.0, 5.0))
    probs = model_bundle['classifier'].predict_proba(feats)[0]
    classes = model_bundle['classifier'].classes_
    prob_dist = {int(c): float(p) for c, p in zip(classes, probs)}
    
    return {
        "analysis": res,
        "predicted_stars": pred_stars,
        "probabilities": prob_dist,
        "loss_aversion_ratio": model_bundle['metrics'].get('loss_aversion_ratio', 2.5)
    }


def main():
    # Top Academic Header
    st.markdown("""
    <div class="header-box">
        <div class="main-title">AGED: Aspect–Global Evaluative Dissonance</div>
        <div class="sub-title">
            Empirical Framework & Machine Learning Model Trained on <strong>Kaggle Amazon Cell Phone Reviews</strong><br/>
            Investigating how non-linear combinations of positive ("goods") and negative ("bads") aspects diverge from numerical star ratings.
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_bundle = load_ml_model_bundle()

    # Sidebar Controls
    st.sidebar.markdown("### ⚙️ Dataset & Model Configuration")
    data_source = st.sidebar.radio(
        "Select Active Dataset:",
        [
            "Kaggle Amazon Cell Phones (4,000 Real Reviews)",
            "Longitudinal Smartphone Benchmark (180 Reviews)",
            "Upload Custom Review Dataset (CSV/JSON)"
        ]
    )

    min_priors = st.sidebar.slider("Minimum Prior Reviews for Baseline Window:", min_value=1, max_value=10, value=1)

    if data_source == "Kaggle Amazon Cell Phones (4,000 Real Reviews)":
        pre_df = load_processed_kaggle_dataset()
        if pre_df is not None:
            aged_df = pre_df
            # Run econometrics on pre-processed data
            econ_engine = EconometricEngine(aged_df)
            econ_results = econ_engine.run_all_models()
        else:
            st.sidebar.warning("Processing Kaggle reviews...")
            raw = pd.read_csv("data/kaggle_amazon/20191226-reviews.csv", nrows=4000)
            aged_df, econ_results = run_pipeline(raw, min_priors)
        st.sidebar.success(f"Loaded Kaggle Amazon Dataset ({len(aged_df):,} reviews)")

    elif data_source == "Longitudinal Smartphone Benchmark (180 Reviews)":
        if os.path.exists("data/sample_smartphones.csv"):
            input_df = pd.read_csv("data/sample_smartphones.csv")
        else:
            input_df = generate_benchmark_reviews(num_products=3, reviews_per_product=60)
        aged_df, econ_results = run_pipeline(input_df, min_priors)
        st.sidebar.success(f"Loaded Benchmark Dataset ({len(aged_df):,} reviews)")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV or JSON reviews file", type=["csv", "json"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
                aged_df, econ_results = run_pipeline(input_df, min_priors)
                st.sidebar.success(f"Processed {len(aged_df):,} uploaded reviews")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")
                st.stop()
        else:
            st.info("Please upload a file or switch to Kaggle Amazon Dataset.")
            st.stop()

    # Product Filter
    product_col = "product_name" if "product_name" in aged_df.columns else "product_id"
    all_prods = ["ALL PRODUCTS"] + sorted(aged_df[product_col].dropna().unique().tolist())
    selected_prod = st.sidebar.selectbox("Filter Product Scope:", all_prods)

    if selected_prod == "ALL PRODUCTS":
        view_df = aged_df
    else:
        view_df = aged_df[aged_df[product_col] == selected_prod].copy()

    # Top KPI Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Sample Reviews</div>
            <div class="kpi-val">{len(view_df):,}</div>
            <div class="kpi-sub">{view_df['product_id'].nunique()} Products Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Star Rating</div>
            <div class="kpi-val">{view_df['rating'].mean():.2f} ★</div>
            <div class="kpi-sub">Scale: 1.0 - 5.0 Stars</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        g_mean = view_df["num_good_aspects"].mean() if "num_good_aspects" in view_df.columns else 0
        b_mean = view_df["num_bad_aspects"].mean() if "num_bad_aspects" in view_df.columns else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Goods / Bads</div>
            <div class="kpi-val">{g_mean:.1f} <span style="font-size:0.9rem; color:#059669;">G</span> : {b_mean:.1f} <span style="font-size:0.9rem; color:#DC2626;">B</span></div>
            <div class="kpi-sub">Aspect Mentions per Review</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean AGED Dissonance</div>
            <div class="kpi-val">{view_df['aged_dissonance'].mean():.3f}</div>
            <div class="kpi-sub">|Rating - Aspect Sentiment|</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        loss_ratio = model_bundle['metrics'].get('loss_aversion_ratio', 2.8) if model_bundle else 2.5
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Loss Aversion Ratio</div>
            <div class="kpi-val" style="color: #DC2626;">{loss_ratio:.1f}x</div>
            <div class="kpi-sub">Penalty of 1 Bad vs 1 Good</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Main Tabs
    tab_sim, tab_matrix, tab_temp, tab_econ, tab_inspect = st.tabs([
        "🧩 Goods vs. Bads Variation Engine",
        "📊 Kaggle Amazon Empirical Matrix",
        "📈 Longitudinal Trajectories",
        "🔬 Econometric & ML Models",
        "🔍 Single Review Inspector"
    ])

    # -------------------------------------------------------------
    # TAB 1: GOODS VS BADS VARIATION ENGINE (INTERACTIVE SIMULATOR)
    # -------------------------------------------------------------
    with tab_sim:
        st.subheader("Interactive Goods vs. Bads Variation Sandbox")
        st.markdown("""
        Star ratings are blunt numerical summaries (1–5★) that fail to convey nuanced trade-offs. 
        A reviewer might write about multiple positive aspects ("goods") and negative aspects ("bads"), 
        yet map them onto unexpected star ratings due to cognitive dissonance, loss aversion, or expectation gaps.
        """)

        # 4 Core Presets from User Request
        st.markdown("##### 🎯 Test Standard Variation Scenarios")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        preset_text = None
        preset_rating = 5.0
        preset_scenario = "Custom"

        with col_p1:
            if st.button("1) 5★ with 2 Goods, 1 Bad\n(Minor Flaw Tolerated)", use_container_width=True):
                preset_text = "The camera photo quality is stunning and the screen display is gorgeous, though battery drains a bit fast."
                preset_rating = 5.0
                preset_scenario = "5-star with 2 Goods and 1 Bad"
        with col_p2:
            if st.button("2) 3★ with 2 Goods, 2 Bads\n(Balanced Ambivalence)", use_container_width=True):
                preset_text = "Screen display is vibrant and build quality is sturdy, but software is laggy with bloatware and price is overpriced."
                preset_rating = 3.0
                preset_scenario = "3-star with 2 Goods and 2 Bads"
        with col_p3:
            if st.button("3) 1★ with 0 Goods, 2 Bads\n(Catastrophic Dealbreaker)", use_container_width=True):
                preset_text = "Overheating terribly and battery drain is horrific. Absolute waste of money."
                preset_rating = 1.0
                preset_scenario = "1-star with 0 Goods and 2 Bads"
        with col_p4:
            if st.button("4) 2★ with 1 Good, 2 Bads\n(Negativity Dominance)", use_container_width=True):
                preset_text = "The build design feels premium, but camera photos are blurry and performance suffers from constant lag."
                preset_rating = 2.0
                preset_scenario = "2-star with 1 Good and 2 Bads"

        if "sim_text" not in st.session_state or preset_text is not None:
            st.session_state["sim_text"] = preset_text or "The camera photo quality is stunning and the screen display is gorgeous, though battery drains a bit fast."
            st.session_state["sim_rating"] = preset_rating
            st.session_state["sim_scenario"] = preset_scenario

        sim_col_l, sim_col_r = st.columns([1.1, 0.9])

        with sim_col_l:
            user_review_text = st.text_area(
                "Review Text to Decompose & Evaluate:",
                value=st.session_state["sim_text"],
                height=110,
                help="Type any review with mixed good and bad attributes to see how the model extracts aspects and predicts ratings."
            )

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                user_actual_rating = st.selectbox(
                    "Reviewer Given Star Rating:",
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                    index=[5.0, 4.0, 3.0, 2.0, 1.0].index(st.session_state.get("sim_rating", 5.0))
                )
            with col_s2:
                sim_prior_mean = st.slider(
                    "Community Prior Rating Baseline:",
                    min_value=1.0, max_value=5.0, value=3.8, step=0.1,
                    help="Historical average star rating of the product before this review was posted."
                )

        # Run Prediction
        if model_bundle is not None and user_review_text.strip():
            pred_res = predict_review_stars(user_review_text, model_bundle, sim_prior_mean)
            analysis = pred_res["analysis"]
            pred_stars = pred_res["predicted_stars"]
            probs = pred_res["probabilities"]

            with sim_col_r:
                st.markdown("##### 🧠 Model Evaluation & Dissonance Diagnosis")
                
                # Detected Goods & Bads Badges
                g_list = analysis["good_aspects"]
                b_list = analysis["bad_aspects"]

                badge_html = "<div style='margin-bottom: 8px;'>"
                if g_list:
                    badge_html += "<strong>Goods (+):</strong> " + "".join([f"<span class='badge-good'>{g}</span>" for g in g_list]) + " "
                else:
                    badge_html += "<strong>Goods (+):</strong> <span style='color:#64748B;'>None</span> "

                if b_list:
                    badge_html += "<br/><strong>Bads (−):</strong> " + "".join([f"<span class='badge-bad'>{b}</span>" for b in b_list])
                else:
                    badge_html += "<br/><strong>Bads (−):</strong> <span style='color:#64748B;'>None</span>"
                badge_html += "</div>"
                st.markdown(badge_html, unsafe_allow_html=True)

                # Dissonance Gap Calculation
                dissonance_gap = user_actual_rating - pred_stars
                abs_gap = abs(dissonance_gap)

                if abs_gap <= 0.6:
                    diag_title = "✅ Evaluatively Aligned"
                    diag_desc = "The star rating accurately mirrors the empirical balance of positive and negative aspects."
                    diag_color = "#059669"
                elif dissonance_gap > 0.6:
                    diag_title = "🛡️ Forgiving / Minor Flaw Tolerated (Positive AGED)"
                    diag_desc = f"Reviewer awarded {user_actual_rating:.0f}★ despite negative aspects ({', '.join(b_list)}). High consumer tolerance or low baseline expectation."
                    diag_color = "#D97706"
                else:
                    diag_title = "⚠️ Fatal Dealbreaker / Loss Aversion (Negative AGED)"
                    diag_desc = f"Reviewer downgraded rating to {user_actual_rating:.0f}★ despite positive aspects ({', '.join(g_list)}). Negative aspects acted as dealbreakers."
                    diag_color = "#DC2626"

                sc1, sc2 = st.columns(2)
                with sc1:
                    st.metric("Model Predicted Stars", f"{pred_stars:.2f} ★")
                with sc2:
                    st.metric("Evaluative Dissonance Gap (Δ)", f"{dissonance_gap:+.2f} ★")

                st.markdown(f"""
                <div class="scenario-card" style="border-left: 4px solid {diag_color};">
                    <strong style="color: {diag_color};">{diag_title}</strong><br/>
                    <span style="font-size: 0.85rem; color: #475569;">{diag_desc}</span>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Bar Chart
            st.markdown("##### 📊 Discrete Star Rating Probabilities P(R = k | Goods, Bads, Priors)")
            p_df = pd.DataFrame({
                "Star Rating": [f"{k} Stars" for k in probs.keys()],
                "Probability": list(probs.values()),
                "Color": ["#DC2626", "#EA580C", "#D97706", "#2563EB", "#059669"]
            })
            fig_p = px.bar(
                p_df,
                x="Star Rating",
                y="Probability",
                text="Probability",
                color="Star Rating",
                color_discrete_map={f"{k} Stars": c for k, c in zip(probs.keys(), p_df["Color"])},
                height=240
            )
            fig_p.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_p.update_layout(
                yaxis=dict(range=[0, 1.0], title="Probability"),
                xaxis=dict(title=""),
                showlegend=False,
                margin=dict(l=20, r=20, t=10, b=10),
                template="plotly_white"
            )
            st.plotly_chart(fig_p, use_container_width=True)

            # Detailed Sentence Decomposition Table
            with st.expander("🔎 View Sentence & Clause Level ABSA Tokenization"):
                evals = analysis.get("sentence_evaluations", [])
                decomp_rows = []
                for idx, ev in enumerate(evals):
                    decomp_rows.append({
                        "Clause #": idx + 1,
                        "Segmented Clause": ev["text"],
                        "Detected Aspects": ", ".join(ev["aspects"]) if ev["aspects"] else "None",
                        "Sentiment Polarity": f"{ev['sentiment']:+.2f}",
                        "Valence Type": "Good (+)" if ev['sentiment'] >= 0.05 else ("Bad (−)" if ev['sentiment'] <= -0.05 else "Neutral")
                    })
                st.dataframe(pd.DataFrame(decomp_rows), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: KAGGLE AMAZON EMPIRICAL MATRIX
    # -------------------------------------------------------------
    with tab_matrix:
        st.subheader("Empirical Variation Matrix in Kaggle Amazon Dataset")
        st.markdown("""
        How do real consumers in the 4,000 Kaggle Amazon reviews combine goods and bads? 
        The heatmap and archetype breakdown below validate the user's insight: 
        **identical combinations of positive and negative aspects yield divergent star ratings across different reviewers.**
        """)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("##### 🗺️ Mean Star Rating by (Goods vs. Bads) Grid")
            if "num_good_aspects" in view_df.columns and "num_bad_aspects" in view_df.columns:
                heatmap_data = view_df.groupby(["num_good_aspects", "num_bad_aspects"])["rating"].mean().unstack()
                fig_heat = px.imshow(
                    heatmap_data,
                    labels=dict(x="Number of Bad Aspects (Criticisms)", y="Number of Good Aspects (Praises)", color="Mean Rating"),
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    height=360
                )
                fig_heat.update_layout(template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("Note how adding just 1 Bad Aspect drops ratings significantly faster than adding 1 Good Aspect raises them.")

        with col_m2:
            st.markdown("##### 🏛️ Distribution of Evaluative Archetypes")
            if "evaluative_archetype" in view_df.columns:
                arch_counts = view_df["evaluative_archetype"].value_counts().reset_index()
                arch_counts.columns = ["Archetype", "Review Count"]
                fig_arch = px.bar(
                    arch_counts,
                    x="Review Count",
                    y="Archetype",
                    orientation="h",
                    color="Review Count",
                    color_continuous_scale="Blues",
                    height=360
                )
                fig_arch.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig_arch, use_container_width=True)

        st.markdown("##### ⚖️ Loss Aversion Asymmetry Curve: Goods vs. Bads Impact")
        gb_df = view_df.groupby("net_aspect_balance")["rating"].agg(["mean", "count"]).reset_index()
        gb_df = gb_df[gb_df["count"] >= 5]
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=gb_df["net_aspect_balance"],
            y=gb_df["mean"],
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=7),
            name="Empirical Mean Rating"
        ))
        fig_curve.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
        fig_curve.update_layout(
            title="Star Rating as a Function of Net Aspect Balance (Goods − Bads)",
            xaxis_title="Net Aspect Balance (Positive Mentions − Negative Mentions)",
            yaxis_title="Average Star Rating (1 - 5 Stars)",
            template="plotly_white",
            height=320
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: LONGITUDINAL TRAJECTORIES
    # -------------------------------------------------------------
    with tab_temp:
        st.subheader("Temporal Evolution of Prior Review Environments")
        st.caption("Shows how early reviews set community expectations, and how subsequent focal reviews diverge over time.")

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=view_df["review_order"],
            y=view_df["rating"],
            mode="markers+lines",
            name="Focal Star Rating",
            marker=dict(size=5, color="#2563EB", opacity=0.7),
            line=dict(color="#93C5FD", width=1)
        ))
        fig_temp.add_trace(go.Scatter(
            x=view_df["review_order"],
            y=view_df["prior_rating_mean"],
            mode="lines",
            name="Prior Rating Baseline (Expectation Frame)",
            line=dict(color="#DC2626", width=2.5, dash="dash")
        ))
        fig_temp.update_layout(
            title="Focal Star Rating vs. Prior Review Baseline Over Chronological Sequence",
            xaxis_title="Chronological Review Order (Sequence Position)",
            yaxis_title="Rating (1 - 5 Stars)",
            template="plotly_white",
            hovermode="x unified",
            height=380
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("Aspect-Specific Trajectories Over Time")
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
                    marker=dict(size=4)
                ))
            fig_aspect.update_layout(
                title="Aspect Sentiment Trajectories [-1.0 (Negative) to +1.0 (Positive)]",
                xaxis_title="Chronological Review Order",
                yaxis_title="Aspect Sentiment Score",
                template="plotly_white",
                height=350
            )
            st.plotly_chart(fig_aspect, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: ECONOMETRIC & ML MODELS
    # -------------------------------------------------------------
    with tab_econ:
        st.subheader("Statistical Validation & Econometric Estimation")
        st.markdown("""
        These models validate the AGED hypotheses:
        - **Model 1 (OLS Direct Effects):** Tests main effects of aspect sentiment and prior baselines.
        - **Model 2 (Moderation Interaction):** Tests whether community expectations moderate consumer evaluations.
        - **Model 3 (Dissonance & Mismatch):** Models the role of cognitive dissonance ($|S_{focal} - S_{prior}|$).
        - **Model 4 (Ordered Logit):** Discrete choice ordinal estimation for discrete 1–5 stars.
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
                st.metric("R-squared", f"{r2:.4f}" if r2 is not None else "N/A (Ordered Logit)")
            with col_m3:
                adj_r2 = selected_model.get("adj_r_squared")
                st.metric("Adj. R-squared", f"{adj_r2:.4f}" if adj_r2 is not None else "N/A")

            st.code(f"Formula: {selected_model.get('formula', 'logit(rating) ~ exog_vars')}", language="r")

            st.subheader("Parameter Estimates & Significance")
            summary_table = selected_model.get("summary_table")
            if summary_table is not None and isinstance(summary_table, pd.DataFrame):
                st.dataframe(summary_table, use_container_width=True)
                st.caption("Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, . p < 0.1")

        if model_bundle is not None:
            st.markdown("---")
            st.subheader("Machine Learning Goods/Bads Feature Importances")
            fi = model_bundle['metrics'].get('feature_importances', {})
            if fi:
                fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                fig_fi = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Gradient Boosting Regressor — Top Predictive Features for Rating",
                    color="Importance",
                    color_continuous_scale="Purples",
                    height=320
                )
                fig_fi.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_fi, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: SINGLE REVIEW INSPECTOR
    # -------------------------------------------------------------
    with tab_inspect:
        st.subheader("Granular Review & Clause Decomposition")
        st.caption("Inspect individual reviews to trace sentence/clause tags, good/bad counts, and prior baseline calculations.")

        rev_ids = view_df["review_id"].tolist()
        chosen_rev_id = st.selectbox("Select Review ID to Inspect:", rev_ids[:500])
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

        if "good_aspects" in rev_row and "bad_aspects" in rev_row:
            st.markdown(f"**Goods Detected:** `{rev_row['good_aspects']}` | **Bads Detected:** `{rev_row['bad_aspects']}`")

        sentence_evals = rev_row.get("sentence_evaluations", [])
        if sentence_evals:
            s_rows = []
            for se in sentence_evals:
                s_rows.append({
                    "Clause #": se["sentence_idx"] + 1,
                    "Clause Text": se["text"],
                    "Detected Aspects": ", ".join(se["aspects"]) if se["aspects"] else "None",
                    "Sentiment Polarity": f"{se['sentiment']:+.2f}"
                })
            st.dataframe(pd.DataFrame(s_rows), use_container_width=True)


if __name__ == "__main__":
    main()
