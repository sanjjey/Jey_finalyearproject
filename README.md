# AGED: Aspect–Global Evaluative Dissonance in Online Product Reviews

### A Temporal Analysis of Consumer Experience and Prior Review Environments
Based on the research framework by **A Jey Pradeep (22MIS1225)** and **Dr. V. Pandiyaraju**.

---

## 📌 1. Theoretical Overview & What AGED Is

Online product reviews provide two complementary sources of evaluation:
1. **Aspect-Level Experience:** Granular evaluations of specific product attributes (e.g., *Battery, Camera, Display, Performance, Build, Price*).
2. **Global Evaluation:** Overall product star rating (1 to 5 stars) and high-level impression.

In conventional literature, these have often been treated as static, independent processes. The **AGED Framework** integrates them into a dynamic temporal model:
- **Prior Review Environment:** Consumers write reviews under the influence of earlier ratings and community sentiment, which form an *expectation frame*.
- **Expectation Mismatch / Dissonance:** When a consumer's focal aspect experience diverges from the prior environment ($|S_{focal} - S_{prior}|$), cognitive dissonance is triggered.
- **Aspect–Global Dissonance:** This expectation gap systematically moderates how consumers translate their own feature satisfaction into overall star ratings, explaining review polarization, bandwagon effects, and rating inflation/deflation over product lifecycles.

---

## 🏗️ 2. Modular Architecture

```
d:/brother_project/
├── .venv/                         # Lightweight virtual environment (via uv)
├── aged/                          # Modular AGED core package
│   ├── __init__.py                # Package declaration
│   ├── config.py                  # Default aspect taxonomies & rating constants
│   ├── ingestion.py               # Phase 1: Ingestion, schema normalization & sentence segmentation
│   ├── absa.py                    # Phase 2: Aspect extraction & domain-aware sentiment intensity
│   ├── temporal_engine.py         # Phase 3: O(N) prior review environment & AGED dissonance engine
│   ├── econometrics.py            # Phase 4: Econometric modeling (OLS, Moderation, Ordered Logit)
│   └── synthetic_generator.py     # Phase 5: Benchmark longitudinal data generator
├── data/                          # Data directory
│   ├── sample_smartphones.csv     # Pre-generated longitudinal review stream (180 reviews)
│   └── aged_processed.csv         # Processed panel dataset with regression variables
├── app.py                         # Phase 5: Interactive Streamlit analytics dashboard
├── run_pipeline.py                # Command-line end-to-end execution script
├── test_aged.py                   # Automated verification unit test suite (5/5 passing)
└── README.md                      # Complete system documentation
```

---

## ⚡ 3. Performance & $O(N)$ Optimization

Calculating historical prior baselines naively requires an $O(N^2)$ search across all past reviews for every focal review. 
This implementation uses **cumulative running sums, squares, and counts**:
- Prior rating mean: $\mu_{R, prior}(t) = \frac{\sum_{k < t} R_k}{N_{prior}}$
- Prior rating variance: $\sigma^2_{R, prior}(t) = \frac{\sum R_k^2 - \frac{(\sum R_k)^2}{N_{prior}}}{N_{prior} - 1}$
- Prior aspect environment: $\mu_{S, prior}(a, t) = \frac{\sum_{k < t} S_k(a)}{N_{prior}(a)}$

This enables **$O(1)$ lookup per review** and **$O(N)$ linear time complexity** across the entire dataset. A dataset of hundreds of reviews processes in **under 1.5 seconds**.

---

## 🚀 4. Quickstart Guide

### Step 1: Activate Virtual Environment
The virtual environment is already prepared using `uv`:
```powershell
.venv\Scripts\activate
```

*(If you ever need to re-install packages with uv)*:
```powershell
uv --cache-dir .uv_cache pip install pandas numpy scipy statsmodels nltk streamlit plotly pyarrow scikit-learn
```

### Step 2: Run Headless Pipeline (CLI)
To run the complete data ingestion, ABSA, temporal aggregation, and econometrics from the terminal:
```powershell
.venv\Scripts\python run_pipeline.py --input data/sample_smartphones.csv
```

### Step 3: Run Automated Test Suite
To verify all unit tests and mathematical formulas:
```powershell
.venv\Scripts\python test_aged.py
```

### Step 4: Launch Interactive Dashboard
To open the interactive visual analytics dashboard in your browser:
```powershell
.venv\Scripts\streamlit run app.py
```

---

## 📊 5. Econometric Models Implemented

The system estimates 4 regression models directly implementing the research methodology:

1. **Model 1: Direct Effects (OLS)**
   $$Rating_i = \beta_0 + \beta_1 S_{focal} + \beta_2 \mu_{R, prior} + \beta_3 \sigma^2_{R, prior} + \gamma Controls + \epsilon$$

2. **Model 2: Moderation by Prior Environment (Interaction OLS)**
   $$Rating_i = \beta_0 + \beta_1 S_{focal} + \beta_2 \mu_{R, prior} + \beta_3 (S_{focal} \times \mu_{R, prior}) + \gamma Controls + \epsilon$$
   *Validates whether prior ratings amplify or dampen consumer aspect-level reactions.*

3. **Model 3: AGED Dissonance & Mismatch Dynamics (OLS)**
   $$Rating_i = \beta_0 + \beta_1 S_{focal} + \beta_2 \mu_{R, prior} + \beta_3 \overline{D}_{env} + \beta_4 (S_{focal} \times \overline{D}_{env}) + \gamma Controls + \epsilon$$
   *Directly tests if expectation disconfirmation ($|S_{focal} - S_{prior}|$) drives rating penalties.*

4. **Model 4: Ordered Logistic Regression**
   *Models star ratings as discrete ordinal choice categories ($1, 2, 3, 4, 5$).*

---

## 💰 6. Cost & Privacy Guarantee
- **100% Free & Open Source:** Zero paid API keys, zero subscription dependencies.
- **100% Local Execution:** All data, text processing, and statistical estimations run entirely on your local machine.
