"""
================================================================================
AI DATA SCIENCE COMMAND CENTER
================================================================================
A single-file Streamlit application that turns exploratory data analysis,
visualization, machine learning, and AI-style insight generation into a
futuristic, mission-control-style dashboard experience.

Run with:
    streamlit run app.py

Everything (styling, state, data generation, charts, ML, "AI" assistant)
lives in this one file, organized into clearly commented sections and
reusable functions.
================================================================================
"""

import os
import re
import time
import random
import platform
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    classification_report, r2_score, mean_absolute_error,
    mean_squared_error, silhouette_score,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# --- AI chat providers (both optional; the app degrades gracefully if either
#     package or API key is missing — see the AI PROVIDER INTEGRATION section) ---
try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False


# ==============================================================================
# PAGE CONFIG  (must be the first Streamlit call)
# ==============================================================================
st.set_page_config(
    page_title="AI Data Science Command Center",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# THEMES  — distinct "mission control" color palettes, selectable in Settings
# ==============================================================================
THEMES = {
    "Aurora Nightfall": {"bg1": "#060912", "bg2": "#0d1326", "accent1": "#5eead4", "accent2": "#a78bfa", "accent3": "#f472b6", "text": "#e5e7eb"},
    "Cyberpunk Ember":  {"bg1": "#0a0505", "bg2": "#1a0a12", "accent1": "#fb923c", "accent2": "#f43f5e", "accent3": "#facc15", "text": "#f5f5f4"},
    "Deep Ocean":       {"bg1": "#020617", "bg2": "#0b1225", "accent1": "#38bdf8", "accent2": "#22d3ee", "accent3": "#818cf8", "text": "#e2e8f0"},
    "Midnight Violet":  {"bg1": "#0a0512", "bg2": "#180a2b", "accent1": "#c084fc", "accent2": "#818cf8", "accent3": "#f0abfc", "text": "#ede9fe"},
}

QUOTES = [
    "Data is the new oil, but insight is the refinery.",
    "In God we trust. All others must bring data.",
    "The goal is to turn data into information, and information into insight.",
    "Without data, you're just another person with an opinion.",
    "Machine learning is the last invention humanity will ever need to make.",
    "The best thing about being a statistician is you get to play in everyone's backyard.",
    "Torture the data long enough and it will confess to anything.",
    "It is a capital mistake to theorize before one has data.",
]

FACTS = [
    "The term 'Big Data' was first popularized in a 1997 NASA research paper.",
    "Over 300 million terabytes of data are generated across the world every day.",
    "Random Forests were introduced by statistician Leo Breiman in 2001.",
    "The k-Nearest-Neighbors algorithm dates all the way back to 1951.",
    "Python overtook R as the most popular data science language around 2016.",
    "The first neural network, the Perceptron, was built in 1958.",
    "A single Google search touches hundreds of machine learning models.",
]


# ==============================================================================
# CSS INJECTION — glassmorphism, gradients, animated cards, custom fonts
# ==============================================================================
def inject_css(theme_name):
    t = THEMES[theme_name]
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {{
        --bg1: {t['bg1']};
        --bg2: {t['bg2']};
        --accent1: {t['accent1']};
        --accent2: {t['accent2']};
        --accent3: {t['accent3']};
        --text: {t['text']};
    }}

    html, body, .stApp {{
        background: radial-gradient(circle at 12% 8%, var(--bg2), var(--bg1) 55%) fixed;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}

    .stApp::before {{
        content: "";
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
        background-size: 42px 42px;
        pointer-events: none;
        z-index: 0;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--bg2), var(--bg1));
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    .sidebar-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 1.1rem;
        letter-spacing: 0.06em;
        background: linear-gradient(90deg, var(--accent1), var(--accent2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 0.4rem 0 1rem 0;
    }}

    .hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 2.6rem; text-align: center;
        letter-spacing: 0.03em;
        background: linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3), var(--accent1));
        background-size: 300% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: hueShift 8s linear infinite;
        margin-bottom: 0;
    }}
    .hero-subtitle {{
        text-align: center; color: rgba(255,255,255,0.55);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem; letter-spacing: 0.05em;
        margin-top: 0.2rem; margin-bottom: 1.6rem;
    }}
    @keyframes hueShift {{
        0% {{ background-position: 0% 50%; }}
        100% {{ background-position: 300% 50%; }}
    }}

    .glass-card {{
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }}
    .glass-card:hover {{ border-color: var(--accent1); transform: translateY(-2px); }}
    .quote-card {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.92rem;
        animation: colorCycle 6s ease-in-out infinite;
    }}
    @keyframes colorCycle {{
        0%   {{ border-color: var(--accent1); }}
        33%  {{ border-color: var(--accent2); }}
        66%  {{ border-color: var(--accent3); }}
        100% {{ border-color: var(--accent1); }}
    }}

    .kpi-card {{
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 1rem; text-align: center;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.5s ease both;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px) scale(1.02);
        border-color: var(--accent2);
        box-shadow: 0 8px 24px rgba(0,0,0,0.35), 0 0 18px -4px var(--accent2);
    }}
    .kpi-icon {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 700;
        background: linear-gradient(90deg, var(--accent1), var(--accent2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .kpi-label {{ font-size: 0.76rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    .ai-bubble {{
        display: flex; align-items: flex-start; gap: 0.7rem;
        background: rgba(255,255,255,0.04);
        border-left: 3px solid var(--accent1);
        border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
        animation: fadeInUp 0.5s ease both;
    }}
    .ai-bubble.ai-warning {{ border-left-color: #facc15; }}
    .ai-bubble.ai-success {{ border-left-color: #4ade80; }}
    .ai-bubble.ai-info    {{ border-left-color: var(--accent2); }}
    .ai-icon {{ font-size: 1.2rem; }}
    .ai-text {{ font-size: 0.92rem; line-height: 1.4; }}

    .log-entry {{
        background: rgba(255,255,255,0.03); border-radius: 10px;
        padding: 0.5rem 0.9rem; margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    }}

    .footer {{
        text-align: center; color: rgba(255,255,255,0.4); font-size: 0.78rem;
        padding: 1rem 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;
    }}

    .fab {{
        position: fixed; bottom: 24px; right: 24px;
        width: 50px; height: 50px; border-radius: 50%;
        background: linear-gradient(135deg, var(--accent1), var(--accent2));
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        z-index: 999; animation: floatPulse 3s ease-in-out infinite;
    }}
    @keyframes floatPulse {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-8px); }}
    }}

    .stButton>button {{
        background: linear-gradient(90deg, var(--accent1), var(--accent2));
        color: #06090f; font-weight: 600; border: none;
        border-radius: 10px; padding: 0.5rem 1.1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 18px -4px var(--accent2);
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,0.03); border-radius: 10px 10px 0 0; padding: 0.5rem 1rem; }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
        color: #06090f !important;
    }}

    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: var(--accent2); border-radius: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg1); }}

    [data-testid="stMetricValue"] {{ color: var(--accent1); font-family: 'JetBrains Mono', monospace; }}

    /* ---- AI Data Chat page ---- */
    .ai-chat-header {{
        display: flex; align-items: center; justify-content: space-between;
        background: linear-gradient(120deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1rem 1.4rem; margin-bottom: 1rem;
    }}
    .ai-chat-header .ai-chat-title {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem;
        background: linear-gradient(90deg, var(--accent1), var(--accent2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .ai-chat-subtitle {{
        color: rgba(255,255,255,0.55); font-size: 0.85rem; margin-top: 0.15rem;
    }}
    .provider-pill {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600;
        padding: 0.35rem 0.8rem; border-radius: 999px;
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
        display: inline-flex; align-items: center; gap: 0.4rem; white-space: nowrap;
    }}
    .provider-pill .dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent1); box-shadow: 0 0 8px var(--accent1);
        animation: floatPulse 2s ease-in-out infinite;
    }}
    .chat-quick-label {{
        font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: rgba(255,255,255,0.4); margin: 0.2rem 0 0.5rem 0;
    }}
    .chat-provider-caption {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        color: rgba(255,255,255,0.45); margin-top: 0.3rem;
    }}
    [data-testid="stChatMessage"] {{
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 0.3rem 0.4rem; margin-bottom: 0.5rem;
    }}
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE
# ==============================================================================
def init_session_state():
    """Initialize all session-state keys the app relies on, exactly once."""
    defaults = {
        "df": None,
        "dataset_source": None,
        "page": "🏠 Dashboard",
        "theme": "Aurora Nightfall",
        "start_time": time.time(),
        "interactions": 0,
        "logs": [],
        "chat_history": [],
        "last_accuracy": None,
        "last_task_type": None,
        "last_inference_time": None,
        "upload_time": None,
        "point_size": 8,
        "opacity": 0.75,
        "num_bins": 30,
        "animation_speed": 1.0,
        "engineered_df": None,
        # --- AI chat provider settings (see Settings page) ---
        "ai_provider": "Gemini",                                   # "Gemini" or "Groq"
        "ai_auto_fallback": True,                                  # fall back to the other provider on failure
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        "gemini_model": "gemini-2.5-flash",
        "groq_api_key": os.environ.get("GROQ_API_KEY", "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"),
        "groq_model": "openai/gpt-oss-120b",
        "last_ai_provider_used": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state.interactions += 1


def add_log(action, icon="📝"):
    """Append an entry to the activity log (capped at 50 entries)."""
    st.session_state.logs.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "icon": icon,
    })
    st.session_state.logs = st.session_state.logs[:50]


# ==============================================================================
# SYNTHETIC DATASET  (bonus requirement: usable with zero uploads)
# ==============================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(n_rows=600, seed=42):
    """Generate a realistic, slightly messy customer-analytics dataset."""
    rng = np.random.default_rng(seed)

    customer_id = np.arange(1, n_rows + 1)
    age = rng.normal(38, 12, n_rows).clip(18, 80).round().astype(float)
    gender = rng.choice(["Male", "Female", "Other"], n_rows, p=[0.47, 0.47, 0.06])
    region = rng.choice(["North", "South", "East", "West", "Central"], n_rows)
    country = np.array(["United States"] * n_rows)  # intentionally constant

    device_types = [f"Device_{i}" for i in range(1, 15)]  # intentionally high-cardinality
    device = rng.choice(device_types, n_rows)

    start_date = pd.Timestamp("2022-01-01")
    signup_offsets = rng.integers(0, 1000, n_rows)
    signup_date = start_date + pd.to_timedelta(signup_offsets, unit="D")

    tenure_months = rng.integers(1, 60, n_rows).astype(float)
    monthly_income = rng.lognormal(mean=8.5, sigma=0.5, size=n_rows).round(2)
    support_tickets = rng.poisson(2, n_rows).astype(float)
    satisfaction = np.clip(5 - support_tickets * 0.4 + rng.normal(0, 0.8, n_rows), 1, 5).round().astype(float)
    monthly_spend = (monthly_income * 0.02 + tenure_months * 1.5 + rng.normal(0, 15, n_rows)).clip(5, None).round(2)
    is_premium = rng.choice(["Yes", "No"], n_rows, p=[0.3, 0.7])

    churn_logit = (-0.35 * satisfaction + 0.25 * support_tickets - 0.015 * tenure_months
                   + rng.normal(0, 0.6, n_rows))
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn_labels = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    lifetime_value = (monthly_spend * tenure_months + rng.normal(0, 200, n_rows)).clip(0, None).round(2)

    df = pd.DataFrame({
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "Region": region,
        "Country": country,
        "DeviceType": device,
        "SignupDate": signup_date,
        "TenureMonths": tenure_months,
        "MonthlyIncome": monthly_income,
        "SupportTickets": support_tickets,
        "SatisfactionScore": satisfaction,
        "MonthlySpend": monthly_spend,
        "IsPremium": is_premium,
        "Churn": churn_labels,
        "LifetimeValue": lifetime_value,
    })

    # Inject realistic missing values
    for col in ["Age", "MonthlyIncome", "SatisfactionScore"]:
        mask = rng.random(n_rows) < 0.04
        df.loc[mask, col] = np.nan

    # Inject duplicate rows on purpose (data-quality demo)
    dup_sample = df.sample(n=min(15, n_rows), random_state=seed)
    df = pd.concat([df, dup_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ==============================================================================
# DATA PROFILING HELPERS
# ==============================================================================
def detect_column_types(df):
    """Classify columns and flag common data-quality issues."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    hc_threshold = max(10, int(0.02 * len(df))) if len(df) else 10
    high_card_cols = [c for c in categorical_cols if df[c].nunique() > hc_threshold]
    missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    duplicates = int(df.duplicated().sum())
    return {
        "numeric": numeric_cols, "categorical": categorical_cols, "datetime": datetime_cols,
        "constant": constant_cols, "high_cardinality": high_card_cols,
        "missing": missing, "duplicates": duplicates,
    }


def get_dataset_summary(df):
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
    }


def calculate_health_score(df):
    """A rough 0-100 'data health' score penalizing common quality issues."""
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        return 0
    score = 100.0
    missing_pct = df.isna().sum().sum() / (n_rows * n_cols) * 100
    dup_pct = df.duplicated().sum() / n_rows * 100
    const_cols = sum(df[c].nunique(dropna=False) <= 1 for c in df.columns)
    score -= min(missing_pct * 1.5, 30)
    score -= min(dup_pct * 1.2, 20)
    score -= const_cols * 5
    return int(max(0, min(100, round(score))))


def random_quote():
    return random.choice(QUOTES)


def random_fact():
    return random.choice(FACTS)


# ==============================================================================
# RULE-BASED INSIGHTS  (lightweight, no API calls — powers the exported .txt report)
# ==============================================================================
def generate_ai_insights(df, target=None):
    """Produce a list of plain-language observations about the dataset."""
    insights = []
    types = detect_column_types(df)

    for col in types["numeric"]:
        skew = df[col].skew()
        if pd.notna(skew) and abs(skew) > 1:
            direction = "right" if skew > 0 else "left"
            insights.append({"icon": "📈", "type": "warning",
                              "text": f"Column '{col}' shows significant {direction}-skewness (skew = {skew:.2f})."})

    for col, cnt in types["missing"].items():
        pct = cnt / len(df) * 100
        insights.append({"icon": "🕳️", "type": "warning",
                          "text": f"Column '{col}' contains {cnt} missing values ({pct:.1f}% of rows)."})

    if types["duplicates"] > 0:
        insights.append({"icon": "🪞", "type": "warning",
                          "text": f"Dataset contains {types['duplicates']} duplicate rows — removing them may improve model quality."})

    for col in types["constant"]:
        insights.append({"icon": "🚫", "type": "info",
                          "text": f"Column '{col}' is constant across all rows and adds no predictive value."})

    for col in types["high_cardinality"]:
        insights.append({"icon": "🔑", "type": "info",
                          "text": f"Column '{col}' has high cardinality and may need special encoding before modeling."})

    if target and target in df.columns:
        if df[target].dtype == object or df[target].nunique() < 15:
            vc = df[target].value_counts(normalize=True)
            if len(vc) and vc.max() > 0.65:
                insights.append({"icon": "⚖️", "type": "warning",
                                  "text": f"The target '{target}' appears imbalanced ('{vc.idxmax()}' makes up {vc.max()*100:.1f}% of rows)."})
        numeric_feats = [c for c in types["numeric"] if c != target]
        if numeric_feats and pd.api.types.is_numeric_dtype(df[target]):
            corrs = df[numeric_feats + [target]].corr(numeric_only=True)[target].drop(target).abs().dropna().sort_values(ascending=False)
            if len(corrs):
                top = corrs.index[0]
                insights.append({"icon": "🔗", "type": "success",
                                  "text": f"Feature '{top}' strongly correlates with target '{target}' (|r| = {corrs.iloc[0]:.2f})."})

    if not insights:
        insights.append({"icon": "✅", "type": "success",
                          "text": "No major data quality issues detected — this dataset looks healthy!"})
    return insights


# ==============================================================================
# AI PROVIDER INTEGRATION  —  Gemini (primary) + Groq (fallback / switchable)
# ==============================================================================
# This section powers the "🧠 AI Data Chat" page. A single master prompt asks
# whichever model is active to either (a) answer in Markdown, or (b) return
# executable pandas/matplotlib/seaborn code that operates on the in-memory
# `df`. The code path mirrors the exec()-based pattern from the original
# notebook snippet, with a light keyword guard before anything is executed.

MASTER_PROMPT_TEMPLATE = """You are a senior data analyst AI embedded inside a Streamlit dashboard.
You are analyzing a pandas DataFrame that is ALREADY loaded in memory as `df`. Do not reload,
recreate, or simulate it — just use `df` directly.

Dataset overview:
- Shape: {n_rows} rows x {n_cols} columns
- Columns and dtypes: {dtypes}
- Missing values per column: {missing}
- Summary statistics:
{describe}

Decide the best way to answer the user's question:
1. If it can be answered with an explanation, numbers, or a short table, respond in **Markdown only**
   (no code, no code fences).
2. If it needs a computed statistic, transformation, or chart, respond with **executable Python code
   only** that uses the existing variable `df`, plus the already-imported `pd`, `np`, `plt`, `sns`,
   and the Streamlit module `st`.
   - Do NOT import anything, do NOT read/write files, do NOT access the network or the OS.
   - Do NOT redefine or reload `df`.
   - Do NOT include markdown code fences (no ```).
   - For a chart: build it with `fig, ax = plt.subplots()` (matplotlib) or seaborn on an `ax` you
     created, then display it with `st.pyplot(fig)`.
   - For a table or number: display it with `st.dataframe(...)`, `st.write(...)`, or `st.metric(...)`.

Respond with EXACTLY one of these two formats — start your reply with one of these two lines,
literally, as the very first line:
MODE: MARKDOWN
MODE: CODE
...followed by a newline and then the content (markdown text, or raw python code — nothing else).

User question: {question}
"""


def build_master_prompt(df, question):
    """Fill the master prompt template with live stats about the active dataset."""
    dtypes_str = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
    missing = df.isna().sum()
    missing = missing[missing > 0]
    missing_str = ", ".join(f"{c}: {int(v)}" for c, v in missing.items()) or "None"
    try:
        describe_str = df.describe(include="all").T.to_string()
    except Exception:
        describe_str = "N/A"
    # Keep the prompt from growing unbounded on very wide describe() outputs
    if len(describe_str) > 6000:
        describe_str = describe_str[:6000] + "\n... (truncated)"

    return MASTER_PROMPT_TEMPLATE.format(
        n_rows=df.shape[0], n_cols=df.shape[1],
        dtypes=dtypes_str, missing=missing_str, describe=describe_str,
        question=question,
    )


def call_gemini(prompt, api_key, model_name):
    """Call Google's Gemini API and return the raw text response."""
    if not GEMINI_SDK_AVAILABLE:
        raise RuntimeError("The 'google-generativeai' package isn't installed (pip install google-generativeai).")
    if not api_key:
        raise RuntimeError("No Gemini API key configured on the server.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def call_groq(prompt, api_key, model_name):
    """Call the Groq API and return the raw text response (mirrors the notebook snippet)."""
    if not GROQ_SDK_AVAILABLE:
        raise RuntimeError("The 'groq' package isn't installed (pip install groq).")
    if not api_key:
        raise RuntimeError("No Groq API key configured on the server.")
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        stream=False,
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")
    return content


def get_ai_response(prompt):
    """
    Route the prompt to the active provider (Settings page), automatically
    falling back to the other provider if the primary one fails and
    'ai_auto_fallback' is enabled. Returns (raw_text, provider_label_used).
    """
    provider = st.session_state.ai_provider
    fallback_enabled = st.session_state.ai_auto_fallback

    providers = {
        "Gemini": lambda: call_gemini(prompt, st.session_state.gemini_api_key, st.session_state.gemini_model),
        "Groq": lambda: call_groq(prompt, st.session_state.groq_api_key, st.session_state.groq_model),
    }
    other = "Groq" if provider == "Gemini" else "Gemini"

    try:
        return providers[provider](), provider
    except Exception as primary_err:
        if not fallback_enabled:
            raise
        try:
            result = providers[other]()
            return result, f"{other} (fallback — {provider} failed: {primary_err})"
        except Exception as fallback_err:
            raise RuntimeError(
                f"{provider} failed ({primary_err}); fallback {other} also failed ({fallback_err})."
            )


def strip_code_fences(text):
    """Remove accidental ```python ... ``` fences if the model added them anyway."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def parse_ai_response(raw):
    """Split the model's reply into ('CODE'|'MARKDOWN', content)."""
    raw = raw.strip()
    upper = raw.upper()
    if upper.startswith("MODE: CODE"):
        rest = raw.split("\n", 1)[1] if "\n" in raw else ""
        return "CODE", strip_code_fences(rest)
    if upper.startswith("MODE: MARKDOWN"):
        rest = raw.split("\n", 1)[1] if "\n" in raw else ""
        return "MARKDOWN", strip_code_fences(rest)
    # Model didn't follow the MODE: prefix — fall back to a simple heuristic.
    looks_like_code = any(tok in raw for tok in ["st.pyplot(", "st.dataframe(", "plt.subplots(", "df["]) 
    return ("CODE", strip_code_fences(raw)) if looks_like_code else ("MARKDOWN", strip_code_fences(raw))


# Keyword blocklist used before exec()-ing any AI-generated code. This is a
# basic guard rail, NOT a real sandbox — for production use, run generated
# code in an isolated subprocess/container instead of the app process.
_FORBIDDEN_CODE_TOKENS = [
    "import ", "__import__", "open(", "eval(", "exec(", "os.", "sys.",
    "subprocess", "socket", "shutil", "pathlib", "Path(", "input(",
    "globals(", "locals(", "getattr(", "setattr(", "delattr(",
    "compile(", "requests", "urllib", ".system(", "pip ",
]


def is_code_safe(code):
    for tok in _FORBIDDEN_CODE_TOKENS:
        if tok in code:
            return False, tok
    return True, None


def execute_generated_code(code, df):
    """Run AI-generated analysis code against the active dataframe inside Streamlit."""
    safe, bad_token = is_code_safe(code)
    with st.expander("🔍 View generated code", expanded=False):
        st.code(code, language="python")
    if not safe:
        st.error(f"⚠️ Generated code was blocked for safety (contains '{bad_token.strip()}'). Try rephrasing your question.")
        return

    sandbox_globals = {"__builtins__": __builtins__}
    sandbox_locals = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns, "st": st}
    try:
        exec(code, sandbox_globals, sandbox_locals)
        # Safety net: if the code drew a matplotlib figure without explicitly
        # calling st.pyplot(), show whatever is on the current figure anyway.
        if plt.get_fignums() and "st.pyplot(" not in code:
            st.pyplot(plt.gcf())
        plt.close("all")
    except Exception as e:
        st.error(f"Error running the generated analysis code: {e}")


# ==============================================================================
# CHART / UI HELPER FUNCTIONS
# ==============================================================================
def style_fig(fig, height=420):
    theme = THEMES[st.session_state.theme]
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["text"], family="Inter, sans-serif"),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def kpi_card_html(icon, label, value):
    return f"""<div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def circular_progress_html(label, percent, color="var(--accent1)"):
    percent = max(0, min(100, percent))
    deg = percent * 3.6
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;">
        <div style="width:110px;height:110px;border-radius:50%;
             background:conic-gradient({color} {deg}deg, rgba(255,255,255,0.08) {deg}deg);
             display:flex;align-items:center;justify-content:center;">
            <div style="width:86px;height:86px;border-radius:50%;background:var(--bg1);
                 display:flex;align-items:center;justify-content:center;
                 font-family:'JetBrains Mono',monospace;font-weight:700;font-size:1.05rem;color:{color};">
                {percent:.0f}%
            </div>
        </div>
        <div style="margin-top:0.5rem;font-size:0.8rem;color:rgba(255,255,255,0.6);">{label}</div>
    </div>
    """


def build_gauge_chart(value, title="Health Score"):
    theme = THEMES[st.session_state.theme]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": theme["text"]}},
        number={"font": {"color": theme["accent1"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": theme["text"]},
            "bar": {"color": theme["accent1"]},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 40], "color": "rgba(244,63,94,0.3)"},
                {"range": [40, 70], "color": "rgba(250,204,21,0.3)"},
                {"range": [70, 100], "color": "rgba(94,234,212,0.3)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=theme["text"]),
                       height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_timeline_chart(logs):
    if not logs:
        return None
    log_df = pd.DataFrame(logs[:15]).iloc[::-1].reset_index(drop=True)
    log_df["idx"] = range(len(log_df))
    theme = THEMES[st.session_state.theme]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_df["idx"], y=[1] * len(log_df), mode="markers+text",
        text=log_df["icon"], textposition="top center",
        marker=dict(size=18, color=theme["accent1"]),
        hovertext=log_df["action"] + " @ " + log_df["time"], hoverinfo="text",
    ))
    fig.add_shape(type="line", x0=0, x1=max(len(log_df) - 1, 0), y0=1, y1=1,
                  line=dict(color="rgba(255,255,255,0.2)", width=2))
    fig.update_yaxes(visible=False, range=[0.5, 1.5])
    fig.update_xaxes(visible=False)
    return style_fig(fig, 220)


def generate_text_report(df):
    types = detect_column_types(df)
    summary = get_dataset_summary(df)
    health = calculate_health_score(df)
    lines = [
        "=" * 60,
        "AI DATA SCIENCE COMMAND CENTER — DATASET REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Rows: {summary['rows']}",
        f"Columns: {summary['cols']}",
        f"Memory Usage: {summary['memory_mb']} MB",
        f"Missing Values: {summary['missing']}",
        f"Duplicate Rows: {summary['duplicates']}",
        f"Health Score: {health}/100",
        "",
        f"Numeric Columns: {', '.join(types['numeric']) or 'None'}",
        f"Categorical Columns: {', '.join(types['categorical']) or 'None'}",
        f"Datetime Columns: {', '.join(types['datetime']) or 'None'}",
        f"Constant Columns: {', '.join(types['constant']) or 'None'}",
        f"High-Cardinality Columns: {', '.join(types['high_cardinality']) or 'None'}",
        "",
        "AI Insights:",
    ]
    for ins in generate_ai_insights(df):
        lines.append(f"  - {ins['text']}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ==============================================================================
# MACHINE LEARNING — preprocessing, training, and result-rendering functions
# ==============================================================================
def detect_task_type(df, target):
    col = df[target]
    if col.dtype == object or str(col.dtype) == "category":
        return "Classification"
    if pd.api.types.is_bool_dtype(col):
        return "Classification"
    if pd.api.types.is_integer_dtype(col) and col.nunique() <= 15:
        return "Classification"
    return "Regression"


def preprocess_for_ml(df, target, feature_cols):
    """Impute numeric features, one-hot encode categoricals, return X, y."""
    data = df[feature_cols + [target]].copy()
    data = data.dropna(subset=[target])
    y = data[target]
    X = data[feature_cols]

    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in X.columns if c not in numeric_feats]

    if numeric_feats:
        X[numeric_feats] = SimpleImputer(strategy="mean").fit_transform(X[numeric_feats])
    if cat_feats:
        X[cat_feats] = X[cat_feats].astype(str).fillna("Missing")
        X = pd.get_dummies(X, columns=cat_feats, drop_first=True)

    X = X.astype(float)
    return X, y


def train_classification_model(X, y, algo_name, test_size, random_state):
    scaler_needed = algo_name in ["Logistic Regression", "SVM", "KNN"]
    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    if scaler_needed:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train.values, X_test.values

    le = None
    if y.dtype == object:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
    else:
        y_test_enc = y_test.values

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=random_state)
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(random_state=random_state, verbose=-1)

    model = models[algo_name]
    t0 = time.time()
    model.fit(X_train_s, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    preds = model.predict(X_test_s)
    inference_time = time.time() - t0

    proba = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None

    return {
        "model": model, "X_test": X_test, "y_test": y_test_enc, "preds": preds,
        "proba": proba, "label_encoder": le, "fit_time": fit_time,
        "inference_time": inference_time, "feature_names": X.columns.tolist(),
    }


def run_classification(df, target, features, algo, test_size, random_state):
    X, y = preprocess_for_ml(df, target, features)
    res = train_classification_model(X, y, algo, test_size, random_state)
    acc = accuracy_score(res["y_test"], res["preds"])
    res["accuracy"] = acc
    st.session_state.last_accuracy = round(acc * 100, 2)
    st.session_state.last_task_type = "Classification"
    st.session_state.last_inference_time = round(res["inference_time"] * 1000, 2)
    return res


def render_classification_results(res):
    theme = THEMES[st.session_state.theme]
    y_test, preds = res["y_test"], res["preds"]

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi_card_html("🎯", "Accuracy", f"{res['accuracy']*100:.2f}%"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card_html("⚡", "Inference Time", f"{res['inference_time']*1000:.2f} ms"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card_html("🧬", "Fit Time", f"{res['fit_time']*1000:.2f} ms"), unsafe_allow_html=True)

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Plasma",
                     labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(style_fig(fig), use_container_width=True)

    if res["proba"] is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, res["proba"][:, 1])
        roc_auc = auc(fpr, tpr)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.2f})", line=dict(color=theme["accent1"])))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray")))
        st.markdown("#### ROC Curve")
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    model = res["model"]
    st.markdown("#### Feature Importance")
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": res["feature_names"], "Importance": model.feature_importances_}).sort_values("Importance")
        fig3 = px.bar(imp.tail(15), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig3), use_container_width=True)
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0] if np.ndim(model.coef_) > 1 else model.coef_
        imp = pd.DataFrame({"Feature": res["feature_names"], "Coefficient": coefs}).sort_values("Coefficient")
        fig3 = px.bar(imp, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig3), use_container_width=True)
    else:
        st.info("Feature importance is not available for this algorithm.")

    st.markdown("#### Classification Report")
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    st.markdown("#### Sample Predictions")
    sample = res["X_test"].copy()
    sample["Actual"] = y_test
    sample["Predicted"] = preds
    st.dataframe(sample.head(10), use_container_width=True)


def run_regression(df, target, features, algo, test_size, random_state):
    X, y = preprocess_for_ml(df, target, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler_needed = algo in ["Linear Regression", "SVM", "KNN"]
    if scaler_needed:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train.values, X_test.values

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "SVM": SVR(),
        "KNN": KNeighborsRegressor(),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(random_state=random_state)
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(random_state=random_state, verbose=-1)

    model = models[algo]
    t0 = time.time(); model.fit(X_train_s, y_train); fit_time = time.time() - t0
    t0 = time.time(); preds = model.predict(X_test_s); inference_time = time.time() - t0

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    st.session_state.last_accuracy = round(r2 * 100, 2)
    st.session_state.last_task_type = "Regression"
    st.session_state.last_inference_time = round(inference_time * 1000, 2)

    return {
        "model": model, "X_test": X_test, "y_test": y_test, "preds": preds,
        "r2": r2, "mae": mae, "mse": mse, "rmse": rmse,
        "fit_time": fit_time, "inference_time": inference_time,
        "feature_names": X.columns.tolist(),
    }


def render_regression_results(res):
    theme = THEMES[st.session_state.theme]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card_html("📐", "R² Score", f"{res['r2']:.3f}"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card_html("📏", "MAE", f"{res['mae']:.2f}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card_html("📊", "RMSE", f"{res['rmse']:.2f}"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card_html("⚡", "Inference", f"{res['inference_time']*1000:.2f} ms"), unsafe_allow_html=True)

    st.markdown("#### Predicted vs Actual")
    plot_df = pd.DataFrame({"Actual": res["y_test"], "Predicted": res["preds"]})
    fig = px.scatter(plot_df, x="Actual", y="Predicted", opacity=st.session_state.opacity,
                      color_discrete_sequence=[theme["accent1"]])
    fig.add_trace(go.Scatter(x=plot_df["Actual"], y=plot_df["Actual"], mode="lines",
                              name="Ideal Fit", line=dict(color=theme["accent3"], dash="dash")))
    fig.update_traces(marker=dict(size=st.session_state.point_size), selector=dict(mode="markers"))
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("#### Residuals")
    residuals = plot_df["Actual"] - plot_df["Predicted"]
    fig2 = px.histogram(residuals, nbins=st.session_state.num_bins, color_discrete_sequence=[theme["accent2"]])
    fig2.update_layout(showlegend=False, xaxis_title="Residual")
    st.plotly_chart(style_fig(fig2), use_container_width=True)

    model = res["model"]
    st.markdown("#### Feature Importance")
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": res["feature_names"], "Importance": model.feature_importances_}).sort_values("Importance")
        fig3 = px.bar(imp.tail(15), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig3), use_container_width=True)
    elif hasattr(model, "coef_"):
        imp = pd.DataFrame({"Feature": res["feature_names"], "Coefficient": model.coef_}).sort_values("Coefficient")
        fig3 = px.bar(imp, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig3), use_container_width=True)
    else:
        st.info("Feature importance is not available for this algorithm.")


def run_clustering(df, features, k):
    X = df[features].copy()
    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in X.columns if c not in numeric_feats]
    if numeric_feats:
        X[numeric_feats] = SimpleImputer(strategy="mean").fit_transform(X[numeric_feats])
    if cat_feats:
        X[cat_feats] = X[cat_feats].astype(str).fillna("Missing")
        X = pd.get_dummies(X, columns=cat_feats, drop_first=True)
    X = X.astype(float)

    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
    coords = PCA(n_components=2).fit_transform(X_scaled)

    st.session_state.last_task_type = "Clustering"
    st.session_state.last_accuracy = round(sil * 100, 2)
    return {"labels": labels, "coords": coords, "silhouette": sil, "k": k, "X": X}


def render_clustering_results(res):
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi_card_html("🧩", "Clusters (k)", str(res["k"])), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card_html("📐", "Silhouette Score", f"{res['silhouette']:.3f}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card_html("🔢", "Points", str(len(res["labels"]))), unsafe_allow_html=True)

    plot_df = pd.DataFrame(res["coords"], columns=["PC1", "PC2"])
    plot_df["Cluster"] = res["labels"].astype(str)
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", opacity=st.session_state.opacity,
                      color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_traces(marker=dict(size=st.session_state.point_size))
    st.markdown("#### Cluster Visualization (PCA 2D Projection)")
    st.plotly_chart(style_fig(fig, 500), use_container_width=True)

    sizes = pd.Series(res["labels"]).value_counts().sort_index().reset_index()
    sizes.columns = ["Cluster", "Count"]
    fig2 = px.bar(sizes, x="Cluster", y="Count", color="Count", color_continuous_scale="Plasma")
    st.plotly_chart(style_fig(fig2), use_container_width=True)

    result_df = res["X"].copy()
    result_df["Cluster"] = res["labels"]
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Clustered Data", csv, "clustered_dataset.csv", "text/csv")


# ==============================================================================
# VISUAL ANALYTICS TAB RENDERERS
# ==============================================================================
def render_overview_tab(df, types):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)
    with col2:
        st.markdown("#### Column Type Breakdown")
        type_counts = pd.DataFrame({
            "Type": ["Numeric", "Categorical", "Datetime"],
            "Count": [len(types["numeric"]), len(types["categorical"]), len(types["datetime"])],
        })
        fig = px.pie(type_counts, names="Type", values="Count", hole=0.5,
                     color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)


def render_missing_tab(df, types):
    if not types["missing"]:
        st.success("✅ No missing values found in this dataset!")
        return
    miss_df = pd.DataFrame(list(types["missing"].items()), columns=["Column", "Missing"])
    miss_df["Percent"] = (miss_df["Missing"] / len(df) * 100).round(2)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(miss_df.sort_values("Missing"), x="Missing", y="Column", orientation="h",
                     color="Percent", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        st.markdown("#### Missingness Matrix")
        miss_matrix = df[list(types["missing"].keys())].isna().astype(int)
        fig2 = px.imshow(miss_matrix.T, aspect="auto", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    st.dataframe(miss_df, use_container_width=True)


def render_correlation_tab(df, types, palette):
    if len(types["numeric"]) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    corr = df[types["numeric"]].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=palette, aspect="auto")
    st.plotly_chart(style_fig(fig, 500), use_container_width=True)

    st.markdown("#### Scatter Explorer")
    c1, c2 = st.columns(2)
    with c1: x = st.selectbox("X axis", types["numeric"], key="corr_x")
    with c2: y = st.selectbox("Y axis", types["numeric"], index=min(1, len(types["numeric"]) - 1), key="corr_y")
    fig2 = px.scatter(df, x=x, y=y, opacity=st.session_state.opacity,
                       color_discrete_sequence=[THEMES[st.session_state.theme]["accent1"]])
    fig2.update_traces(marker=dict(size=st.session_state.point_size))
    st.plotly_chart(style_fig(fig2), use_container_width=True)


def render_distribution_tab(df, types, palette):
    if not types["numeric"]:
        st.warning("No numeric columns available.")
        return
    theme = THEMES[st.session_state.theme]
    col = st.selectbox("Select numeric column", types["numeric"], key="dist_col")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x=col, nbins=st.session_state.num_bins, color_discrete_sequence=[theme["accent1"]])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig2 = px.violin(df, y=col, box=True, points="outliers", color_discrete_sequence=[theme["accent2"]])
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.box(df, y=col, color_discrete_sequence=[theme["accent3"]])
        st.plotly_chart(style_fig(fig3), use_container_width=True)
    with c4:
        try:
            data = df[col].dropna()
            fig4 = ff.create_distplot([data], [col], show_hist=False)
            st.plotly_chart(style_fig(fig4), use_container_width=True)
        except Exception:
            st.info("Density plot unavailable for this column.")


def render_outliers_tab(df, types):
    if not types["numeric"]:
        st.warning("No numeric columns available.")
        return
    rows = []
    for col in types["numeric"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(df[(df[col] < lower) | (df[col] > upper)][col].count())
        rows.append({"Column": col, "Outliers": outliers, "Lower Bound": round(lower, 2), "Upper Bound": round(upper, 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    col = st.selectbox("Inspect column", types["numeric"], key="out_col")
    fig = px.box(df, y=col, points="all", color_discrete_sequence=[THEMES[st.session_state.theme]["accent1"]])
    st.plotly_chart(style_fig(fig), use_container_width=True)


def render_relationships_tab(df, types, palette):
    chart_type = st.selectbox("Chart Type", [
        "Scatter / Bubble", "Line", "Parallel Coordinates", "Radar", "Pair Plot (matrix)", "Density Heatmap",
    ], key="rel_chart")
    numeric = types["numeric"]
    theme = THEMES[st.session_state.theme]

    if chart_type == "Scatter / Bubble":
        if len(numeric) < 2:
            st.warning("Need 2+ numeric columns."); return
        c1, c2, c3 = st.columns(3)
        with c1: x = st.selectbox("X", numeric, key="rel_x")
        with c2: y = st.selectbox("Y", numeric, index=min(1, len(numeric) - 1), key="rel_y")
        with c3: size = st.selectbox("Size (bubble)", ["None"] + numeric, key="rel_size")
        color_col = st.selectbox("Color by", ["None"] + types["categorical"] + numeric, key="rel_color")
        fig = px.scatter(
            df, x=x, y=y,
            size=None if size == "None" else size,
            color=None if color_col == "None" else color_col,
            opacity=st.session_state.opacity,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        if size == "None":
            fig.update_traces(marker=dict(size=st.session_state.point_size))
        st.plotly_chart(style_fig(fig, 500), use_container_width=True)

    elif chart_type == "Line":
        work_df = df
        if types["datetime"]:
            xcol = types["datetime"][0]
        else:
            work_df = df.reset_index()
            xcol = "index"
        if not numeric:
            st.warning("Need at least one numeric column."); return
        ycol = st.selectbox("Y value", numeric, key="rel_line_y")
        plot_df = work_df.sort_values(xcol)
        fig = px.line(plot_df, x=xcol, y=ycol, color_discrete_sequence=[theme["accent1"]])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    elif chart_type == "Parallel Coordinates":
        if len(numeric) < 3:
            st.warning("Need 3+ numeric columns."); return
        cols = st.multiselect("Columns", numeric, default=numeric[:5], key="rel_parallel")
        if cols:
            fig = px.parallel_coordinates(df, dimensions=cols, color=cols[0], color_continuous_scale=palette)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    elif chart_type == "Radar":
        if types["categorical"] and numeric:
            cat_col = st.selectbox("Group by (categorical)", types["categorical"], key="rel_radar_cat")
            agg = df.groupby(cat_col)[numeric].mean(numeric_only=True).reset_index()
            fig = go.Figure()
            for _, row in agg.head(6).iterrows():
                fig.add_trace(go.Scatterpolar(r=row[numeric].values, theta=numeric, fill="toself", name=str(row[cat_col])))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)
        else:
            st.warning("Need at least one categorical and one numeric column.")

    elif chart_type == "Pair Plot (matrix)":
        cols = st.multiselect("Columns (max 5)", numeric, default=numeric[:4], key="rel_pair")
        if len(cols) >= 2:
            fig = px.scatter_matrix(df, dimensions=cols[:5], color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(style_fig(fig, 600), use_container_width=True)

    elif chart_type == "Density Heatmap":
        if len(numeric) < 2:
            st.warning("Need 2+ numeric columns."); return
        c1, c2 = st.columns(2)
        with c1: x = st.selectbox("X", numeric, key="rel_hex_x")
        with c2: y = st.selectbox("Y", numeric, index=min(1, len(numeric) - 1), key="rel_hex_y")
        fig = px.density_heatmap(df, x=x, y=y, color_continuous_scale=palette,
                                  nbinsx=st.session_state.num_bins, nbinsy=st.session_state.num_bins)
        st.plotly_chart(style_fig(fig), use_container_width=True)


def render_categorical_tab(df, types, palette):
    if not types["categorical"]:
        st.warning("No categorical columns available.")
        return
    col = st.selectbox("Select categorical column", types["categorical"], key="cat_col")
    vc = df[col].value_counts().head(15).reset_index()
    vc.columns = [col, "Count"]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(vc, names=col, values="Count", hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig2 = px.bar(vc, x=col, y="Count", color="Count", color_continuous_scale=palette)
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    other_cats = [c for c in types["categorical"] if c != col]
    if other_cats:
        col2 = st.selectbox("Second category (sunburst / treemap)", other_cats, key="cat_col2")
        path_df = df[[col, col2]].dropna()
        fig3 = px.sunburst(path_df, path=[col, col2], color_discrete_sequence=px.colors.qualitative.Prism)
        fig4 = px.treemap(path_df, path=[col, col2], color_discrete_sequence=px.colors.qualitative.Prism)
        c5, c6 = st.columns(2)
        with c5: st.plotly_chart(style_fig(fig3), use_container_width=True)
        with c6: st.plotly_chart(style_fig(fig4), use_container_width=True)


# ==============================================================================
# PAGE: DASHBOARD
# ==============================================================================
def page_dashboard():
    df = st.session_state.df
    types = detect_column_types(df)
    summary = get_dataset_summary(df)

    st.markdown('<h1 class="hero-title">AI DATA SCIENCE COMMAND CENTER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Built entirely in one Streamlit file.</p>', unsafe_allow_html=True)

    now = datetime.now()
    elapsed = int(time.time() - st.session_state.start_time)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card_html("🕐", "Current Time", now.strftime("%H:%M:%S")), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card_html("📅", "Current Date", now.strftime("%b %d, %Y")), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card_html("⏱️", "Session Timer", f"{elapsed//60}m {elapsed%60}s"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card_html("🔄", "Interactions", str(st.session_state.interactions)), unsafe_allow_html=True)

    st.markdown(f'<div class="glass-card quote-card">💡 <i>"{random_quote()}"</i></div>', unsafe_allow_html=True)

    st.markdown("### 🛰️ Live Dataset Metrics")
    d1, d2, d3, d4 = st.columns(4)
    with d1: st.markdown(kpi_card_html("📏", "Rows", f"{summary['rows']:,}"), unsafe_allow_html=True)
    with d2: st.markdown(kpi_card_html("📊", "Columns", str(summary['cols'])), unsafe_allow_html=True)
    with d3: st.markdown(kpi_card_html("🕳️", "Missing Values", str(summary['missing'])), unsafe_allow_html=True)
    with d4: st.markdown(kpi_card_html("💾", "Memory Usage", f"{summary['memory_mb']} MB"), unsafe_allow_html=True)

    st.markdown("### 🧠 Model & Performance Metrics")
    e1, e2, e3, e4 = st.columns(4)
    acc_display = f"{st.session_state.last_accuracy}%" if st.session_state.last_accuracy is not None else "—"
    with e1: st.markdown(kpi_card_html("🎯", f"Last {st.session_state.last_task_type or 'Model'} Score", acc_display), unsafe_allow_html=True)
    upload_display = f"{st.session_state.upload_time}s" if st.session_state.upload_time else "—"
    with e2: st.markdown(kpi_card_html("⬆️", "Upload Time", upload_display), unsafe_allow_html=True)
    inf_display = f"{st.session_state.last_inference_time} ms" if st.session_state.last_inference_time else "—"
    with e3: st.markdown(kpi_card_html("⚡", "Inference Time", inf_display), unsafe_allow_html=True)
    speed = round(summary["rows"] / max(elapsed, 1), 1)
    with e4: st.markdown(kpi_card_html("🚀", "Processing Speed", f"{speed} rows/s"), unsafe_allow_html=True)

    st.markdown("### 🩺 Dataset Health Score")
    health = calculate_health_score(df)
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(build_gauge_chart(health), use_container_width=True)
    with g2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Health Breakdown**")
        st.write(f"- Missing values: {summary['missing']}")
        st.write(f"- Duplicate rows: {summary['duplicates']}")
        st.write(f"- Constant columns: {len(types['constant'])}")
        st.write(f"- High-cardinality columns: {len(types['high_cardinality'])}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### ✨ Quick Actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("🎉 Celebrate"):
            st.balloons()
            add_log("Celebration button pressed", "🎉")
    with q2:
        if st.button("🔮 Random Fact"):
            st.info(random_fact())
    with q3:
        if st.button("❄️ Snow Effect"):
            st.snow()


# ==============================================================================
# PAGE: UPLOAD DATASET
# ==============================================================================
def page_upload():
    st.markdown("## 📂 Upload Dataset")
    st.markdown("Upload your own CSV, or explore the auto-generated synthetic dataset already loaded.")

    colA, colB = st.columns([3, 1])
    with colA:
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    with colB:
        st.write("")
        st.write("")
        if st.button("🔄 Reset to Synthetic Data"):
            st.session_state.df = generate_synthetic_dataset()
            st.session_state.dataset_source = None
            st.session_state.engineered_df = None
            add_log("Reset to synthetic dataset", "🔄")
            st.toast("Synthetic dataset restored!", icon="✅")
            st.rerun()

    if uploaded is not None:
        t0 = time.time()
        try:
            new_df = pd.read_csv(uploaded)
            st.session_state.df = new_df
            st.session_state.dataset_source = uploaded.name
            st.session_state.upload_time = round(time.time() - t0, 3)
            st.session_state.engineered_df = None
            add_log(f"Uploaded dataset: {uploaded.name}", "📂")
            st.toast(f"Loaded {uploaded.name}", icon="✅")
            st.balloons()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    df = st.session_state.df
    types = detect_column_types(df)
    summary = get_dataset_summary(df)

    st.markdown(f"**Current source:** `{st.session_state.dataset_source or 'synthetic (auto-generated)'}`")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card_html("📏", "Rows", f"{summary['rows']:,}"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card_html("📊", "Columns", str(summary['cols'])), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card_html("💾", "Memory", f"{summary['memory_mb']} MB"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card_html("🪞", "Duplicates", str(summary['duplicates'])), unsafe_allow_html=True)

    st.markdown("#### Dataset Health")
    total_cells = df.shape[0] * df.shape[1] if df.shape[0] and df.shape[1] else 1
    completeness = 100 - (summary["missing"] / total_cells * 100)
    uniqueness = 100 - (summary["duplicates"] / max(len(df), 1) * 100)
    cc1, cc2 = st.columns(2)
    with cc1: st.markdown(circular_progress_html("Data Completeness", completeness), unsafe_allow_html=True)
    with cc2: st.markdown(circular_progress_html("Row Uniqueness", uniqueness, "var(--accent2)"), unsafe_allow_html=True)

    tabs = st.tabs(["👀 Preview", "🎲 Random Sample", "📈 Statistics", "🧬 Feature Types", "⚠️ Data Quality"])
    with tabs[0]:
        st.dataframe(df.head(20), use_container_width=True)
    with tabs[1]:
        st.dataframe(df.sample(min(20, len(df))), use_container_width=True)
    with tabs[2]:
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    with tabs[3]:
        st.write("**Numeric:**", types["numeric"] or "None")
        st.write("**Categorical:**", types["categorical"] or "None")
        st.write("**Datetime:**", types["datetime"] or "None")
    with tabs[4]:
        st.write("**Constant columns:**", types["constant"] or "None")
        st.write("**High-cardinality columns:**", types["high_cardinality"] or "None")
        st.write("**Duplicate rows:**", types["duplicates"])
        if types["missing"]:
            miss_df = pd.DataFrame(list(types["missing"].items()), columns=["Column", "Missing Count"])
            st.dataframe(miss_df, use_container_width=True)
        else:
            st.success("No missing values detected!")

    st.markdown("#### Export")
    report = generate_text_report(df)
    st.download_button("📄 Export Text Report", report, "dataset_report.txt", "text/plain")


# ==============================================================================
# PAGE: VISUAL ANALYTICS
# ==============================================================================
def page_visual_analytics():
    df = st.session_state.df
    types = detect_column_types(df)
    st.markdown("## 📊 Visual Analytics")

    with st.expander("🎛️ Chart Customization"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            palette = st.selectbox("Color Palette", ["Plasma", "Viridis", "Turbo", "Sunset", "Magma", "Inferno", "Rainbow", "Blues"], key="va_palette")
        with c2:
            st.session_state.point_size = st.slider("Point Size", 3, 20, st.session_state.point_size)
        with c3:
            st.session_state.opacity = st.slider("Opacity", 0.1, 1.0, st.session_state.opacity)
        with c4:
            st.session_state.num_bins = st.slider("Bins", 5, 100, st.session_state.num_bins)

    tabs = st.tabs(["🗺️ Overview", "🕳️ Missing Values", "🔗 Correlation", "📉 Distribution", "🎯 Outliers", "🫧 Relationships", "🥧 Categorical"])
    with tabs[0]: render_overview_tab(df, types)
    with tabs[1]: render_missing_tab(df, types)
    with tabs[2]: render_correlation_tab(df, types, palette)
    with tabs[3]: render_distribution_tab(df, types, palette)
    with tabs[4]: render_outliers_tab(df, types)
    with tabs[5]: render_relationships_tab(df, types, palette)
    with tabs[6]: render_categorical_tab(df, types, palette)


# ==============================================================================
# PAGE: MACHINE LEARNING
# ==============================================================================
def page_machine_learning():
    df = st.session_state.df
    types = detect_column_types(df)
    st.markdown("## 🤖 Machine Learning")

    all_cols = df.columns.tolist()
    default_target = "Churn" if "Churn" in all_cols else all_cols[-1]
    target = st.selectbox("🎯 Select Target Variable", all_cols, index=all_cols.index(default_target))

    task_choice = st.radio("Task Type", ["Auto-Detect", "Classification", "Regression", "Cluster Analysis"], horizontal=True)
    if task_choice == "Auto-Detect":
        task_type = detect_task_type(df, target)
        st.info(f"🔍 Auto-detected task type: **{task_type}**")
    else:
        task_type = task_choice

    feature_candidates = [c for c in all_cols if c != target and c not in types["datetime"]]
    features = st.multiselect("Feature Columns", feature_candidates, default=feature_candidates[:min(8, len(feature_candidates))])

    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with c2:
        random_state = st.number_input("Random State", 0, 999, 42)
    with c3:
        if task_type == "Classification":
            algo_options = ["Logistic Regression", "Random Forest", "Decision Tree", "SVM", "KNN"]
        elif task_type == "Regression":
            algo_options = ["Linear Regression", "Random Forest", "Decision Tree", "SVM", "KNN"]
        else:
            algo_options = ["KMeans"]
        if XGBOOST_AVAILABLE and task_type in ["Classification", "Regression"]:
            algo_options.append("XGBoost")
        if LIGHTGBM_AVAILABLE and task_type in ["Classification", "Regression"]:
            algo_options.append("LightGBM")
        algo = st.selectbox("Algorithm", algo_options)

    k = 3
    if task_type == "Cluster Analysis":
        k = st.slider("Number of Clusters (k)", 2, 10, 3)

    if st.button("🚀 Train Model", type="primary"):
        if not features:
            st.error("Select at least one feature column.")
        else:
            progress = st.progress(0, text="Preparing data...")
            for pct in (20, 45, 70):
                time.sleep(0.05 * st.session_state.animation_speed)
                progress.progress(pct, text="Training model...")
            with st.spinner("Finalizing results..."):
                if task_type == "Classification":
                    result = run_classification(df, target, features, algo, test_size, random_state)
                    progress.progress(100, text="Done!")
                    render_classification_results(result)
                elif task_type == "Regression":
                    result = run_regression(df, target, features, algo, test_size, random_state)
                    progress.progress(100, text="Done!")
                    render_regression_results(result)
                else:
                    result = run_clustering(df, features, k)
                    progress.progress(100, text="Done!")
                    render_clustering_results(result)
            add_log(f"Trained {algo} ({task_type})", "🤖")
            st.toast("Model trained successfully!", icon="✅")


# ==============================================================================
# PAGE: FEATURE ENGINEERING
# ==============================================================================
def page_feature_engineering():
    base_df = st.session_state.df
    types = detect_column_types(base_df)
    st.markdown("## 📈 Feature Engineering")

    working = st.session_state.engineered_df if st.session_state.engineered_df is not None else base_df.copy()

    st.markdown("#### Numeric Transformations")
    if types["numeric"]:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: col = st.selectbox("Column", types["numeric"], key="fe_col")
        with c2: transform = st.selectbox("Transform", ["Log Transform", "Standard Scale", "Min-Max Scale", "Binning"], key="fe_transform")
        with c3:
            st.write("")
            st.write("")
            apply_transform = st.button("Apply")
        if apply_transform:
            suffix = transform.split()[0].lower()
            new_col = f"{col}_{suffix}"
            filled = working[col].fillna(working[col].mean())
            if transform == "Log Transform":
                working[new_col] = np.log1p(filled.clip(lower=0))
            elif transform == "Standard Scale":
                working[new_col] = StandardScaler().fit_transform(filled.values.reshape(-1, 1))
            elif transform == "Min-Max Scale":
                working[new_col] = MinMaxScaler().fit_transform(filled.values.reshape(-1, 1))
            elif transform == "Binning":
                n_bins = max(3, st.session_state.num_bins // 5)
                working[new_col] = pd.cut(working[col], bins=n_bins).astype(str)
            st.session_state.engineered_df = working
            add_log(f"Applied {transform} to {col}", "🧪")
            st.success(f"Created new column '{new_col}'")

    st.markdown("#### Categorical Encoding")
    if types["categorical"]:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: col2 = st.selectbox("Column", types["categorical"], key="fe_cat_col")
        with c2: method = st.selectbox("Method", ["One-Hot Encode", "Label Encode"], key="fe_cat_method")
        with c3:
            st.write("")
            st.write("")
            apply_encode = st.button("Encode")
        if apply_encode:
            if method == "One-Hot Encode":
                working = pd.get_dummies(working, columns=[col2], prefix=col2)
            else:
                working[f"{col2}_encoded"] = LabelEncoder().fit_transform(working[col2].astype(str))
            st.session_state.engineered_df = working
            add_log(f"Applied {method} to {col2}", "🧪")
            st.success("Encoding applied!")

    st.markdown("#### Data Cleaning")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🧹 Drop Duplicate Rows"):
            working = working.drop_duplicates()
            st.session_state.engineered_df = working
            add_log("Dropped duplicate rows", "🧹")
            st.success(f"Removed duplicates. New shape: {working.shape}")
    with c2:
        fill_method = st.selectbox("Fill Missing With", ["Mean", "Median", "Mode", "Drop Rows"], key="fe_fill")
    with c3:
        if st.button("🩹 Fill/Drop Missing"):
            if fill_method == "Drop Rows":
                working = working.dropna()
            else:
                for c in working.select_dtypes(include=[np.number]).columns:
                    if fill_method == "Mean":
                        working[c] = working[c].fillna(working[c].mean())
                    elif fill_method == "Median":
                        working[c] = working[c].fillna(working[c].median())
                    elif fill_method == "Mode" and not working[c].mode().empty:
                        working[c] = working[c].fillna(working[c].mode().iloc[0])
            st.session_state.engineered_df = working
            add_log(f"Missing values handled: {fill_method}", "🩹")
            st.success("Missing values handled!")

    st.markdown("#### Preview & Export")
    st.dataframe(working.head(15), use_container_width=True)
    csv = working.to_csv(index=False).encode("utf-8")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ Download Engineered Dataset", csv, "engineered_dataset.csv", "text/csv")
    with d2:
        if st.button("✅ Use as Active Dataset"):
            st.session_state.df = working
            st.session_state.engineered_df = None
            add_log("Engineered dataset set as active", "✅")
            st.success("Active dataset updated!")
            st.rerun()


# ==============================================================================
# PAGE: AI DATA CHAT  (real Gemini / Groq models, replaces AI Insights + Smart Assistant)
# ==============================================================================
def handle_ai_chat_query(question, df):
    """Send the question + dataset context to the active AI provider and render the reply."""
    st.session_state.chat_history.append({"role": "user", "content": question, "kind": "text"})

    master_prompt = build_master_prompt(df, question)
    try:
        raw_response, provider_used = get_ai_response(master_prompt)
    except Exception as e:
        error_msg = f"❌ AI request failed: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "kind": "text"})
        add_log("AI chat request failed", "⚠️")
        st.rerun()
        return

    st.session_state.last_ai_provider_used = provider_used
    mode, content = parse_ai_response(raw_response)
    st.session_state.chat_history.append({
        "role": "assistant", "content": content, "kind": mode.lower(), "provider": provider_used,
    })
    add_log(f"AI chat via {provider_used}: {question[:40]}", "🧠")
    st.rerun()


def page_ai_chatbot():
    df = st.session_state.df

    fallback_label = "auto-fallback on" if st.session_state.ai_auto_fallback else "auto-fallback off"
    st.markdown(f"""
    <div class="ai-chat-header">
        <div>
            <div class="ai-chat-title">🧠 AI Data Chat</div>
            <div class="ai-chat-subtitle">Ask anything about your dataset — get answers, tables, or on-the-fly charts.</div>
        </div>
        <div class="provider-pill"><span class="dot"></span>{st.session_state.ai_provider} · {fallback_label}</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.gemini_api_key and not st.session_state.groq_api_key:
        st.warning("No AI provider is configured on the server yet. Contact an admin to enable AI Data Chat.")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg.get("kind") == "code":
                    execute_generated_code(msg["content"], df)
                else:
                    st.markdown(msg["content"])
                if msg.get("provider"):
                    st.markdown(f'<div class="chat-provider-caption">via {msg["provider"]}</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown('<div class="chat-quick-label">Try asking</div>', unsafe_allow_html=True)
        quick_qs = [
            ("💡", "Key takeaways from this dataset?"),
            ("🔗", "Plot a correlation heatmap"),
            ("🕳️", "Which column has the most missing values?"),
            ("📊", "Show the distribution of a key numeric column"),
        ]
        quick_cols = st.columns(4)
        for i, (icon, q) in enumerate(quick_qs):
            if quick_cols[i].button(f"{icon}  {q}", key=f"aichat_quick_{i}", use_container_width=True):
                with st.spinner(f"Thinking with {st.session_state.ai_provider}..."):
                    handle_ai_chat_query(q, df)

    prompt = st.chat_input("Ask me anything about your dataset...")
    if prompt:
        with st.spinner(f"Thinking with {st.session_state.ai_provider}..."):
            handle_ai_chat_query(prompt, df)


# ==============================================================================
# PAGE: SETTINGS
# ==============================================================================
def page_settings():
    st.markdown("## ⚙ Settings")

    theme = st.selectbox("🎨 Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        add_log(f"Theme changed to {theme}", "🎨")
        st.rerun()

    st.markdown("#### Chart Defaults")
    c1, c2 = st.columns(2)
    with c1: st.session_state.point_size = st.slider("Default Point Size", 3, 20, st.session_state.point_size)
    with c2: st.session_state.opacity = st.slider("Default Opacity", 0.1, 1.0, st.session_state.opacity)
    c3, c4 = st.columns(2)
    with c3: st.session_state.num_bins = st.slider("Default Bins", 5, 100, st.session_state.num_bins)
    with c4: st.session_state.animation_speed = st.slider("Animation Speed", 0.5, 3.0, st.session_state.animation_speed)

    st.markdown("#### 🤖 AI Data Chat — Provider")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    provider = st.radio(
        "Default provider", ["Gemini", "Groq"],
        index=["Gemini", "Groq"].index(st.session_state.ai_provider),
        horizontal=True,
        help="Which model answers first on the 🧠 AI Data Chat page.",
    )
    if provider != st.session_state.ai_provider:
        st.session_state.ai_provider = provider
        add_log(f"AI provider switched to {provider}", "🤖")

    st.session_state.ai_auto_fallback = st.toggle(
        "Auto-fallback to the other provider if the primary one fails",
        value=st.session_state.ai_auto_fallback,
    )

    sdk_status = []
    sdk_status.append("✅ google-generativeai installed" if GEMINI_SDK_AVAILABLE else "❌ google-generativeai not installed (`pip install google-generativeai`)")
    sdk_status.append("✅ groq installed" if GROQ_SDK_AVAILABLE else "❌ groq not installed (`pip install groq`)")
    st.caption(" · ".join(sdk_status))

    key_status = []
    key_status.append("✅ Gemini key configured" if st.session_state.gemini_api_key else "⚠️ Gemini key not configured")
    key_status.append("✅ Groq key configured" if st.session_state.groq_api_key else "⚠️ Groq key not configured")
    st.caption(" · ".join(key_status) + " — keys are set server-side via environment variables, not editable here.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Session")
    if st.button("🗑️ Reset Entire Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ==============================================================================
# PAGE: LOGS
# ==============================================================================
def page_logs():
    st.markdown("## 📜 Logs & Recent Activity")
    if not st.session_state.logs:
        st.info("No activity yet — interact with the app to see logs here.")
        return

    timeline_fig = render_timeline_chart(st.session_state.logs)
    if timeline_fig is not None:
        st.markdown("#### Activity Timeline")
        st.plotly_chart(timeline_fig, use_container_width=True)

    st.markdown("#### Detailed Log")
    for log in st.session_state.logs[:25]:
        st.markdown(f'<div class="log-entry">{log["icon"]} <b>{log["time"]}</b> — {log["action"]}</div>', unsafe_allow_html=True)

    if st.button("🧹 Clear Logs"):
        st.session_state.logs = []
        st.rerun()


# ==============================================================================
# PAGE: ABOUT
# ==============================================================================
def page_about():
    st.markdown("## 👤 About")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        "**AI Data Science Command Center** is a single-file Streamlit application "
        "combining exploratory data analysis, visualization, machine learning, and a "
        "rule-based AI assistant into one futuristic dashboard."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### ⌨️ Keyboard Shortcuts")
    st.markdown("""
- `R` — Rerun the app (Streamlit built-in shortcut)
- `C` — Open the Streamlit menu to clear cache
- Use the sidebar to jump between sections at any time
    """)

    st.markdown("#### 🔔 Notification Center")
    if st.session_state.logs:
        for log in st.session_state.logs[:5]:
            st.markdown(f"{log['icon']} {log['action']} — *{log['time']}*")
    else:
        st.caption("No notifications yet.")

    st.markdown("#### 📄 Export")
    report = generate_text_report(st.session_state.df)
    st.download_button("📄 Export Full Report", report, "dataset_report.txt", "text/plain")


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🛸 COMMAND CENTER</div>', unsafe_allow_html=True)

        pages = [
            "🏠 Dashboard", "📂 Upload Dataset", "📊 Visual Analytics",
            "📈 Feature Engineering", "🤖 Machine Learning", "🧠 AI Data Chat",
            "⚙ Settings", "📜 Logs", "👤 About",
        ]
        current = st.session_state.page if st.session_state.page in pages else pages[0]
        selected = st.radio("Navigate", pages, index=pages.index(current), label_visibility="collapsed")
        st.session_state.page = selected

        st.markdown("---")
        df = st.session_state.df
        if df is not None:
            st.markdown(f"**Active dataset:** `{st.session_state.dataset_source or 'synthetic'}`")
            st.markdown(f"**Shape:** {df.shape[0]:,} × {df.shape[1]}")
        st.markdown("---")
        st.caption(f"Session interactions: {st.session_state.interactions}")
        st.caption(f"Theme: {st.session_state.theme}")


# ==============================================================================
# FOOTER
# ==============================================================================
def render_footer():
    st.markdown("---")
    exec_time = round(time.time() - st.session_state.start_time, 2)
    df = st.session_state.df
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3) if df is not None else 0
    st.markdown(f"""
    <div class="footer">
        <span>Made with ❤️ using Streamlit</span> ·
        <span>Version 1.0.0</span> ·
        <span>Python {platform.python_version()}</span> ·
        <span>Session runtime: {exec_time}s</span> ·
        <span>Dataset memory: {mem_mb} MB</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="fab" title="AI Command Center">🛸</div>', unsafe_allow_html=True)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    init_session_state()

    if st.session_state.df is None:
        st.session_state.df = generate_synthetic_dataset()
        st.session_state.dataset_source = None

    inject_css(st.session_state.theme)
    render_sidebar()

    page = st.session_state.page
    page_map = {
        "🏠 Dashboard": page_dashboard,
        "📂 Upload Dataset": page_upload,
        "📊 Visual Analytics": page_visual_analytics,
        "📈 Feature Engineering": page_feature_engineering,
        "🤖 Machine Learning": page_machine_learning,
        "🧠 AI Data Chat": page_ai_chatbot,
        "⚙ Settings": page_settings,
        "📜 Logs": page_logs,
        "👤 About": page_about,
    }
    page_map.get(page, page_dashboard)()

    render_footer()


if __name__ == "__main__":
    main()
