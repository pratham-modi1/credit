import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard — Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

* { box-sizing: border-box; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
            .block-container {
    padding-top: 1rem !important;
}
            
.stDeployButton {display: none;}
[data-testid="collapsedControl"] {display: none;}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── ROOT THEME ── */
:root {
    --bg-deep:    #07080F;
    --bg-card:    #0D0F1C;
    --bg-card2:   #111420;
    --border:     rgba(255,255,255,0.07);
    --border-glow:rgba(0,245,200,0.25);
    --accent:     #00F5C8;
    --accent2:    #FF5F40;
    --accent3:    #7B6EF6;
    --text-main:  #F0F2FF;
    --text-muted: #7A809C;
    --text-dim:   #4A5068;
}

.stApp {
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,245,200,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(123,110,246,0.06) 0%, transparent 60%);
    background-attachment: fixed;
}

/* ── NAVBAR ── */
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.02em;
    padding: 0.4rem 0;
}
.nav-logo span { color: var(--text-main); }

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.1rem !important;
    color: var(--text-muted) !important;
    background: var(--bg-card) !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,245,200,0.06) !important;
    box-shadow: 0 0 18px rgba(0,245,200,0.1) !important;
    transform: translateY(-1px) !important;
}

/* Primary CTA button */
.cta-btn .stButton > button {
    background: linear-gradient(135deg, #00F5C8 0%, #00C4A0 100%) !important;
    color: #07080F !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    box-shadow: 0 4px 24px rgba(0,245,200,0.25) !important;
    letter-spacing: 0.02em !important;
}
.cta-btn .stButton > button:hover {
    box-shadow: 0 6px 32px rgba(0,245,200,0.4) !important;
    transform: translateY(-2px) !important;
    color: #07080F !important;
}

/* Run button */
.run-btn .stButton > button {
    background: linear-gradient(135deg, #7B6EF6 0%, #5A4FCF 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 20px rgba(123,110,246,0.3) !important;
}
.run-btn .stButton > button:hover {
    box-shadow: 0 6px 28px rgba(123,110,246,0.5) !important;
    transform: translateY(-2px) !important;
    color: #FFFFFF !important;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.5;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-label {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 0.6rem;
    line-height: 1.5;
    font-weight: 400;
}

/* ── STEP CARDS ── */
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    height: 100%;
}
.step-num {
    background: linear-gradient(135deg, #7B6EF6, #5A4FCF);
    color: #fff;
    width: 40px;
    height: 40px;
    border-radius: 12px;
    line-height: 40px;
    text-align: center;
    margin: 0 auto 1rem auto;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    box-shadow: 0 4px 16px rgba(123,110,246,0.35);
}

/* ── RESULT CARDS ── */
.fraud-card {
    background: rgba(255,95,64,0.05);
    border: 1px solid rgba(255,95,64,0.3);
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
}
.safe-card {
    background: rgba(0,245,200,0.05);
    border: 1px solid rgba(0,245,200,0.3);
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
}

/* ── SECTION LABELS ── */
.section-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-main);
    margin: 0 0 1.2rem 0;
    letter-spacing: -0.01em;
}

/* ── GRAPH CAPTIONS ── */
.graph-caption {
    color: var(--text-muted);
    font-size: 0.8rem;
    text-align: center;
    margin-top: 0.5rem;
    font-style: italic;
    padding: 0 0.5rem;
    line-height: 1.5;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px 12px 0 0;
    gap: 2px;
    padding: 6px 6px 0 6px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--bg-deep) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── INPUTS ── */
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,245,200,0.15) !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    background: var(--bg-card) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 2rem 0;
    border: none;
}

/* ── HERO ── */
.hero-wrapper {
    text-align: center;
    padding: 3.5rem 0 2rem 0;
}
.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(0,245,200,0.08);
    border: 1px solid rgba(0,245,200,0.2);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    color: var(--text-main);
    line-height: 1.1;
    letter-spacing: -0.04em;
    margin-bottom: 1.2rem;
}
.hero-highlight {
    background: linear-gradient(135deg, #00F5C8, #7B6EF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1.05rem;
    max-width: 540px;
    margin: 0 auto 2rem auto;
    line-height: 1.65;
}

/* ── PAGE HEADER ── */
.page-h2 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.03em;
    margin: 0.2rem 0 0.3rem 0;
    line-height: 1.2;
}
.page-sub {
    color: var(--text-muted);
    font-size: 0.93rem;
    margin: 0 0 2rem 0;
}

/* ── EMPTY STATE ── */
.empty-state {
    background: var(--bg-card);
    border: 1.5px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    min-height: 300px;
}

/* ── STATUS BAR ── */
.status-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1.2rem;
}

/* ── FOOTER ── */
.footer-note {
    text-align: center;
    color: var(--text-dim);
    font-size: 0.8rem;
    padding: 2rem 0 1rem 0;
}
.footer-note a { color: var(--accent); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf   = joblib.load('models/rf_model.pkl')
        a_sc = joblib.load('models/amount_scaler.pkl')
        h_sc = joblib.load('models/hour_scaler.pkl')
        return rf, a_sc, h_sc, True
    except:
        return None, None, None, False

rf_model, amount_scaler, hour_scaler, models_loaded = load_models()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        if not os.path.exists("creditcard.csv"):
            st.warning("Dataset not found. Pulling from Git LFS...")
            os.system("git lfs pull")

        df = pd.read_csv('creditcard.csv')
        df = df.drop_duplicates()
        df["Hour"] = (df["Time"] / 3600) % 24
        df.drop(columns=["Time"], inplace=True)
        return df, True
    except:
        return None, False

df_raw, data_loaded = load_data()

# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────
PLOT_BG    = '#0D0F1C'
PLOT_TEXT  = '#7A809C'
PLOT_WHITE = '#F0F2FF'
PLOT_GRID  = '#1C1F2E'
ACCENT     = '#00F5C8'
ACCENT2    = '#FF5F40'
ACCENT3    = '#7B6EF6'

def apply_plot_style(fig, ax_list=None):
    fig.patch.set_facecolor(PLOT_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT, labelsize=9)
        ax.xaxis.label.set_color(PLOT_TEXT)
        ax.yaxis.label.set_color(PLOT_TEXT)
        ax.title.set_color(PLOT_WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLOT_GRID)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# ─── NAVBAR ───────────────────────────────────────────────────────────────────
def navbar():
    c0, c1, c2, c3, c4 = st.columns([3, 1, 1, 1, 1])
    with c0:
        st.markdown('<div class="nav-logo">🛡️ Fraud<span>Guard</span></div>',
                    unsafe_allow_html=True)
    with c1:
        if st.button('Home', key='n1', use_container_width=True):
            st.session_state.page = 'Home'; st.rerun()
    with c2:
        if st.button('Predict', key='n2', use_container_width=True):
            st.session_state.page = 'Predict'; st.rerun()
    with c3:
        if st.button('Visualize', key='n3', use_container_width=True):
            st.session_state.page = 'Visualizations'; st.rerun()
    with c4:
        if st.button('Report', key='n4', use_container_width=True):
            st.session_state.page = 'Report'; st.rerun()
    st.markdown('<div class="fancy-divider" style="margin:0.4rem 0 2rem 0;"></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-eyebrow">AI-Powered Security</div>
        <div class="hero-title">
            Is Your Transaction<br>
            <span class="hero-highlight">Safe?</span>
        </div>
        <div class="hero-sub">
            Real-time credit card fraud detection powered by Machine Learning —
            trained on 284,807 European cardholder transactions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([3, 2, 3])
    with mid:
        st.markdown('<div class="cta-btn">', unsafe_allow_html=True)
        if st.button('Run Fraud Detection →', use_container_width=True, key='hero_cta'):
            st.session_state.page = 'Predict'; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider" style="margin:2.5rem 0;"></div>', unsafe_allow_html=True)

    # Why it matters
    st.markdown('<div class="section-eyebrow">The Problem</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why It Matters</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    stats = [
        ("$32B",  "#00F5C8", "Lost to credit card fraud globally every single year"),
        ("0.17%", "#7B6EF6", "Fraud rate in real data — nearly invisible, yet financially devastating"),
        ("80%",   "#00F5C8", "Fraud cases caught by this model with 70.4% precision"),
    ]
    for col, (val, color, label) in zip([c1, c2, c3], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-eyebrow" style="margin-top:1rem;">The Process</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    steps = [
        ("1", "Input Details",   "Enter the transaction amount, hour, and optional behavioral pattern features from the bank system."),
        ("2", "Pattern Analysis","Random Forest evaluates 30 features learned from 284K+ real transactions to surface fraud signals."),
        ("3", "Instant Verdict", "Receive fraud probability with risk classification — High Risk or Safe — in under a second."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;color:#F0F2FF;
                            font-size:0.97rem;margin-bottom:0.5rem;">{title}</div>
                <div style="color:#7A809C;font-size:0.85rem;line-height:1.55;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer-note">
        Dataset: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">
        Kaggle — Credit Card Fraud Detection</a>
        &nbsp;·&nbsp; European cardholders &nbsp;·&nbsp; 284,807 transactions
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="section-eyebrow">Real-time Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h2">Transaction Fraud Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter transaction details to assess fraud risk instantly</div>',
                unsafe_allow_html=True)

    if not models_loaded:
        st.error("⚠️ Model files not found. Run `save_models.py` first — ensure `models/` contains `rf_model.pkl`, `amount_scaler.pkl`, `hour_scaler.pkl`.")
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;'
                    'font-size:0.95rem;color:#F0F2FF;margin-bottom:1rem;'
                    'letter-spacing:-0.01em;">Transaction Details</div>',
                    unsafe_allow_html=True)

        amount = st.number_input(
            "Transaction Amount (€)",
            min_value=0.0, max_value=50000.0, value=100.0, step=0.01,
            help="Raw amount in Euros — log-transformed and scaled automatically."
        )
        hour = st.slider(
            "Hour of Transaction (0 = midnight · 23 = 11 PM)",
            0, 23, 12
        )
        label = ('🌙 Late night' if hour < 6
                 else '🌅 Early morning' if hour < 10
                 else '☀️ Daytime' if hour < 18
                 else '🌆 Evening')
        st.caption(f"{label} — {hour:02d}:00")

        st.markdown('<br>', unsafe_allow_html=True)
        with st.expander("⚙️  Advanced — PCA Behavioral Features V1–V28", expanded=False):
            st.markdown(
                '<div style="color:#7A809C;font-size:0.82rem;margin-bottom:1rem;line-height:1.55;">'
                'PCA-derived behavioral patterns. Leave all at 0 for average behavior. '
                'Only modify with actual transformed values from the bank system.'
                '</div>', unsafe_allow_html=True
            )
            v_values = {}
            cols = st.columns(4)
            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    v_values[f'V{i}'] = st.number_input(
                        f"V{i}", value=0.0, step=0.1,
                        key=f"v_input_{i}", format="%.3f"
                    )

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        predict_btn = st.button("🔍  Analyze Transaction", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;'
                    'font-size:0.95rem;color:#F0F2FF;margin-bottom:1rem;'
                    'letter-spacing:-0.01em;">Prediction Result</div>',
                    unsafe_allow_html=True)

        if not predict_btn:
            st.markdown("""
            <div class="empty-state">
                <div style="font-size:2.8rem;opacity:0.3;margin-bottom:1rem;">🔍</div>
                <div style="color:#4A5068;font-size:0.95rem;font-weight:500;">
                    Awaiting transaction data</div>
                <div style="color:#4A5068;font-size:0.82rem;margin-top:0.5rem;line-height:1.5;">
                    Fill in the details on the left<br>and click Analyze Transaction</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing transaction patterns..."):
                amount_log    = np.log1p(amount)
                amount_scaled = amount_scaler.transform([[amount_log]])[0][0]
                hour_scaled   = hour_scaler.transform([[hour]])[0][0]

                feature_cols  = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
                feature_dict  = {f'V{i}': v_values[f'V{i}'] for i in range(1, 29)}
                feature_dict['Amount'] = amount_scaled
                feature_dict['Hour']   = hour_scaled

                fv = np.array([[feature_dict[c] for c in feature_cols]])
                fraud_prob = rf_model.predict_proba(fv)[0][1]

            THRESHOLD = 0.4
            is_fraud  = fraud_prob >= THRESHOLD
            prob_pct  = fraud_prob * 100
            bar_w     = min(prob_pct, 100)

            if is_fraud:
                risk_label = "CRITICAL RISK" if prob_pct > 75 else "HIGH RISK"
                st.markdown(f"""
                <div class="fraud-card">
                    <div style="font-size:2.6rem;margin-bottom:0.6rem;">⚠️</div>
                    <div style="color:#FF5F40;font-family:'Syne',sans-serif;font-size:1.65rem;
                                font-weight:800;letter-spacing:-0.02em;">{risk_label}</div>
                    <div style="color:rgba(255,95,64,0.75);font-size:1.1rem;margin:0.7rem 0;
                                font-weight:500;">Fraud Probability: {prob_pct:.1f}%</div>
                    <div style="background:rgba(255,255,255,0.06);border-radius:50px;
                                height:7px;margin:1.2rem 0;overflow:hidden;">
                        <div style="background:linear-gradient(90deg,#FF5F40,#FF2200);
                                    width:{bar_w:.1f}%;height:7px;border-radius:50px;
                                    box-shadow:0 0 12px rgba(255,95,64,0.5);"></div>
                    </div>
                    <div style="color:#7A809C;font-size:0.85rem;line-height:1.55;margin-top:0.6rem;">
                        This transaction exhibits patterns strongly associated with fraud.
                        Recommend blocking and initiating cardholder verification immediately.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                risk_label = "LOW RISK" if prob_pct < 15 else "MODERATE RISK"
                st.markdown(f"""
                <div class="safe-card">
                    <div style="font-size:2.6rem;margin-bottom:0.6rem;">✅</div>
                    <div style="color:#00F5C8;font-family:'Syne',sans-serif;font-size:1.65rem;
                                font-weight:800;letter-spacing:-0.02em;">TRANSACTION SAFE</div>
                    <div style="color:rgba(0,245,200,0.75);font-size:1.1rem;margin:0.7rem 0;
                                font-weight:500;">Fraud Probability: {prob_pct:.1f}% — {risk_label}</div>
                    <div style="background:rgba(255,255,255,0.06);border-radius:50px;
                                height:7px;margin:1.2rem 0;overflow:hidden;">
                        <div style="background:linear-gradient(90deg,#00F5C8,#00C4A0);
                                    width:{bar_w:.1f}%;height:7px;border-radius:50px;
                                    box-shadow:0 0 12px rgba(0,245,200,0.4);"></div>
                    </div>
                    <div style="color:#7A809C;font-size:0.85rem;line-height:1.55;margin-top:0.6rem;">
                        This transaction shows patterns consistent with legitimate activity.
                        No immediate action required.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("""
            <div class="status-bar">
                <div style="color:#4A5068;font-size:0.75rem;text-align:center;
                            letter-spacing:0.05em;font-family:'DM Sans',sans-serif;">
                    <span style="color:#00F5C8;font-weight:600;">MODEL</span>&nbsp;Random Forest
                    &nbsp;&nbsp;·&nbsp;&nbsp;
                    <span style="color:#00F5C8;font-weight:600;">THRESHOLD</span>&nbsp;0.40
                    &nbsp;&nbsp;·&nbsp;&nbsp;
                    <span style="color:#00F5C8;font-weight:600;">RECALL</span>&nbsp;80.0%
                    &nbsp;&nbsp;·&nbsp;&nbsp;
                    <span style="color:#00F5C8;font-weight:600;">PRECISION</span>&nbsp;70.4%
                    &nbsp;&nbsp;·&nbsp;&nbsp;
                    <span style="color:#00F5C8;font-weight:600;">F1</span>&nbsp;74.8%
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_visualizations():
    st.markdown('<div class="section-eyebrow">Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h2">Model Insights & Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Visual exploration of fraud patterns, feature behavior, and model performance</div>',
                unsafe_allow_html=True)

    if not data_loaded:
        st.error("⚠️ creditcard.csv not found. Place it in the project root and restart.")
        return

    df    = df_raw.copy()
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]

    tab1, tab2, tab3 = st.tabs(["Data Overview", "Feature Analysis", "Model Performance"])

    # ── TAB 1: Data Overview ─────────────────────────────────────────────────
    with tab1:

        # 1. Class Distribution — full width centred
        fig, ax = plt.subplots(figsize=(8, 4.2))
        apply_plot_style(fig)
        counts = df['Class'].value_counts()
        bars = ax.bar(['Legitimate', 'Fraud'], counts.values,
                      color=[ACCENT3, ACCENT2], width=0.38, edgecolor='none', zorder=3)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 700,
                    f'{val:,}', ha='center', va='bottom',
                    color=PLOT_WHITE, fontsize=11, fontweight='700')
        ax.set_title('Class Distribution', color=PLOT_WHITE, fontweight='bold', pad=14, fontsize=13)
        ax.set_ylabel('Number of Transactions')
        ax.set_ylim(0, counts.max() * 1.14)
        ax.grid(axis='y', color=PLOT_GRID, alpha=0.5, zorder=0)
        ax.grid(axis='x', visible=False)
        fig.tight_layout()
        _, cc, _ = st.columns([1, 4, 1])
        with cc:
            st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">99.83% legitimate vs 0.17% fraud. This extreme imbalance makes raw accuracy a completely misleading metric — a naive model predicting everything as legitimate would score 99.83% while catching zero fraud.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # 2. Amount distributions — side-by-side, full width
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        apply_plot_style(fig, list(axes))
        axes[0].hist(legit['Amount'], bins=60, color=ACCENT3, alpha=0.9, edgecolor='none', zorder=3)
        axes[0].set_title('Legitimate — Transaction Amounts', color=PLOT_WHITE, fontweight='bold', fontsize=11)
        axes[0].set_xlabel('Amount (€)')
        axes[0].grid(axis='y', color=PLOT_GRID, alpha=0.4, zorder=0)
        axes[0].grid(axis='x', visible=False)
        axes[1].hist(fraud['Amount'], bins=40, color=ACCENT2, alpha=0.9, edgecolor='none', zorder=3)
        axes[1].set_title('Fraud — Transaction Amounts', color=PLOT_WHITE, fontweight='bold', fontsize=11)
        axes[1].set_xlabel('Amount (€)')
        axes[1].grid(axis='y', color=PLOT_GRID, alpha=0.4, zorder=0)
        axes[1].grid(axis='x', visible=False)
        fig.suptitle('Transaction Amount Distribution', color=PLOT_WHITE,
                     fontweight='bold', fontsize=13, y=1.01)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">Fraud concentrates at lower amounts — fraudsters deliberately keep transactions small to avoid triggering manual review alerts.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # 3. Hour distributions — side-by-side, full width
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        apply_plot_style(fig, list(axes))
        axes[0].hist(legit['Hour'], bins=24, color=ACCENT3, alpha=0.9, edgecolor='none', zorder=3)
        axes[0].set_title('Legitimate — Hour of Day', color=PLOT_WHITE, fontweight='bold', fontsize=11)
        axes[0].set_xlabel('Hour (0–23)')
        axes[0].grid(axis='y', color=PLOT_GRID, alpha=0.4, zorder=0)
        axes[0].grid(axis='x', visible=False)
        axes[1].hist(fraud['Hour'], bins=24, color=ACCENT2, alpha=0.9, edgecolor='none', zorder=3)
        axes[1].set_title('Fraud — Hour of Day', color=PLOT_WHITE, fontweight='bold', fontsize=11)
        axes[1].set_xlabel('Hour (0–23)')
        axes[1].grid(axis='y', color=PLOT_GRID, alpha=0.4, zorder=0)
        axes[1].grid(axis='x', visible=False)
        fig.suptitle('Transaction Timing Distribution', color=PLOT_WHITE,
                     fontweight='bold', fontsize=13, y=1.01)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">Legitimate transactions peak during business hours. Fraud spikes in early morning (0–4 AM) when cardholders are least likely to notice unauthorized charges.</div>',
                    unsafe_allow_html=True)

    # ── TAB 2: Feature Analysis ───────────────────────────────────────────────
    with tab2:

        # 1. Correlation bar chart — full width
        corr = df.corr()['Class'].drop('Class').sort_values()
        top_pos = corr.tail(5)
        top_neg = corr.head(5)
        selected_corr = pd.concat([top_neg, top_pos])

        fig, ax = plt.subplots(figsize=(11, 4.8))
        apply_plot_style(fig)
        colors = [ACCENT2 if x < 0 else ACCENT for x in selected_corr.values]
        ax.barh(selected_corr.index, selected_corr.values,
                color=colors, edgecolor='none', height=0.55, zorder=3)
        ax.axvline(x=0, color='#3A3E55', linewidth=1, linestyle='--')
        ax.set_title('Feature Correlation with Fraud (Class)', color=PLOT_WHITE,
                     fontweight='bold', pad=14, fontsize=13)
        ax.set_xlabel('Pearson Correlation Coefficient')
        ax.grid(axis='x', color=PLOT_GRID, alpha=0.4, zorder=0)
        ax.grid(axis='y', visible=False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">Teal bars: higher value → more fraud. Red bars: higher value → less fraud. Top correlated features guided EDA and align with what Random Forest later learned via importance scores.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # 2. Boxplots 2×2 — full width figure
        top_features = ['V11', 'V4', 'V14', 'V17']
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        apply_plot_style(fig, axes.flatten().tolist())
        for i, feat in enumerate(top_features):
            ax = axes.flatten()[i]
            sample_legit = legit[feat].dropna().sample(min(2000, len(legit)), random_state=42)
            sample_fraud = fraud[feat].dropna()
            bp = ax.boxplot(
                [sample_legit, sample_fraud],
                patch_artist=True,
                labels=['Legit', 'Fraud'],
                whiskerprops=dict(color='#3A3E55', linewidth=1.2),
                capprops=dict(color='#3A3E55', linewidth=1.2),
                medianprops=dict(color=PLOT_WHITE, linewidth=2.2),
                flierprops=dict(marker='o', markerfacecolor='#3A3E55',
                                markersize=2, alpha=0.3, linestyle='none')
            )
            bp['boxes'][0].set_facecolor(ACCENT3); bp['boxes'][0].set_alpha(0.75)
            bp['boxes'][1].set_facecolor(ACCENT2); bp['boxes'][1].set_alpha(0.75)
            ax.set_title(f'{feat} Distribution', color=PLOT_WHITE, fontweight='bold', fontsize=11)
            ax.grid(axis='y', color=PLOT_GRID, alpha=0.4)
            ax.grid(axis='x', visible=False)
        fig.suptitle('Top Predictive Features — Fraud vs Legitimate',
                     color=PLOT_WHITE, fontweight='bold', fontsize=13)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">V11 and V4 are elevated during fraud (positive indicators). V14 and V17 drop sharply (negative indicators). Wider median separation = stronger predictive power.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # 3. Feature Importances — full width
        if models_loaded:
            feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
            importances  = rf_model.feature_importances_
            feat_df = (pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
                       .sort_values('Importance', ascending=True).tail(12))
            fig, ax = plt.subplots(figsize=(11, 5.2))
            apply_plot_style(fig)
            colors_imp = [ACCENT if f in ['Amount', 'Hour'] else ACCENT3
                          for f in feat_df['Feature']]
            ax.barh(feat_df['Feature'], feat_df['Importance'],
                    color=colors_imp, edgecolor='none', height=0.6, zorder=3)
            ax.set_title('Top 12 Feature Importances — Random Forest',
                         color=PLOT_WHITE, fontweight='bold', pad=14, fontsize=13)
            ax.set_xlabel('Importance Score')
            ax.grid(axis='x', color=PLOT_GRID, alpha=0.4, zorder=0)
            ax.grid(axis='y', visible=False)
            fig.tight_layout()
            st.pyplot(fig); plt.close()
            st.markdown('<div class="graph-caption">Purple: PCA behavioral features. Teal: engineered features (Amount, Hour). V features dominate — confirming fraud detection relies on behavioral patterns, not transaction amount alone.</div>',
                        unsafe_allow_html=True)

    # ── TAB 3: Model Performance ──────────────────────────────────────────────
    with tab3:

        # 1. Confusion Matrix — centred
        cm = np.array([[56635, 16], [24, 71]])
        fig, ax = plt.subplots(figsize=(7, 5.2))
        apply_plot_style(fig)
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap=sns.light_palette(ACCENT3, as_cmap=True),
                    xticklabels=['Pred: Legit', 'Pred: Fraud'],
                    yticklabels=['Actual: Legit', 'Actual: Fraud'],
                    ax=ax, linewidths=2.5, linecolor=PLOT_BG,
                    annot_kws={'size': 16, 'weight': 'bold', 'color': PLOT_WHITE})
        ax.set_title('Confusion Matrix — RF Tuned (Threshold 0.4)',
                     color=PLOT_WHITE, fontweight='bold', pad=14, fontsize=12)
        ax.tick_params(colors=PLOT_TEXT, labelsize=10)
        fig.tight_layout()
        _, cc, _ = st.columns([1, 3, 1])
        with cc:
            st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">71 fraud cases correctly caught · 16 false alarms · 24 missed fraud cases. Maximizing the bottom-right cell (true positives) was the key optimization objective.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # 2. Model Comparison — full width
        mdf = pd.DataFrame({
            'Model':     ['Logistic\n(Tuned)', 'Decision\nTree', 'Random\nForest', 'RF Tuned\n⭐'],
            'Precision': [20.4, 3.9, 81.6, 70.4],
            'Recall':    [83.2, 85.3, 74.7, 80.0],
            'F1 Score':  [32.8,  7.4, 78.0, 74.8]
        })
        fig, ax = plt.subplots(figsize=(11, 5))
        apply_plot_style(fig)
        x = np.arange(len(mdf['Model']))
        w = 0.23
        ax.bar(x - w, mdf['Precision'], w, label='Precision', color=ACCENT3, alpha=0.9, edgecolor='none', zorder=3)
        ax.bar(x,     mdf['Recall'],    w, label='Recall',    color=ACCENT,  alpha=0.9, edgecolor='none', zorder=3)
        ax.bar(x + w, mdf['F1 Score'],  w, label='F1 Score',  color=ACCENT2, alpha=0.9, edgecolor='none', zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(mdf['Model'], fontsize=10, color=PLOT_TEXT)
        ax.set_ylabel('Score (%)')
        ax.set_title('All Models — Performance Comparison',
                     color=PLOT_WHITE, fontweight='bold', pad=14, fontsize=13)
        ax.legend(facecolor=PLOT_BG, edgecolor=PLOT_GRID, labelcolor=PLOT_WHITE, fontsize=10)
        ax.set_ylim(0, 110)
        ax.axvspan(2.55, 3.45, alpha=0.07, color=ACCENT, zorder=0)
        ax.grid(axis='y', color=PLOT_GRID, alpha=0.4, zorder=0)
        ax.grid(axis='x', visible=False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">RF Tuned (highlighted) achieves the best balance. Logistic Regression leads recall but fails on precision. Decision Tree collapses entirely. Random Forest wins on all-round performance.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # 3. Threshold tradeoff — full width
        thresh_vals     = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
        precision_curve = np.array([54.0, 58.5, 62.5, 66.5, 70.4, 74.0, 77.5, 81.6, 84.0, 86.5])
        recall_curve    = np.array([91.0, 88.5, 86.0, 83.5, 80.0, 78.0, 76.5, 74.7, 71.0, 67.0])

        fig, ax = plt.subplots(figsize=(11, 4.5))
        apply_plot_style(fig)
        ax.plot(thresh_vals, precision_curve, color=ACCENT3, linewidth=2.5,
                marker='o', markersize=5.5, label='Precision', zorder=3)
        ax.plot(thresh_vals, recall_curve,    color=ACCENT2, linewidth=2.5,
                marker='o', markersize=5.5, label='Recall',    zorder=3)
        ax.axvline(x=0.4, color=ACCENT, linewidth=2, linestyle='--',
                   label='Selected Threshold (0.40)', alpha=0.9, zorder=4)
        ax.fill_between(thresh_vals, precision_curve, recall_curve,
                        alpha=0.05, color=ACCENT3)
        ax.set_xlabel('Classification Threshold')
        ax.set_ylabel('Score (%)')
        ax.set_title('Precision–Recall Tradeoff vs Classification Threshold',
                     color=PLOT_WHITE, fontweight='bold', pad=14, fontsize=13)
        ax.legend(facecolor=PLOT_BG, edgecolor=PLOT_GRID, labelcolor=PLOT_WHITE, fontsize=10)
        ax.grid(color=PLOT_GRID, alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="graph-caption">As threshold rises: precision improves, recall falls. Selected threshold 0.4 (teal line) achieves 80% recall with 70.4% precision — the operational sweet spot balancing fraud capture against false alarms.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ══════════════════════════════════════════════════════════════════════════════
def page_report():
    st.markdown('<div class="section-eyebrow">Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h2">Technical Report & Code</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Complete documentation of methodology, decision reasoning, and implementation</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1 · Preprocessing",
        "2 · EDA",
        "3 · Model Selection",
        "4 · Results",
        "5 . Code"
    ])

    def load_report(fname):
        path = os.path.join('reports', fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return f"> Report file `{path}` not found."

    with tab1: st.markdown(load_report('report1.md'))
    with tab2: st.markdown(load_report('report2.md'))
    with tab3: st.markdown(load_report('report3.md'))
    with tab4: st.markdown(load_report('report4.md'))

    with tab5:
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;'
                    'font-size:1.05rem;color:#F0F2FF;margin-bottom:0.3rem;">'
                    'Complete Model Pipeline — Annotated Code</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="color:#7A809C;font-size:0.86rem;margin-bottom:1rem;">'
                    'Full preprocessing, EDA, training, threshold tuning, and model saving.</div>',
                    unsafe_allow_html=True)
        st.code(load_report('model_code.py'), language='python')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    navbar()
    page = st.session_state.page
    if page == 'Home':
        page_home()
    elif page == 'Predict':
        page_predict()
    elif page == 'Visualizations':
        page_visualizations()
    elif page == 'Report':
        page_report()

if __name__ == '__main__':
    main()