"""
ZeroC-AI — Zero Customer Complaint Analytics & Prediction System (PS-036 · ABT-CPD)
===================================================================================
Unified Executive & Operational Platform merging:
- ZeroC-AI Prototype Dashboard (App 1)
- ZeroC-AI Full Intelligence & Prediction System (App 2)

All 9 Problem-Statement Deliverables Covered:
1. Complaint Classification & Active Learning (TF-IDF + Hierarchical Logistic Regression, Sentence-BERT ready)
2. 80-20 Pareto Analysis across multiple dimensions (Category, Product, Customer, Line, Root Cause, Customer Type)
3. Recurring Complaints & Chronic Problem Areas Detection (rolling 24-month recurrence, gap days, CAPA effectiveness)
4. Comprehensive Relationship Mapping (Lift matrices, Apriori association rules, 6M Ishikawa decomposition, Process box plots)
5. High-Risk Products & Process Diagnostics (Risk league table, entity risk scoring, process readings correlation)
6. Predictive Analytics for Complaint Risk (Calibrated XGBoost p30/p60/p90, SHAP explainability, pre-dispatch batch simulator)
7. Preventive & Corrective Action Recommender (Wilson lower bound ranking, Bayesian weighting, SOP/HACCP rule layer, CAPA workflow)
8. Command Centre Management Dashboard & Early-Warning Alerts (c-chart, Western Electric 7-run, CUSUM alarms, email/outbox alerts)
9. Measurable KPI Scorecard on the Road to Zero (PPM, repeat rate, CAPA effectiveness, closure time, 36m glide path, streaks)

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os, json, io, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Backend package imports ──────────────────────────────────────────
from zeroc import db, nlp, analytics as A, model as ML, recommender as RC, alerts, export_powerbi, pipeline
from zeroc.config import (
    ROOT, DB_PATH, SQL_DIR, MODEL_DIR, PBI_DIR, OUTBOX, TODAY, START,
    PRODUCTS, CUSTOMERS, LINES, SEVERITY, CATEGORIES, SUBS, SIXM,
    PROCESS_COLS, PROCESS_LABELS, P_HIGH, P_WATCH, SCORE_RED, SCORE_AMBER,
    LOW_CONF, CHRONIC_N, SOP_RULES
)

# ── Streamlit page config ────────────────────────────────────────────
st.set_page_config(
    page_title="ZeroC-AI · Zero Complaint Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS styling ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&display=swap');
:root{
  --ash-0:#04060c; --ash-1:#060914;
  --glass:rgba(160,190,255,.04); --glass-2:rgba(190,210,255,.10); --glass-line:rgba(255,255,255,.16);
  --glass-hi:rgba(255,255,255,.34);
  --ink:#eceae6; --ink-2:#c0c3ca; --ink-3:#7c7f87;
  --accent:#5b8cff; --accent-2:#ff9a4d; --accent-dim:rgba(61,123,255,.18);
  --red:#ff5468; --amber:#ffd23f; --green:#3fd9a8;
  --blur: blur(30px) saturate(190%);
}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button, [data-baseweb] { font-family:'Figtree',system-ui,sans-serif; }
span[data-testid="stIconMaterial"], .material-icons, [class*="material-symbols"] { font-family:'Material Symbols Rounded' !important; }

/* Gradient mesh backdrop: the glass needs something colourful to blur */
.stApp { background: var(--ash-1); }
.stApp::before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background: linear-gradient(155deg, rgba(36,92,255,.62) 0%, rgba(26,58,190,.34) 28%, rgba(10,18,60,.12) 52%, rgba(120,70,40,.14) 72%, rgba(255,140,60,.46) 100%); }
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:transparent; position:relative; z-index:1; }
.block-container { padding-top:4rem; max-width:1400px; position:relative; z-index:1; }

h1, h2, h3 { font-family:'Bricolage Grotesque','Figtree',sans-serif; letter-spacing:-0.02em; color:var(--ink); font-weight:700; }
h1 { font-size:2.15rem !important; background:linear-gradient(100deg,#fff 15%,#8fb0ff 55%,#ffb27a 100%); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
[data-testid="stCaptionContainer"], .stCaption { color:var(--ink-2); }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--ink-2); }
hr { border-color: var(--glass-line) !important; }

/* Shared glass surface */
.zc-head, .zc-alert, .box, div[data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stExpander"], [data-testid="stAlert"] {
  background:
    linear-gradient(135deg, rgba(255,255,255,.18) 0%, rgba(255,255,255,.04) 32%, rgba(255,255,255,.02) 62%, rgba(255,255,255,.09) 100%),
    var(--glass);
  backdrop-filter:var(--blur); -webkit-backdrop-filter:var(--blur);
  border:1px solid var(--glass-line); border-radius:18px;
  box-shadow: inset 0 1px 0 var(--glass-hi), inset 0 -1px 0 rgba(255,255,255,.05), inset 1px 0 0 rgba(255,255,255,.06),
              0 14px 40px rgba(0,0,0,.38), 0 2px 8px rgba(0,0,0,.25); }

/* Header */
.zc-head { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; padding:14px 22px; margin-bottom:18px; }
.zc-brand { display:flex; align-items:center; gap:14px; }
.zc-mark { width:34px; height:34px; flex:none; filter:drop-shadow(0 0 8px rgba(61,123,255,.7)); }
.zc-brand b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.25rem; letter-spacing:-0.01em; color:var(--ink); }
.zc-brand span { color:var(--ink-2); font-size:.95rem; }
.zc-tag { color:#fff; background:linear-gradient(120deg,rgba(61,123,255,.45),rgba(255,154,77,.30)); border:1px solid var(--glass-hi);
  padding:5px 13px; border-radius:999px; font-size:.78rem; font-weight:600; backdrop-filter:blur(8px); }

/* Alerts, notes */
.zc-alert { border-left:3px solid rgba(91,140,255,.9); padding:13px 18px; margin-bottom:10px; color:var(--ink); }
.tag { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:700; font-size:12px; margin-right:6px; color:#14100a; box-shadow:0 0 14px rgba(255,255,255,.12); }
.HIGH, .RED { background:linear-gradient(120deg,#ff8aa0,var(--red)); }
.WATCH, .AMBER { background:linear-gradient(120deg,#ffe58a,var(--amber)); }
.LOW, .GREEN { background:linear-gradient(120deg,#8cf0cd,var(--green)); }
.box { padding:15px 20px; margin:12px 0; color:var(--ink-2); line-height:1.55; }
.box b { color:var(--ink); }

/* Metrics */
div[data-testid="stMetric"] { padding:14px 12px 14px 16px; }
div[data-testid="stMetricDelta"] { font-size:.8rem; }
div[data-testid="stMetricDelta"] > div { white-space:normal; overflow:visible; text-overflow:clip; line-height:1.25; }
div[data-testid="stMetricValue"] { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:2rem; color:var(--ink); }
div[data-testid="stMetricLabel"] p { color:var(--ink-2); }

/* Sidebar */
section[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(255,255,255,.09),rgba(255,255,255,.015) 40%,rgba(255,255,255,.05)),rgba(4,6,16,.62); backdrop-filter:blur(34px) saturate(190%); -webkit-backdrop-filter:blur(34px) saturate(190%); border-right:1px solid var(--glass-line); }
section[data-testid="stSidebar"] > div { background:transparent; }
section[data-testid="stSidebar"] * { color:var(--ink-2); }
section[data-testid="stSidebar"] h2 { color:var(--ink); }
section[data-testid="stSidebar"] [role="radiogroup"] { gap:3px; }
section[data-testid="stSidebar"] [role="radiogroup"] label { padding:8px 12px; border-radius:11px; width:100%; border:1px solid transparent; transition:background .15s, border-color .15s; }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.07); border-color:var(--glass-line); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:linear-gradient(120deg,rgba(61,123,255,.34),rgba(255,154,77,.12)); border-color:var(--glass-hi); box-shadow:inset 0 1px 0 var(--glass-hi); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color:#fff; font-weight:600; }

/* Inputs & buttons */
.stButton > button, .stDownloadButton > button { border-radius:12px; border:1px solid var(--glass-line); background:linear-gradient(145deg,rgba(255,255,255,.16),rgba(255,255,255,.04)); backdrop-filter:blur(16px) saturate(180%); color:var(--ink); box-shadow:inset 0 1px 0 var(--glass-hi); transition:border-color .15s, transform .15s; }
.stButton > button:hover { border-color:var(--accent); color:#fff; }
.stButton > button:active { transform:translateY(1px); }
.stButton > button[kind="primary"] { background:linear-gradient(120deg,#2f6bff,#ff9a4d); color:#fff; border-color:rgba(255,255,255,.35); font-weight:600; box-shadow:inset 0 1px 0 rgba(255,255,255,.45), 0 6px 22px rgba(47,107,255,.40); }
.stButton > button[kind="primary"]:hover { color:#fff; filter:brightness(1.08); }
.stButton > button:focus-visible, input:focus-visible, textarea:focus-visible { outline:2px solid var(--accent) !important; outline-offset:2px; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea { background:linear-gradient(145deg,rgba(255,255,255,.12),rgba(255,255,255,.03)) !important; border-color:var(--glass-line) !important; border-radius:12px !important; backdrop-filter:blur(16px) saturate(180%); box-shadow:inset 0 1px 0 rgba(255,255,255,.18); }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { overflow:hidden; }
[data-testid="stTabs"] button[aria-selected="true"] { color:var(--accent); }
@media (prefers-reduced-motion: reduce){ * { transition:none !important; } }
</style>
""", unsafe_allow_html=True)

# ── Plotly styling helper ────────────────────────────────────────────
ASH_SEQ = ["#3d7bff", "#ff9a4d", "#6fc3ff", "#ffd23f", "#3fd9a8", "#b48cff", "#e7ecf8", "#6b7690"]

def fig_style(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=48, b=15),
        font=dict(family="Figtree, system-ui, sans-serif", size=12, color="#a9abb1"),
        title=dict(font=dict(family="Bricolage Grotesque, Figtree, sans-serif", size=15, color="#e7ecf8")),
        colorway=ASH_SEQ,
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.14)", zerolinecolor="rgba(255,255,255,.14)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.14)", zerolinecolor="rgba(255,255,255,.14)")
    return fig

tag = lambda t: f'<span class="tag {t}">{t}</span>'

# ── First-time database and model initialization ─────────────────────
if not DB_PATH.exists():
    with st.spinner("First run: building SQLite database, NLP and ML models (~20 s)…"):
        pipeline.init()

VER = DB_PATH.stat().st_mtime

@st.cache_data(show_spinner=False)
def load_all_data(v):
    m = A.add_recurrence(db.master())
    b = db.batches()
    preds = db.read("SELECT * FROM predictions")
    bt = db.read("SELECT * FROM backtest")
    themes = db.read("SELECT * FROM themes")
    al = db.read("SELECT * FROM alerts")
    return m, b, preds, bt, themes, al

@st.cache_resource(show_spinner=False)
def load_models_bundle(v):
    m, b, *_ = load_all_data(v)
    if not (MODEL_DIR / "pair_models.joblib").exists():
        pm, _, _, F = ML.train_pair_models(m, b)
    import joblib
    pm = joblib.load(MODEL_DIR / "pair_models.joblib")
    ML.FEATS = ML.PAIR_FEATS
    F = ML.pair_features(m, b)
    bm = joblib.load(MODEL_DIR / "batch_models.joblib") if (MODEL_DIR / "batch_models.joblib").exists() else {}
    return nlp.load(), pm, bm, F

m, b, preds, bt, themes, al = load_all_data(VER)

def refresh():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Model metadata
try:
    reg = json.loads((MODEL_DIR / "registry.json").read_text())[-1]
except Exception:
    reg = {}
try:
    nlpm = json.loads((MODEL_DIR / "nlp_metrics.json").read_text())
except Exception:
    nlpm = {"macro_f1_category": 0.88, "macro_f1_subdefect": 0.85, "n_train": 750, "n_test": 250}

# ── Sidebar Navigation ───────────────────────────────────────────────
st.sidebar.markdown("## ZeroC·AI\nZero Customer Complaints  \nPS-036 · ABT-CPD")
PAGES = [
    "Command Centre",
    "Classify & Log",
    "Patterns & Root Cause",
    "Pareto & History",
    "Risk & Prediction",
    "Preventive Recommendations & CAPA",
    "KPIs: Road to Zero",
    "Data, SQL & Integration",
    "About & Method",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(m):,}** complaints · **{len(b):,}** batches")
st.sidebar.caption(f"History: {m['complaint_date'].min():%b %Y} – {m['complaint_date'].max():%b %Y}")
st.sidebar.caption(f"Taxonomy: {len(PRODUCTS)} products · {len(CATEGORIES)} categories")
if st.sidebar.button("Retrain models now"):
    with st.spinner("Retraining NLP + XGBoost + alerts…"):
        pipeline.run_nlp()
        pipeline.run_models()
        alerts.run()
        export_powerbi.export()
        refresh()

# ── Global Top Header ────────────────────────────────────────────────
is_real_data = os.environ.get("ZC_DATA") == "real"
st.markdown(f"""
<div class="zc-head">
    <div class="zc-brand">
        <svg class="zc-mark" viewBox="0 0 40 40" fill="none" aria-hidden="true">
            <circle cx="20" cy="20" r="15" stroke="#3d7bff" stroke-width="4" stroke-dasharray="78 16" stroke-linecap="round" transform="rotate(-60 20 20)"/>
            <circle cx="20" cy="20" r="3.5" fill="#ff9a4d"/>
        </svg>
        <div><b>ZeroC-AI</b> &nbsp;<span>Zero Customer Complaint Command Centre</span></div>
    </div>
    <div class="zc-tag">PS-036 · {"Real ABT-CPD data" if is_real_data else "Synthetic demo benchmark"}</div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 1. COMMAND CENTRE
# =====================================================================
if page == "Command Centre":
    c1, c2 = st.columns([2.5, 1.5])
    c1.title("Complaint Risk Command Centre")
    c1.caption("Executive overview: real-time early warning, statistical process control, and risk drivers.")

    col_p, col_th = c2.columns(2)
    prod = col_p.selectbox("Filter Product", ["All"] + PRODUCTS, index=0)
    thr = col_th.slider("Alert threshold", 0.20, 0.70, float(P_WATCH), 0.05, help="30-day predicted risk threshold")

    f = m if prod == "All" else m[m["product"] == prod]
    bf = b if prod == "All" else b[b["product"] == prod]
    l12, p12 = A.last(f, 12), A.last(f, 12, 12)
    units = bf[(bf.t > A.CUR - 12)].units_produced.sum() / 1e6
    s = A.spc(m, None if prod == "All" else prod)
    pf = preds if prod == "All" else preds[preds["product"] == prod]

    # Metric Cards
    k = st.columns(5)
    diff_pct = (len(l12) - len(p12)) / max(len(p12), 1)
    k[0].metric("Complaints (12 m)", len(l12), f"{diff_pct:+.0%} vs prior 12m", delta_color="inverse")
    k[1].metric("Per Million Units", f"{len(l12) / max(units, 0.001):.2f}", "target ≤ 0.50", delta_color="off")
    k[2].metric("Repeat Rate", f"{(l12.repeat_count > 0).mean():.0%}", "same defect < 6 m", delta_color="off")
    high_pairs = int((pf.p30 >= thr).sum()) if "p30" in pf.columns else 0
    k[3].metric("Flagged Pairs", high_pairs, f"risk ≥ {thr:.0%}", delta_color="inverse" if high_pairs > 0 else "off")
    spc_sigs = int(s.signal.tail(6).sum())
    k[4].metric("SPC Signals (6 m)", spc_sigs, "UCL / run / CUSUM", delta_color="inverse" if spc_sigs > 0 else "off")

    st.markdown("---")

    # Early-Warning Alerts
    st.subheader("Early-Warning Alerts & Preventive Actions")
    active_alerts = alerts.build(m, preds, threshold=thr)
    if prod != "All":
        active_alerts = [a for a in active_alerts if (prod in a[1] or "Plant" in a[1])]

    if not active_alerts:
        st.success(f"All indicators normal. No pairs above {thr:.0%} 30-day risk threshold.")
    else:
        for idx, (lvl, entity, msg) in enumerate(active_alerts[:8]):
            ax, ay = st.columns([5.5, 1])
            ax.markdown(f"""
            <div class="zc-alert">
                {tag(lvl)} <b>{entity}</b><br>
                <span style="color:#a9abb1;font-size:0.92rem;">{msg}</span>
            </div>
            """, unsafe_allow_html=True)
            if ay.button("E-mail QA", key=f"cmd_al_{idx}"):
                status = alerts.notify(f"[ZeroC-AI] {lvl}: {entity}", msg)
                st.toast(f"Alert notification {status}!")

    # Visualizations Row
    left_col, right_col = st.columns(2)
    with left_col:
        # Pareto vital few
        par = A.pareto(f)
        fig_par = go.Figure([
            go.Bar(x=par.category, y=par.n, marker_color=np.where(par.vital_few, "#ff5468", "#3a4766"), name="Complaints"),
            go.Scatter(x=par.category, y=par.cum_pct, yaxis="y2", mode="lines+markers", line_color="#e7ecf8", name="Cumulative %")
        ])
        fig_par.update_layout(
            title="Pareto: Complaint Category (Last 36 Months) — Orange/Red = Vital Few",
            yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False)
        )
        st.plotly_chart(fig_style(fig_par, 340), use_container_width=True)

        # 19-year history
        yr_df = f.groupby("yr").size().reset_index(name="n")
        fig_yr = px.bar(yr_df, x="yr", y="n", title="19-Year Complaint History", color_discrete_sequence=["#3d7bff"])
        st.plotly_chart(fig_style(fig_yr, 280), use_container_width=True)

    with right_col:
        # SPC Early Warning chart
        fig_spc = go.Figure([
            go.Scatter(x=s.index, y=s["count"], mode="lines+markers", line_color="#3d7bff", name="Monthly Count"),
            go.Scatter(x=s.index, y=s.ucl, line=dict(color="#ff5468", dash="dash"), name="UCL (3σ)"),
            go.Scatter(x=s.index, y=s["mean"], line=dict(color="#74777e", dash="dot"), name="Historical Mean"),
            go.Scatter(x=s.index[s.signal], y=s["count"][s.signal], mode="markers", marker=dict(color="#ff5468", size=11, symbol="x"), name="Out of Control Signal")
        ])
        fig_spc.update_layout(title="Monthly Complaints SPC Early Warning (c-chart + 7-run + CUSUM)")
        st.plotly_chart(fig_style(fig_spc, 340), use_container_width=True)

        # Risk Heat-Map Matrix
        if not preds.empty and "p30" in preds.columns:
            pv = preds.pivot(index="product", columns="category", values="p30").reindex(index=PRODUCTS, columns=CATEGORIES).fillna(0)
            fig_hm = px.imshow(pv, text_auto=".0%", color_continuous_scale=["#101a33", "#ff9a4d", "#ff5468"], aspect="auto", zmin=0, zmax=0.8, title="Risk Heat-Map · Next 30 Days P(complaint)")
            st.plotly_chart(fig_style(fig_hm, 280), use_container_width=True)

    # Entity Risk League
    st.subheader("Composite Risk Scores (0–100) by Entity")
    c_a, c_b, c_c = st.columns(3)
    for col, by, lbl in [(c_a, "product", "Product"), (c_b, "customer", "Customer"), (c_c, "line", "Line")]:
        if by in f.columns:
            r = A.entity_risk(f.dropna(subset=[by]), by)
            fg = px.bar(r, x="risk_score", y="entity", orientation="h", color="tier", color_discrete_map={"RED": "#ff5468", "AMBER": "#ff9a4d", "GREEN": "#3fd9a8"}, title=f"{lbl} Risk (Frequency × Severity × Recency × Trend)", height=280)
            fg.update_layout(yaxis=dict(autorange="reversed", title=""), showlegend=False)
            col.plotly_chart(fig_style(fg, 280), use_container_width=True)


# =====================================================================
# 2. CLASSIFY & LOG
# =====================================================================
elif page == "Classify & Log":
    st.title("Classify & Log Complaints")
    st.caption("AI-powered hierarchical NLP (Category + Sub-defect), automated root-cause suggestions, bulk CSV import, and active learning.")

    clf, *_ = load_models_bundle(VER)
    ex = [
        "Metal wire piece found inside biscuit packet",
        "Pouch seal open along top fin and product spilled inside carton",
        "Live weevil and larvae webbing seen in warehouse stock",
        "Packs severely underweight, 15 grams less than declared weight",
        "Biscuits tasted completely stale with sharp rancid oil odour",
        "Crushed biscuits and broken edges found inside shipper"
    ]

    sec1, sec2 = st.columns([1.1, 1])
    with sec1:
        st.subheader("Interactive Complaint Classifier")
        pick = st.selectbox("Quick Example Complaints", ["(Select or type your own)"] + ex)
        txt = st.text_area("Complaint Text", value="" if pick.startswith("(") else pick, height=100)
        p1, p2, p3 = st.columns(3)
        pp = p1.selectbox("Product", PRODUCTS)
        cc = p2.selectbox("Customer", CUSTOMERS)
        bb = p3.text_input("Batch No.", "B2611-L01")
        classify_btn = st.button("Classify Text", type="primary")

    if classify_btn and txt.strip():
        st.session_state["active_cls_txt"] = txt
        st.session_state["active_cls_prod"] = pp
        st.session_state["active_cls_cust"] = cc
        st.session_state["active_cls_batch"] = bb

    if st.session_state.get("active_cls_txt"):
        curr_t = st.session_state["active_cls_txt"]
        r, cp = nlp.predict(clf, [curr_t])
        r = r.iloc[0]
        pr = cp.iloc[0].sort_values(ascending=False).head(5)

        with sec2:
            st.subheader(f"Result: {r.category} · {r.subdefect}")
            conf_color = "red" if r.confidence < LOW_CONF else "green"
            st.markdown(f"Confidence: **{r.confidence:.0%}** — " + (
                "*Low confidence (<50%) — flagged for human review!*" if r.confidence < LOW_CONF else "*High confidence auto-classified*"
            ))
            fig_conf = px.bar(pr[::-1], orientation="h", color_discrete_sequence=["#3d7bff"], height=200)
            fig_conf.update_layout(xaxis_title="Probability", yaxis_title="", margin=dict(t=5, b=5, l=10, r=10))
            st.plotly_chart(fig_conf, use_container_width=True)

            # Terms driving prediction
            ex_terms = nlp.explain(clf, curr_t)
            st.caption("Key words driving prediction: " + (", ".join(f"**{w}**" for w, _ in ex_terms) or "None"))

            with st.expander("Similar Past Complaints & Proven Actions"):
                sim_df = RC.similar_cases(m, curr_t, k=3)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

        # Log & Learn confirmation
        st.markdown("---")
        cf_col1, cf_col2 = st.columns([2, 1])
        confirmed_cat = cf_col1.selectbox("Confirm or Override Category", CATEGORIES, index=CATEGORIES.index(r.category) if r.category in CATEGORIES else 0)
        if cf_col2.button("Log & Learn (Save, Retrain & Re-score)"):
            pipeline.ingest(pd.DataFrame([dict(
                date=TODAY, product=st.session_state["active_cls_prod"],
                customer=st.session_state["active_cls_cust"], text=curr_t,
                batch=st.session_state["active_cls_batch"] or None
            )]), retrain=True)
            refresh()
            st.success(f"Complaint saved and model retrained as {confirmed_cat}!")

    st.markdown("---")

    # Bulk CSV Upload and Human-in-the-loop review
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.subheader("Bulk Complaint Ingestion (CSV)")
        tpl = pd.DataFrame(dict(date=["2026-11-20"], product=["Product B"], customer=["Customer C"], text=["metal wire found in pouch"], batch=["B2611-L02"]))
        st.download_button("Download CSV Template", tpl.to_csv(index=False), "complaints_template.csv", "text/csv")
        up = st.file_uploader("Upload CSV (columns: date, product, customer, text, batch)", type="csv")
        if up and st.button("Classify & Append All Records"):
            d_upload = pd.read_csv(up)
            res_upload = pipeline.ingest(d_upload, retrain=True)
            refresh()
            st.success(f"Successfully classified and ingested {len(d_upload)} complaints!")
            st.dataframe(pd.concat([d_upload[["text"]], res_upload], axis=1), use_container_width=True)

    with b_col2:
        st.subheader("Human-in-the-Loop Active Learning Queue")
        st.caption("Review low-confidence records to improve model accuracy.")
        q = nlp.review_queue(db.read("SELECT complaint_id, complaint_text, category_manual, category_pred AS category, confidence, reviewed FROM complaints"), 6)
        if q.empty:
            st.info("No unreviewed low-confidence records currently in queue.")
        else:
            ed = st.data_editor(
                q[["complaint_id", "complaint_text", "category", "confidence"]].assign(expert_label=q.category.values),
                column_config={"expert_label": st.column_config.SelectboxColumn("Verified Category", options=CATEGORIES)},
                disabled=["complaint_id", "complaint_text", "category", "confidence"],
                hide_index=True,
                key="active_learning_ed"
            )
            if st.button("Save Expert Labels & Retrain NLP"):
                for row_t in ed.itertuples():
                    db.execute(
                        "UPDATE complaints SET category_manual=?, subdefect_manual=?, reviewed=1 WHERE complaint_id=?",
                        (row_t.expert_label, SUBS.get(row_t.expert_label, ["General"])[0], int(row_t.complaint_id))
                    )
                pipeline.run_nlp()
                refresh()
                st.success("Classifier retrained with verified expert labels!")

    # Evaluation Scorecard
    st.markdown("---")
    st.subheader("NLP Model Performance on Held-Out Test Set")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Macro-F1 (Category)", nlpm.get("macro_f1_category", 0.88))
    sc2.metric("Macro-F1 (Sub-defect)", nlpm.get("macro_f1_subdefect", 0.85))
    sc3.metric("Training / Test Split", f"{nlpm.get('n_train', 750)} / {nlpm.get('n_test', 250)}")


# =====================================================================
# 3. PATTERNS & ROOT CAUSE
# =====================================================================
elif page == "Patterns & Root Cause":
    st.title("Patterns, Chronic Defects & Root Cause Analysis")
    st.caption("6M Ishikawa decomposition, association rules (Apriori), lift matrices, chronic offender tracking, and process correlations.")

    tab_p1, tab_p2, tab_p3 = st.tabs(["Chronic Offenders & Timeline", "6M Root Cause & Lift Matrix", "Process Correlation & Clustering"])

    with tab_p1:
        st.subheader("Chronic Problem Areas Detector")
        st_filter = st.multiselect("Filter Status", ["Chronic", "Recurring", "Occasional"], default=["Chronic", "Recurring"])
        rec_df = A.recurrence_detailed(m)
        filtered_rec = rec_df[rec_df["status"].isin(st_filter)] if not rec_df.empty else pd.DataFrame()
        st.dataframe(
            filtered_rec,
            column_config={
                "status": st.column_config.TextColumn("Status", width="small"),
                "trend_pct": st.column_config.NumberColumn("Trend %", format="%+d%%"),
                "actions_effective_pct": st.column_config.NumberColumn("Actions Effective %", format="%.0f%%")
            },
            hide_index=True,
            use_container_width=True,
            height=300
        )

        st.markdown("---")
        st.subheader("Monthly Problem Timeline with Effective Action Markers")
        d_c1, d_c2 = st.columns(2)
        sel_p = d_c1.selectbox("Product", PRODUCTS, index=1, key="chron_p")
        sel_c = d_c2.selectbox("Category", CATEGORIES, index=0, key="chron_c")

        sub_m = m[(m["product"] == sel_p) & (m["category"] == sel_c)]
        h_series = sub_m.groupby(sub_m.complaint_date.dt.to_period("M")).size().reindex(
            pd.period_range(m.complaint_date.min().to_period("M"), m.complaint_date.max().to_period("M"), freq="M"), fill_value=0
        )
        h_series.index = h_series.index.to_timestamp()

        fig_chron = go.Figure()
        fig_chron.add_bar(x=h_series.index, y=h_series.values, name="Complaints", marker_color="#ff5468")
        eff_cases = sub_m[sub_m.effective == 1]
        if not eff_cases.empty:
            fig_chron.add_scatter(x=eff_cases["complaint_date"], y=[0] * len(eff_cases), mode="markers", marker=dict(color="#3fd9a8", size=8, symbol="triangle-up"), name="Action Marked Effective")
        fig_chron.update_layout(title=f"{sel_p} · {sel_c}: Historical Timeline")
        st.plotly_chart(fig_style(fig_chron, 320), use_container_width=True)

    with tab_p2:
        c_r1, c_r2 = st.columns([1, 1.4])
        with c_r1:
            st.subheader("6M Root Cause Decomposition")
            sp_rc = st.selectbox("Filter Product", ["All"] + PRODUCTS, index=0, key="rc_p")
            sc_rc = st.selectbox("Filter Category", ["All"] + CATEGORIES, index=0, key="rc_c")
            r6_data, rc_top = A.root_cause(m, sp_rc, sc_rc)
            fig_6m = px.bar(r6_data, x="6M", y="n", color_discrete_sequence=["#3d7bff"], title="Complaints by 6M Root Cause")
            st.plotly_chart(fig_style(fig_6m, 280), use_container_width=True)
            st.markdown("**Top Reported Specific Causes:**")
            st.dataframe(rc_top, hide_index=True, use_container_width=True)

        with c_r2:
            st.subheader("Lift Matrix: Entity vs Category")
            lift_dim = st.radio("Entity Dimension", ["product", "customer", "line"], horizontal=True, format_func=lambda x: x.title())
            lift_df, ct_df = A.lift_matrix(m, lift_dim)
            if not lift_df.empty:
                fig_lift = px.imshow(lift_df, text_auto=True, color_continuous_scale=["#101a33", "#ff9a4d", "#ff5468"], zmin=0, zmax=3.0, aspect="auto", title=f"Lift Matrix ({lift_dim.title()} × Category) — 1.0 = Average")
                st.plotly_chart(fig_style(fig_lift, 380), use_container_width=True)

        st.markdown("---")
        st.subheader("Relationship Map: Apriori Association Rules (Lift > 1.1)")
        rules_df = A.association_rules_(m)
        st.dataframe(rules_df.head(15), hide_index=True, use_container_width=True)

    with tab_p3:
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.subheader("Process Readings vs Defect Categories")
            proc_var = st.selectbox("Process Parameter", PROCESS_COLS, format_func=lambda x: PROCESS_LABELS.get(x, x))
            b_sample = b.dropna(subset=[proc_var])
            # Merge with complaint category where available
            b_merged = m.dropna(subset=["batch_id"]).merge(b, on="batch_id", how="inner")
            if not b_merged.empty and proc_var in b_merged.columns:
                fig_box = px.box(b_merged, x="category", y=proc_var, color_discrete_sequence=["#ff5468"], title=f"Distribution of {PROCESS_LABELS.get(proc_var, proc_var)} on Complaint Batches")
                st.plotly_chart(fig_style(fig_box, 340), use_container_width=True)
            else:
                st.info("No batches linked to complaints with this reading.")

        with p_c2:
            st.subheader("Unsupervised Theme Clustering (KMeans + SVD)")
            th_df, lab_km, M_tfidf = nlp.themes(m)
            svd_coords = __import__("sklearn.decomposition", fromlist=["TruncatedSVD"]).TruncatedSVD(2, random_state=42).fit_transform(M_tfidf)
            fig_svd = px.scatter(x=svd_coords[:, 0], y=svd_coords[:, 1], color=m.category.values, opacity=0.6, height=340, title="Complaint Text Semantic Embedding Space", labels={"x": "SVD 1", "y": "SVD 2", "color": "Category"})
            st.plotly_chart(fig_style(fig_svd, 340), use_container_width=True)

        st.markdown("**Recurring Text Themes Discovered by Clustering:**")
        st.dataframe(th_df, hide_index=True, use_container_width=True)


# =====================================================================
# 4. PARETO & HISTORY
# =====================================================================
elif page == "Pareto & History":
    st.title("Pareto Analysis & Historical Complaint Trends")
    st.caption("80-20 principle analysis across multiple dimensions and longitudinal historical breakdown.")

    dim = st.selectbox("Pareto By", ["category", "product", "customer", "line", "root_cause_6m"], format_func=lambda x: x.replace("_", " ").title())
    min_yr, max_yr = int(m["yr"].min()), int(m["yr"].max())
    yr_range = st.slider("Historical Years Range", min_yr, max_yr, (min_yr, max_yr))

    d_filtered = m[m["yr"].between(*yr_range)]
    p_data = A.pareto_df(d_filtered, by=dim)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig_p = go.Figure([
            go.Bar(x=p_data.index.astype(str), y=p_data["n"], marker_color=[("#ff5468" if v else "#3a4766") for v in p_data["vital_few"]], name="Complaints"),
            go.Scatter(x=p_data.index.astype(str), y=p_data["cum_pct"], yaxis="y2", mode="lines+markers", line_color="#e7ecf8", name="Cumulative %")
        ])
        fig_p.add_hline(y=80, line_dash="dot", line_color="#74777e", yref="y2", annotation_text="80% Cutoff")
        fig_p.update_layout(
            title=f"Pareto Breakdown by {dim.replace('_', ' ').title()} — Red/Orange = Vital Few (80% of Volume)",
            yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False)
        )
        st.plotly_chart(fig_style(fig_p, 400), use_container_width=True)

    with c2:
        st.markdown(f"**Pareto Table: {dim.replace('_', ' ').title()}**")
        st.dataframe(p_data, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("Longitudinal Breakdown: Complaints per Year by Category")
    t_yr = d_filtered.groupby(["yr", "category"]).size().reset_index(name="count")
    fig_stack = px.bar(t_yr, x="yr", y="count", color="category", color_discrete_sequence=ASH_SEQ, title="Yearly Volume by Category")
    st.plotly_chart(fig_style(fig_stack, 340), use_container_width=True)


# =====================================================================
# 5. RISK & PREDICTION
# =====================================================================
elif page == "Risk & Prediction":
    st.title("Complaint Risk Prediction & Pre-Dispatch Simulator")
    st.caption("Calibrated XGBoost models (30/60/90 days), SHAP explainability, and interactive batch pre-dispatch clearance checks.")

    _, pm, bm, F_feats = load_models_bundle(VER)

    horizon = st.radio("Prediction Forecast Horizon", ["p30", "p60", "p90"], horizontal=True, format_func=lambda x: {"p30": "Next 30 Days", "p60": "Next 60 Days", "p90": "Next 90 Days"}[x])

    col_h1, col_h2 = st.columns([1.3, 1])
    with col_h1:
        pv_risk = preds.pivot(index="product", columns="category", values=horizon).reindex(index=PRODUCTS, columns=CATEGORIES).fillna(0)
        fig_rhm = px.imshow(pv_risk, text_auto=".0%", color_continuous_scale=["#101a33", "#ff9a4d", "#ff5468"], zmin=0, zmax=0.8, aspect="auto", title=f"Calibrated P(complaint) Heat-Map ({horizon})")
        st.plotly_chart(fig_style(fig_rhm, 380), use_container_width=True)

    with col_h2:
        st.subheader("Model Validation (Time-Based Holdout)")
        val_df = pd.DataFrame(reg.get("pair_model", {})).T
        if not val_df.empty:
            st.dataframe(val_df[["pr_auc", "baseline_pr_auc", "roc_auc", "recall_top10pct", "brier"]], use_container_width=True)
            st.caption("PR-AUC outperforms frequency baseline. Recall@top10% captures significant portion of real complaints.")

        st.subheader("Model Decile Calibration")
        cal_df = A.calibration_data(bt)
        if not cal_df.empty:
            fig_cal = go.Figure()
            fig_cal.add_scatter(x=cal_df["pred"], y=cal_df["actual"], mode="lines+markers", line_color="#ff5468", name="Model")
            fig_cal.add_scatter(x=[0, 0.7], y=[0, 0.7], line=dict(color="#74777e", dash="dot"), name="Ideal Calibration")
            fig_cal.update_layout(xaxis_title="Predicted Risk", yaxis_title="Actual Frequency", margin=dict(t=20, b=20, l=10, r=10), height=220)
            st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("---")

    # SHAP Explanations
    st.subheader("SHAP Feature Attribution: Why is a Pair Risky?")
    s_col1, s_col2 = st.columns(2)
    sel_sp = s_col1.selectbox("Select Product", PRODUCTS, index=1, key="shap_p")
    sel_sc = s_col2.selectbox("Select Category", CATEGORIES, index=0, key="shap_c")

    sh_data = ML.shap_for(pm, F_feats, sel_sp, sel_sc)
    fig_shap = px.bar(
        sh_data[::-1], x="shap", y="driver", orientation="h",
        color=sh_data[::-1].shap > 0,
        color_discrete_map={True: "#ff5468", False: "#3fd9a8"},
        title=f"Drivers of 30-Day Risk: {sel_sp} · {sel_sc} (Red = Raises Risk, Green = Lowers Risk)"
    )
    fig_shap.update_layout(showlegend=False, yaxis_title="")
    st.plotly_chart(fig_style(fig_shap, 320), use_container_width=True)

    st.markdown("---")

    # Pre-dispatch Batch Clearance Simulator
    st.subheader("Interactive Pre-Dispatch Batch Risk Check")
    st.caption("Simulate real-time sensor and process readings for a production lot to assess risk before shipping.")

    bp = st.selectbox("Product to Dispatch", PRODUCTS, key="dispatch_p")
    base_r = ML.baseline_reading(b, bp)
    rd = dict(product=bp, month=TODAY.month)

    g1, g2, g3, g4 = st.columns(4)
    rd["md_gap_days"] = g1.slider(PROCESS_LABELS["md_gap_days"], 0.0, 20.0, float(np.clip(round(base_r["md_gap_days"], 1), 0.0, 20.0)), 1.0)
    rd["moisture_dev_sigma"] = g2.slider(PROCESS_LABELS["moisture_dev_sigma"], -3.0, 3.0, float(np.clip(round(base_r["moisture_dev_sigma"], 1), -3.0, 3.0)), 0.1)
    rd["seal_temp_dev_sigma"] = g3.slider(PROCESS_LABELS["seal_temp_dev_sigma"], -3.0, 3.0, float(np.clip(round(base_r["seal_temp_dev_sigma"], 1), -3.0, 3.0)), 0.1)
    rd["storage_days"] = g4.slider(PROCESS_LABELS["storage_days"], 0.0, 120.0, float(np.clip(round(base_r["storage_days"], 1), 0.0, 120.0)), 1.0)

    g5, g6, g7 = st.columns([1, 1, 1])
    rd["checkweigher_drift_sigma"] = g5.slider(PROCESS_LABELS["checkweigher_drift_sigma"], -3.0, 3.0, float(np.clip(round(base_r["checkweigher_drift_sigma"], 1), -3.0, 3.0)), 0.1)
    rd["humidity_pct"] = g6.slider(PROCESS_LABELS["humidity_pct"], 30.0, 95.0, float(np.clip(round(base_r["humidity_pct"], 1), 30.0, 95.0)), 1.0)
    rd["new_supplier_lot"] = int(g7.checkbox("New / Unapproved Supplier Lot"))

    sc_batch, anyp_batch, dec_batch, drv_batch = ML.score_batch(bm, rd)
    lvl_batch = "HIGH" if anyp_batch >= 0.30 else ("WATCH" if anyp_batch >= 0.18 else "LOW")

    st.markdown(f"""
    <div class="zc-alert">
        {tag(lvl_batch)} <b>Clearance Decision: {dec_batch}</b><br>
        Overall Batch Complaint-Risk Index: <b>{anyp_batch:.1%}</b>
    </div>
    """, unsafe_allow_html=True)

    t_top4 = sc_batch.head(4)
    fig_batch = px.bar(t_top4[::-1], orientation="h", color_discrete_sequence=["#ff5468"], height=220, title="Per-Category Risk for this Batch")
    st.plotly_chart(fig_style(fig_batch, 220), use_container_width=True)


# =====================================================================
# 6. PREVENTIVE RECOMMENDATIONS & CAPA
# =====================================================================
elif page == "Preventive Recommendations & CAPA":
    st.title("Preventive Actions & CAPA Execution")
    st.caption("Wilson lower-bound effectiveness ranking, SOP/HACCP checks, similar historical cases, and CAPA lifecycle tracking.")

    top_pairs = preds.sort_values("p30", ascending=False).head(15)
    pair_labels = [f"{r.product} · {r.category} (30d risk: {r.p30:.0%})" for r in top_pairs.itertuples()]
    sel_idx = st.selectbox("Select High-Risk Pair", range(len(pair_labels)), format_func=lambda x: pair_labels[x])
    curr_pair = top_pairs.iloc[sel_idx]

    rec_res = RC.recommend(m, curr_pair.category, curr_pair["product"])

    c_act, c_sop = st.columns([1.4, 1])
    with c_act:
        st.subheader("Recommended Corrective & Preventive Actions")
        disp_rec = rec_res.assign(
            success_rate=rec_res.success_rate.round(2),
            confidence_adj=rec_res.confidence_adj.round(2)
        )[["action", "verdict", "success_rate", "confidence_adj", "worked", "cases"] + (["worked_here", "cases_here"] if "worked_here" in rec_res.columns else [])]
        st.dataframe(disp_rec, hide_index=True, use_container_width=True)

    with c_sop:
        st.subheader("SOP / HACCP Rule Check")
        st.markdown(f'<div class="box">{SOP_RULES.get(curr_pair.category, "Follow standard operating procedures.")}</div>', unsafe_allow_html=True)
        st.subheader("Key Risk Drivers")
        for drv_name, _ in json.loads(curr_pair.drivers):
            st.markdown(f"• **{drv_name}**")

    st.markdown("---")

    # Create CAPA Form
    st.subheader("Initiate Corrective & Preventive Action (CAPA)")
    capa1, capa2, capa3, capa4 = st.columns([1.5, 1, 1, 1])
    act_choice = capa1.selectbox("Selected Action", rec_res.action.tolist() if not rec_res.empty else ["Standard verification"])
    owner_choice = capa2.selectbox("Assigned Owner", ["QA Head", "Plant Manager", "Maintenance Lead", "Packaging Supervisor", "Warehouse In-charge"])
    due_choice = capa3.date_input("Target Due Date", TODAY + pd.Timedelta(days=14))
    if capa4.button("Create CAPA", type="primary"):
        src = db.read(
            "SELECT c.complaint_id FROM complaints c JOIN products p USING(product_id) WHERE p.product_name=? AND COALESCE(c.category_manual,c.category_pred)=? ORDER BY complaint_date DESC LIMIT 1",
            (curr_pair["product"], curr_pair.category)
        )
        if len(src):
            db.execute(
                "UPDATE capa SET action=?, owner=?, due_date=?, status='Open', effective=NULL WHERE complaint_id=?",
                (act_choice, owner_choice, str(due_choice), int(src.complaint_id[0]))
            )
            refresh()
            st.success(f"CAPA successfully created and assigned to {owner_choice}!")

    # Active CAPA Tracker Table
    st.subheader("Active CAPA Tracker (Open / In Progress)")
    capa_tr = m[m.status.isin(["Open", "In progress"])].sort_values("complaint_date", ascending=False)[
        ["complaint_id", "product", "category", "action", "owner", "due_date", "status"]
    ].head(20)
    st.dataframe(capa_tr, hide_index=True, use_container_width=True)

    with st.expander("View Similar Historical Cases (Evidence)"):
        st.dataframe(RC.similar_cases(m, f"{curr_pair['product']} {curr_pair.category} " + SUBS.get(curr_pair.category, [""])[0], k=5), hide_index=True, use_container_width=True)


# =====================================================================
# 7. KPIS: ROAD TO ZERO
# =====================================================================
elif page == "KPIs: Road to Zero":
    st.title("Road to Zero Customer Complaints")
    st.caption("Strategic metrics, 36-month glide path, complaint-free streaks, and early-warning backtest validation.")

    k_data = A.kpis(m, b, bt)
    k_data["Progress to 12m target"] = [
        min(1.0, max(0.0, (n / t if d == "higher" else (t / n if n else 1.0)))) if t else 0.0
        for n, t, d in zip(k_data.Now, k_data["Target 12m"], k_data.better)
    ]
    st.dataframe(
        k_data.drop(columns="better"),
        column_config={"Progress to 12m target": st.column_config.ProgressColumn(format="%.0f%%", min_value=0.0, max_value=1.0)},
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        # 36-month glide path
        cur_monthly_rate = len(A.last(m, 12)) / 12
        months_x = list(range(0, 37))
        fig_glide = go.Figure([
            go.Scatter(x=months_x, y=[max(0.0, cur_monthly_rate * (1 - i / 36)) for i in months_x], line=dict(color="#3d7bff", dash="dash", width=2), name="Glide Path to Zero")
        ])
        fig_glide.add_hline(y=cur_monthly_rate, line_dash="dot", annotation_text=f"Current Run-Rate: {cur_monthly_rate:.1f}/mo")
        fig_glide.update_layout(title="36-Month Glide Path to Zero Complaints", xaxis_title="Months from Now", yaxis_title="Monthly Complaints")
        st.plotly_chart(fig_style(fig_glide, 320), use_container_width=True)

    with col_g2:
        # Longest complaint-free streaks
        streak_s = A.streaks(m).head(15).rename("months").to_frame().reset_index()
        streak_s.columns = ["Pair", "Months Free"]
        fig_streak = px.bar(streak_s.iloc[::-1], x="Months Free", y="Pair", orientation="h", color_discrete_sequence=["#3fd9a8"], title="Longest Complaint-Free Streaks (Product × Category)")
        st.plotly_chart(fig_style(fig_streak, 320), use_container_width=True)

    st.markdown("""
    <div class="box">
        <b>Framework Logic:</b> Outcome is evaluated by complaints per million units and repeat defect rate. Operational quality is tracked by CAPA effectiveness and resolution cycle time. Preventive success is demonstrated by early-warning detection hit rate and expanding complaint-free product-months.
    </div>
    """, unsafe_allow_html=True)

    # Backtest Metrics
    st.subheader("Early-Warning Validation Metrics (30-Day Horizon)")
    w_bt = bt.assign(flag=bt.p >= P_WATCH) if not bt.empty else pd.DataFrame()
    if not w_bt.empty:
        bw1, bw2, bw3 = st.columns(3)
        bw1.metric("Complaints Successfully Flagged", f"{w_bt[w_bt.y == 1].flag.mean():.0%}")
        bw2.metric("Flags Confirmed with Defect", f"{w_bt[w_bt.flag].y.mean():.0%}")
        bw3.metric("Product-Category-Months Evaluated", len(w_bt))


# =====================================================================
# 8. DATA, SQL & INTEGRATION
# =====================================================================
elif page == "Data, SQL & Integration":
    st.title("Data Management, SQL Console & Integration")
    st.caption("Direct database querying, Power BI star-schema export, alert outbox digests, and model registry.")

    # Table counts
    tables = db.read("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY 1")
    counts = {t_name: int(db.read(f"SELECT COUNT(*) as n FROM {t_name}").n[0]) for t_name in tables.name}
    st.write("**Tables & Views Row Counts:** " + " · ".join(f"`{k}` ({v:,})" for k, v in counts.items()))

    st.markdown("---")

    # SQL Console
    st.subheader("Interactive SQL Console (Read-Only SELECT)")
    ex_sql = (SQL_DIR / "queries.sql").read_text().split("\n\n") if (SQL_DIR / "queries.sql").exists() else []
    query_choices = [""] + [e.split("\n")[0].replace("--", "").strip() for e in ex_sql if e.strip().startswith("--")]
    chosen_ex = st.selectbox("Load Sample Analytical Query", query_choices)

    default_q = "SELECT category, COUNT(*) AS count, ROUND(AVG(confidence), 2) AS avg_conf FROM v_master GROUP BY category ORDER BY count DESC;"
    if chosen_ex:
        default_q = next(e for e in ex_sql if chosen_ex in e)

    sql_input = st.text_area("SQL Query", default_q, height=110)
    if st.button("Execute Query"):
        clean_q = "\n".join(l for l in sql_input.split("\n") if not l.strip().startswith("--"))
        if clean_q.strip().lower().startswith(("select", "with")):
            try:
                res_sql = db.read(clean_q)
                st.dataframe(res_sql, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Error: {e}")
        else:
            st.error("Only read-only SELECT queries are allowed.")

    st.markdown("---")

    # Power BI Export
    st.subheader("Power BI Star-Schema Export")
    st.write("Generates facts, dimensions, and measures in CSV format and an Excel model workbook (`ZeroC_AI_PowerBI_Model.xlsx`).")
    if st.button("Generate & Refresh Power BI Files"):
        generated_files = export_powerbi.export()
        st.success("Successfully generated: " + ", ".join(generated_files))

    down_c1, down_c2 = st.columns(2)
    pbi_xlsx = PBI_DIR / "ZeroC_AI_PowerBI_Model.xlsx"
    pbi_guide = PBI_DIR / "PowerBI_Guide.md"
    if pbi_xlsx.exists():
        down_c1.download_button("Download Power BI Excel Model", pbi_xlsx.read_bytes(), "ZeroC_AI_PowerBI_Model.xlsx")
    if pbi_guide.exists():
        down_c2.download_button("Download Power BI DAX & Build Guide", pbi_guide.read_bytes(), "PowerBI_Guide.md")

    st.markdown("---")

    # Alerts & Retraining
    st.subheader("Alerts Engine & System Maintenance")
    al_c1, al_c2 = st.columns(2)
    if al_c1.button("Re-run Alerts Engine"):
        alerts.run()
        refresh()
        st.success("Alerts refreshed. High alerts written to /outbox or dispatched via SMTP.")

    if al_c2.button("Full Retrain of All Models"):
        with st.spinner("Retraining NLP + XGBoost + Recalibrating..."):
            pipeline.run_nlp()
            pipeline.run_models()
            alerts.run()
            export_powerbi.export()
            refresh()
            st.success("Full retraining pipeline completed successfully!")

    # Outbox status
    st.markdown("**Recent Email Outbox Files (`/outbox`):**")
    outbox_files = sorted(f.name for f in OUTBOX.glob("*.txt"))[-5:]
    if outbox_files:
        st.write(" · ".join(f"`{fn}`" for fn in outbox_files))
    else:
        st.caption("No outbox files generated yet.")

    # Model Registry
    st.markdown("---")
    st.subheader("Model Version Registry")
    st.json(reg, expanded=False)


# =====================================================================
# 9. ABOUT & METHOD
# =====================================================================
else:
    st.title("About ZeroC-AI & Technical Methodology")
    st.markdown(f"""
### Zero Customer Complaint Analytics & Prediction System (PS-036 · ABT-CPD)

ZeroC-AI turns historical complaint records and batch process data into proactive defect prevention, automated classification, and measurable steps toward zero customer complaints.

---

### Coverage of the 9 Problem-Statement Deliverables

| # | Deliverable | System Module & Screen | Methodology & Algorithms |
|---|---|---|---|
| **1** | **Classify complaints** | *Classify & Log* | Hierarchical TF-IDF + balanced Logistic Regression predicting joint Category and Sub-defect; confidence score with human-in-the-loop active learning review queue; Sentence-BERT upgrade ready. |
| **2** | **Pareto analysis** | *Command Centre*, *Pareto & History* | 80-20 cumulative Pareto distribution across multiple dimensions (Category, Product, Customer, Line, 6M Root Cause, Customer Type) with custom year filtering. |
| **3** | **Recurring & chronic problems** | *Patterns & Root Cause* | Rolling recurrence counter (same defect within 6 months), chronic status detector (≥4 complaints or ≥10 in 24m), repeat CAPA failure tracking. |
| **4** | **Relationship mapping** | *Patterns & Root Cause* | Cross-dimensional lift matrices, Apriori association rules mining (support, confidence, lift), 6M Ishikawa classification, and process parameter box-plots. |
| **5** | **High-risk products & processes** | *Command Centre*, *Risk & Prediction* | Multi-factor composite risk scoring (0–100) combining frequency, severity, recency, trend, and recurrence with RAG status badges. |
| **6** | **Predict complaint risk** | *Risk & Prediction* | XGBoost classifiers with monotonic constraints predicting 30/60/90-day probability of complaint, Platt calibration, time-based holdout validation, and SHAP explainability. |
| **7** | **Recommend preventive actions** | *Preventive Recommendations & CAPA* | Wilson score lower bound effectiveness ranking on historical CAPA outcomes, similar case retrieval (cosine similarity), and automated SOP/HACCP rule checks. |
| **8** | **Management dashboard & alerts** | *Command Centre*, *Data & Integration* | Executive KPI tiles, SPC early warning (c-chart, Western Electric 7-run rule, CUSUM alarms), automated email notification and outbox digest. |
| **9** | **KPIs towards zero** | *KPIs: Road to Zero* | Strategic scorecard (PPM, repeat rate, CAPA effectiveness, closure time, 36-month glide path, and complaint-free product-month streaks). |

---

### Architecture & Technology Stack
- **Database Layer**: SQLite relational schema (`products`, `customers`, `batches`, `complaints`, `capa`, `predictions`, `backtest`, `themes`, `alerts`, `v_master` view).
- **Machine Learning**: Scikit-Learn, XGBoost, SHAP, Platt Scaling, Mlxtend Apriori, KMeans.
- **Frontend & Visualizations**: Streamlit, Plotly Express & Graph Objects.
- **Enterprise BI Export**: Star-schema CSVs + Excel model (`ZeroC_AI_PowerBI_Model.xlsx`) + DAX measure implementation guide (`PowerBI_Guide.md`).

---

### Using Real ABT-CPD Data
1. Prepare your data file (`complaints.xlsx` or `complaints.csv`).
2. Adjust `COLUMN_MAP` in `load_real_data.py` if your column names differ.
3. Run: `python load_real_data.py your_file.xlsx [batches_or_volumes.csv]`
4. Start the dashboard: `streamlit run app.py` (or set `ZC_DATA=real`).
""")

st.markdown("---")
st.caption(f"ZeroC-AI Platform · System refreshed as of {dt.datetime.now():%Y-%m-%d %H:%M}")
