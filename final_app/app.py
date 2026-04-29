import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import requests
from streamlit_searchbox import st_searchbox
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from final_app.ensemble_model import (
    DeployableStackedEnsemble,
    EnsembleModel,
    CathyLevelsMultiLabelEncoder,
    RandomSampleImputer,
    FeatureEngineer,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "pred_similar_df" not in st.session_state:
    st.session_state.pred_similar_df = None
if "pred_x_input" not in st.session_state:
    st.session_state.pred_x_input = None
if "_lgbm_price_cache" not in st.session_state:
    st.session_state._lgbm_price_cache = None
if "_xgb_price_cache" not in st.session_state:
    st.session_state._xgb_price_cache = None
if "_ensemble_price_cache" not in st.session_state:
    st.session_state._ensemble_price_cache = None
if "_addr_suggestions" not in st.session_state:
    st.session_state._addr_suggestions = []
if "_addr_sugg_ver" not in st.session_state:
    st.session_state._addr_sugg_ver = 0
if "_addr_last_query" not in st.session_state:
    st.session_state._addr_last_query = ""
if "_address_geo" not in st.session_state:
    st.session_state._address_geo = {}
if "_confirmed_place_id" not in st.session_state:
    st.session_state._confirmed_place_id = ""
if "_addr_api_error" not in st.session_state:
    st.session_state._addr_api_error = ""
if "predict_step" not in st.session_state:
    st.session_state.predict_step = "address"
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0
if "wizard_inputs" not in st.session_state:
    st.session_state.wizard_inputs = {}
if "scatter_x_feat" not in st.session_state:
    st.session_state.scatter_x_feat = "Living Area"
if "pred_results_html" not in st.session_state:
    st.session_state.pred_results_html = None
if "pred_chart_html" not in st.session_state:
    st.session_state.pred_chart_html = None

# ── Design system CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:           #F5F6FA;
  --surface:      #FFFFFF;
  --primary:      #1a2e44;
  --accent:       #2563eb;
  --accent-light: rgba(37,99,235,0.08);
  --accent-mid:   rgba(37,99,235,0.18);
  --ensemble:     #10b981;
  --ensemble-dk:  #059669;
  --text-1:       #111827;
  --text-2:       #4b5563;
  --text-3:       #9ca3af;
  --border:       #e5e7eb;
  --border-focus: #2563eb;
  --success:      #16a34a;
  --radius-sm:    6px;
  --radius-md:    10px;
  --radius-lg:    14px;
}

html, body {
  background-color: var(--bg) !important;
  color: var(--text-1) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
}

.stApp { background-color: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
  padding: 0 2.5rem 4rem 2.5rem !important;
  max-width: 1180px !important;
}

/* ══ NAV BRAND ════════════════════════════════════════════════════════════ */
.nav-brand {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--primary);
  letter-spacing: -0.025em;
  margin: 0;
  padding: 0.5rem 0;
  line-height: 1;
}
.nav-brand span { color: var(--accent); }

/* ══ PANEL ════════════════════════════════════════════════════════════════ */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.75rem 2rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  max-width: 820px;
  margin-left: auto;
  margin-right: auto;
}
.panel-title {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-2);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 1.25rem;
}
.section-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.25rem 0;
}
.section-heading {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-2);
  margin: 0 0 0.9rem 0;
  padding-left: 0.6rem;
  border-left: 3px solid var(--accent);
}

/* ══ FORM INPUTS ══════════════════════════════════════════════════════════ */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  padding: 0.6rem 0.85rem !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 3px var(--accent-light) !important;
  background: var(--surface) !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  color: var(--text-1) !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p,
div[data-testid="stToggle"] label p,
div[data-testid="stCheckbox"] p {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.76rem !important;
  font-weight: 600 !important;
  color: var(--text-2) !important;
}
/* Expander header text */
div[data-testid="stExpander"] summary {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  padding-right: 1.5rem !important;
}
div[data-testid="stExpander"] summary p {
  margin: 0 !important;
  flex-grow: 1;
}
div[data-testid="stExpander"] * {
  color: var(--text-2) !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary:hover,
div[data-testid="stExpander"] summary:focus,
div[data-testid="stExpander"] summary:active,
div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] details[open] summary,
div[data-testid="stExpander"] details[open] summary:hover {
  background-color: transparent !important;
  background: transparent !important;
  color: var(--text-2) !important;
  outline: none !important;
  box-shadow: none !important;
}
div[data-testid="stExpander"] summary:hover {
  color: var(--accent) !important;
}
div[data-testid="stExpander"] summary:hover * {
  color: var(--accent) !important;
}

[data-baseweb="popover"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}

/* Radio buttons */
div[data-testid="stRadio"] > div {
  gap: 0.5rem !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  align-items: center !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] {
  display: flex !important;
  align-items: center !important;
  flex-shrink: 0 !important;
  gap: 0.4rem !important;
  background: transparent !important;
  border: 1px solid #d1d5db !important;
  border-radius: 999px !important;
  padding: 0.35rem 1rem !important;
  cursor: pointer !important;
  transition: border-color 0.15s !important;
  box-sizing: border-box !important;
  min-height: 32px !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
  width: 16px !important;
  height: 16px !important;
  min-width: 16px !important;
  min-height: 16px !important;
  max-width: 16px !important;
  max-height: 16px !important;
  margin: 0 !important;
  padding: 0 !important;
  flex-shrink: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  border-color: #d1d5db !important;
  box-sizing: border-box !important;
  position: relative !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child > div {
  width: 8px !important;
  height: 8px !important;
  min-width: 8px !important;
  min-height: 8px !important;
  max-width: 8px !important;
  max-height: 8px !important;
  margin: 0 !important;
  padding: 0 !important;
  position: absolute !important;
  top: 50% !important;
  left: 50% !important;
  transform: translate(-50%, -50%) !important;
  border-radius: 50% !important;
  background-color: #9ca3af !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"]:has([aria-checked="true"]) > div:first-child > div {
  background-color: var(--accent) !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] label,
div[data-testid="stRadio"] [data-baseweb="radio"] label *,
div[data-testid="stRadio"] [data-baseweb="radio"] span,
div[data-testid="stRadio"] [data-baseweb="radio"] p {
  color: var(--text-2) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 400 !important;
  line-height: 1.3 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  margin: 0 !important;
  padding: 0 !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"]:has([aria-checked="true"]) {
  border-color: var(--accent) !important;
  background: transparent !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"]:has([aria-checked="true"]) label,
div[data-testid="stRadio"] [data-baseweb="radio"]:has([aria-checked="true"]) label *,
div[data-testid="stRadio"] [data-baseweb="radio"]:has([aria-checked="true"]) span,
div[data-testid="stRadio"] [data-baseweb="radio"]:has([aria-checked="true"]) p {
  color: var(--accent) !important;
  font-weight: 600 !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"]:hover {
  border-color: var(--accent) !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] svg {
  color: #9ca3af !important;
  fill: #9ca3af !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"]:has([aria-checked="true"]) svg {
  color: var(--accent) !important;
  fill: var(--accent) !important;
}

/* ══ BUTTONS ══════════════════════════════════════════════════════════════ */
[data-testid="baseButton-primary"] {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  padding: 0.7rem 2rem !important;
  border-radius: 8px !important;
  width: 100% !important;
  transition: background 0.15s, transform 0.12s, box-shadow 0.15s !important;
  cursor: pointer !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
[data-testid="baseButton-primary"]:hover {
  background: #1d4ed8 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: #6b7280 !important;
  font-weight: 500 !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
  background: transparent !important;
  color: #374151 !important;
}

/* ══ RESULT PANEL ═════════════════════════════════════════════════════════ */
.result-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: var(--radius-lg);
  padding: 2.25rem 2.5rem;
  margin-top: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2rem;
  box-shadow: 0 4px 20px rgba(37,99,235,0.10);
  text-align: center;
}
.result-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent);
  margin-bottom: 0.35rem;
}
.result-price {
  font-size: 3.2rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1;
  margin-bottom: 0.5rem;
  color: var(--primary);
  font-variant-numeric: tabular-nums;
  text-align: center;
}
.result-note {
  font-size: 0.76rem;
  color: var(--text-3);
}

/* ══ ENSEMBLE HERO CARD ══════════════════════════════════════════════════ */
.ensemble-hero {
  background: linear-gradient(135deg, #1a2e44 0%, #1e3a5f 100%);
  border-top: 4px solid var(--ensemble);
  border-radius: var(--radius-lg);
  padding: 2.25rem 2.5rem;
  margin-top: 1.5rem;
  text-align: center;
  box-shadow: 0 4px 24px rgba(0,0,0,0.12);
}
.ensemble-hero-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: rgba(255,255,255,0.6);
  margin-bottom: 0.5rem;
}
.ensemble-hero-price {
  font-size: 3.8rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1;
  color: #ffffff;
  font-variant-numeric: tabular-nums;
  margin-bottom: 0.55rem;
}
.ensemble-hero-note {
  font-size: 0.78rem;
  color: rgba(255,255,255,0.45);
}
.model-agreement-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent);
  margin: 1.5rem 0 0.75rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--accent-light);
}

/* ══ METRIC CARDS ═════════════════════════════════════════════════════════ */
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 3px solid var(--accent);
  border-radius: var(--radius-md);
  padding: 1.25rem 1.5rem;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--primary);
  letter-spacing: -0.03em;
}
.metric-label {
  font-size: 0.74rem;
  font-weight: 500;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.2rem;
}

/* ══ HERO ═════════════════════════════════════════════════════════════════ */
.hero {
  background: linear-gradient(135deg, #1a2e44 0%, #1e3a5f 60%, #1e3058 100%);
  border-radius: var(--radius-lg);
  padding: 4rem 3.5rem;
  margin-bottom: 2rem;
  color: #fff;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 2.5rem;
}
.hero-title {
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1.1;
  color: #fff;
  margin: 0 0 0.85rem 0;
}
.hero-sub {
  font-size: 1rem;
  font-weight: 400;
  color: rgba(255,255,255,0.62);
  max-width: 480px;
  line-height: 1.7;
  margin: 0;
}
.hero-stats {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: var(--radius-lg);
  padding: 1.75rem 2rem;
  min-width: 200px;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}
.hero-stat-value {
  font-size: 1.8rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.03em;
  line-height: 1;
}
.hero-stat-label {
  font-size: 0.72rem;
  color: rgba(255,255,255,0.45);
  font-weight: 500;
  margin-top: 0.2rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ══ INFO STRIP ══════════════════════════════════════════════════════════ */
.info-strip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.75rem 2rem;
  margin-bottom: 2rem;
}
.info-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
}
.info-item { display: flex; flex-direction: column; gap: 0.3rem; }
.info-heading { font-size: 0.875rem; font-weight: 600; color: var(--text-1); }
.info-body { font-size: 0.8rem; color: var(--text-2); line-height: 1.55; }

/* ══ MODEL PAGE ══════════════════════════════════════════════════════════ */
.feature-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
}
.feature-row:last-child { border-bottom: none; }
.feature-dot { width: 3px; height: 16px; border-radius: 2px; background: var(--accent); flex-shrink: 0; }
.feature-name { font-size: 0.84rem; font-weight: 600; color: var(--text-1); width: 160px; flex-shrink: 0; }
.feature-desc { font-size: 0.8rem; color: var(--text-2); line-height: 1.5; }

/* ══ PAGE TITLE ══════════════════════════════════════════════════════════ */
.page-title {
  font-size: 1.7rem;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.04em;
  margin: 0 0 0.35rem 0;
}
.page-sub {
  font-size: 0.875rem;
  color: var(--text-2);
  margin: 0 0 2rem 0;
  line-height: 1.65;
}

/* ══ ALERTS ══════════════════════════════════════════════════════════════ */
div[data-testid="stAlert"] {
  background: #fef3cd !important;
  border: 1px solid #fcd34d !important;
  border-radius: var(--radius-md) !important;
}

/* ══ FORM SECTION HEADINGS ═══════════════════════════════════════════════ */
.form-section-heading {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 1.75rem 0 1rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--accent-light);
}
div[data-testid="stForm"] div[data-testid="stExpander"] {
  margin-top: 0.75rem;
}

/* ══ DUAL RESULT CARD ════════════════════════════════════════════════════ */
.dual-result {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  margin-top: 0.75rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  overflow: hidden;
}
.dual-result-body {
  display: grid;
  grid-template-columns: 1fr 80px 1fr;
  align-items: center;
  justify-items: center;
}
.dual-result-side {
  padding: 2rem 1.5rem;
  text-align: center;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.dual-result-divider {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}
.vs-badge {
  background: var(--accent-mid);
  color: var(--accent);
  font-size: 0.7rem;
  font-weight: 700;
  padding: 0.3rem 0.6rem;
  border-radius: 999px;
  letter-spacing: 0.06em;
}
.divider-line { width: 1px; height: 60px; background: var(--border); }
.dual-result-footer {
  border-top: 1px solid var(--border);
  padding: 0.85rem 2.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.diff-badge {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 0.25rem 0.8rem;
  border-radius: 999px;
}
.diff-badge.green { background: #dcfce7; color: #16a34a; }
.diff-badge.amber { background: #fef9c3; color: #ca8a04; }
.diff-badge.red   { background: #fee2e2; color: #dc2626; }
.diff-label { font-size: 0.8rem; color: var(--text-2); }

/* ══ CSS COMPARISON BARS ═════════════════════════════════════════════════ */
.compare-bars {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem 2rem;
  margin-top: 1rem;
}
.compare-bar-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.9rem;
}
.compare-bar-row:last-child { margin-bottom: 0; }
.compare-bar-label {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-2);
  width: 140px;
  flex-shrink: 0;
}
.compare-bar-track {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.compare-bar-fill { height: 10px; border-radius: 999px; }
.compare-bar-fill.ensemble { background: #10b981; }
.compare-bar-fill.lgbm     { background: #2563eb; }
.compare-bar-fill.xgb      { background: #93c5fd; }
.compare-bar-value {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-1);
  white-space: nowrap;
}

/* ══ SIMILAR PROPERTY CARDS ══════════════════════════════════════════════ */
.sim-cards-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-top: 0.5rem;
}
.sim-card {
  background: #f8f9fb;
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.75rem 1.25rem 1.75rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  display: flex;
  flex-direction: column;
  gap: 0;
}
.sim-card-price {
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.045em;
  line-height: 1;
  color: var(--primary);
  font-variant-numeric: tabular-nums;
  margin-bottom: 0.7rem;
}
.sim-card-stats {
  display: flex;
  gap: 1.2rem;
  align-items: center;
  margin-bottom: 0.85rem;
}
.sim-stat {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.875rem;
  color: var(--text-3);
  font-weight: 500;
}
.sim-stat-val { color: var(--text-1); font-weight: 600; }
.sim-card-footer { display: flex; align-items: center; gap: 0.5rem; }
.sim-footer-chip {
  border-radius: var(--radius-sm);
  padding: 0.18rem 0.55rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.age-new { background: #dcfce7; color: #15803d; }
.age-mid { background: #fef3c7; color: #b45309; }
.age-old { background: #f1f5f9; color: #64748b; }
.sim-chip-county {
  background: var(--bg);
  color: var(--text-3);
  border-radius: var(--radius-sm);
  padding: 0.18rem 0.55rem;
  font-size: 0.72rem;
  font-weight: 500;
}
.sim-delta {
  font-size: 0.7rem;
  font-weight: 600;
  border-radius: 3px;
  padding: 0.05rem 0.3rem;
  margin-left: 0.2rem;
  vertical-align: middle;
}
.sim-delta.pos { background: #dcfce7; color: #15803d; }
.sim-delta.neg { background: #fee2e2; color: #b91c1c; }

/* ══ COMPARISON PAGE ═════════════════════════════════════════════════════ */
.winner-strip {
  display: flex;
  align-items: center;
  gap: 1rem;
  background: rgba(16,185,129,0.08);
  border-radius: var(--radius-md);
  padding: 0.85rem 1.5rem;
  margin-bottom: 2rem;
  border-left: 4px solid var(--ensemble);
}
.winner-strip-label {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ensemble);
  white-space: nowrap;
}
.winner-strip-name {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-1);
}
.winner-strip-dot {
  color: var(--border);
  font-size: 1rem;
}
.winner-strip-stat {
  font-size: 0.82rem;
  color: var(--text-2);
  font-weight: 500;
}
.score-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}
.score-table thead th {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-2);
  padding: 0 0 0.6rem 0.5rem;
  border-bottom: 2px solid var(--border);
  text-align: left;
}
.score-table tbody tr { border-bottom: 1px solid var(--border); }
.score-table tbody tr:last-child { border-bottom: none; }
.score-table tbody td { padding: 0.5rem 0.5rem; color: var(--text-2); }
.score-table tbody td:first-child { color: var(--text-1); font-weight: 500; }
.score-table .row-winner td {
  background: rgba(16,185,129,0.06);
  color: var(--text-1);
  font-weight: 600;
}
.score-table .row-winner td:first-child {
  border-left: 3px solid var(--ensemble);
  padding-left: calc(0.5rem - 3px);
}
.score-val-neg { color: #ef4444 !important; }

/* ══ MODEL PAGE ══════════════════════════════════════════════════════════ */
.model-hero-banner {
  background: linear-gradient(135deg, #1a2e44 0%, #1e3a5f 100%);
  border-top: 4px solid var(--ensemble);
  border-radius: var(--radius-lg);
  padding: 1.75rem 2.25rem;
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  box-shadow: 0 4px 24px rgba(0,0,0,0.12);
}
.model-hero-left {}
.model-hero-name {
  font-size: 1.25rem;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: -0.01em;
  margin: 0 0 0.3rem 0;
}
.model-hero-type {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--ensemble);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 0;
}
.model-hero-stats {
  display: flex;
  gap: 1.5rem;
  align-items: center;
}
.model-hero-stat {
  text-align: center;
}
.model-hero-stat-val {
  font-size: 1.5rem;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: -0.02em;
  line-height: 1;
}
.model-hero-stat-lbl {
  font-size: 0.68rem;
  font-weight: 500;
  color: rgba(255,255,255,0.55);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.25rem;
}
.model-hero-divider {
  width: 1px;
  height: 2.5rem;
  background: rgba(255,255,255,0.15);
}
.model-section {
  margin-bottom: 2.25rem;
}
.model-section-title {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text-2);
  margin: 0 0 1rem 0;
  padding-bottom: 0.6rem;
  border-bottom: 2px solid var(--border);
}
.arch-row {
  display: flex;
  gap: 1.25rem;
  padding: 0.7rem 0;
  border-bottom: 1px solid var(--border);
  align-items: flex-start;
}
.arch-row:last-child { border-bottom: none; }
.arch-term {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-1);
  min-width: 130px;
  flex-shrink: 0;
}
.arch-def {
  font-size: 0.8rem;
  color: var(--text-2);
  line-height: 1.6;
}
.feat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.6rem;
}
.feat-chip {
  padding: 0.7rem 0.9rem;
  border-left: 3px solid var(--accent);
  background: var(--bg);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.feat-chip-name {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 0.2rem;
}
.feat-chip-desc {
  font-size: 0.74rem;
  color: var(--text-2);
  line-height: 1.5;
}
.r2-callout {
  border-left: 3px solid var(--ensemble);
  padding: 1rem 1.15rem;
  background: rgba(16,185,129,0.06);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}

/* ══ ADDRESS LOOKUP ══════════════════════════════════════════════════════ */
.addr-confirmed {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  background: rgba(16, 185, 129, 0.06);
  border: 1px solid rgba(16, 185, 129, 0.25);
  border-radius: var(--radius-md);
  padding: 0.85rem 1rem;
  margin-top: 0.6rem;
}
.addr-check {
  color: #10b981;
  font-size: 1rem;
  font-weight: 700;
  flex-shrink: 0;
  margin-top: 0.05rem;
}
.addr-formatted {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 0.45rem;
  font-family: 'Inter', sans-serif;
}
.addr-chips {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.addr-chip {
  background: rgba(37, 99, 235, 0.07);
  border: 1px solid rgba(37, 99, 235, 0.18);
  border-radius: 999px;
  padding: 0.18rem 0.6rem;
  font-size: 0.71rem;
  font-weight: 600;
  color: var(--accent);
  font-family: 'Inter', sans-serif;
  white-space: nowrap;
}

/* ══ SEARCHBOX STYLING ════════════════════════════════════════════════════ */
/* Component renders inside an iframe — inner styles go via style_overrides prop.
   These selectors cover the outer iframe container only. */
div[data-testid="stCustomComponentV1"] {
  border-radius: var(--radius-md) !important;
}
iframe[title="streamlit_searchbox.searchbox"] {
  border-radius: var(--radius-md) !important;
  display: block;
}

/* ══ ADDRESS BANNER (in wizard steps) ════════════════════════════════════ */
.addr-banner {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  background: rgba(16, 185, 129, 0.06);
  border: 1px solid rgba(16, 185, 129, 0.22);
  border-radius: var(--radius-md);
  padding: 0.6rem 0.85rem;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
}
.addr-banner-addr {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 0.3rem;
}
.addr-api-error {
  background: rgba(220,38,38,0.06);
  border: 1px solid rgba(220,38,38,0.22);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.75rem;
  font-size: 0.76rem;
  color: #dc2626;
  font-family: 'Inter', sans-serif;
  margin-top: 0.4rem;
}

/* ══ WIZARD PROGRESS ═════════════════════════════════════════════════════ */
.wizard-progress {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 0;
  margin-bottom: 1.25rem;
}
.wizard-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
}
.wizard-connector {
  height: 2px;
  width: 2.5rem;
  background: var(--border);
  margin: 0 0.15rem;
  margin-top: 0.9rem;
  flex-shrink: 0;
  border-radius: 1px;
  transition: background 0.2s;
}
.wizard-connector.done { background: var(--accent); }
.wz-num {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 2px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  color: var(--text-3);
  background: transparent;
  transition: background 0.2s, border-color 0.2s, color 0.2s;
}
.wizard-step.done .wz-num,
.wizard-step.active .wz-num {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.wz-lbl {
  font-size: 0.67rem;
  font-weight: 500;
  color: var(--text-3);
  font-family: 'Inter', sans-serif;
  white-space: nowrap;
}
.wizard-step.done .wz-lbl,
.wizard-step.active .wz-lbl {
  color: var(--text-2);
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def nav():
    brand_col, h_col, pr_col, m_col, c_col = st.columns([2, 1, 1, 1, 1])
    with brand_col:
        st.markdown('<p class="nav-brand"> <span></span></p>', unsafe_allow_html=True)
    with h_col:
        if st.button("Home", key="nav_home", type="secondary"):
            st.session_state.page = "Home"; st.rerun()
    with pr_col:
        if st.button("Predict", key="nav_predict", type="secondary"):
            st.session_state.page = "Predict"
            st.session_state.predict_step = "address"
            st.session_state.wizard_step = 0
            st.session_state.wizard_inputs = {}
            st.session_state.pop("addr_searchbox", None)
            st.rerun()
    with m_col:
        if st.button("Model", key="nav_model", type="secondary"):
            st.session_state.page = "Model"; st.rerun()
    with c_col:
        if st.button("Comparison", key="nav_comparison", type="secondary"):
            st.session_state.page = "Comparison"; st.rerun()


def panel(title: str):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)


def _delta_chip(delta, fmt_int=True) -> str:
    """Return a colored +/- chip HTML span, or empty string if delta is 0."""
    if delta == 0:
        return ""
    cls = "pos" if delta > 0 else "neg"
    label = f"+{delta}" if delta > 0 else str(delta)
    if not fmt_int:
        label = f"+{delta}%" if delta > 0 else f"{delta}%"
    return f'<span class="sim-delta {cls}">{label}</span>'


def _sim_property_card(row, subj_beds=None, subj_baths=None, subj_sqft=None) -> str:
    """HTML for one similar-property card."""
    import datetime
    price     = f"${int(row.get('ClosePrice', 0) or 0):,}"
    sqft_raw  = int(row.get('LivingArea', 0) or 0)
    sqft      = f"{sqft_raw:,}"
    beds      = int(row.get('BedroomsTotal', 0) or 0)
    baths     = int(row.get('BathroomsTotalInteger', 0) or 0)
    year      = int(row.get('YearBuilt', 0) or 0)
    county    = str(row.get('CountyOrParish', 'N/A'))

    bed_chip  = _delta_chip(beds - subj_beds) if subj_beds is not None else ""
    bath_chip = _delta_chip(baths - subj_baths) if subj_baths is not None else ""
    if subj_sqft:
        sqft_pct  = round((sqft_raw - subj_sqft) / subj_sqft * 100)
        sqft_chip = _delta_chip(sqft_pct, fmt_int=False)
    else:
        sqft_chip = ""

    age = datetime.datetime.now().year - year if year else 999
    if age <= 5:
        age_class, age_label = "age-new", f"Built {year} · New"
    elif age <= 20:
        age_class, age_label = "age-mid", f"Built {year}"
    else:
        age_class, age_label = "age-old", f"Built {year}"

    return f"""
<div class="sim-card">
  <div class="sim-card-price">{price}</div>
  <div class="sim-card-stats">
    <div class="sim-stat"><span class="sim-stat-val">{beds}</span>&nbsp;bed{bed_chip}</div>
    <div class="sim-stat"><span class="sim-stat-val">{baths}</span>&nbsp;bath{bath_chip}</div>
    <div class="sim-stat"><span class="sim-stat-val">{sqft}</span>&nbsp;sqft{sqft_chip}</div>
  </div>
  <div class="sim-card-footer">
    <span class="sim-footer-chip {age_class}">{age_label}</span>
    <span class="sim-chip-county">{county}</span>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
_APP_DIR                    = os.path.dirname(os.path.abspath(__file__))
NATIVE_ARTIFACTS_DIR        = os.path.join(_APP_DIR, "model_artifacts")
NATIVE_BUNDLE_DIR           = os.path.join(_APP_DIR, "similarity_bundle")
STACKED_ENSEMBLE_MODEL_PATH = os.path.join(_APP_DIR, "stacked_ensemble_model.joblib")
LEGACY_ENSEMBLE_MODEL_PATH  = os.path.join(_APP_DIR, "ensemble_model.pkl")
SIMILAR_BUNDLE_PATH         = os.path.join(_APP_DIR, "similar_houses_bundle_new.pkl")

LGBM_OUTPUT_IS_LOG     = False
XGB_OUTPUT_IS_LOG      = False
ENSEMBLE_OUTPUT_IS_LOG = False


@st.cache_resource
def load_similar_bundle():
    """Load the similar-homes bundle. Prefers split native format; falls back to legacy pkl."""
    npy_path = os.path.join(NATIVE_BUNDLE_DIR, "sim_matrix.npy")
    if os.path.exists(npy_path):
        import numpy as _np
        import pandas as _pd
        meta = joblib.load(os.path.join(NATIVE_BUNDLE_DIR, "sim_meta.joblib"))
        return {
            "encoded_matrix": _np.load(npy_path),
            "reference_df":   _pd.read_parquet(os.path.join(NATIVE_BUNDLE_DIR, "sim_ref.parquet")),
            **meta,
        }

    bundle = joblib.load(SIMILAR_BUNDLE_PATH)
    if "CountyOrParish" not in bundle["reference_df"].columns:
        try:
            _xtrain = pd.read_csv("sold_data/test_data/X_train_imputed.csv", index_col=0)
            _postal_to_county = (
                _xtrain.groupby("PostalCode")["CountyOrParish"]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
            )
            ref = bundle["reference_df"]
            ref["CountyOrParish"] = ref["PostalCode"].map(_postal_to_county).fillna("N/A")
            bundle["reference_df"] = ref
            joblib.dump(bundle, SIMILAR_BUNDLE_PATH)
        except Exception:
            pass
    return bundle


@st.cache_resource
def load_ensemble_model():
    """Load the stacked ensemble. Prefers native artifacts; falls back to joblib then legacy pkl."""
    if os.path.exists(os.path.join(NATIVE_ARTIFACTS_DIR, "preprocessor.joblib")):
        ensemble = DeployableStackedEnsemble.load_native(NATIVE_ARTIFACTS_DIR)
        return ensemble, "stacked"

    if os.path.exists(STACKED_ENSEMBLE_MODEL_PATH):
        ensemble = joblib.load(STACKED_ENSEMBLE_MODEL_PATH)
        return ensemble, "stacked"

    sys.modules['__main__'].LevelsMultiLabelEncoder = CathyLevelsMultiLabelEncoder
    sys.modules['__main__'].RandomSampleImputer     = RandomSampleImputer
    sys.modules['__main__'].FeatureEngineer         = FeatureEngineer
    sys.modules['__main__'].EnsembleModel           = EnsembleModel
    with open(LEGACY_ENSEMBLE_MODEL_PATH, "rb") as f:
        ensemble = pickle.load(f)
    return ensemble, "legacy"


def run_prediction(model, X, output_is_log=False):
    pred = model.predict(X)[0]
    return float(np.expm1(pred)) if output_is_log else float(pred)


_SIM_BOOL_COLS   = ["NewConstructionYN", "AttachedGarageYN", "PoolPrivateYN", "ViewYN", "FireplaceYN"]
_LEVELS_ORDINAL  = {"One": 1.0, "Two": 2.0, "ThreeOrMore": 3.0, "MultiSplit": 1.5}


def _sim_get(user_row, key, default=np.nan):
    """Get a scalar from either a dict or a single-row DataFrame."""
    if isinstance(user_row, pd.DataFrame):
        return user_row[key].iloc[0] if key in user_row.columns else default
    return user_row.get(key, default)


def _encode_user_for_sim(user_row, bundle):
    """Encode a user input (dict or single-row DataFrame) into a scaled feature vector."""
    sim_feats  = bundle["similarity_features"]
    scaler     = bundle["scaler"]
    postal_med = bundle.get("postal_median_price", {})

    vals = {}
    for col in ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet",
                "YearBuilt", "Stories", "ParkingTotal", "GarageSpaces", "Latitude", "Longitude"]:
        try:
            vals[col] = float(_sim_get(user_row, col))
        except (TypeError, ValueError):
            vals[col] = np.nan

    postal = str(_sim_get(user_row, "PostalCode", ""))
    vals["postal_median_price"] = float(postal_med.get(postal, np.nan))

    for col in _SIM_BOOL_COLS:
        v = str(_sim_get(user_row, col, ""))
        vals[f"{col}_enc"] = 1.0 if v in ("True", "1") else (0.0 if v in ("False", "0") else np.nan)

    raw_lvl = str(_sim_get(user_row, "Levels", "")).split(",")[0].strip()
    vals["Levels_enc"] = _LEVELS_ORDINAL.get(raw_lvl, np.nan)

    vec = np.array([vals.get(f, np.nan) for f in sim_feats], dtype=float)

    # Distribution imputation: NaN → training mean (stored in scaler.mean_)
    nan_mask = np.isnan(vec)
    if nan_mask.any():
        vec[nan_mask] = scaler.mean_[nan_mask]

    return scaler.transform(vec.reshape(1, -1))[0]


def find_similar_houses(user_row, bundle, top_k=10):
    ref            = bundle["reference_df"]
    weights        = bundle.get("feature_weights")
    encoded_matrix = bundle.get("encoded_matrix")
    postal_med     = bundle.get("postal_median_price", {})

    # ── Fallback: old bundle format ────────────────────────────────────────────
    if encoded_matrix is None:
        sim_feats = bundle["similarity_features"]
        scaler    = bundle["scaler"]
        nn_model  = bundle["nn_model"]
        user_features = user_row[sim_feats].copy()
        user_features = user_features.fillna(ref[sim_feats].median(numeric_only=True))
        user_scaled   = scaler.transform(user_features)
        distances, indices = nn_model.kneighbors(user_scaled, n_neighbors=top_k)
        similar_df = ref.iloc[indices[0]].copy()
        similar_df["distance"] = distances[0]
        return similar_df

    postal     = str(_sim_get(user_row, "PostalCode", ""))
    county     = str(_sim_get(user_row, "CountyOrParish", ""))
    MIN_POOL   = top_k * 4
    ref_postal = ref["PostalCode"].astype(str)

    # Same-zip candidates
    postal_idx = np.where(ref_postal == postal)[0]

    # Price-band candidates: postal codes whose median is within ±20% of subject zip
    subj_median = postal_med.get(postal)
    if subj_median is not None and subj_median > 0:
        lo, hi    = subj_median * 0.80, subj_median * 1.20
        band_mask = ref_postal.map(postal_med).between(lo, hi, inclusive="both")
        band_idx  = np.where(band_mask.fillna(False))[0]
    else:
        band_idx  = np.array([], dtype=int)

    has_county  = "CountyOrParish" in ref.columns
    county_idx  = np.where(ref["CountyOrParish"].astype(str) == county)[0] if has_county else np.array([], dtype=int)
    band_county = np.intersect1d(band_idx, county_idx)

    # Priority: same zip → (price-band ∩ county) → county → price-band → all
    if len(postal_idx) >= MIN_POOL:
        search_idx = postal_idx
    elif len(band_county) >= MIN_POOL:
        search_idx = band_county
    elif len(county_idx) >= MIN_POOL:
        search_idx = county_idx
    elif len(band_idx) >= MIN_POOL:
        search_idx = band_idx
    else:
        search_idx = np.arange(len(ref))

    # ── Weighted Euclidean distance ────────────────────────────────────────────
    user_vec   = _encode_user_for_sim(user_row, bundle)
    search_mat = encoded_matrix[search_idx]
    diff       = (search_mat - user_vec) * weights
    dists      = np.linalg.norm(diff, axis=1)

    top_k_local = np.argpartition(dists, top_k)[:top_k]
    similar_df  = ref.iloc[search_idx[top_k_local]].copy()
    similar_df["distance"] = dists[top_k_local]
    return similar_df


def _maps_autocomplete(query, api_key):
    """Return (suggestions, error_msg). suggestions = [{description, place_id}]."""
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/autocomplete/json",
            params={"input": query, "types": "geocode", "components": "country:us", "key": api_key},
            headers={"Referer": "http://localhost:8501"},
            timeout=5,
        )
        data = resp.json()
        status = data.get("status")
        if status == "OK":
            return [
                {"description": p["description"], "place_id": p["place_id"]}
                for p in data.get("predictions", [])
            ], None
        if status == "REQUEST_DENIED":
            return [], data.get("error_message", "API key denied — check Google Cloud Console.")
        return [], None
    except Exception:
        pass
    return [], None


def _maps_place_details(place_id, api_key):
    """Fetch lat/lng, postal code, county, and formatted address for a place_id."""
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={
                "place_id": place_id,
                "fields": "geometry,address_components,formatted_address",
                "key": api_key,
            },
            headers={"Referer": "http://localhost:8501"},
            timeout=5,
        )
        data = resp.json()
        if data.get("status") == "OK":
            result = data["result"]
            loc = result.get("geometry", {}).get("location", {})
            postal_code = county = ""
            for comp in result.get("address_components", []):
                types = comp.get("types", [])
                if "postal_code" in types:
                    postal_code = comp["short_name"]
                if "administrative_area_level_2" in types:
                    county = comp["long_name"].replace(" County", "")
            return {
                "formatted_address": result.get("formatted_address", ""),
                "latitude": loc.get("lat"),
                "longitude": loc.get("lng"),
                "postal_code": postal_code,
                "county": county,
            }
    except Exception:
        pass
    return {}


try:
    similar_bundle = load_similar_bundle()
except Exception:
    similar_bundle = None

try:
    ensemble_model, ensemble_model_kind = load_ensemble_model()
    ensemble_loaded = True
except Exception as _ensemble_err:
    ensemble_model = None
    ensemble_model_kind = None
    ensemble_loaded = False
    _ensemble_err_str = str(_ensemble_err)

ensemble_display_name = "Stacked Ensemble" if ensemble_model_kind == "stacked" else "Gradient Boosting Ensemble"
ensemble_display_note = (
    "XGBoost + LightGBM · Gradient-boosted meta-model"
    if ensemble_model_kind == "stacked"
    else "XGBoost + LightGBM · Equal weights"
)
ensemble_info_body = (
    "XGBoost and LightGBM are stacked with a gradient-boosted meta-model, improving robustness across price segments and market regimes."
    if ensemble_model_kind == "stacked"
    else "XGBoost and LightGBM are combined into an equal-weight ensemble to improve prediction robustness across diverse property types and price ranges."
)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER NAV
# ─────────────────────────────────────────────────────────────────────────────
nav()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">
      <div class="hero-left">
        <h1 class="hero-title">House Price<br>Prediction</h1>
        <p class="hero-sub">
          Estimate property values using a stacked ensemble model (XGBoost + LightGBM)
          trained on regional MLS transaction data.
        </p>
      </div>
      <div class="hero-stats">
        <div class="hero-stat-item">
          <div class="hero-stat-value">0.88</div>
          <div class="hero-stat-label">Test R&sup2;</div>
        </div>
        <div class="hero-stat-item">
          <div class="hero-stat-value">17</div>
          <div class="hero-stat-label">Features</div>
        </div>
        <div class="hero-stat-item">
          <div class="hero-stat-value">68k+</div>
          <div class="hero-stat-label">Training rows</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, cta_col, _ = st.columns([2, 1, 2])
    with cta_col:
        if st.button("Estimate a Property  \u2192", key="cta_predict", type="primary"):
            st.session_state.page = "Predict"
            st.session_state.predict_step = "address"
            st.session_state.wizard_step = 0
            st.session_state.wizard_inputs = {}
            st.session_state.pop("addr_searchbox", None)
            st.rerun()

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-strip">
      <div class="info-grid">
        <div class="info-item">
          <div class="info-heading">Trained on Real Data</div>
          <div class="info-body">The model is trained on regional MLS transaction records, capturing real market dynamics across property types and locations.</div>
        </div>
        <div class="info-item">
          <div class="info-heading">Gradient Boosting Ensemble</div>
          <div class="info-body">%s</div>
        </div>
        <div class="info-item">
          <div class="info-heading">Location-Aware</div>
          <div class="info-body">Latitude, longitude, and postal code are included as features, allowing the model to capture geographic price variation.</div>
        </div>
      </div>
    </div>
    """ % ensemble_info_body, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Predict":

    if not ensemble_loaded:
        st.error(f"Ensemble model could not be loaded: {_ensemble_err_str}")
        st.stop()

    LEVELS_MAP = {"One Story": "One", "Two Story": "Two", "Three or More": "ThreeOrMore", "Multi-Split": "MultiSplit"}
    _LEVELS_DISPLAY = {v: k for k, v in LEVELS_MAP.items()}

    def _build_results_html(ensemble_price, lgbm_price, xgb_price):
        html = ""
        if ensemble_price is not None:
            html += f"""
            <div class="ensemble-hero">
              <div class="ensemble-hero-label">Ensemble Estimate</div>
              <div class="ensemble-hero-price">${ensemble_price:,.0f}</div>
              <div class="ensemble-hero-note">{ensemble_display_note}</div>
            </div>
            """
        if lgbm_price is not None and xgb_price is not None:
            diff           = xgb_price - lgbm_price
            pct_diff       = (diff / lgbm_price * 100) if lgbm_price != 0 else 0.0
            abs_pct        = abs(pct_diff)
            badge_class    = "green" if abs_pct < 5 else "amber" if abs_pct < 15 else "red"
            agreement_text = "Models closely agree" if abs_pct < 5 else f"Models differ by {abs_pct:.1f}%"
            higher_model   = "XGBoost" if xgb_price > lgbm_price else "LightGBM"
            sign           = "+" if diff >= 0 else ""
            if ensemble_price is not None:
                html += '<div class="model-agreement-label">Model Agreement</div>'
            html += f"""
            <div class="dual-result">
              <div class="dual-result-body">
                <div class="dual-result-side">
                  <div class="result-label">LightGBM</div>
                  <div class="result-price">${lgbm_price:,.0f}</div>
                </div>
                <div class="dual-result-divider">
                  <div class="divider-line"></div><div class="vs-badge">VS</div><div class="divider-line"></div>
                </div>
                <div class="dual-result-side">
                  <div class="result-label">XGBoost</div>
                  <div class="result-price">${xgb_price:,.0f}</div>
                </div>
              </div>
              <div class="dual-result-footer">
                <span class="diff-badge {badge_class}">{sign}${diff:,.0f} ({sign}{pct_diff:.1f}%)</span>
                <span class="diff-label">{agreement_text} &middot; {higher_model} is higher</span>
              </div>
            </div>
            """
        elif lgbm_price is not None and ensemble_price is None:
            html += f'<div class="result-panel"><div><div class="result-label">LightGBM</div><div class="result-price">${lgbm_price:,.0f}</div></div></div>'
        elif xgb_price is not None and ensemble_price is None:
            html += f'<div class="result-panel"><div><div class="result-label">XGBoost</div><div class="result-price">${xgb_price:,.0f}</div></div></div>'
        return html

    def _run_prediction_and_store(row, geo):
        X_input = pd.DataFrame([row])
        X_input["CountyOrParish"] = geo.get("county", "") or "Unknown"
        lgbm_price = xgb_price = ensemble_price = None
        try:
            ensemble_price = run_prediction(ensemble_model, X_input, output_is_log=ENSEMBLE_OUTPUT_IS_LOG)
            if hasattr(ensemble_model, "predict_base_models"):
                xgb_base_preds, lgbm_base_preds = ensemble_model.predict_base_models(X_input)
                xgb_price  = float(xgb_base_preds[0])
                lgbm_price = float(lgbm_base_preds[0])
            else:
                xgb_price  = run_prediction(ensemble_model.xgb_model,  X_input, output_is_log=XGB_OUTPUT_IS_LOG)
                lgbm_price = run_prediction(ensemble_model.lgbm_model, X_input, output_is_log=LGBM_OUTPUT_IS_LOG)
            st.session_state._ensemble_price_cache = ensemble_price
            st.session_state._xgb_price_cache      = xgb_price
            st.session_state._lgbm_price_cache     = lgbm_price
        except Exception as e:
            st.error(f"Ensemble failed: {e}")
            st.session_state._ensemble_price_cache = None
            st.session_state._xgb_price_cache      = None
            st.session_state._lgbm_price_cache     = None
        if similar_bundle is not None:
            try:
                _similar_df = find_similar_houses(X_input, similar_bundle, top_k=10)
                st.session_state.pred_similar_df = _similar_df
                st.session_state.pred_x_input    = X_input
            except Exception:
                st.session_state.pred_similar_df = None
        results_html = _build_results_html(ensemble_price, lgbm_price, xgb_price)
        all_prices = {}
        if ensemble_price is not None: all_prices["Ensemble"] = ensemble_price
        if lgbm_price     is not None: all_prices["LightGBM"] = lgbm_price
        if xgb_price      is not None: all_prices["XGBoost"]  = xgb_price
        if len(all_prices) >= 2:
            max_p  = max(all_prices.values())
            css_map = {"Ensemble": "ensemble", "LightGBM": "lgbm", "XGBoost": "xgb"}
            chart_html = '<div class="compare-bars">'
            for name, price in all_prices.items():
                w = price / max_p * 100
                chart_html += (
                    f'<div class="compare-bar-row">'
                    f'<div class="compare-bar-label">{name}</div>'
                    f'<div class="compare-bar-track">'
                    f'<div class="compare-bar-fill {css_map[name]}" style="width:{w:.1f}%"></div>'
                    f'<span class="compare-bar-value">${price:,.0f}</span>'
                    f'</div></div>'
                )
            chart_html += '</div>'
            st.session_state.pred_chart_html = chart_html
        else:
            st.session_state.pred_chart_html = None
        if results_html:
            st.session_state.pred_results_html = results_html
            st.session_state.predict_step = "results"
            st.rerun()

    def _lbl(label, key, wi):
        """Append '(default)' to a field label when the user hasn't yet saved that field."""
        return label if key in wi else f"{label} (default)"

    def _parse_float(s, default, mn=None, mx=None):
        """Convert a text-input string to float; clamp to [mn, mx] and fall back to default."""
        try:
            v = float(str(s).replace(",", "").strip())
            if mn is not None and v < mn:
                v = mn
            if mx is not None and v > mx:
                v = mx
            return v
        except (ValueError, AttributeError):
            return float(default)

    def _addr_banner(geo):
        """Render a compact confirmed-address strip for the top of wizard steps."""
        if not geo:
            return ""
        lat_str = f"{geo['latitude']:.5f}" if geo.get("latitude") is not None else "—"
        lng_str = f"{geo['longitude']:.5f}" if geo.get("longitude") is not None else "—"
        return (
            f'<div class="addr-banner">'
            f'<span class="addr-check">✓</span>'
            f'<div>'
            f'<div class="addr-banner-addr">{geo.get("formatted_address", "")}</div>'
            f'<div class="addr-chips">'
            f'<span class="addr-chip">📮 {geo.get("postal_code", "—")}</span>'
            f'<span class="addr-chip">🏛 {geo.get("county", "—")} County</span>'
            f'<span class="addr-chip">📍 {lat_str}, {lng_str}</span>'
            f'</div></div></div>'
        )

    def _wizard_progress(current_step):
        labels = ["Address", "Basic Info", "Structure", "Amenities"]
        html   = '<div class="wizard-progress">'
        for i, lbl in enumerate(labels):
            cls = "done" if i <= current_step else ("active" if i == current_step + 1 else "upcoming")
            html += f'<div class="wizard-step {cls}"><div class="wz-num">{i + 1}</div><div class="wz-lbl">{lbl}</div></div>'
            if i < len(labels) - 1:
                conn_cls = "done" if i < current_step else ""
                html += f'<div class="wizard-connector {conn_cls}"></div>'
        html += '</div>'
        return html

    # ══════════════════════════════════════════════════════════════════════════
    # STEP: address
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.predict_step == "address":
        st.markdown('<p class="page-title">Property Valuation</p>', unsafe_allow_html=True)
        st.markdown('<p class="page-sub">Start by entering the property address — we\'ll pull the location data automatically.</p>', unsafe_allow_html=True)

        _gmaps_key = (
            st.secrets.get("GOOGLE_MAPS_API_KEY", "")
            or os.environ.get("GOOGLE_MAPS_API_KEY", "")
        )

        def _search_fn(query):
            if len(query) < 3 or not _gmaps_key:
                return []
            suggestions, err = _maps_autocomplete(query, _gmaps_key)
            if err:
                st.session_state._addr_api_error = err
            return [(s["description"], s["place_id"]) for s in suggestions]

        _, addr_col, _ = st.columns([1, 3, 1])
        with addr_col:
            st.markdown('<div class="panel-title">Property Address</div>', unsafe_allow_html=True)
            selected_place = st_searchbox(
                _search_fn,
                placeholder="",
                key="addr_searchbox",
                label="",
                default=None,
                style_overrides={
                    "wrapper": {"backgroundColor": "#F5F6FA"},
                    "searchbox": {
                        "control": {
                            "backgroundColor": "#FFFFFF",
                            "border": "1px solid #e5e7eb",
                            "borderRadius": "10px",
                            "boxShadow": "none",
                            "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
                            "fontSize": "0.9rem",
                            "minHeight": "44px",
                            "cursor": "text",
                        },
                        "input": {
                            "color": "#111827",
                            "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
                        },
                        "placeholder": {
                            "color": "#9ca3af",
                            "fontSize": "0.9rem",
                        },
                        "singleValue": {
                            "color": "#111827",
                        },
                        "menuList": {
                            "backgroundColor": "#FFFFFF",
                            "borderRadius": "10px",
                            "padding": "4px 0",
                        },
                        "option": {
                            "color": "#111827",
                            "backgroundColor": "#FFFFFF",
                            "highlightColor": "rgba(37,99,235,0.08)",
                        },
                    },
                    "clear": {"fill": "#9ca3af"},
                    "dropdown": {"fill": "#9ca3af"},
                },
            )

            if not _gmaps_key:
                st.caption("Add `GOOGLE_MAPS_API_KEY` to `.streamlit/secrets.toml` to enable autocomplete.")

            if selected_place is not None:
                place_id = selected_place
                if place_id != st.session_state._confirmed_place_id:
                    _fetched = _maps_place_details(place_id, _gmaps_key)
                    if _fetched:
                        st.session_state._address_geo        = _fetched
                        st.session_state._confirmed_place_id = place_id
                        st.session_state._addr_api_error     = ""
                    else:
                        st.session_state._addr_api_error = "Could not fetch address details — try again."
            else:
                if st.session_state._confirmed_place_id:
                    st.session_state._address_geo        = {}
                    st.session_state._confirmed_place_id = ""

            if st.session_state._addr_api_error:
                st.markdown(
                    f'<div class="addr-api-error">⚠ {st.session_state._addr_api_error}</div>',
                    unsafe_allow_html=True,
                )

            _geo = st.session_state._address_geo
            if _geo:
                _lat_str = f"{_geo['latitude']:.5f}" if _geo.get("latitude") is not None else "—"
                _lng_str = f"{_geo['longitude']:.5f}" if _geo.get("longitude") is not None else "—"
                st.markdown(
                    f'<div class="addr-confirmed">'
                    f'<span class="addr-check">✓</span>'
                    f'<div>'
                    f'<div class="addr-formatted">{_geo.get("formatted_address", "")}</div>'
                    f'<div class="addr-chips">'
                    f'<span class="addr-chip">📮 {_geo.get("postal_code", "—")}</span>'
                    f'<span class="addr-chip">🏛 {_geo.get("county", "—")} County</span>'
                    f'<span class="addr-chip">📍 {_lat_str}, {_lng_str}</span>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                if st.button("Confirm Address  →", type="primary", use_container_width=True, key="addr_confirm"):
                    st.session_state.wizard_inputs["_geo"] = _geo
                    st.session_state.predict_step = "wizard"
                    st.session_state.wizard_step  = 0
                    st.rerun()

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            if st.button("Skip — enter location manually", type="secondary", use_container_width=True, key="addr_skip"):
                st.session_state.wizard_inputs["_geo"] = {}
                st.session_state._address_geo = {}
                st.session_state.predict_step = "wizard"
                st.session_state.wizard_step  = 0
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP: wizard
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.predict_step == "wizard":
        _step = st.session_state.wizard_step
        _wi   = st.session_state.wizard_inputs
        _geo  = _wi.get("_geo", {})

        _, prog_col, _ = st.columns([1, 3, 1])
        with prog_col:
            st.markdown(_wizard_progress(_step), unsafe_allow_html=True)

        _, back_col, _ = st.columns([1, 3, 1])
        with back_col:
            if st.button("← Back", key="wiz_back", type="secondary"):
                if _step == 0:
                    st.session_state.predict_step = "address"
                else:
                    st.session_state.wizard_step -= 1
                st.rerun()

        _, form_col, _ = st.columns([1, 3, 1])
        with form_col:

            # ── Step 0: Basic Info ─────────────────────────────────────────────
            if _step == 0:
                st.markdown('<div class="panel-title">Basic Information</div>', unsafe_allow_html=True)
                if _geo:
                    st.markdown(_addr_banner(_geo), unsafe_allow_html=True)
                with st.form("wiz_step0"):
                    c1, c2 = st.columns(2)
                    with c1:
                        living_area_str = st.text_input(
                            "Living Area (sqft)",
                            value="" if "living_area" not in _wi else str(int(_wi["living_area"])),
                            placeholder="1,500",
                        )
                    with c2:
                        lot_size_str = st.text_input(
                            "Lot Size (sqft)",
                            value="" if "lot_size" not in _wi else str(int(_wi["lot_size"])),
                            placeholder="5,000",
                        )
                    c1, c2 = st.columns(2)
                    with c1:
                        beds  = st.number_input(_lbl("Bedrooms",  "beds",  _wi), 0, 20, int(_wi.get("beds",  3)))
                    with c2:
                        baths = st.number_input(_lbl("Bathrooms", "baths", _wi), 0, 20, int(_wi.get("baths", 2)))
                    if st.form_submit_button("Next: Structure  →", type="primary", use_container_width=True):
                        living_area = _parse_float(living_area_str, 1500.0, mn=200.0, mx=20000.0)
                        lot_size    = _parse_float(lot_size_str,    5000.0, mn=0.0,   mx=200000.0)
                        _wi.update({"living_area": living_area, "lot_size": lot_size,
                                    "beds": beds, "baths": baths})
                        st.session_state.wizard_step = 1
                        st.rerun()

            # ── Step 1: Structure & Parking ────────────────────────────────────
            elif _step == 1:
                st.markdown('<div class="panel-title">Structure &amp; Parking</div>', unsafe_allow_html=True)
                if _geo:
                    st.markdown(_addr_banner(_geo), unsafe_allow_html=True)
                _lv_default = _LEVELS_DISPLAY.get(_wi.get("levels", "One"), "One Story")
                with st.form("wiz_step1"):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        year_built_str = st.text_input(
                            _lbl("Year Built", "year_built", _wi),
                            value="" if "year_built" not in _wi else str(int(_wi["year_built"])),
                            placeholder="2000",
                        )
                    with c2:
                        stories       = st.number_input(_lbl("Stories",       "stories",       _wi), 1, 5,  int(_wi.get("stories",       1)))
                    with c3:
                        garage_spaces = st.number_input(_lbl("Garage Spaces", "garage_spaces", _wi), 0, 6,  int(_wi.get("garage_spaces", 1)))
                    with c4:
                        parking_total = st.number_input(_lbl("Total Parking", "parking_total", _wi), 0, 10, int(_wi.get("parking_total", 2)))
                    levels_display = st.selectbox("Levels", list(LEVELS_MAP.keys()),
                        index=list(LEVELS_MAP.keys()).index(_lv_default))
                    if st.form_submit_button("Next: Amenities  →", type="primary", use_container_width=True):
                        year_built = int(_parse_float(year_built_str, 2000.0, mn=1800.0, mx=2030.0))
                        _wi.update({"year_built": year_built, "stories": stories,
                                    "garage_spaces": garage_spaces, "parking_total": parking_total,
                                    "levels": LEVELS_MAP[levels_display]})
                        st.session_state.wizard_step = 2
                        st.rerun()

            # ── Step 2: Amenities & Location ───────────────────────────────────
            elif _step == 2:
                st.markdown('<div class="panel-title">Amenities &amp; Location</div>', unsafe_allow_html=True)
                if _geo:
                    st.markdown(_addr_banner(_geo), unsafe_allow_html=True)
                with st.form("wiz_step2"):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_construction = "True" if st.toggle("New Construction", value=_wi.get("new_construction", "False") == "True", key="nc") else "False"
                        pool_private     = "True" if st.toggle("Private Pool",     value=_wi.get("pool_private",     "False") == "True", key="pp") else "False"
                        fireplace_yn     = "True" if st.toggle("Fireplace",        value=_wi.get("fireplace_yn",     "False") == "True", key="fy") else "False"
                    with col2:
                        attached_garage  = "True" if st.toggle("Attached Garage",  value=_wi.get("attached_garage",  "False") == "True", key="ag") else "False"
                        view_yn          = "True" if st.toggle("View",             value=_wi.get("view_yn",          "False") == "True", key="vy") else "False"

                    if not _geo:
                        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                        postal_code_inp = st.text_input(
                            _lbl("Postal Code", "postal_code", _wi),
                            value=_wi.get("postal_code", ""),
                            placeholder="90210", max_chars=5,
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            latitude_inp = st.number_input(
                                _lbl("Latitude", "latitude", _wi),
                                value=float(_wi.get("latitude", 34.4140)),
                                format="%.6f",
                            )
                        with c2:
                            longitude_inp = st.number_input(
                                _lbl("Longitude", "longitude", _wi),
                                value=float(_wi.get("longitude", -119.8489)),
                                format="%.6f",
                            )
                        st.caption("Latitude & longitude default to Santa Barbara, CA — update for your property.")

                    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                    if st.form_submit_button("Estimate Price  →", type="primary", use_container_width=True):
                        if _geo:
                            postal_code = _geo.get("postal_code", "")
                            latitude    = float(_geo.get("latitude")  or 34.4140)
                            longitude   = float(_geo.get("longitude") or -119.8489)
                        else:
                            postal_code = postal_code_inp
                            latitude    = latitude_inp
                            longitude   = longitude_inp
                        _wi.update({"new_construction": new_construction, "pool_private": pool_private,
                                    "fireplace_yn": fireplace_yn, "attached_garage": attached_garage,
                                    "view_yn": view_yn, "postal_code": postal_code,
                                    "latitude": latitude, "longitude": longitude})
                        row = {
                            "LivingArea":            _wi.get("living_area",    1500.0),
                            "BedroomsTotal":         _wi.get("beds",           3),
                            "BathroomsTotalInteger": _wi.get("baths",          2),
                            "LotSizeSquareFeet":     _wi.get("lot_size",       5000.0),
                            "ParkingTotal":          _wi.get("parking_total",  2),
                            "GarageSpaces":          _wi.get("garage_spaces",  1),
                            "Latitude":              latitude,
                            "Longitude":             longitude,
                            "YearBuilt":             _wi.get("year_built",     2000),
                            "NewConstructionYN":     new_construction,
                            "AttachedGarageYN":      attached_garage,
                            "PoolPrivateYN":         pool_private,
                            "ViewYN":                view_yn,
                            "FireplaceYN":           fireplace_yn,
                            "Stories":               _wi.get("stories",        1),
                            "Levels":                _wi.get("levels",         "One"),
                            "PostalCode":            postal_code,
                        }
                        _run_prediction_and_store(row, _geo)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP: results
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.predict_step == "results":
        if st.button("← New Estimate", type="secondary"):
            st.session_state.predict_step  = "address"
            st.session_state.wizard_step   = 0
            st.session_state.wizard_inputs = {}
            st.session_state._address_geo  = {}
            st.session_state._confirmed_place_id = ""
            st.session_state.pred_results_html   = None
            st.session_state.pred_chart_html     = None
            st.session_state.pop("addr_searchbox", None)
            st.rerun()

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        _, res_col, _ = st.columns([1, 3, 1])
        with res_col:
            if st.session_state.pred_results_html:
                st.markdown(st.session_state.pred_results_html, unsafe_allow_html=True)
            if st.session_state.pred_chart_html:
                st.markdown(st.session_state.pred_chart_html, unsafe_allow_html=True)

            if st.session_state.pred_similar_df is not None and st.session_state.pred_x_input is not None:
                st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-heading">Market Context</div>', unsafe_allow_html=True)

                _similar_df   = st.session_state.pred_similar_df
                _x_input      = st.session_state.pred_x_input
                _reference_df = similar_bundle["reference_df"]

                _background_df = _reference_df.copy()
                if len(_background_df) > 2000:
                    _background_df = _background_df.sample(2000, random_state=42)

                _FEATURE_OPTIONS = {
                    "Living Area":   {"col": "LivingArea",            "unit": "sqft",   "fmt": ",.0f", "jitter": False},
                    "Lot Size":      {"col": "LotSizeSquareFeet",     "unit": "sqft",   "fmt": ",.0f", "jitter": False},
                    "Bedrooms":      {"col": "BedroomsTotal",         "unit": "bd",     "fmt": ".0f",  "jitter": True},
                    "Bathrooms":     {"col": "BathroomsTotalInteger", "unit": "ba",     "fmt": ".0f",  "jitter": True},
                    "Year Built":    {"col": "YearBuilt",             "unit": "",       "fmt": ".0f",  "jitter": False},
                    "Garage Spaces": {"col": "GarageSpaces",          "unit": "spaces", "fmt": ".0f",  "jitter": True},
                }

                _feat_label = st.session_state.scatter_x_feat
                _fcfg       = _FEATURE_OPTIONS[_feat_label]
                _x_col      = _fcfg["col"]
                _x_unit     = _fcfg["unit"]
                _x_fmt      = _fcfg["fmt"]
                _do_jitter  = _fcfg["jitter"]
                _x_title    = f"{_feat_label} ({_x_unit})" if _x_unit else _feat_label

                def _add_jitter(series, scale=0.12):
                    rng = np.random.default_rng(seed=42)
                    return series + rng.uniform(-scale, scale, len(series))

                _bg_clean  = _background_df.dropna(subset=[_x_col, "ClosePrice"])
                _sim_clean = _similar_df.dropna(subset=[_x_col, "ClosePrice"])
                _bg_x      = _add_jitter(_bg_clean[_x_col])  if _do_jitter else _bg_clean[_x_col]
                _sim_x     = _add_jitter(_sim_clean[_x_col]) if _do_jitter else _sim_clean[_x_col]

                _fig = go.Figure()
                _fig.add_trace(go.Scatter(
                    x=_bg_x, y=_bg_clean["ClosePrice"],
                    mode="markers", name="All Sales",
                    marker=dict(size=5, color="#9ca3af", opacity=0.18),
                    hovertemplate=f"{_feat_label}: %{{x:{_x_fmt}}} {_x_unit}<br>Sale Price: $%{{y:,.0f}}<extra></extra>",
                ))
                _fig.add_trace(go.Scatter(
                    x=_sim_x, y=_sim_clean["ClosePrice"],
                    mode="markers", name="Similar Properties",
                    marker=dict(size=7, color="#92C5FD", line=dict(width=1.5, color="#6891BD")),
                    hovertemplate=f"{_feat_label}: %{{x:{_x_fmt}}} {_x_unit}<br>Sale Price: $%{{y:,.0f}}<extra></extra>",
                ))
                _ensemble_p = st.session_state._ensemble_price_cache
                if _ensemble_p is not None:
                    _fig.add_trace(go.Scatter(
                        x=[_x_input.iloc[0][_x_col]], y=[_ensemble_p],
                        mode="markers", name="This Property (Ensemble)",
                        marker=dict(size=14, symbol="star", color="#2463EB", line=dict(width=1.5, color="#1E52C3")),
                        hovertemplate="Ensemble Estimate: $%{y:,.0f}<extra></extra>",
                    ))

                _x_lo  = float(_bg_clean[_x_col].quantile(0.01))
                _x_hi  = float(_bg_clean[_x_col].quantile(0.98))
                _y_lo  = float(_bg_clean["ClosePrice"].quantile(0.01))
                _y_hi  = float(_bg_clean["ClosePrice"].quantile(0.98))
                _x_pad = (_x_hi - _x_lo) * 0.05
                _y_pad = (_y_hi - _y_lo) * 0.05

                _fig.update_layout(
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", size=12, color="#111827"),
                    title=None,
                    xaxis=dict(title_text=_x_title, title_font=dict(color="#1a2e44", size=12),
                               gridcolor="#f3f4f6", zeroline=False, linecolor="#e5e7eb", linewidth=1,
                               showline=True, tickfont=dict(color="#4b5563", size=11),
                               range=[_x_lo - _x_pad, _x_hi + _x_pad]),
                    yaxis=dict(title_text="Sale Price", title_font=dict(color="#1a2e44", size=12),
                               gridcolor="#f3f4f6", zeroline=False, tickprefix="$", tickformat=",",
                               linecolor="#e5e7eb", linewidth=1, showline=True,
                               tickfont=dict(color="#4b5563", size=11),
                               range=[_y_lo - _y_pad, _y_hi + _y_pad]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                bgcolor="rgba(255,255,255,0.9)", bordercolor="#e5e7eb", borderwidth=1,
                                font=dict(color="#111827", size=11)),
                    margin=dict(t=56, b=40, l=10, r=10),
                )
                st.plotly_chart(_fig, use_container_width=True)

                _sel_col, _radio_col = st.columns([1, 7])
                with _sel_col:
                    st.markdown(
                        "<div style='padding-top:6px;font-size:12px;color:#6b7280;font-family:Inter,sans-serif;"
                        "font-weight:500;white-space:nowrap'>Explore by</div>",
                        unsafe_allow_html=True,
                    )
                with _radio_col:
                    st.radio("X-axis feature", list(_FEATURE_OPTIONS.keys()),
                             horizontal=True, label_visibility="collapsed", key="scatter_x_feat")

                with st.expander("View similar properties"):
                    _subj       = st.session_state.pred_x_input.iloc[0]
                    _subj_beds  = int(_subj.get("BedroomsTotal",         0) or 0)
                    _subj_baths = int(_subj.get("BathroomsTotalInteger",  0) or 0)
                    _subj_sqft  = int(_subj.get("LivingArea",             0) or 0)
                    _cards_html = '<div class="sim-cards-grid">'
                    for _, _row in _similar_df.iterrows():
                        _cards_html += _sim_property_card(_row, _subj_beds, _subj_baths, _subj_sqft)
                    _cards_html += "</div>"
                    st.markdown(_cards_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Model":

    st.markdown('<p class="page-title">Model Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Technical details about the machine learning model used for prediction.</p>', unsafe_allow_html=True)

    # Hero banner
    st.markdown("""
    <div class="model-hero-banner">
      <div class="model-hero-left">
        <p class="model-hero-type">Ensemble · XGBoost + LightGBM</p>
        <p class="model-hero-name">%s</p>
      </div>
      <div class="model-hero-stats">
        <div class="model-hero-stat">
          <div class="model-hero-stat-val">7.74%</div>
          <div class="model-hero-stat-lbl">Test MdAPE</div>
        </div>
        <div class="model-hero-divider"></div>
        <div class="model-hero-stat">
          <div class="model-hero-stat-val">0.8796</div>
          <div class="model-hero-stat-lbl">Test R²</div>
        </div>
        <div class="model-hero-divider"></div>
        <div class="model-hero-stat">
          <div class="model-hero-stat-val">17</div>
          <div class="model-hero-stat-lbl">Input Features</div>
        </div>
      </div>
    </div>
    """ % ensemble_display_name, unsafe_allow_html=True)

    info_col, metrics_col = st.columns([3, 2], gap="large")

    with info_col:
        # Model Architecture — borderless section
        st.markdown("""
        <div class="model-section">
          <p class="model-section-title">Model Architecture</p>
          <div class="arch-row">
            <div class="arch-term">Algorithm</div>
            <div class="arch-def">XGBoost and LightGBM base learners feed a GradientBoostingRegressor meta-model, with prices trained on a log1p target transform and served through a full preprocessing pipeline.</div>
          </div>
          <div class="arch-row">
            <div class="arch-term">Target Transform</div>
            <div class="arch-def">Prices are transformed using log1p(Price) before training to normalize the right-skewed distribution of home values.</div>
          </div>
          <div class="arch-row">
            <div class="arch-term">Inverse Transform</div>
            <div class="arch-def">Predictions are converted back to dollar values using expm1(), ensuring accurate final estimates.</div>
          </div>
          <div class="arch-row">
            <div class="arch-term">Training Data</div>
            <div class="arch-def">Regional MLS transaction records covering a range of residential property types and price points.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col:
        # R² callout — borderless
        st.markdown("""
        <div class="model-section">
          <p class="model-section-title">Interpreting the Metrics</p>
          <div class="r2-callout">
            <p style="font-size:0.82rem; color:var(--text-2); line-height:1.65; margin:0;">
              <strong style="color:var(--text-1);">R² of 0.8796</strong> means the model accounts for
              87.96% of the variation in house prices — strong performance for real estate data.
            </p>
            <p style="font-size:0.82rem; color:var(--text-2); line-height:1.65; margin:0.75rem 0 0 0;">
              <strong style="color:var(--text-1);">MdAPE of 7.74%</strong> means the typical prediction
              is within 7.74% of the actual sale price.
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Key Features — full-width 3-column chip grid below the columns
    features = [
        ("Living Area",   "Primary size signal — strongly correlated with sale price."),
        ("Bedrooms",      "Room count used alongside bathrooms to capture functional size."),
        ("Bathrooms",     "Higher counts associated with luxury and larger properties."),
        ("Lot Size",      "Captures land value, especially in suburban and rural areas."),
        ("Location",      "Latitude, longitude, and postal code encode geographic variation."),
        ("Year Built",    "Newer construction commands a premium; older homes may carry a discount."),
        ("Garage / Pool", "Amenity flags provide incremental adjustment beyond size and location."),
    ]
    chips_html = "".join(
        f'<div class="feat-chip"><div class="feat-chip-name">{n}</div><div class="feat-chip-desc">{d}</div></div>'
        for n, d in features
    )
    st.markdown(f"""
    <div class="model-section">
      <p class="model-section-title">Key Features</p>
      <div class="feat-grid">{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Comparison":

    st.markdown('<p class="page-title">Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Performance benchmarks across candidate models evaluated during development.</p>', unsafe_allow_html=True)

    # ── Dataset selector ─────────────────────────────────────────────────────
    dataset_view = st.radio(
        "Dataset",
        ["Validation Set", "Holdout Set"],
        horizontal=True,
        key="comparison_dataset_view",
        label_visibility="collapsed",
    )

    # ── Data definitions ─────────────────────────────────────────────────────
    # Validation set: all candidate models on the held-out test split
    val_models  = ["Baseline LR", "Ridge", "Log-Linear", "Decision Tree",
                   "Random Forest", "XGBoost", "LightGBM", "Ensemble"]
    val_r2      = [0.5452, 0.4718, 0.2209, -4.9413, -3.3187, 0.8748, 0.8830, 0.8796]
    val_mdape_models = ["Ridge", "Log-Linear", "Decision Tree", "Random Forest",
                        "XGBoost", "LightGBM", "Ensemble"]
    val_mdape_vals   = [38.27, 22.80, 15.41, 9.01, 7.62, 7.74, 7.74]
    val_mdape_lookup = {
        "Baseline LR": "—", "Ridge": "38.27%", "Log-Linear": "22.80%",
        "Decision Tree": "15.41%", "Random Forest": "9.01%",
        "XGBoost": "7.62%", "LightGBM": "7.74%", "Ensemble": "7.74%",
    }

    # Holdout set: 3-fold rolling forward validation (XGBoost / LightGBM / Ensemble only)
    holdout_folds   = ["Dec 2025", "Jan 2026", "Feb 2026"]
    holdout_xgb_r2  = [0.8748, 0.8789, 0.8772]
    holdout_lgbm_r2 = [0.8830, 0.8869, 0.8834]
    holdout_ens_r2  = [0.8796, 0.8855, 0.8824]
    holdout_xgb_mdape  = [7.62, 7.77, 7.80]
    holdout_lgbm_mdape = [7.74, 7.78, 8.12]
    holdout_ens_mdape  = [7.74, 7.74, 7.88]

    PAL_PRIMARY = "#2563eb"
    PAL_LGBM    = "#93c5fd"
    PAL_ENS     = "#10b981"

    # Winner strip
    st.markdown("""
    <div class="winner-strip">
      <span class="winner-strip-label">Best Model</span>
      <span class="winner-strip-name">%s</span>
      <span class="winner-strip-dot">·</span>
      <span class="winner-strip-stat">R² 0.8796</span>
      <span class="winner-strip-dot">·</span>
      <span class="winner-strip-stat">MdAPE 7.74%</span>
    </div>
    """ % ensemble_display_name, unsafe_allow_html=True)

    # ── R² chart ─────────────────────────────────────────────────────────────
    if dataset_view == "Validation Set":
        st.markdown('<div class="model-section"><p class="model-section-title">R² by Model — Validation Set</p></div>', unsafe_allow_html=True)

        x     = np.arange(len(val_models))
        width = 0.5

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#F5F6FA")
        ax.set_facecolor("#F5F6FA")

        bar_colors = ["#ef4444" if v < 0 else PAL_PRIMARY for v in val_r2]
        bars = ax.bar(x, val_r2, width, color=bar_colors, zorder=3)

        ax.axhline(0, color="#94a3b8", linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(val_models, fontsize=8.5, color="#475569", rotation=15, ha="right")
        ax.set_ylim(-5.5, 1.15)
        ax.set_ylabel("R²", fontsize=9, color="#475569")
        ax.yaxis.set_tick_params(labelsize=8, labelcolor="#94a3b8")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#dde1e8")
        ax.yaxis.grid(True, color="#dde1e8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, val_r2):
            offset = 0.1 if val >= 0 else -0.25
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#1d4ed8" if val >= 0 else "#ef4444", fontweight="600")

        ax.legend(handles=[mpatches.Patch(color=PAL_PRIMARY, label="R²")],
                  frameon=False, fontsize=9, loc="upper left")
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    else:  # Holdout Set
        st.markdown('<div class="model-section"><p class="model-section-title">R² by Model — Holdout Set (3-Fold Rolling Forward)</p></div>', unsafe_allow_html=True)

        x     = np.arange(len(holdout_folds))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#F5F6FA")
        ax.set_facecolor("#F5F6FA")

        bars_xgb  = ax.bar(x - width, holdout_xgb_r2,  width, color=PAL_PRIMARY, label="XGBoost",  zorder=3)
        bars_lgbm = ax.bar(x,         holdout_lgbm_r2, width, color=PAL_LGBM,    label="LightGBM", zorder=3)
        bars_ens  = ax.bar(x + width, holdout_ens_r2,  width, color=PAL_ENS,     label="Ensemble", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(holdout_folds, fontsize=9, color="#475569")
        ax.set_ylim(0.85, 0.92)
        ax.set_ylabel("R²", fontsize=9, color="#475569")
        ax.yaxis.set_tick_params(labelsize=8, labelcolor="#94a3b8")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#dde1e8")
        ax.yaxis.grid(True, color="#dde1e8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        for bar_group in [bars_xgb, bars_lgbm, bars_ens]:
            for bar in bar_group:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7, color="#1a2e44", fontweight="600")

        ax.legend(frameon=False, fontsize=9, loc="lower right")
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    chart_col, text_col = st.columns([3, 2], gap="large")

    with chart_col:
        # ── MdAPE chart ───────────────────────────────────────────────────────
        if dataset_view == "Validation Set":
            st.markdown('<div class="model-section"><p class="model-section-title">MdAPE by Model — Validation Set (lower is better)</p></div>', unsafe_allow_html=True)

            mdape_colors = ["#ef4444" if v > 20 else "#f59e0b" if v > 10 else PAL_PRIMARY
                            for v in val_mdape_vals]

            fig2, ax2 = plt.subplots(figsize=(7, 2.8))
            fig2.patch.set_facecolor("#F5F6FA")
            ax2.set_facecolor("#F5F6FA")

            h_bars = ax2.barh(val_mdape_models, val_mdape_vals, color=mdape_colors, zorder=3, height=0.5)
            ax2.set_xlim(0, max(val_mdape_vals) * 1.25)
            ax2.xaxis.grid(True, color="#dde1e8", linewidth=0.6, zorder=0)
            ax2.set_axisbelow(True)
            ax2.spines[["top", "right", "bottom"]].set_visible(False)
            ax2.spines["left"].set_color("#dde1e8")
            ax2.tick_params(axis="y", labelsize=8.5, labelcolor="#475569")
            ax2.tick_params(axis="x", labelsize=8,   labelcolor="#94a3b8")
            ax2.set_xlabel("MdAPE % (lower is better)", fontsize=8.5, color="#94a3b8")

            for bar, val in zip(h_bars, val_mdape_vals):
                ax2.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                         f"{val:.2f}%", va="center", fontsize=8, color="#1a2e44", fontweight="600")

            plt.tight_layout(pad=1.0)
            st.pyplot(fig2, width='stretch')
            plt.close(fig2)

        else:  # Holdout Set
            st.markdown('<div class="model-section"><p class="model-section-title">MdAPE by Model — Holdout Set (lower is better)</p></div>', unsafe_allow_html=True)

            x     = np.arange(len(holdout_folds))
            width = 0.25

            fig2, ax2 = plt.subplots(figsize=(7, 3.2))
            fig2.patch.set_facecolor("#F5F6FA")
            ax2.set_facecolor("#F5F6FA")

            bars_xgb  = ax2.bar(x - width, holdout_xgb_mdape,  width, color=PAL_PRIMARY, label="XGBoost",  zorder=3)
            bars_lgbm = ax2.bar(x,          holdout_lgbm_mdape, width, color=PAL_LGBM,    label="LightGBM", zorder=3)
            bars_ens  = ax2.bar(x + width,  holdout_ens_mdape,  width, color=PAL_ENS,     label="Ensemble", zorder=3)

            ax2.set_xticks(x)
            ax2.set_xticklabels(holdout_folds, fontsize=9, color="#475569")
            ax2.set_ylim(6.5, 9.0)
            ax2.set_ylabel("MdAPE %", fontsize=9, color="#475569")
            ax2.yaxis.set_tick_params(labelsize=8, labelcolor="#94a3b8")
            ax2.spines[["top", "right", "left"]].set_visible(False)
            ax2.spines["bottom"].set_color("#dde1e8")
            ax2.yaxis.grid(True, color="#dde1e8", linewidth=0.6, zorder=0)
            ax2.set_axisbelow(True)

            for bar_group in [bars_xgb, bars_lgbm, bars_ens]:
                for bar in bar_group:
                    h = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}%",
                             ha="center", va="bottom", fontsize=7, color="#1a2e44", fontweight="600")

            ax2.legend(frameon=False, fontsize=9, loc="upper right")
            plt.tight_layout(pad=1.0)
            st.pyplot(fig2, width='stretch')
            plt.close(fig2)

    with text_col:
        # Why the Ensemble — green callout, no box
        st.markdown("""
        <div class="model-section">
          <p class="model-section-title">Why the Ensemble?</p>
          <div class="r2-callout">
            <p style="font-size:0.83rem; color:var(--text-2); line-height:1.7; margin:0 0 0.75rem 0;">
              Baseline linear models (Ridge, Log-Linear) achieved poor R² (0.22–0.47) and
              high MdAPE (23–38%), showing they underfit the complexity of real estate pricing.
            </p>
            <p style="font-size:0.83rem; color:var(--text-2); line-height:1.7; margin:0 0 0.75rem 0;">
              Tree models reduced MdAPE (9–15%) but severely overfit — Decision Tree and
              Random Forest both produced negative test R².
            </p>
            <p style="font-size:0.83rem; color:var(--text-2); line-height:1.7; margin:0;">
              <strong style="color:var(--text-1)">XGBoost + LightGBM</strong> each hit ~8% MdAPE
              with strong holdout R². A stacked meta-model then blends them nonlinearly,
              improving robustness across property types and forward months.
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Score summary — switches between validation table and holdout table
        if dataset_view == "Validation Set":
            rows_html = ""
            for m, r2 in zip(val_models, val_r2):
                row_class = "row-winner" if m == "Ensemble" else ""
                r2_class  = "score-val-neg" if r2 < 0 else ""
                rows_html += (
                    f'<tr class="{row_class}">'
                    f'<td>{m}</td>'
                    f'<td class="{r2_class}">{r2:.2f}</td>'
                    f'<td>{val_mdape_lookup[m]}</td>'
                    f'</tr>'
                )
            st.markdown(f"""
            <div class="model-section">
              <p class="model-section-title">Score Summary — Validation Set</p>
              <table class="score-table">
                <thead><tr><th>Model</th><th>R²</th><th>MdAPE</th></tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

        else:
            holdout_rows = [
                ("XGBoost",  holdout_xgb_r2,  holdout_xgb_mdape),
                ("LightGBM", holdout_lgbm_r2, holdout_lgbm_mdape),
                ("Ensemble", holdout_ens_r2,  holdout_ens_mdape),
            ]
            rows_html = ""
            for model_name, r2s, mdapes in holdout_rows:
                row_class = "row-winner" if model_name == "Ensemble" else ""
                for fold, r2, mdape in zip(holdout_folds, r2s, mdapes):
                    rows_html += (
                        f'<tr class="{row_class}">'
                        f'<td>{model_name}</td>'
                        f'<td>{fold}</td>'
                        f'<td>{r2:.4f}</td>'
                        f'<td>{mdape:.2f}%</td>'
                        f'</tr>'
                    )
            st.markdown(f"""
            <div class="model-section">
              <p class="model-section-title">Score Summary — Holdout Set</p>
              <table class="score-table">
                <thead><tr><th>Model</th><th>Fold</th><th>R²</th><th>MdAPE</th></tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)
