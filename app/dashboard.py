import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Set page configuration as first Streamlit command
st.set_page_config(
    page_title="GPT Analytics • Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress noisy NLTK download messages in stdout
import nltk
@st.cache_resource(show_spinner=False)
def setup_nlp_resources():
    for pkg in ['punkt', 'punkt_tab', 'brown', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

setup_nlp_resources()

from src.parser import load_chatgpt_export, generate_sample_chat_data
from src.analytics import (
    compute_basic_stats,
    get_activity_over_time,
    activity_heatmap_data,
    get_hourly_activity,
    get_weekday_activity,
    get_conversation_leaderboard,
    get_token_cost_estimation
)
from src.nlp import (
    add_sentiment_to_df,
    extract_top_ngrams,
    cluster_topics_with_labels,
    detect_code_languages
)
from src.visualization import (
    activity_timeline_chart,
    role_distribution_chart,
    hourly_heatmap,
    weekday_bar_chart,
    top_ngrams_bar_chart,
    sentiment_breakdown_chart,
    sentiment_trend_chart,
    message_length_distribution,
    code_languages_chart
)

# --- SESSION STATE INITIALIZATION ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "Demo Dataset"
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None

current_theme = st.session_state.theme

# --- GLOBAL STYLES (RESPONSIVE, ZERO-GRADIENT, HIGH-CONTRAST LIGHT & DARK) ---
if current_theme == "Dark":
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
            box-sizing: border-box;
        }

        [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }

        .stApp {
            background-color: #0A0D14 !important;
            color: #F8FAFC !important;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        .stApp p, .stApp span, .stApp div, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #F8FAFC;
        }

        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
            color: #E2E8F0 !important;
        }

        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #10141E !important;
            border: 1px solid #1E2638 !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            box-shadow: none !important;
        }
        
        div[data-testid="metric-container"] label {
            color: #94A3B8 !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
            color: #38BDF8 !important;
            font-size: 0.8rem !important;
        }

        /* Hero & Surface Cards */
        .landing-hero {
            background-color: #10141E;
            border: 1px solid #1E2638;
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 1rem 0 1.5rem 0;
        }

        .hero-badge {
            background-color: rgba(56, 189, 248, 0.12);
            color: #38BDF8;
            font-weight: 600;
            font-size: 0.82rem;
            padding: 5px 14px;
            border-radius: 20px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            display: inline-block;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 1.2rem 0 0.8rem 0;
            line-height: 1.2;
            color: #F8FAFC !important;
        }

        .hero-desc {
            font-size: 1.05rem;
            color: #94A3B8 !important;
            max-width: 680px;
            margin: 0 auto 1.5rem auto;
            line-height: 1.6;
        }

        .feature-card {
            background-color: #10141E;
            border: 1px solid #1E2638;
            border-radius: 8px;
            padding: 1.5rem;
            height: 100%;
        }

        .feature-card h4 {
            color: #F8FAFC !important;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 8px 0;
        }

        .feature-desc {
            font-size: 0.88rem;
            color: #94A3B8 !important;
            line-height: 1.5;
            margin: 0;
        }

        .feature-icon-box {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            margin-bottom: 12px;
        }

        .hub-surface {
            background-color: #10141E;
            border: 1px solid #1E2638;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .hub-surface h3, .hub-surface h4 {
            color: #F8FAFC !important;
        }

        .hub-subtext {
            color: #94A3B8 !important;
            font-size: 0.88rem;
        }

        /* High-Contrast File Uploader (Dark Mode) */
        [data-testid="stFileUploader"] section {
            background-color: #10141E !important;
            border: 2px dashed #334155 !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #38BDF8 !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 0.45rem 1rem !important;
        }
        [data-testid="stFileUploader"] * {
            color: #E2E8F0 !important;
        }

        /* Chat Messages */
        .chat-bubble-user {
            background-color: #141E2F !important;
            border: 1px solid #1E3A5F !important;
            border-left: 4px solid #38BDF8 !important;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 10px;
            color: #F8FAFC !important;
        }

        .chat-bubble-ai {
            background-color: #10141E !important;
            border: 1px solid #1E2638 !important;
            border-left: 4px solid #34D399 !important;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 14px;
            color: #F8FAFC !important;
        }

        /* Inputs, Datepickers & Selects (Dark Mode) */
        [data-testid="stDateInput"],
        [data-testid="stTextInput"],
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"],
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="input"] *,
        [data-baseweb="base-input"] *,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] div,
        [data-testid="stTextInput"] div,
        [data-baseweb="select"],
        [data-baseweb="select"] *,
        [data-baseweb="popover"],
        [data-baseweb="popover"] *,
        ul[role="listbox"],
        li[role="option"],
        input,
        textarea {
            background-color: #10141E !important;
            color: #F8FAFC !important;
            -webkit-text-fill-color: #F8FAFC !important;
            border-color: #1E2638 !important;
            font-weight: 500 !important;
        }

        /* Multiselect Tags (Dark Mode) */
        div[data-baseweb="tag"],
        span[data-baseweb="tag"],
        div[data-baseweb="tag"] *,
        span[data-baseweb="tag"] * {
            background-color: #1E3A5F !important;
            color: #38BDF8 !important;
            -webkit-text-fill-color: #38BDF8 !important;
            border-color: #0284C7 !important;
            font-weight: 600 !important;
        }

        /* SVG Icons in inputs (Dark Mode) */
        [data-baseweb="select"] svg,
        [data-testid="stDateInput"] svg,
        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg,
        svg[data-icon="calendar"] {
            fill: #94A3B8 !important;
            color: #94A3B8 !important;
        }

        /* Checkbox (Dark Mode) */
        [data-testid="stCheckbox"] label span,
        [data-testid="stCheckbox"] p {
            color: #F8FAFC !important;
            font-weight: 500 !important;
        }

        /* Chat Explorer (Dark Mode) */
        .chat-bubble-user {
            background-color: #141E2F !important;
            border: 1px solid #1E3A5F !important;
            border-left: 4px solid #38BDF8 !important;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 8px;
            color: #F8FAFC !important;
        }
        .chat-bubble-ai {
            background-color: #10141E !important;
            border: 1px solid #1E2638 !important;
            border-left: 4px solid #34D399 !important;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            color: #F8FAFC !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] p,
        [data-testid="stVerticalBlockBorderWrapper"] li,
        [data-testid="stVerticalBlockBorderWrapper"] code {
            color: #F8FAFC !important;
        }

        /* Tabs (Dark Mode) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            border-bottom: 1px solid #1E2638;
            padding-bottom: 6px;
            margin-bottom: 16px;
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #94A3B8 !important;
            border: 1px solid #1E2638 !important;
            background-color: #10141E !important;
            white-space: nowrap !important;
        }
        .stTabs [data-baseweb="tab"] * {
            color: #94A3B8 !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #38BDF8 !important;
            border-color: #0284C7 !important;
            background-color: #141E2F !important;
        }
        .stTabs [data-baseweb="tab"]:hover * {
            color: #38BDF8 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;
            background-color: #0284C7 !important;
            border: 1px solid #0284C7 !important;
            font-weight: 700 !important;
            box-shadow: 0 1px 3px rgba(2, 132, 199, 0.4) !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Buttons (Dark Mode) */
        [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"], button[kind="primary"] {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.25rem !important;
        }
        [data-testid="baseButton-secondary"], [data-testid="stBaseButton-secondary"], button[kind="secondary"], .stButton > button:not([kind="primary"]) {
            background-color: #10141E !important;
            color: #F8FAFC !important;
            border: 1px solid #1E2638 !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
        }
        [data-testid="baseButton-secondary"] *, [data-testid="stBaseButton-secondary"] *, button[kind="secondary"] * {
            color: #F8FAFC !important;
        }
        [data-testid="baseButton-secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
            background-color: #141E2F !important;
            border-color: #0284C7 !important;
            color: #38BDF8 !important;
        }
        [data-testid="baseButton-secondary"]:hover *, [data-testid="stBaseButton-secondary"]:hover *, button[kind="secondary"]:hover * {
            color: #38BDF8 !important;
        }

        /* Responsive Media Queries */
        @media (max-width: 768px) {
            .landing-hero {
                padding: 1.75rem 1rem !important;
            }
            .hero-title {
                font-size: 1.85rem !important;
            }
            div[data-testid="metric-container"] [data-testid="stMetricValue"] {
                font-size: 1.25rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
            box-sizing: border-box;
        }

        [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }

        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #0F172A !important;
            font-weight: 700 !important;
        }

        .stApp p, .stApp span, .stApp label, .stApp div, .stApp li {
            color: #0F172A;
        }

        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: #1E293B !important;
        }

        [data-testid="stCaptionContainer"], small {
            color: #334155 !important;
            font-weight: 500 !important;
        }

        /* Metric Cards (Light Mode) */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        }
        
        div[data-testid="metric-container"] label {
            color: #334155 !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-size: 1.45rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
            color: #0284C7 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }

        /* Hero & Surface Cards (Light Mode) */
        .landing-hero {
            background-color: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 1rem 0 1.5rem 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }

        .hero-badge {
            background-color: #EFF6FF;
            color: #0284C7;
            font-weight: 700;
            font-size: 0.82rem;
            padding: 5px 14px;
            border-radius: 20px;
            border: 1px solid #BFDBFE;
            display: inline-block;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 1.2rem 0 0.8rem 0;
            line-height: 1.2;
            color: #0F172A !important;
        }

        .hero-desc {
            font-size: 1.05rem;
            color: #334155 !important;
            max-width: 680px;
            margin: 0 auto 1.5rem auto;
            line-height: 1.6;
        }

        .feature-card {
            background-color: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            padding: 1.5rem;
            height: 100%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .feature-card h4 {
            color: #0F172A !important;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 8px 0;
        }

        .feature-desc {
            font-size: 0.88rem;
            color: #334155 !important;
            line-height: 1.5;
            margin: 0;
        }

        .feature-icon-box {
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 8px;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            margin-bottom: 12px;
        }

        .hub-surface {
            background-color: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .hub-surface h3, .hub-surface h4 {
            color: #0F172A !important;
        }

        .hub-subtext {
            color: #334155 !important;
            font-size: 0.88rem;
        }

        /* High-Contrast File Uploader (Light Mode) */
        [data-testid="stFileUploader"] section {
            background-color: #F8FAFC !important;
            border: 2px dashed #94A3B8 !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #0284C7 !important;
            background-color: #EFF6FF !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            border: 1px solid #0284C7 !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.25rem !important;
            box-shadow: 0 1px 2px rgba(2, 132, 199, 0.2) !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #0369A1 !important;
            border-color: #0369A1 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] div {
            color: #1E293B !important;
            font-weight: 500 !important;
        }

        /* Chat Messages (Light Mode) */
        .chat-bubble-user {
            background-color: #EFF6FF !important;
            border: 1px solid #BFDBFE !important;
            border-left: 4px solid #0284C7 !important;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 8px;
            color: #0F172A !important;
        }

        .chat-bubble-ai {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-left: 4px solid #059669 !important;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            color: #0F172A !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] p,
        [data-testid="stVerticalBlockBorderWrapper"] li,
        [data-testid="stVerticalBlockBorderWrapper"] code {
            color: #0F172A !important;
        }

        /* COMPLETE INPUTS, DATEPICKER & MULTISELECT OVERRIDES (LIGHT MODE) */
        [data-testid="stDateInput"],
        [data-testid="stTextInput"],
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"],
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="input"] *,
        [data-baseweb="base-input"] *,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] div,
        [data-testid="stTextInput"] div,
        [data-baseweb="select"],
        [data-baseweb="select"] *,
        [data-baseweb="popover"],
        [data-baseweb="popover"] *,
        ul[role="listbox"],
        li[role="option"],
        input,
        textarea {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            font-weight: 600 !important;
        }

        /* MultiSelect Tags / Chips (High-Contrast Sky Blue Pills) */
        div[data-baseweb="tag"],
        span[data-baseweb="tag"],
        div[data-baseweb="tag"] *,
        span[data-baseweb="tag"] * {
            background-color: #0284C7 !important;
            border-color: #0284C7 !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border-radius: 4px !important;
            padding: 2px 6px !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="tag"] svg,
        span[data-baseweb="tag"] svg {
            fill: #FFFFFF !important;
            color: #FFFFFF !important;
        }

        /* SVG Icons inside inputs (calendar icon, down arrow, search) */
        [data-baseweb="select"] svg,
        [data-testid="stDateInput"] svg,
        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg,
        svg[data-icon="calendar"] {
            fill: #334155 !important;
            color: #334155 !important;
        }

        /* Checkbox (Code Only) */
        [data-testid="stCheckbox"] label span,
        [data-testid="stCheckbox"] p {
            color: #0F172A !important;
            font-weight: 600 !important;
        }

        /* Tabs (High-Contrast Pill Tabs in Light Mode) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 6px;
            margin-bottom: 16px;
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #1E293B !important;
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
            white-space: nowrap !important;
        }
        .stTabs [data-baseweb="tab"] * {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #0284C7 !important;
            border-color: #0284C7 !important;
            background-color: #EFF6FF !important;
        }
        .stTabs [data-baseweb="tab"]:hover * {
            color: #0284C7 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;
            background-color: #0284C7 !important;
            border: 1px solid #0284C7 !important;
            font-weight: 700 !important;
            box-shadow: 0 1px 3px rgba(2, 132, 199, 0.3) !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Secondary & Primary Buttons (Light Mode) */
        [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"], button[kind="primary"] {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.25rem !important;
            box-shadow: 0 1px 2px rgba(2, 132, 199, 0.2) !important;
        }
        [data-testid="baseButton-secondary"],
        [data-testid="stBaseButton-secondary"],
        button[kind="secondary"],
        .stButton > button:not([kind="primary"]) {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        }
        [data-testid="baseButton-secondary"] *,
        [data-testid="stBaseButton-secondary"] *,
        button[kind="secondary"] *,
        .stButton > button:not([kind="primary"]) * {
            color: #0F172A !important;
            font-weight: 600 !important;
        }
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        button[kind="secondary"]:hover,
        .stButton > button:not([kind="primary"]):hover {
            background-color: #EFF6FF !important;
            border-color: #0284C7 !important;
            color: #0284C7 !important;
        }
        [data-testid="baseButton-secondary"]:hover *,
        [data-testid="stBaseButton-secondary"]:hover *,
        button[kind="secondary"]:hover *,
        .stButton > button:not([kind="primary"]):hover * {
            color: #0284C7 !important;
        }

        /* Responsive Media Queries */
        @media (max-width: 768px) {
            .landing-hero {
                padding: 1.75rem 1rem !important;
            }
            .hero-title {
                font-size: 1.85rem !important;
            }
            div[data-testid="metric-container"] [data-testid="stMetricValue"] {
                font-size: 1.25rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# --- TOP NAVBAR HEADER (THEME AWARE) ---
nav_col1, nav_col2, nav_col3 = st.columns([2.2, 1.8, 0.9])

with nav_col1:
    icon_bg = "#1E293B" if current_theme == "Dark" else "#EFF6FF"
    icon_border = "#334155" if current_theme == "Dark" else "#BFDBFE"
    icon_color = "#38BDF8" if current_theme == "Dark" else "#0284C7"
    title_color = "#F8FAFC" if current_theme == "Dark" else "#0F172A"
    
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 2px;">
            <div style="background-color: {icon_bg}; border: 1px solid {icon_border}; color: {icon_color}; border-radius: 6px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; font-weight: 700;">⚡</div>
            <div>
                <span style="font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; color: {title_color};">GPT Analytics</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with nav_col2:
    btn_home_col, btn_dash_col = st.columns(2)
    with btn_home_col:
        is_home = (st.session_state.active_tab == "Home")
        if st.button("🏠 Home & Upload", type="primary" if is_home else "secondary", use_container_width=True):
            st.session_state.active_tab = "Home"
            st.rerun()
    with btn_dash_col:
        is_dash = (st.session_state.active_tab == "Dashboard")
        if st.button("📊 Dashboard", type="primary" if is_dash else "secondary", use_container_width=True):
            st.session_state.active_tab = "Dashboard"
            st.rerun()

with nav_col3:
    def on_theme_toggle():
        st.session_state.theme = st.session_state.theme_selector

    st.selectbox(
        "Theme",
        ["Dark", "Light"],
        index=0 if current_theme == "Dark" else 1,
        key="theme_selector",
        on_change=on_theme_toggle,
        label_visibility="collapsed"
    )

# --- DATA PROCESSING FUNCTION ---
@st.cache_data(show_spinner=False)
def load_and_enrich_data(file_bytes_or_none, is_demo=False):
    if is_demo or file_bytes_or_none is None:
        raw_df = generate_sample_chat_data()
    else:
        raw_df = load_chatgpt_export(file_bytes_or_none)
    
    if raw_df.empty:
        return raw_df
        
    enriched_df = add_sentiment_to_df(raw_df)
    return enriched_df

# ==============================================================================
# VIEW 1: CLEAN RESPONSIVE LANDING PAGE
# ==============================================================================
if st.session_state.active_tab == "Home":
    
    # Hero Section
    st.markdown("""
        <div class="landing-hero">
            <h1 class="hero-title" style="margin-top: 0;">
                Understand Your AI Habits & Conversation Patterns
            </h1>
            <p class="hero-desc">
                Transform your raw ChatGPT export into interactive analytics. Discover usage streaks, peak productivity windows, NLP topic clusters, and token usage with privacy-first local processing.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Hero Action Grid
    cta_col1, cta_col2 = st.columns(2)
    with cta_col1:
        st.markdown("""
            <div class="hub-surface" style="text-align: center; padding: 1.75rem 1.25rem;">
                <div class="feature-icon-box" style="margin: 0 auto 10px auto;">🚀</div>
                <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 700;">Explore with Demo Data</h3>
                <p class="hub-subtext" style="margin-bottom: 1.25rem;">
                    Instantly load mock conversation history to test all interactive charts, heatmaps, and sentiment tools.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Launch Live Demo", use_container_width=True, type="primary"):
            st.session_state.data_mode = "Demo Dataset"
            st.session_state.active_tab = "Dashboard"
            st.rerun()

    with cta_col2:
        st.markdown("""
            <div class="hub-surface" style="text-align: center; padding: 1.75rem 1.25rem;">
                <div class="feature-icon-box" style="margin: 0 auto 10px auto;">📁</div>
                <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 700;">Upload conversations.json</h3>
                <p class="hub-subtext" style="margin-bottom: 0.8rem;">
                    Load your personal ChatGPT export file. Processed 100% locally on your machine.
                </p>
            </div>
        """, unsafe_allow_html=True)
        uploaded_landing_file = st.file_uploader(
            "Upload JSON",
            type=["json"],
            label_visibility="collapsed",
            help="Extract from ChatGPT export ZIP -> conversations.json"
        )
        if uploaded_landing_file is not None:
            st.session_state.uploaded_file_bytes = uploaded_landing_file.getvalue()
            st.session_state.data_mode = "Upload Export (.json)"
            st.session_state.active_tab = "Dashboard"
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Feature Grid
    st.markdown("<h3 style='text-align: center; font-size: 1.35rem; font-weight: 700; margin-bottom: 1.25rem;'>Everything You Need To Know</h3>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    
    with f1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box">⏰</div>
                <h4>Habit & Streak Matrices</h4>
                <p class="feature-desc">
                    Visualize when you brainstorm most with Day-of-Week vs Hour heatmaps, daily volume trends, and consecutive active streaks.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box">🧠</div>
                <h4>NLP & Topic Discovery</h4>
                <p class="feature-desc">
                    Automatic conversation clustering with theme keyword tags, frequent phrase rankings, and sentiment scoring.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box">💬</div>
                <h4>Deep Reader & Search</h4>
                <p class="feature-desc">
                    Instant keyword search across your entire archive with an interactive conversation viewer and token/effort estimation.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # 3-Step Export Guide
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    guide_link_color = "#38BDF8" if current_theme == "Dark" else "#0284C7"
    st.markdown(f"""
        <div class="hub-surface">
            <h4 style="margin: 0 0 12px 0; font-size: 1.1rem; font-weight: 700;">📖 How to Export Your Data from ChatGPT</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; font-size: 0.88rem; line-height: 1.6;" class="hub-subtext">
                <div>
                    <b style="color: inherit;">1. Request Export</b><br>
                    Go to <a href="https://chatgpt.com" target="_blank" style="color: {guide_link_color}; text-decoration: none; font-weight: 600;">chatgpt.com</a> &rarr; Settings &rarr; Data Controls &rarr; Export Data.
                </div>
                <div>
                    <b style="color: inherit;">2. Download ZIP</b><br>
                    Open the confirmation email sent by OpenAI and download your archive file.
                </div>
                <div>
                    <b style="color: inherit;">3. Drop conversations.json</b><br>
                    Extract the ZIP, upload <code>conversations.json</code> above, and explore your personal dashboard.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: ANALYTICS DASHBOARD
# ==============================================================================
else:
    # Resolve dataset
    if st.session_state.data_mode == "Demo Dataset":
        df = load_and_enrich_data(None, is_demo=True)
    elif st.session_state.uploaded_file_bytes is not None:
        df = load_and_enrich_data(st.session_state.uploaded_file_bytes, is_demo=False)
    else:
        df = load_and_enrich_data(None, is_demo=True)

    if df.empty:
        st.warning("No data found or failed to parse JSON file. Switch to Demo mode or upload a valid conversations.json.")
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("🚀 Load Demo Dataset", type="primary"):
                st.session_state.data_mode = "Demo Dataset"
                st.rerun()
        with col_fb2:
            if st.button("📁 Return to Upload Screen"):
                st.session_state.active_tab = "Home"
                st.rerun()
        st.stop()

    # --- TOP TOOLBAR FILTERS (RESPONSIVE INLINE) ---
    min_date = df["message_time"].min().date()
    max_date = df["message_time"].max().date()

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.2, 1, 0.8])
    with f_col1:
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed"
        )
    with f_col2:
        role_filter = st.multiselect(
            "Roles",
            options=df["role"].unique().tolist(),
            default=df["role"].unique().tolist(),
            label_visibility="collapsed",
            placeholder="Filter Roles"
        )
    with f_col3:
        freq_option = st.selectbox(
            "Aggregation Interval",
            ["Daily", "Weekly", "Monthly"],
            index=0,
            label_visibility="collapsed"
        )
        freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
    with f_col4:
        only_code = st.checkbox("Code Only", value=False)

    # Filter dataframe
    filtered_df = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[
            (filtered_df["message_time"].dt.date >= start_d) & 
            (filtered_df["message_time"].dt.date <= end_d)
        ]
        
    if role_filter:
        filtered_df = filtered_df[filtered_df["role"].isin(role_filter)]
        
    if only_code:
        filtered_df = filtered_df[filtered_df["has_code"] == True]

    if filtered_df.empty:
        st.warning("No messages match the filter selection.")
        st.stop()

    stats = compute_basic_stats(filtered_df)

    # --- KPI METRICS BAR ---
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Conversations", f"{stats['total_conversations']:,}")
    k2.metric("Total Messages", f"{stats['total_messages']:,}")
    k3.metric("Total Words", f"{stats['total_words']:,}")
    k4.metric("Est. Tokens", f"{stats['total_tokens']:,}")
    k5.metric("Avg Thread", f"{stats['avg_conversation_length']} msgs")
    k6.metric("Active Streak", f"{stats['current_streak']} days", delta=f"Max {stats['max_streak']}d")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --- MODULAR TABS ---
    t_overview, t_habits, t_nlp, t_chat, t_tokens, t_export = st.tabs([
        "📊 Overview",
        "⏰ Habits & Timing",
        "🧠 NLP & Topics",
        "💬 Conversation Explorer",
        "⚡ Tokens & Effort",
        "📥 Export"
    ])

    # -------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------
    with t_overview:
        c_trend, c_dist = st.columns([2.2, 1])
        with c_trend:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Activity Over Time</p>", unsafe_allow_html=True)
            timeline_data = get_activity_over_time(filtered_df, freq=freq_map[freq_option])
            st.plotly_chart(activity_timeline_chart(timeline_data, theme=current_theme), use_container_width=True)
        with c_dist:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Role Proportion</p>", unsafe_allow_html=True)
            st.plotly_chart(role_distribution_chart(filtered_df, theme=current_theme), use_container_width=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        c_len, c_top = st.columns([1.2, 1.8])
        with c_len:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Message Length Distribution</p>", unsafe_allow_html=True)
            st.plotly_chart(message_length_distribution(filtered_df, theme=current_theme), use_container_width=True)
        with c_top:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>🏆 Top Active Conversations</p>", unsafe_allow_html=True)
            leaderboard = get_conversation_leaderboard(filtered_df, top_n=6, sort_by="messages")
            if not leaderboard.empty:
                st.dataframe(
                    leaderboard[["title", "messages", "total_words", "start_time"]].rename(
                        columns={"title": "Conversation Title", "messages": "Messages", "total_words": "Words", "start_time": "Started"}
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    # -------------------------------------------------------------
    # TAB 2: HABITS & TIMING
    # -------------------------------------------------------------
    with t_habits:
        st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Interaction Heatmap (Day of Week vs. Hour)</p>", unsafe_allow_html=True)
        heat_data = activity_heatmap_data(filtered_df)
        st.plotly_chart(hourly_heatmap(heat_data, theme=current_theme), use_container_width=True)

        col_hb1, col_hb2 = st.columns([1.5, 1])
        with col_hb1:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Volume by Weekday</p>", unsafe_allow_html=True)
            weekday_df = get_weekday_activity(filtered_df)
            st.plotly_chart(weekday_bar_chart(weekday_df, theme=current_theme), use_container_width=True)
        with col_hb2:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Productivity Highlights</p>", unsafe_allow_html=True)
            hourly_df = get_hourly_activity(filtered_df)
            peak_row = hourly_df.loc[hourly_df["count"].idxmax()] if not hourly_df.empty else None
            peak_str = f"{int(peak_row['hour']):02d}:00 - {int(peak_row['hour'])+1:02d}:00" if peak_row is not None else "N/A"
            
            sub_label_color = "#94A3B8" if current_theme == "Dark" else "#475569"
            st.markdown(f"""
            <div class="hub-surface">
                <p style="margin: 0 0 6px 0; font-size: 0.78rem; font-weight: 700; color: {sub_label_color}; letter-spacing: 0.04em;">PEAK INTERACTION WINDOW</p>
                <p style="font-size: 1.45rem; font-weight: 800; margin: 0 0 12px 0;">{peak_str}</p>
                <div style="border-top: 1px solid rgba(128,128,128,0.2); padding-top: 10px; font-size: 0.88rem; line-height: 1.9;">
                    <div>• <b>Total Active Days:</b> {stats['active_days']} days</div>
                    <div>• <b>Longest Consecutive Streak:</b> {stats['max_streak']} days</div>
                    <div>• <b>Current Streak:</b> {stats['current_streak']} days</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: NLP & TOPICS
    # -------------------------------------------------------------
    with t_nlp:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Frequent User Concepts</p>", unsafe_allow_html=True)
            ngrams = extract_top_ngrams(filtered_df, ngram_range=(1, 1), top_n=10, role="user")
            st.plotly_chart(top_ngrams_bar_chart(ngrams, theme=current_theme), use_container_width=True)
        with col_t2:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>🎯 Topic Clusters (Keyword Grouping)</p>", unsafe_allow_html=True)
            clusters_df = cluster_topics_with_labels(filtered_df, n_clusters=4)
            if not clusters_df.empty and "cluster_name" in clusters_df.columns:
                c_summary = clusters_df.groupby("cluster_name").size().reset_index(name="Threads").sort_values("Threads", ascending=False)
                st.dataframe(c_summary.rename(columns={"cluster_name": "Cluster Keywords"}), use_container_width=True, hide_index=True)
            else:
                st.info("Insufficient data for clustering.")

        st.divider()
        col_s1, col_s2, col_s3 = st.columns([1, 1.2, 1])
        with col_s1:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>User Sentiment Breakdown</p>", unsafe_allow_html=True)
            st.plotly_chart(sentiment_breakdown_chart(filtered_df, theme=current_theme), use_container_width=True)
        with col_s2:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Sentiment Trend</p>", unsafe_allow_html=True)
            sent_plot = sentiment_trend_chart(filtered_df, theme=current_theme)
            if sent_plot:
                st.plotly_chart(sent_plot, use_container_width=True)
            else:
                st.info("Sentiment trend unavailable for selected range.")
        with col_s3:
            st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Detected Languages & Code</p>", unsafe_allow_html=True)
            lang_df = detect_code_languages(filtered_df)
            st.plotly_chart(code_languages_chart(lang_df, theme=current_theme), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: CONVERSATION EXPLORER
    # -------------------------------------------------------------
    with t_chat:
        st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Search & Thread Inspector</p>", unsafe_allow_html=True)
        search_kw = st.text_input(
            "Search keywords",
            placeholder="Type search terms (e.g. 'Postgres', 'FastAPI', 'Itinerary')...",
            label_visibility="collapsed"
        )

        col_nav, col_body = st.columns([1.1, 2])
        
        all_convs = filtered_df[["conversation_id", "conversation_title"]].drop_duplicates()
        if search_kw.strip():
            matched_ids = filtered_df[filtered_df["content"].str.contains(search_kw, case=False, na=False)]["conversation_id"].unique()
            all_convs = all_convs[all_convs["conversation_id"].isin(matched_ids)]

        with col_nav:
            st.markdown(f"**Threads ({len(all_convs)})**")
            if not all_convs.empty:
                selected_cid = st.selectbox(
                    "Select thread",
                    options=all_convs["conversation_id"].tolist(),
                    format_func=lambda cid: all_convs.loc[all_convs["conversation_id"] == cid, "conversation_title"].iloc[0],
                    label_visibility="collapsed"
                )
            else:
                st.warning("No conversations match your search.")
                selected_cid = None

        with col_body:
            if selected_cid:
                thread_msgs = filtered_df[filtered_df["conversation_id"] == selected_cid].sort_values("message_time")
                header_msg = thread_msgs.iloc[0]
                
                st.markdown(f"#### {header_msg['conversation_title']}")
                st.caption(f"Started {header_msg['message_time'].strftime('%b %d, %Y at %H:%M')} • {len(thread_msgs)} messages")
                
                box = st.container(height=480)
                with box:
                    for _, msg_row in thread_msgs.iterrows():
                        is_user = msg_row["role"] == "user"
                        sender_tag = "👤 You" if is_user else "🤖 Assistant"
                        box_class = "chat-bubble-user" if is_user else "chat-bubble-ai"
                        t_str = msg_row["message_time"].strftime("%H:%M") if pd.notna(msg_row["message_time"]) else ""
                        tag_color = "#0284C7" if is_user else "#059669"
                        time_color = "#94A3B8" if current_theme == "Dark" else "#64748B"
                        
                        st.markdown(f"""
                        <div class="{box_class}">
                            <div style="font-size: 0.8rem; font-weight: 700; color: {tag_color}; margin-bottom: 4px;">
                                {sender_tag} <span style="color: {time_color}; font-weight: 500; font-size: 0.75rem; float: right;">{t_str}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(msg_row["content"])
                        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 5: TOKENS & EFFORT
    # -------------------------------------------------------------
    with t_tokens:
        t_estimates = get_token_cost_estimation(filtered_df)
        if t_estimates:
            c_tk1, c_tk2, c_tk3, c_tk4 = st.columns(4)
            c_tk1.metric("Input Tokens (Prompt)", f"{t_estimates['input_tokens']:,}")
            c_tk2.metric("Output Tokens (AI)", f"{t_estimates['output_tokens']:,}")
            c_tk3.metric("Est. Cost (GPT-4o)", f"${t_estimates['gpt4o_cost']:.2f}")
            c_tk4.metric("Est. Cost (GPT-4o-mini)", f"${t_estimates['mini_cost']:.2f}")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            col_hr1, col_hr2 = st.columns(2)
            sub_label_color = "#94A3B8" if current_theme == "Dark" else "#475569"
            with col_hr1:
                st.markdown(f"""
                <div class="hub-surface">
                    <p style="margin: 0 0 6px 0; font-size: 0.78rem; font-weight: 700; color: {sub_label_color}; letter-spacing: 0.04em;">HUMAN EFFORT INVESTED</p>
                    <p style="font-size: 1.8rem; font-weight: 800; color: #0284C7; margin: 0 0 10px 0;">
                        {t_estimates['typing_hours']} <span style="font-size: 0.95rem; font-weight: 600; color: inherit;">hours typing</span>
                    </p>
                    <p style="font-size: 0.88rem; line-height: 1.5; margin: 0;">
                        Estimated typing time based on ~40 WPM. You wrote <b>{stats['total_user_words']:,} words</b> and asked <b>{stats['total_questions']:,} questions</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col_hr2:
                st.markdown(f"""
                <div class="hub-surface">
                    <p style="margin: 0 0 6px 0; font-size: 0.78rem; font-weight: 700; color: {sub_label_color}; letter-spacing: 0.04em;">ASSISTANT READING TIME</p>
                    <p style="font-size: 1.8rem; font-weight: 800; color: #059669; margin: 0 0 10px 0;">
                        {t_estimates['reading_hours']} <span style="font-size: 0.95rem; font-weight: 600; color: inherit;">hours reading</span>
                    </p>
                    <p style="font-size: 0.88rem; line-height: 1.5; margin: 0;">
                        Estimated human reading time based on ~250 WPM. Assistant provided <b>{stats['total_assistant_words']:,} words</b> across all conversations.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 6: EXPORT
    # -------------------------------------------------------------
    with t_export:
        st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 8px;'>Export Structured History</p>", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_payload = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Structured CSV",
                data=csv_payload,
                file_name=f"chatgpt_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        with col_dl2:
            summary_str = f"""ChatGPT Intelligence Hub Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
--------------------------------------------------
Total Conversations: {stats['total_conversations']}
Total Messages: {stats['total_messages']}
User Messages: {stats['user_messages']}
Assistant Messages: {stats['assistant_messages']}
Total Words: {stats['total_words']}
Estimated Tokens: {stats['total_tokens']}
Longest Streak: {stats['max_streak']} days
Active Days: {stats['active_days']}
Estimated Cost (GPT-4o equivalent): ${t_estimates.get('gpt4o_cost', 0):.2f}
"""
            st.download_button(
                label="📄 Download Summary Report (TXT)",
                data=summary_str,
                file_name=f"chatgpt_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>Raw Data Table</p>", unsafe_allow_html=True)
        st.dataframe(
            filtered_df[["message_time", "conversation_title", "role", "word_count", "estimated_tokens", "sentiment_category", "content"]],
            use_container_width=True,
            height=300
        )

# --- FOOTER ---
footer_color = "#94A3B8" if current_theme == "Dark" else "#64748B"
st.markdown(f"""
<div style="text-align: center; margin-top: 3.5rem; padding-bottom: 1.5rem; color: {footer_color}; font-size: 0.8rem; border-top: 1px solid rgba(128,128,128,0.15); padding-top: 1rem; font-weight: 500;">
    GPT Analytics • Clean Modern Intelligence
</div>
""", unsafe_allow_html=True)
