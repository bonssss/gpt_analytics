
import streamlit as st

st.set_page_config(
    page_title="ChatGPT Intelligence Dashboard",
    page_icon="🧠",
    layout="centered"
)

import pandas as pd
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Download necessary NLP data
import nltk
@st.cache_resource
def download_nlp_data():
    try:
        nltk.download('punkt')
        nltk.download('brown')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
    except:
        pass

download_nlp_data()

from src.parser import load_chatgpt_export
from src.analytics import compute_basic_stats, activity_heatmap_data, get_activity_over_time
from src.visualization import (
    activity_line_chart,
    role_distribution_chart,
    generate_wordcloud,
    hourly_heatmap,
    sentiment_trend_chart,
    message_length_distribution
)
from src.nlp import add_sentiment_to_df, cluster_topics

# --- PAGE CONFIG ---


# --- CUSTOM STYLING ---
st.markdown("""
<style>
    /* Gradient Title */
    .title-text {
        background: -webkit-linear-gradient(45deg, #1cb5e0, #000851);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        padding-bottom: 10px;
        margin-top: -40px;
    }
    
    @media (prefers-color-scheme: dark) {
        .title-text {
            background: -webkit-linear-gradient(45deg, #4ECDC4, #FF6B6B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }

    .subtitle-text {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(28, 131, 225, 0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .stMetric:hover {
        transform: translateY(-2px);
    }
    
    /* Make File Uploader look premium (colors applied dynamically by theme) */
    [data-testid='stFileUploader'] section {
        padding: 3rem;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown("<h1 class='title-text'>🧠 ChatGPT Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Unlock insights and deep patterns from your conversation history</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload ChatGPT Export (JSON) directly below", type=["json"])

# Consolidated Settings
with st.expander("⚙️ Settings & Visualization Options", expanded=False):
    col_setup1, col_setup2 = st.columns(2)
    with col_setup1:
        theme = st.selectbox("App Theme", ["Dark", "Light"])
    with col_setup2:
        clustering_on = st.checkbox("Enable Topic Clustering", value=False)

# Apply global theme CSS
if theme == "Light":
    st.markdown('''
        <style>
            /* Light Theme Overrides */
            .stApp { background-color: #F0F2F6 !important; }
            .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp label { color: #31333F !important; text-shadow: none !important; }
            header[data-testid="stHeader"] { background-color: transparent !important; }
            
            /* UI Elements Fixes for Light Mode */
            [data-testid="stExpander"] details summary { background-color: #FFFFFF !important; color: #31333F !important; border-radius: 8px; border: 1px solid #E6E9EF !important; }
            [data-testid="stExpander"] details summary p { color: #31333F !important; }
            [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #31333F !important; border-color: #E6E9EF !important; }
            [data-baseweb="select"] span { color: #31333F !important; }
            
            /* Buttons */
            [data-testid="baseButton-secondary"], [data-testid="stFileUploader"] button { background-color: #FFFFFF !important; color: #31333F !important; border: 1px solid #E6E9EF !important; }
            [data-testid="baseButton-secondary"]:hover, [data-testid="stFileUploader"] button:hover { border-color: #FF4B4B !important; color: #FF4B4B !important; background-color: #FFFFFF !important; }
            [data-testid="baseButton-primary"] { background-color: #FF4B4B !important; color: #FFFFFF !important; border-color: #FF4B4B !important; }
            [data-testid="baseButton-primary"]:hover { background-color: #FF3333 !important; border-color: #FF3333 !important; color: #FFFFFF !important; }
            
            /* Inputs */
            [data-baseweb="input"], [data-baseweb="base-input"] { background-color: #FFFFFF !important; }
            [data-baseweb="input"] input { color: #31333F !important; background-color: #FFFFFF !important; }
            
            .stMetric { background-color: #FFFFFF !important; border: 1px solid #E6E9EF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important; }
            [data-testid='stFileUploader'] section { background-color: #FFFFFF !important; border: 2px dashed #D2D6DF !important; }
            [data-testid='stFileUploader'] section:hover { border-color: #4ECDC4 !important; background-color: rgba(78, 205, 196, 0.1) !important; }
            .subtitle-text, [data-testid="stMarkdownContainer"] p.subtitle-text { color: #555 !important; }
        </style>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
        <style>
            /* Dark Theme Overrides */
            .stApp { background-color: #0E1117 !important; }
            .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp label { color: #FAFAFA !important; }
            header[data-testid="stHeader"] { background-color: transparent !important; }
            
            /* UI Elements Fixes for Dark Mode */
            [data-testid="stExpander"] details summary { background-color: #262730 !important; color: #FAFAFA !important; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1) !important; }
            [data-testid="stExpander"] details summary p { color: #FAFAFA !important; }
            [data-baseweb="select"] > div { background-color: #262730 !important; color: #FAFAFA !important; border-color: rgba(255,255,255,0.1) !important; }
            [data-baseweb="select"] span { color: #FAFAFA !important; }
            ul[data-baseweb="menu"] { background-color: #262730 !important; color: #FAFAFA !important; }
            [data-baseweb="checkbox"] div { color: #FAFAFA !important; }
            
            /* Buttons */
            [data-testid="baseButton-secondary"], [data-testid="stFileUploader"] button { background-color: #262730 !important; color: #FAFAFA !important; border: 1px solid rgba(255,255,255,0.1) !important; }
            [data-testid="baseButton-secondary"]:hover, [data-testid="stFileUploader"] button:hover { border-color: #FF4B4B !important; color: #FF4B4B !important; background-color: #262730 !important; }
            [data-testid="baseButton-primary"] { background-color: #FF4B4B !important; color: #FFFFFF !important; border-color: #FF4B4B !important; }
            [data-testid="baseButton-primary"]:hover { background-color: #FF3333 !important; border-color: #FF3333 !important; color: #FFFFFF !important; }
            
            /* Inputs */
            [data-baseweb="input"], [data-baseweb="base-input"] { background-color: #262730 !important; }
            [data-baseweb="input"] input { color: #FAFAFA !important; background-color: #262730 !important; -webkit-text-fill-color: #FAFAFA !important; }

            .stMetric { background-color: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }
            [data-testid='stFileUploader'] section { background-color: rgba(128, 128, 128, 0.05) !important; border: 2px dashed rgba(128, 128, 128, 0.3) !important; }
            [data-testid='stFileUploader'] section:hover { border-color: #4ECDC4 !important; background-color: rgba(78, 205, 196, 0.05) !important; }
            .subtitle-text, [data-testid="stMarkdownContainer"] p.subtitle-text { color: #888 !important; }
        </style>
    ''', unsafe_allow_html=True)

if uploaded_file:

    with st.spinner("Processing your intelligence data..."):
        df = load_chatgpt_export(uploaded_file)
        
        if df.empty:
            st.error("Could not parse the JSON file. Please ensure it's a valid ChatGPT export.")
        else:
            # Enrich data
            df = add_sentiment_to_df(df)
            
            # Basic Stats
            stats = compute_basic_stats(df)
            
            # --- ROW 1: METRICS ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conversations", f"{stats['total_conversations']:,}")
            m2.metric("Total Messages", f"{stats['total_messages']:,}")
            m3.metric("Avg Conversation Length", f"{stats['avg_conversation_length']:.1f} msgs")
            m4.metric("User Word Count", f"{stats['total_user_words']:,}")

            st.markdown("### 📈 Engagement Trends")
            
            # --- ROW 2: ACTIVITY OVER TIME ---
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(activity_line_chart(df, theme=theme), use_container_width=True)
            with c2:
                st.plotly_chart(role_distribution_chart(df, theme=theme), use_container_width=True)

            # --- ROW 3: HEATMAP & WORDCLOUD ---
            st.markdown("### 🧬 Content Patterns")
            col_a, col_b = st.columns(2)
            with col_a:
                pivot = activity_heatmap_data(df)
                st.plotly_chart(hourly_heatmap(pivot, theme=theme), use_container_width=True)
            with col_b:
                st.write("#### Most Frequent Terms")
                fig_wc = generate_wordcloud(df, theme=theme)
                st.pyplot(fig_wc)

            # --- ROW 3.5: NEW FEATURE ---
            st.markdown("### 📏 Message Analytics")
            st.plotly_chart(message_length_distribution(df, theme=theme), use_container_width=True)

            # --- ROW 4: NLP INSIGHTS ---
            st.markdown("### 🧠 Advanced Insights")
            col_x, col_y = st.columns(2)
            
            with col_x:
                sent_fig = sentiment_trend_chart(df, theme=theme)
                if sent_fig:
                    st.plotly_chart(sent_fig, use_container_width=True)
            
            with col_y:
                if clustering_on:
                    st.write("#### Topic Clusters")
                    clusters = cluster_topics(df)
                    st.write(clusters.value_counts().rename_axis("Cluster").reset_index(name="Count"))
                else:
                    st.info("Enable 'Topic Clustering' in the settings expander to view conversation groupings.")

            # --- SEARCH SECTION ---
            st.divider()
            st.subheader("🔍 Intelligence Search")
            q = st.text_input("Search through your history...", placeholder="Type keywords here (e.g. 'Python', 'Travel', 'Recipe')")
            if q:
                search_results = df[df["content"].str.contains(q, case=False)]
                st.success(f"Found {len(search_results)} matching messages.")
                st.dataframe(search_results[["message_time", "conversation_title", "role", "content"]], use_container_width=True)

            # --- EXPORT ---
            st.divider()
            st.download_button(
                label="📥 Download Processed CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='chatgpt_history_processed.csv',
                mime='text/csv',
                type="primary",
            )
else:
    # Custom colored warning container
    st.markdown(
        """
        <div style="background-color: rgba(255, 170, 0, 0.15); padding: 1rem; border-radius: 0.5rem; border: 1px solid rgba(255, 170, 0, 0.3); margin-bottom: 1rem;">
            ⚠️ Please upload a 'conversations.json' file using the uploader above to begin.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()
    st.info("Export your data from ChatGPT settings and upload 'conversations.json' here.")

# --- FOOTER ---
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; padding-bottom: 20px; color: #888; font-size: 0.9rem;">
        Developed by Bonsa
    </div>
    """,
    unsafe_allow_html=True
)
