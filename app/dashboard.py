
import streamlit as st
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
    sentiment_trend_chart
)
from src.nlp import add_sentiment_to_df, cluster_topics

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ChatGPT Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload ChatGPT Export (JSON)", type=["json"])
    
    st.divider()
    st.markdown("### Visualization Options")
    theme = st.selectbox("Chart Theme", ["Dark", "Light"])
    clustering_on = st.checkbox("Enable Topic Clustering", value=False)
    
    st.divider()
    st.info("Export your data from ChatGPT settings and upload 'conversations.json' here.")

# --- MAIN CONTENT ---
st.title("📊 ChatGPT Analytics Dashboard")
st.markdown("---")

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
                st.plotly_chart(activity_line_chart(df), use_container_width=True)
            with c2:
                st.plotly_chart(role_distribution_chart(df), use_container_width=True)

            # --- ROW 3: HEATMAP & WORDCLOUD ---
            st.markdown("### 🧬 Content Patterns")
            col_a, col_b = st.columns(2)
            with col_a:
                pivot = activity_heatmap_data(df)
                st.plotly_chart(hourly_heatmap(pivot), use_container_width=True)
            with col_b:
                st.write("#### Most Frequent Terms")
                fig_wc = generate_wordcloud(df)
                st.pyplot(fig_wc)

            # --- ROW 4: NLP INSIGHTS ---
            st.markdown("### 🧠 Advanced Insights")
            col_x, col_y = st.columns(2)
            
            with col_x:
                sent_fig = sentiment_trend_chart(df)
                if sent_fig:
                    st.plotly_chart(sent_fig, use_container_width=True)
            
            with col_y:
                if clustering_on:
                    st.write("#### Topic Clusters")
                    clusters = cluster_topics(df)
                    st.write(clusters.value_counts().rename_axis("Cluster").reset_index(name="Count"))
                else:
                    st.info("Enable 'Topic Clustering' in the sidebar to view conversation groupings.")

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
            )
else:
    st.warning("Please upload a 'conversations.json' file using the sidebar to begin.")
    
    # Show sample empty layout
    st.info("Preview of the dashboard structure below:")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=2070", caption="Interactive Analytics Preview")
