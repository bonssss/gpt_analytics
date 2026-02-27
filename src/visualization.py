
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

def activity_line_chart(df):
    daily = df.set_index("message_time").groupby(pd.Grouper(freq="D")).size().reset_index(name="messages")
    fig = px.area(daily, x="message_time", y="messages", 
                  title="Interaction Volume Over Time",
                  template="plotly_dark",
                  color_discrete_sequence=["#00D4FF"])
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Messages")
    return fig

def role_distribution_chart(df):
    role_counts = df["role"].value_counts().reset_index()
    role_counts.columns = ["role", "count"]
    fig = px.pie(role_counts, values="count", names="role", 
                 title="User vs Assistant Distribution",
                 hole=0.4,
                 template="plotly_dark",
                 color_discrete_sequence=px.colors.sequential.Tealgrn)
    return fig

def hourly_heatmap(pivot_data):
    fig = px.imshow(pivot_data, 
                    labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    aspect="auto",
                    title="Usage Peak Hours",
                    template="plotly_dark",
                    color_continuous_scale="Viridis")
    return fig

def sentiment_trend_chart(df):
    # Filter for user messages and resample
    user_df = df[df["role"] == "user"].copy()
    if "sentiment" not in user_df.columns:
        return None
    
    daily_sentiment = user_df.set_index("message_time").groupby(pd.Grouper(freq="D"))["sentiment"].mean().reset_index()
    fig = px.line(daily_sentiment, x="message_time", y="sentiment", 
                  title="User Sentiment Trend",
                  template="plotly_dark",
                  line_shape="spline")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig

def generate_wordcloud(df):
    text = " ".join(df[df["role"] == "user"]["content"].astype(str))
    if not text.strip():
        text = "No messages found"
        
    wc = WordCloud(width=1200, height=600, 
                   background_color="#0E1117", 
                   colormap="cool",
                   max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0E1117")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig
