import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

def get_template(theme):
    return "plotly_dark" if theme == "Dark" else "plotly_white"

def activity_line_chart(df, theme="Dark"):
    daily = df.set_index("message_time").groupby(pd.Grouper(freq="D")).size().reset_index(name="messages")
    fig = px.area(daily, x="message_time", y="messages", 
                  title="Interaction Volume Over Time",
                  template=get_template(theme),
                  color_discrete_sequence=["#00D4FF"])
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Messages")
    return fig

def role_distribution_chart(df, theme="Dark"):
    role_counts = df["role"].value_counts().reset_index()
    role_counts.columns = ["role", "count"]
    fig = px.pie(role_counts, values="count", names="role", 
                 title="User vs Assistant Distribution",
                 hole=0.4,
                 template=get_template(theme),
                 color_discrete_sequence=px.colors.sequential.Tealgrn)
    return fig

def hourly_heatmap(pivot_data, theme="Dark"):
    fig = px.imshow(pivot_data, 
                    labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    aspect="auto",
                    title="Usage Peak Hours",
                    template=get_template(theme),
                    color_continuous_scale="Viridis")
    return fig

def sentiment_trend_chart(df, theme="Dark"):
    # Filter for user messages and resample
    user_df = df[df["role"] == "user"].copy()
    if "sentiment" not in user_df.columns:
        return None
    
    daily_sentiment = user_df.set_index("message_time").groupby(pd.Grouper(freq="D"))["sentiment"].mean().reset_index()
    fig = px.line(daily_sentiment, x="message_time", y="sentiment", 
                  title="User Sentiment Trend",
                  template=get_template(theme),
                  line_shape="spline")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig

def generate_wordcloud(df, theme="Dark"):
    text = " ".join(df[df["role"] == "user"]["content"].astype(str))
    if not text.strip():
        text = "No messages found"
        
    bg_color = "#0E1117" if theme == "Dark" else "white"
    
    wc = WordCloud(width=1200, height=600, 
                   background_color=bg_color, 
                   colormap="cool",
                   max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

def message_length_distribution(df, theme="Dark"):
    # Filter out very long outliers for better visualization, limit to 95th percentile
    q95 = df["word_count"].quantile(0.95)
    filtered_df = df[df["word_count"] <= q95] if not pd.isna(q95) else df
    
    fig = px.histogram(filtered_df, x="word_count", color="role", 
                       barmode="overlay",
                       title="Message Length Distribution (Words)",
                       template=get_template(theme),
                       opacity=0.7,
                       color_discrete_map={"user": "#00D4FF", "assistant": "#FF4B4B"})
    fig.update_xaxes(title="Word Count")
    fig.update_yaxes(title="Frequency")
    return fig
