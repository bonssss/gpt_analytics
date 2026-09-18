import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

THEME_CONFIG = {
    "Dark": {
        "text": "#E2E8F0",
        "subtext": "#94A3B8",
        "card_bg": "#111827",
        "border": "#1F2937",
        "grid": "#1F2937",
        "primary": "#38BDF8",      # Sky 400
        "secondary": "#34D399",    # Emerald 400
        "accent": "#818CF8",       # Indigo 400
        "warning": "#FBBF24",      # Amber 400
        "danger": "#F87171",       # Red 400
        "neutral": "#64748B",
        "bar_sequence": ["#38BDF8", "#34D399", "#818CF8", "#FBBF24", "#F472B6", "#A78BFA"]
    },
    "Light": {
        "text": "#0F172A",
        "subtext": "#334155",
        "card_bg": "#FFFFFF",
        "border": "#CBD5E1",
        "grid": "#E2E8F0",
        "primary": "#0284C7",      # Sky 600
        "secondary": "#059669",    # Emerald 600
        "accent": "#4F46E5",       # Indigo 600
        "warning": "#D97706",      # Amber 600
        "danger": "#DC2626",       # Red 600
        "neutral": "#64748B",
        "bar_sequence": ["#0284C7", "#059669", "#4F46E5", "#D97706", "#DB2777", "#7C3AED"]
    }
}

def clean_chart_base(fig, theme="Dark", height=340):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color=t["text"], size=12),
        margin=dict(l=15, r=15, t=35, b=20),
        height=height,
        hoverlabel=dict(
            bgcolor=t["card_bg"],
            bordercolor=t["border"],
            font_size=12,
            font_family="Inter, -apple-system, sans-serif",
            font_color=t["text"]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=t["subtext"], size=11)
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=t["grid"],
        gridwidth=1,
        zerolinecolor=t["grid"],
        linecolor=t["grid"],
        tickfont=dict(color=t["subtext"], size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["grid"],
        gridwidth=1,
        zerolinecolor=t["grid"],
        linecolor=t["grid"],
        tickfont=dict(color=t["subtext"], size=11)
    )
    return fig

def activity_timeline_chart(timeline_df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    fig = go.Figure()

    if not timeline_df.empty:
        fig.add_trace(go.Bar(
            x=timeline_df["message_time"],
            y=timeline_df["user_messages"],
            name="Your Prompts",
            marker=dict(color=t["primary"], line=dict(width=0))
        ))
        fig.add_trace(go.Bar(
            x=timeline_df["message_time"],
            y=timeline_df["assistant_messages"],
            name="AI Responses",
            marker=dict(color=t["secondary"], line=dict(width=0))
        ))
        fig.add_trace(go.Scatter(
            x=timeline_df["message_time"],
            y=timeline_df["cumulative_messages"],
            name="Cumulative Volume",
            yaxis="y2",
            mode="lines",
            line=dict(color=t["accent"], width=2.5)
        ))

    fig.update_layout(
        barmode="stack",
        yaxis=dict(title=dict(text="Messages", font=dict(color=t["subtext"], size=11))),
        yaxis2=dict(
            title=dict(text="Cumulative", font=dict(color=t["accent"], size=11)),
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color=t["accent"], size=11)
        )
    )
    return clean_chart_base(fig, theme=theme, height=360)

def role_distribution_chart(df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    role_counts = df["role"].value_counts().reset_index()
    role_counts.columns = ["role", "count"]
    
    color_map = {
        "user": t["primary"],
        "assistant": t["secondary"],
        "system": t["neutral"],
        "tool": t["accent"]
    }
    colors = [color_map.get(r, t["neutral"]) for r in role_counts["role"]]

    fig = go.Figure(data=[go.Pie(
        labels=role_counts["role"].str.capitalize(),
        values=role_counts["count"],
        hole=0.68,
        marker=dict(colors=colors, line=dict(color=t["border"], width=1.5)),
        textinfo="percent",
        textposition="outside",
        hoverinfo="label+value+percent",
        showlegend=True
    )])
    
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{len(df):,}</b><br><span style='font-size:11px; color:{t['subtext']}'>Messages</span>",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
            font_color=t["text"]
        )]
    )
    return clean_chart_base(fig, theme=theme, height=360)

def hourly_heatmap(pivot_data, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    
    # Clean solid step ramp
    scale = [
        [0.0, t["border"]],
        [0.25, "#0E7490" if theme == "Dark" else "#E0F2FE"],
        [0.60, "#0284C7" if theme == "Dark" else "#38BDF8"],
        [1.0, "#38BDF8" if theme == "Dark" else "#0284C7"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f"{h:02d}:00" for h in pivot_data.columns],
        y=pivot_data.index,
        colorscale=scale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Msgs", font=dict(color=t["subtext"], size=10)),
            tickfont=dict(color=t["subtext"], size=10),
            thickness=12,
            len=0.8
        )
    ))
    
    fig.update_layout(
        xaxis=dict(tickangle=-45, title=dict(text="Hour of Day", font=dict(color=t["subtext"], size=11))),
        yaxis=dict(title="")
    )
    return clean_chart_base(fig, theme=theme, height=360)

def weekday_bar_chart(weekday_df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    fig = go.Figure(data=[go.Bar(
        x=weekday_df["weekday"],
        y=weekday_df["count"],
        marker=dict(color=t["primary"], line=dict(width=0)),
        text=weekday_df["count"],
        textposition="outside",
        textfont=dict(color=t["subtext"], size=11)
    )])
    fig.update_layout(
        xaxis=dict(title=""),
        yaxis=dict(title=dict(text="Messages", font=dict(color=t["subtext"], size=11)))
    )
    return clean_chart_base(fig, theme=theme, height=320)

def top_ngrams_bar_chart(ngram_df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    if ngram_df.empty:
        return clean_chart_base(go.Figure(), theme=theme, height=340)

    sorted_df = ngram_df.sort_values("count", ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        y=sorted_df["phrase"],
        x=sorted_df["count"],
        orientation="h",
        marker=dict(color=t["accent"], line=dict(width=0)),
        text=sorted_df["count"],
        textposition="outside",
        textfont=dict(color=t["subtext"], size=11)
    )])
    fig.update_layout(
        xaxis=dict(title=dict(text="Occurrences", font=dict(color=t["subtext"], size=11))),
        yaxis=dict(title="")
    )
    return clean_chart_base(fig, theme=theme, height=340)

def sentiment_breakdown_chart(df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    user_df = df[df["role"] == "user"]
    if "sentiment_category" not in user_df.columns or user_df.empty:
        return clean_chart_base(go.Figure(), theme=theme, height=300)

    counts = user_df["sentiment_category"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).reset_index()
    counts.columns = ["Category", "Count"]
    
    colors = [t["secondary"], t["neutral"], t["danger"]]

    fig = go.Figure(data=[go.Bar(
        x=counts["Category"],
        y=counts["Count"],
        marker=dict(color=colors, line=dict(width=0)),
        text=counts["Count"],
        textposition="outside",
        textfont=dict(color=t["subtext"], size=11)
    )])
    fig.update_layout(
        xaxis=dict(title=""),
        yaxis=dict(title=dict(text="Messages", font=dict(color=t["subtext"], size=11)))
    )
    return clean_chart_base(fig, theme=theme, height=300)

def sentiment_trend_chart(df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    user_df = df[df["role"] == "user"].copy()
    if "sentiment_polarity" not in user_df.columns or user_df.empty:
        return None
    
    daily = user_df.set_index("message_time").groupby(pd.Grouper(freq="D"))["sentiment_polarity"].mean().reset_index().dropna()
    if daily.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["message_time"],
        y=daily["sentiment_polarity"],
        mode="lines+markers",
        line=dict(color=t["primary"], width=2),
        marker=dict(size=4, color=t["primary"]),
        name="Daily Polarity"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=t["subtext"], opacity=0.5)
    
    fig.update_layout(
        xaxis=dict(title=""),
        yaxis=dict(title=dict(text="Polarity Score", font=dict(color=t["subtext"], size=11)), range=[-1.05, 1.05])
    )
    return clean_chart_base(fig, theme=theme, height=300)

def message_length_distribution(df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    q95 = df["word_count"].quantile(0.95)
    filtered = df[df["word_count"] <= q95] if not pd.isna(q95) else df
    
    fig = go.Figure()
    u_words = filtered[filtered["role"] == "user"]["word_count"]
    a_words = filtered[filtered["role"] == "assistant"]["word_count"]

    if not u_words.empty:
        fig.add_trace(go.Histogram(
            x=u_words,
            name="Your Words",
            marker=dict(color=t["primary"]),
            opacity=0.8
        ))
    if not a_words.empty:
        fig.add_trace(go.Histogram(
            x=a_words,
            name="AI Words",
            marker=dict(color=t["secondary"]),
            opacity=0.8
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title=dict(text="Word Count per Message", font=dict(color=t["subtext"], size=11))),
        yaxis=dict(title=dict(text="Frequency", font=dict(color=t["subtext"], size=11)))
    )
    return clean_chart_base(fig, theme=theme, height=340)

def code_languages_chart(lang_df, theme="Dark"):
    t = THEME_CONFIG.get(theme, THEME_CONFIG["Dark"])
    if lang_df.empty:
        return clean_chart_base(go.Figure(), theme=theme, height=300)

    top_langs = lang_df.head(8)
    fig = go.Figure(data=[go.Bar(
        x=top_langs["language"],
        y=top_langs["count"],
        marker=dict(color=t["bar_sequence"][:len(top_langs)], line=dict(width=0)),
        text=top_langs["count"],
        textposition="outside",
        textfont=dict(color=t["subtext"], size=11)
    )])
    fig.update_layout(
        xaxis=dict(title=""),
        yaxis=dict(title=dict(text="Snippets", font=dict(color=t["subtext"], size=11)))
    )
    return clean_chart_base(fig, theme=theme, height=300)
