
import pandas as pd
import numpy as np

def compute_basic_stats(df):
    if df.empty:
        return {
            "total_conversations": 0,
            "total_messages": 0,
            "avg_conversation_length": 0,
            "user_assistant_ratio": 0
        }
    
    stats = {}
    stats["total_conversations"] = df["conversation_id"].nunique()
    stats["total_messages"] = len(df)
    
    # Avg messages per conversation
    conv_lengths = df.groupby("conversation_id").size()
    stats["avg_conversation_length"] = conv_lengths.mean()
    
    # Role distribution
    role_counts = df["role"].value_counts()
    user_msgs = role_counts.get("user", 0)
    assistant_msgs = role_counts.get("assistant", 0)
    
    stats["user_assistant_ratio"] = user_msgs / assistant_msgs if assistant_msgs > 0 else user_msgs
    stats["total_user_words"] = df[df["role"] == "user"]["word_count"].sum()
    stats["total_assistant_words"] = df[df["role"] == "assistant"]["word_count"].sum()
    
    return stats

def get_activity_over_time(df, freq="D"):
    """freq can be 'D' (day), 'W' (week), 'M' (month)"""
    df_temp = df.copy()
    df_temp.set_index("message_time", inplace=True)
    return df_temp.groupby(pd.Grouper(freq=freq)).size().reset_index(name="count")

def get_hourly_activity(df):
    return df["message_time"].dt.hour.value_counts().sort_index().reset_index(name="count")

def get_weekday_activity(df):
    return df["message_time"].dt.day_name().value_counts().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]).reset_index(name="count")

def activity_heatmap_data(df):
    df_temp = df.copy()
    df_temp["hour"] = df_temp["message_time"].dt.hour
    df_temp["weekday"] = df_temp["message_time"].dt.day_name()
    
    # Ensure all days and hours are present
    pivot = df_temp.pivot_table(index="weekday", columns="hour", values="content", aggfunc="count").fillna(0)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(days)
    return pivot
