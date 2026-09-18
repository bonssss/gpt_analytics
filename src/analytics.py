import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_streaks(dates):
    """
    Computes current and maximum consecutive active day streaks.
    """
    if dates.empty:
        return {"current_streak": 0, "max_streak": 0, "active_days": 0}

    unique_dates = pd.Series(dates.dt.date.unique()).sort_values().reset_index(drop=True)
    if unique_dates.empty:
        return {"current_streak": 0, "max_streak": 0, "active_days": 0}

    active_days = len(unique_dates)
    
    # Calculate consecutive day diffs
    diffs = unique_dates.diff()
    
    max_streak = 1
    current_streak = 1
    streak_count = 1

    for diff in diffs.iloc[1:]:
        if diff == timedelta(days=1):
            streak_count += 1
            max_streak = max(max_streak, streak_count)
        else:
            streak_count = 1

    # Check if latest date is today or yesterday for current streak
    today = datetime.now().date()
    latest_date = unique_dates.iloc[-1]
    if (today - latest_date).days <= 1:
        # Determine actual current active streak back from latest_date
        curr = 1
        for i in range(len(unique_dates) - 1, 0, -1):
            if unique_dates.iloc[i] - unique_dates.iloc[i-1] == timedelta(days=1):
                curr += 1
            else:
                break
        current_streak = curr
    else:
        current_streak = 0

    return {
        "current_streak": current_streak,
        "max_streak": max(max_streak, current_streak),
        "active_days": active_days
    }

def compute_basic_stats(df):
    """
    Computes high-level KPIs and summary statistics.
    """
    if df.empty:
        return {
            "total_conversations": 0,
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "avg_conversation_length": 0,
            "user_assistant_ratio": 0.0,
            "total_user_words": 0,
            "total_assistant_words": 0,
            "total_words": 0,
            "total_tokens": 0,
            "avg_user_word_count": 0,
            "avg_assistant_word_count": 0,
            "code_message_count": 0,
            "code_message_pct": 0.0,
            "total_questions": 0,
            "current_streak": 0,
            "max_streak": 0,
            "active_days": 0
        }

    user_mask = df["role"] == "user"
    assistant_mask = df["role"] == "assistant"
    
    user_msgs = int(user_mask.sum())
    assistant_msgs = int(assistant_mask.sum())
    total_messages = len(df)
    total_conversations = int(df["conversation_id"].nunique())
    
    # Message lengths
    conv_lengths = df.groupby("conversation_id").size()
    avg_conv_length = float(conv_lengths.mean()) if not conv_lengths.empty else 0.0
    
    total_user_words = int(df[user_mask]["word_count"].sum())
    total_assistant_words = int(df[assistant_mask]["word_count"].sum())
    total_words = total_user_words + total_assistant_words
    total_tokens = int(df["estimated_tokens"].sum()) if "estimated_tokens" in df.columns else int(total_words / 0.75)
    
    avg_user_words = float(df[user_mask]["word_count"].mean()) if user_msgs > 0 else 0.0
    avg_assistant_words = float(df[assistant_mask]["word_count"].mean()) if assistant_msgs > 0 else 0.0
    
    ratio = (user_msgs / assistant_msgs) if assistant_msgs > 0 else float(user_msgs)
    
    code_msgs = int(df["has_code"].sum()) if "has_code" in df.columns else 0
    code_pct = (code_msgs / total_messages * 100) if total_messages > 0 else 0.0
    
    total_questions = int(df[user_mask]["question_count"].sum()) if "question_count" in df.columns else 0
    
    streak_info = calculate_streaks(df["message_time"])
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "user_messages": user_msgs,
        "assistant_messages": assistant_msgs,
        "avg_conversation_length": round(avg_conv_length, 1),
        "user_assistant_ratio": round(ratio, 2),
        "total_user_words": total_user_words,
        "total_assistant_words": total_assistant_words,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "avg_user_word_count": round(avg_user_words, 1),
        "avg_assistant_word_count": round(avg_assistant_words, 1),
        "code_message_count": code_msgs,
        "code_message_pct": round(code_pct, 1),
        "total_questions": total_questions,
        "current_streak": streak_info["current_streak"],
        "max_streak": streak_info["max_streak"],
        "active_days": streak_info["active_days"]
    }

def get_activity_over_time(df, freq="D"):
    """
    Computes message count aggregated by specified frequency ('D', 'W', 'M').
    Includes user, assistant, and cumulative volume.
    """
    if df.empty:
        return pd.DataFrame()

    df_temp = df.copy().set_index("message_time")
    
    total_series = df_temp.groupby(pd.Grouper(freq=freq)).size().rename("total_messages")
    user_series = df_temp[df_temp["role"] == "user"].groupby(pd.Grouper(freq=freq)).size().rename("user_messages")
    assistant_series = df_temp[df_temp["role"] == "assistant"].groupby(pd.Grouper(freq=freq)).size().rename("assistant_messages")
    
    result = pd.concat([total_series, user_series, assistant_series], axis=1).fillna(0).reset_index()
    result["cumulative_messages"] = result["total_messages"].cumsum()
    return result

def activity_heatmap_data(df):
    """
    Prepares a 7x24 matrix for Weekday vs Hour-of-Day heatmap.
    """
    if df.empty:
        return pd.DataFrame()

    df_temp = df.copy()
    df_temp["hour"] = df_temp["message_time"].dt.hour
    df_temp["weekday"] = df_temp["message_time"].dt.day_name()
    
    pivot = df_temp.pivot_table(index="weekday", columns="hour", values="content", aggfunc="count").fillna(0)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Reindex days
    pivot = pivot.reindex(days).fillna(0)
    # Ensure all 24 hours exist
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot.sort_index(axis=1)
    return pivot

def get_hourly_activity(df):
    """
    Returns message volume aggregated by hour of day (0-23).
    """
    if df.empty:
        return pd.DataFrame()
    hourly = df["message_time"].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index().reset_index()
    hourly.columns = ["hour", "count"]
    return hourly

def get_weekday_activity(df):
    """
    Returns message volume aggregated by weekday (Mon-Sun).
    """
    if df.empty:
        return pd.DataFrame()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = df["message_time"].dt.day_name().value_counts().reindex(days, fill_value=0).reset_index()
    weekday.columns = ["weekday", "count"]
    return weekday

def get_conversation_leaderboard(df, top_n=10, sort_by="messages"):
    """
    Generates a summarized table of top conversations.
    """
    if df.empty:
        return pd.DataFrame()

    def agg_conv(group):
        first_time = group["message_time"].min()
        last_time = group["message_time"].max()
        duration_mins = round((last_time - first_time).total_seconds() / 60, 1) if pd.notna(first_time) and pd.notna(last_time) else 0
        return pd.Series({
            "title": group["conversation_title"].iloc[0],
            "messages": len(group),
            "user_messages": (group["role"] == "user").sum(),
            "assistant_messages": (group["role"] == "assistant").sum(),
            "total_words": group["word_count"].sum(),
            "total_tokens": group["estimated_tokens"].sum() if "estimated_tokens" in group.columns else 0,
            "has_code": group["has_code"].any() if "has_code" in group.columns else False,
            "start_time": first_time.strftime("%Y-%m-%d %H:%M") if pd.notna(first_time) else "N/A",
            "duration_mins": duration_mins
        })

    conv_df = df.groupby("conversation_id").apply(agg_conv, include_groups=False).reset_index()
    
    if sort_by in conv_df.columns:
        conv_df = conv_df.sort_values(sort_by, ascending=False)
    else:
        conv_df = conv_df.sort_values("messages", ascending=False)
        
    return conv_df.head(top_n)

def get_token_cost_estimation(df):
    """
    Estimates token metrics, API cost equivalence, and reading/writing time saved.
    """
    if df.empty:
        return {}

    user_mask = df["role"] == "user"
    assistant_mask = df["role"] == "assistant"

    input_tokens = int(df[user_mask]["estimated_tokens"].sum()) if "estimated_tokens" in df.columns else 0
    output_tokens = int(df[assistant_mask]["estimated_tokens"].sum()) if "estimated_tokens" in df.columns else 0
    total_tokens = input_tokens + output_tokens

    # Approx benchmark pricing (e.g. GPT-4o pricing: $2.50/M input, $10.00/M output)
    gpt4o_cost = (input_tokens / 1_000_000 * 2.50) + (output_tokens / 1_000_000 * 10.00)
    # Approx benchmark GPT-3.5 / mini pricing: $0.15/M input, $0.60/M output
    mini_cost = (input_tokens / 1_000_000 * 0.15) + (output_tokens / 1_000_000 * 0.60)

    # Reading & typing time estimate
    # Avg typing speed: 40 WPM (user). Avg reading speed: 250 WPM (assistant).
    user_words = df[user_mask]["word_count"].sum()
    assistant_words = df[assistant_mask]["word_count"].sum()
    
    typing_hours = round(user_words / (40 * 60), 1)
    reading_hours = round(assistant_words / (250 * 60), 1)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "gpt4o_cost": round(gpt4o_cost, 2),
        "mini_cost": round(mini_cost, 2),
        "typing_hours": typing_hours,
        "reading_hours": reading_hours
    }
