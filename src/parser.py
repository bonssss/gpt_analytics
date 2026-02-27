
import json
import pandas as pd
from datetime import datetime
import logging

def load_chatgpt_export(file):
    """
    Parses ChatGPT export JSON and returns a structured DataFrame.
    """
    try:
        content = file.read()
        data = json.loads(content)
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        return pd.DataFrame()

    records = []

    for conv in data:
        conv_id = conv.get("id")
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time")
        
        mapping = conv.get("mapping", {})
        
        msg_count = 0
        for node in mapping.values():
            message = node.get("message")
            if not message:
                continue

            role = message.get("author", {}).get("role")
            content_data = message.get("content", {})
            
            # Extract content text
            content_parts = content_data.get("parts", [])
            text_content = ""
            for part in content_parts:
                if isinstance(part, str):
                    text_content += part
                elif isinstance(part, dict) and "text" in part:
                    text_content += part["text"]
            
            if not text_content:
                continue

            msg_time = message.get("create_time")
            if msg_time:
                msg_time = datetime.fromtimestamp(msg_time)

            records.append({
                "conversation_id": conv_id,
                "conversation_title": title,
                "message_time": msg_time,
                "role": role,
                "content": text_content,
                "char_count": len(text_content),
                "word_count": len(text_content.split())
            })
            msg_count += 1

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Convert message_time to datetime objects if not already
    df["message_time"] = pd.to_datetime(df["message_time"])
    
    # Sort by time
    df = df.sort_values("message_time")
    
    return df
