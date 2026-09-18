import json
import re
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def extract_message_text(content_data):
    """
    Extracts raw text string from content parts dict/list.
    """
    if not isinstance(content_data, dict):
        return str(content_data) if content_data else ""
        
    content_parts = content_data.get("parts", [])
    text_content = []
    for part in content_parts:
        if isinstance(part, str):
            text_content.append(part)
        elif isinstance(part, dict):
            if "text" in part:
                text_content.append(part["text"])
            elif "content" in part:
                text_content.append(str(part["content"]))
    return "\n".join(text_content).strip()

def load_chatgpt_export(file_input):
    """
    Parses ChatGPT export JSON (conversations.json) and returns a structured DataFrame.
    Accepts: bytes, file-like object, JSON string, list/dict of parsed JSON, or file path.
    """
    data = None
    try:
        if isinstance(file_input, (list, dict)):
            data = file_input
        elif isinstance(file_input, bytes):
            text = file_input.decode("utf-8", errors="ignore")
            data = json.loads(text)
        elif hasattr(file_input, "read"):
            content = file_input.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            data = json.loads(content)
        elif isinstance(file_input, str):
            stripped = file_input.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                data = json.loads(stripped)
            elif os.path.exists(file_input):
                with open(file_input, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
            else:
                data = json.loads(file_input)
    except Exception as e:
        logging.error(f"Failed to load JSON data: {e}")
        return pd.DataFrame()

    if data is None:
        return pd.DataFrame()

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return pd.DataFrame()

    records = []

    for conv in data:
        if not isinstance(conv, dict):
            continue

        conv_id = conv.get("id") or conv.get("conversation_id") or ""
        title = conv.get("title") or "Untitled Conversation"
        conv_create_time = conv.get("create_time")
        if conv_create_time:
            try:
                conv_create_time = datetime.fromtimestamp(float(conv_create_time))
            except Exception:
                conv_create_time = None
            
        mapping = conv.get("mapping", {})
        if not isinstance(mapping, dict):
            continue
        
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue

            message = node.get("message")
            if not message or not isinstance(message, dict):
                continue

            author = message.get("author", {})
            role = author.get("role", "unknown") if isinstance(author, dict) else "unknown"
            # Only keep standard user, assistant, system or tool roles
            if role not in ["user", "assistant", "system", "tool"]:
                continue

            content_data = message.get("content", {})
            text_content = extract_message_text(content_data)
            
            if not text_content:
                continue

            msg_time = message.get("create_time")
            if msg_time:
                try:
                    msg_time = datetime.fromtimestamp(float(msg_time))
                except Exception:
                    msg_time = conv_create_time or datetime.now()
            elif conv_create_time:
                msg_time = conv_create_time
            else:
                msg_time = datetime.now()

            # Metadata extraction
            code_blocks = len(re.findall(r"```[a-zA-Z0-9_-]*\n[\s\S]*?```", text_content))
            question_count = text_content.count("?")
            words = text_content.split()
            word_count = len(words)
            char_count = len(text_content)
            # Standard heuristic: ~1 token per 4 chars or 0.75 words
            estimated_tokens = int(np.ceil(char_count / 4)) if char_count > 0 else 0
            
            # Model name if available
            metadata = message.get("metadata", {})
            model_slug = "Default"
            if isinstance(metadata, dict):
                model_slug = metadata.get("model_slug") or metadata.get("default_model_slug") or "Default"

            records.append({
                "conversation_id": conv_id,
                "conversation_title": title,
                "node_id": node_id,
                "message_time": msg_time,
                "role": role,
                "model": model_slug,
                "content": text_content,
                "char_count": char_count,
                "word_count": word_count,
                "estimated_tokens": estimated_tokens,
                "code_block_count": code_blocks,
                "has_code": code_blocks > 0,
                "question_count": question_count
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["message_time"] = pd.to_datetime(df["message_time"])
    df = df.sort_values("message_time").reset_index(drop=True)
    return df

def generate_sample_chat_data():
    """
    Generates a realistic sample DataFrame of ChatGPT conversations for immediate demo/testing.
    """
    sample_conversations = [
        {
            "title": "FastAPI Microservices Architecture",
            "messages": [
                ("user", "How should I structure a multi-service FastAPI project with Docker and Redis caching?"),
                ("assistant", "Here is a clean layout for FastAPI microservices:\n```python\n# app/main.py\nfrom fastapi import FastAPI\napp = FastAPI(title='Core Service')\n\n@app.get('/health')\ndef health():\n    return {'status': 'healthy'}\n```\n1. Use Docker Compose for service orchestration.\n2. Add Redis for caching expensive database queries."),
                ("user", "Can you show how to configure Redis connection pooling with async support?"),
                ("assistant", "Sure! You can use `redis.asyncio`:\n```python\nimport redis.asyncio as aioredis\n\nasync def get_redis_pool():\n    return aioredis.from_url('redis://localhost:6379', encoding='utf-8', decode_responses=True)\n```")
            ]
        },
        {
            "title": "Debugging React useEffect Memory Leak",
            "messages": [
                ("user", "Why am I getting 'Can't perform a React state update on an unmounted component'?"),
                ("assistant", "This happens when an asynchronous call (like fetch) resolves after the component unmounts. You can fix it with an AbortController:\n```javascript\nuseEffect(() => {\n  const controller = new AbortController();\n  fetchData({ signal: controller.signal });\n  return () => controller.abort();\n}, []);\n```"),
                ("user", "Thanks! That resolved the issue completely.")
            ]
        },
        {
            "title": "Machine Learning Model Evaluation",
            "messages": [
                ("user", "What is the difference between ROC-AUC and Precision-Recall AUC for imbalanced datasets?"),
                ("assistant", "When classes are highly imbalanced (e.g. fraud detection with 99% negative class), PR-AUC is far more informative than ROC-AUC because ROC-AUC can remain deceptively high due to a large number of True Negatives."),
                ("user", "Which Python library should I use to calculate PR-AUC?"),
                ("assistant", "You can use `sklearn.metrics.precision_recall_curve` and `sklearn.metrics.auc`:\n```python\nfrom sklearn.metrics import precision_recall_curve, auc\nprecision, recall, _ = precision_recall_curve(y_true, y_scores)\npr_auc = auc(recall, precision)\n```")
            ]
        },
        {
            "title": "Trip Itinerary: 5 Days in Tokyo",
            "messages": [
                ("user", "Can you make a 5-day itinerary for Tokyo focusing on food, tech, and cultural temples?"),
                ("assistant", "Day 1: Shibuya & Shinjuku - Crossings, ramen, and skyline views.\nDay 2: Akihabara & Ueno - Tech districts, electronic shops, and historical park museums.\nDay 3: Asakusa & Senso-ji - Traditional street snacks and Tokyo Skytree.\nDay 4: Ginza & Tsukiji Outer Market - Fresh sushi and department stores.\nDay 5: Harajuku & Meiji Shrine - Tranquil forest shrine and street fashion."),
                ("user", "What is the best way to get around? Is the Suica card still available?"),
                ("assistant", "Yes! You can add a digital Suica or Pasmo directly to Apple Wallet / Google Wallet for seamless subway transit.")
            ]
        },
        {
            "title": "SQL Query Optimization & Indexes",
            "messages": [
                ("user", "My PostgreSQL query on 10 million rows is taking 12 seconds. How do I optimize a composite WHERE condition?"),
                ("assistant", "First run `EXPLAIN ANALYZE` to inspect whether sequential scans are happening. Then create a composite B-Tree index:\n```sql\nCREATE INDEX idx_orders_user_created ON orders (user_id, created_at DESC);\n```\nEnsure columns with highest selectivity come first."),
                ("user", "That reduced the query time to 42 milliseconds! Amazing.")
            ]
        },
        {
            "title": "Creative Writing: Sci-Fi Short Story",
            "messages": [
                ("user", "Write a 3-paragraph sci-fi opening about an automated relay beacon orbiting a dead star."),
                ("assistant", "Beacon-7 had spent six centuries counting neutrinos in absolute silence. Suspended above the accretion disk of Kepler-442, its copper-alloy plating bore the micro-scars of countless stellar storms.\n\nAt precisely 03:14 galactic standard time, the deep-frequency antenna shuddered. A transmission beam—dense, analog, and impossibly ancient—pierced the void from an uncharted quadrant.\n\nWithin its core memory banks, dormant subroutines began to unfreeze. For the first time in known history, the beacon was commanded to answer."),
                ("user", "I love the tone! Let's continue with the technician who intercepts the echo.")
            ]
        },
        {
            "title": "TypeScript Generics Best Practices",
            "messages": [
                ("user", "How do I constrain generic types in TypeScript to ensure a key exists in an object?"),
                ("assistant", "Use `keyof` and the `extends` keyword:\n```typescript\nfunction getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {\n  return obj[key];\n}\n```"),
                ("user", "Can I make the return type handle nested keys like 'user.profile.name' with template literals?"),
                ("assistant", "Yes! You can build recursive string path types using conditional types and template literals.")
            ]
        },
        {
            "title": "Healthy Weekly Meal Prep Plan",
            "messages": [
                ("user", "Give me a high-protein vegetarian meal prep list for lunches Monday through Friday."),
                ("assistant", "Here is a balanced 5-day plan:\n1. Roasted Chickpea & Quinoa Buddha Bowls with tahini dressing.\n2. Lentil & Black Bean Chili with avocado slices.\n3. Paneer / Tofu Stir-Fry with broccoli and brown rice.\n4. Mediterranean Halloumi & Farro Salad.\n5. Edamame & Peanut Soba Noodle Bowls."),
                ("user", "Can you generate the exact grocery shopping list organized by supermarket aisle?"),
                ("assistant", "Produce: Broccoli, Avocado, Kale, Carrots, Limes.\nGrains & Legumes: Quinoa, Black beans, Brown rice, Soba noodles, Chickpeas.\nRefrigerated/Protein: Extra-firm Tofu, Paneer, Edamame.\nPantry: Tahini, Peanut butter, Soy sauce, Olive oil.")
            ]
        }
    ]

    base_time = datetime.now() - timedelta(days=45)
    records = []

    for i, conv_data in enumerate(sample_conversations):
        conv_id = f"demo-conv-{i+1:03d}"
        conv_time = base_time + timedelta(days=i * 5, hours=(i * 7) % 24, minutes=(i * 23) % 60)
        
        for j, (role, text) in enumerate(conv_data["messages"]):
            msg_time = conv_time + timedelta(minutes=j * 3 + (j * 45) % 15)
            code_blocks = len(re.findall(r"```[a-zA-Z0-9_-]*\n[\s\S]*?```", text))
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            estimated_tokens = int(np.ceil(char_count / 4))
            
            records.append({
                "conversation_id": conv_id,
                "conversation_title": conv_data["title"],
                "node_id": f"{conv_id}-msg-{j+1}",
                "message_time": msg_time,
                "role": role,
                "model": "gpt-4o",
                "content": text,
                "char_count": char_count,
                "word_count": word_count,
                "estimated_tokens": estimated_tokens,
                "code_block_count": code_blocks,
                "has_code": code_blocks > 0,
                "question_count": text.count("?")
            })

    df = pd.DataFrame(records)
    df["message_time"] = pd.to_datetime(df["message_time"])
    df = df.sort_values("message_time").reset_index(drop=True)
    return df
