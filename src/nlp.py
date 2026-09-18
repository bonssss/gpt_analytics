import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

def analyze_sentiment(text):
    """
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0
    try:
        blob = TextBlob(text)
        return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)
    except Exception:
        return 0.0, 0.0

def categorize_sentiment(polarity):
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

def add_sentiment_to_df(df):
    """
    Computes sentiment metrics for all user messages.
    """
    if df.empty:
        return df

    polarities = []
    subjectivities = []
    categories = []

    for _, row in df.iterrows():
        if row["role"] == "user":
            pol, subj = analyze_sentiment(row["content"])
        else:
            pol, subj = 0.0, 0.0
            
        polarities.append(round(pol, 3))
        subjectivities.append(round(subj, 3))
        categories.append(categorize_sentiment(pol) if row["role"] == "user" else "N/A")

    df["sentiment_polarity"] = polarities
    df["sentiment_subjectivity"] = subjectivities
    df["sentiment_category"] = categories
    # Keep legacy column name for backward compatibility if needed
    df["sentiment"] = polarities
    return df

def extract_top_ngrams(df, ngram_range=(1, 1), top_n=15, role="user"):
    """
    Extracts top n-grams and their frequencies for clean, flat bar charts.
    """
    if df.empty:
        return pd.DataFrame(columns=["phrase", "count"])

    subset = df[df["role"] == role] if role else df
    corpus = subset["content"].dropna().astype(str).tolist()
    
    if not corpus or not any(corpus):
        return pd.DataFrame(columns=["phrase", "count"])

    try:
        vec = CountVectorizer(ngram_range=ngram_range, stop_words="english", max_features=top_n * 3)
        X = vec.fit_transform(corpus)
        counts = np.asarray(X.sum(axis=0)).ravel()
        words = vec.get_feature_names_out()
        
        ngram_df = pd.DataFrame({"phrase": words, "count": counts})
        # Filter out purely numeric or short garbage phrases
        ngram_df = ngram_df[ngram_df["phrase"].str.len() > 2]
        ngram_df = ngram_df.sort_values("count", ascending=False).head(top_n)
        return ngram_df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["phrase", "count"])

def cluster_topics_with_labels(df, n_clusters=4):
    """
    Clusters user conversations and auto-assigns top representative keywords per cluster.
    """
    if df.empty:
        return pd.DataFrame()

    user_df = df[df["role"] == "user"]
    if user_df.empty:
        return pd.DataFrame()

    conv_text = user_df.groupby(["conversation_id", "conversation_title"])["content"].apply(lambda x: " ".join(x)).reset_index()
    
    if len(conv_text) < n_clusters:
        n_clusters = max(1, len(conv_text))

    if n_clusters == 1:
        conv_text["cluster"] = 0
        conv_text["cluster_name"] = "General Conversations"
        return conv_text

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1500, min_df=1)
        X = vectorizer.fit_transform(conv_text["content"])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        conv_text["cluster"] = kmeans.fit_predict(X)
        
        # Generate cluster labels from top TF-IDF centroids
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        cluster_names = {}
        for i in range(n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :3]]
            cluster_names[i] = " • ".join(top_terms).title()
            
        conv_text["cluster_name"] = conv_text["cluster"].map(cluster_names)
        return conv_text
    except Exception:
        conv_text["cluster"] = 0
        conv_text["cluster_name"] = "General"
        return conv_text

# Legacy compatibility
def cluster_topics(df, n_clusters=5):
    res = cluster_topics_with_labels(df, n_clusters)
    if res.empty or "cluster" not in res.columns:
        return pd.Series(dtype=int)
    return res.set_index("conversation_id")["cluster"]

def detect_code_languages(df):
    """
    Scans content for markdown code blocks to determine language frequencies (Python, JS, SQL, etc.).
    """
    if df.empty:
        return pd.DataFrame(columns=["language", "count"])

    lang_pattern = re.compile(r"```([a-zA-Z0-9_+-]+)")
    languages = []
    
    for content in df["content"].dropna():
        matches = lang_pattern.findall(content)
        for m in matches:
            lang = m.lower().strip()
            # Normalize common aliases
            if lang in ["py", "python", "python3"]:
                lang = "Python"
            elif lang in ["js", "javascript", "jsx"]:
                lang = "JavaScript"
            elif lang in ["ts", "typescript", "tsx"]:
                lang = "TypeScript"
            elif lang in ["sh", "bash", "shell", "zsh"]:
                lang = "Bash / Shell"
            elif lang in ["sql", "pgsql", "mysql"]:
                lang = "SQL"
            elif lang in ["json", "yaml", "yml", "toml"]:
                lang = "Config / Data"
            elif lang in ["html", "css", "scss"]:
                lang = "HTML / CSS"
            elif lang in ["cpp", "c++", "c", "h"]:
                lang = "C / C++"
            elif lang in ["rust", "rs"]:
                lang = "Rust"
            elif lang in ["go", "golang"]:
                lang = "Go"
            else:
                lang = lang.capitalize()
            languages.append(lang)

    if not languages:
        return pd.DataFrame(columns=["language", "count"])

    lang_df = pd.Series(languages).value_counts().reset_index()
    lang_df.columns = ["language", "count"]
    return lang_df
