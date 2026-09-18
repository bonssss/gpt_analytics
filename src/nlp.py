import re
import math
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from textblob import TextBlob

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "im", "dont", "cant", "like", "just", "also", "want", "help", "need", "make", "using", "use", "please",
    "give", "show", "code", "write", "know", "good", "well", "think", "get", "way", "see", "look"
}

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

    if HAS_SKLEARN:
        try:
            vec = CountVectorizer(ngram_range=ngram_range, stop_words="english", max_features=top_n * 3)
            X = vec.fit_transform(corpus)
            counts = np.asarray(X.sum(axis=0)).ravel()
            words = vec.get_feature_names_out()
            
            ngram_df = pd.DataFrame({"phrase": words, "count": counts})
            ngram_df = ngram_df[ngram_df["phrase"].str.len() > 2]
            ngram_df = ngram_df.sort_values("count", ascending=False).head(top_n)
            return ngram_df.reset_index(drop=True)
        except Exception:
            pass

    # Pure Python Fallback
    n = ngram_range[0]
    token_pattern = re.compile(r"(?u)\b[a-zA-Z]{3,}\b")
    counts = Counter()
    for text in corpus:
        tokens = [w.lower() for w in token_pattern.findall(text) if w.lower() not in ENGLISH_STOP_WORDS]
        if n == 1:
            counts.update(tokens)
        else:
            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            counts.update(ngrams)

    most_common = counts.most_common(top_n)
    if not most_common:
        return pd.DataFrame(columns=["phrase", "count"])
    return pd.DataFrame(most_common, columns=["phrase", "count"])

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

    if HAS_SKLEARN:
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
            pass

    # Pure Python Lightweight Keyword Grouper Fallback
    try:
        token_pattern = re.compile(r"(?u)\b[a-zA-Z]{3,}\b")
        conv_keywords = []
        doc_count = len(conv_text)
        doc_freqs = Counter()

        for content in conv_text["content"]:
            tokens = set(w.lower() for w in token_pattern.findall(content) if w.lower() not in ENGLISH_STOP_WORDS)
            doc_freqs.update(tokens)

        for content in conv_text["content"]:
            tokens = [w.lower() for w in token_pattern.findall(content) if w.lower() not in ENGLISH_STOP_WORDS]
            tf = Counter(tokens)
            tfidf = {word: count * math.log((doc_count + 1) / (doc_freqs[word] + 1)) for word, count in tf.items()}
            sorted_words = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
            top_words = [w[0].title() for w in sorted_words[:3]]
            conv_keywords.append(" • ".join(top_words) if top_words else "General")

        conv_text["cluster"] = [i % n_clusters for i in range(len(conv_text))]
        conv_text["cluster_name"] = conv_keywords
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
