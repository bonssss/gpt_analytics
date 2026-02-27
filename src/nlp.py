
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

def analyze_sentiment(text):
    if not isinstance(text, str):
        return 0
    return TextBlob(text).sentiment.polarity

def add_sentiment_to_df(df):
    # Analyze only user messages for sentiment
    df["sentiment"] = df["content"].apply(lambda x: analyze_sentiment(x))
    return df

def cluster_topics(df, n_clusters=5):
    """
    Groups conversations into clusters based on text content.
    """
    # Group by conversation to get full text
    conv_text = df[df["role"] == "user"].groupby("conversation_id")["content"].apply(lambda x: " ".join(x))
    
    if len(conv_text) < n_clusters:
        return pd.Series(index=conv_text.index, data=0)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(conv_text)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    return pd.Series(clusters, index=conv_text.index)

def extract_top_keywords(df, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(df[df["role"] == "user"]["content"])
    keywords = vectorizer.get_feature_names_out()
    return keywords
