# 🧠 ChatGPT Intelligence Dashboard

A premium analytics platform to visualize and explore your ChatGPT interaction history.

## ✨ Features
- **📊 Engagement Metrics**: Track conversation volume, message counts, and word usage.
- **📈 Trend Analysis**: Interactive Plotly charts for daily/weekly/monthly activity.
- **🔥 Usage Heatmaps**: Identify your most productive hours and days.
- **☁ Content Cloud**: Visualize the most frequent topics and keywords using advanced word clouds.
- **🧠 Advanced NLP**:
    - **Sentiment Analysis**: Track your mood trends across interactions.
    - **Topic Clustering**: Automatically group conversations into themes using Machine Learning.
- **🔍 Deep Search**: Instant full-text search across your entire history.
- **📥 Data Export**: Export processed insights to CSV for further analysis.

## 🚀 Getting Started

### 1. Export your Data
- Go to [ChatGPT](https://chat.openai.com) -> Settings -> Data Controls -> Export Data.
- You will receive an email with a ZIP file.
- Extract the ZIP and find `conversations.json`.

### 2. Installation
Ensure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app/dashboard.py
```

## 📂 Project Structure
- `app/dashboard.py`: Main Streamlit application.
- `src/parser.py`: Robust JSON parsing and data cleaning.
- `src/analytics.py`: Logic for statistical calculations.
- `src/nlp.py`: Sentiment analysis and topic clustering (K-Means).
- `src/visualization.py`: High-performance Plotly and Matplotlib generators.

## 🎨 Design Philosophy
This dashboard uses a **Premium Dark Theme** with interactive components to provide a "Data Hub" feel, ensuring that insights are not only useful but visually stunning.
