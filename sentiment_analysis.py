# sentiment_analysis.py
# -----------------------------------
# Chatbot Feedback Sentiment Analysis using VADER
# -----------------------------------

import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# ------------------------------------------------
# 1️⃣ STEP ONE: Load chatbot feedback
# ------------------------------------------------
data_path = "data/strategy.csv"

if os.path.exists(data_path):
    # Load feedback from CSV file
    df = pd.read_csv(data_path)
    print(f"\n✅ Loaded {len(df)} feedback entries from {data_path}")
else:
    # Fallback: manually entered sample feedback
    print("\n⚠️ CSV file not found! Using sample feedback.")
    feedback_data = [
        "The chatbot was very informative and polite.",
        "It didn’t really answer my question properly.",
        "Great experience! The explanations were clear.",
        "The response was too general and not helpful.",
        "I liked the speed of the reply."
    ]
    df = pd.DataFrame(feedback_data, columns=["Feedback"])


# ------------------------------------------------
# 2️⃣ STEP TWO: Analyze sentiment for each feedback
# ------------------------------------------------
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series([score["pos"], score["neu"], score["neg"], compound, sentiment])

df[["Positive", "Neutral", "Negative", "Compound", "Sentiment"]] = df["Feedback"].apply(analyze_sentiment)

# ------------------------------------------------
# 3️⃣ STEP THREE: Display results
# ------------------------------------------------
print("\nChatbot Feedback Sentiment Analysis Results:")
print(df[["Feedback", "Sentiment", "Compound"]])

# ------------------------------------------------
# 4️⃣ STEP FOUR (Optional): Save to CSV
# ------------------------------------------------
df.to_csv("data/sentiment_results.csv", index=False)
print("\nResults saved to: data/sentiment_results.csv")


# ------------------------------------------------
# 5️⃣ STEP FIVE (Updated): Visualize results with a PIE chart
# ------------------------------------------------
sentiment_counts = df["Sentiment"].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    shadow=True,
    explode=[0.05]*len(sentiment_counts),  # Slightly separate each slice
)
plt.title("Strategy Feedback Sentiment Distribution", fontsize=14)
plt.tight_layout()

# ✅ Save the pie chart as an image (PNG format)
output_path = "data/strategy_sentiment_pie_chart.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n📊 Pie chart saved as: {output_path}")

# Display the chart
plt.show()

