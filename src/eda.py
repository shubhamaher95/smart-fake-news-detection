
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import os

# Create output folder
os.makedirs("outputs/visualizations", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_news_dataset.csv")

print("========== EDA ANALYSIS ==========")

# Dataset information
print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

# Label distribution
print("\nNews Label Distribution:")
print(df["label"].value_counts())

# Calculate text length
df["text_length"] = df["cleaned_text"].astype(str).apply(len)

df["word_count"] = df["cleaned_text"].astype(str).apply(
    lambda x: len(x.split())
)

# 1. Real vs Fake news distribution
label_counts = df["label"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(
    ["Fake News (0)", "Real News (1)"],
    [label_counts.get(0, 0), label_counts.get(1, 0)]
)
plt.title("Real vs Fake News Distribution")
plt.xlabel("News Category")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("outputs/visualizations/label_distribution.png")
plt.show()

# 2. Text length distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["word_count"],
    bins=50
)

plt.title("News Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/visualizations/text_length_distribution.png")
plt.show()

# 3. Average word count by label
avg_words = df.groupby("label")["word_count"].mean()

print("\nAverage Word Count:")
print(avg_words)

plt.figure(figsize=(8, 5))
plt.bar(
    ["Fake News", "Real News"],
    [avg_words.get(0, 0), avg_words.get(1, 0)]
)
plt.title("Average Word Count by News Category")
plt.ylabel("Average Number of Words")
plt.tight_layout()
plt.savefig("outputs/visualizations/average_word_count.png")
plt.show()

# Function to find frequent words
def get_top_words(texts, n=15):

    words = []

    for text in texts:
        words.extend(re.findall(r'\b[a-z]{3,}\b', str(text)))

    # Basic common-word filtering
    stop_words = {
        "the", "and", "for", "that", "this",
        "with", "from", "have", "are", "was",
        "were", "will", "would", "there", "their",
        "about", "they", "you", "your", "has",
        "had", "not", "but", "all", "his", "her"
    }

    words = [word for word in words if word not in stop_words]

    return Counter(words).most_common(n)


# 4. Top words in fake news
fake_texts = df[df["label"] == 0]["cleaned_text"]

fake_words = get_top_words(fake_texts)

print("\nMost Frequent Words in Fake News:")
print(fake_words)

plt.figure(figsize=(10, 5))
plt.bar(
    [word for word, count in fake_words],
    [count for word, count in fake_words]
)
plt.title("Most Frequent Words in Fake News")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/visualizations/fake_news_top_words.png")
plt.show()

# 5. Top words in real news
real_texts = df[df["label"] == 1]["cleaned_text"]

real_words = get_top_words(real_texts)

print("\nMost Frequent Words in Real News:")
print(real_words)

plt.figure(figsize=(10, 5))
plt.bar(
    [word for word, count in real_words],
    [count for word, count in real_words]
)
plt.title("Most Frequent Words in Real News")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/visualizations/real_news_top_words.png")
plt.show()

print("\nEDA completed successfully!")

print("\nCharts saved in outputs/visualizations/")
