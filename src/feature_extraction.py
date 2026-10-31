
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Loading cleaned dataset...")

# Load dataset
df = pd.read_csv("data/cleaned_news_dataset.csv")

# Input text and target labels
X = df["cleaned_text"].fillna("")
y = df["label"]

print("Total records:", len(df))

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

print("\nFitting TF-IDF on training data...")

# Fit ONLY on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data using the same fitted vectorizer
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF extraction completed!")

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)
print("Vocabulary size:", len(tfidf.vocabulary_))

# Save fitted vectorizer
joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.joblib"
)

# Save feature matrices and labels
joblib.dump(
    {
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train,
        "y_test": y_test
    },
    "outputs/feature_data.joblib"
)

print("\nFiles saved successfully!")
print("1. models/tfidf_vectorizer.joblib")
print("2. outputs/feature_data.joblib")

print("\n========== STEP 5 COMPLETED ==========")
