
import pandas as pd
import re
import os

def clean_text(text):
    """
    Clean news text for machine learning.
    """

    if pd.isna(text):
        return ""

    # Convert text to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove unnecessary special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataset(input_path, output_path):

    # Load dataset
    df = pd.read_csv(input_path)

    print("Original dataset shape:", df.shape)

    # Remove unnecessary index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Remove records with missing labels
    df = df.dropna(subset=["label"])

    # Fill missing titles and article text
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title and article text
    df["combined_text"] = (
        df["title"].astype(str) + " " +
        df["text"].astype(str)
    )

    # Clean combined text
    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    # Remove records where combined text is empty
    df = df[df["cleaned_text"].str.strip() != ""]

    # Keep only necessary columns
    df = df[["cleaned_text", "label"]]

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("\nPreprocessing completed!")
    print("Final dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    return df


if __name__ == "__main__":

    input_file = "data/fake_news_dataset.csv"
    output_file = "data/cleaned_news_dataset.csv"

    os.makedirs("data", exist_ok=True)

    df = preprocess_dataset(input_file, output_file)

    print("\nSample cleaned news:")
    print(df["cleaned_text"].iloc[0][:500])
