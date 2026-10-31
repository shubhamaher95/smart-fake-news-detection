
import pandas as pd
import os

# Dataset path
file_path = "data/fake_news_dataset.csv"

# Check whether dataset exists
if not os.path.exists(file_path):
    print("ERROR: Dataset file not found!")
    print("Please place fake_news_dataset.csv inside the data folder.")
    exit()

# Load dataset
df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

# Display first 5 records
print("\n1. First five records:")
print(df.head())

# Dataset shape
print("\n2. Dataset shape:")
print(df.shape)

# Column names
print("\n3. Column names:")
print(df.columns.tolist())

# Data types
print("\n4. Data types:")
print(df.dtypes)

# Missing values
print("\n5. Missing values:")
print(df.isnull().sum())

# Duplicate records
print("\n6. Duplicate records:")
print(df.duplicated().sum())

# Label distribution
if "label" in df.columns:
    print("\n7. Label distribution:")
    print(df["label"].value_counts())

    print("\n8. Label percentages:")
    print(df["label"].value_counts(normalize=True) * 100)

print("\n========== INSPECTION COMPLETED ==========")
