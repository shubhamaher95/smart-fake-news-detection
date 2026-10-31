
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/evaluation_results", exist_ok=True)
os.makedirs("outputs/visualizations", exist_ok=True)

print("========== MACHINE LEARNING TRAINING ==========")

# Load TF-IDF feature data
print("\nLoading feature data...")

data = joblib.load("outputs/feature_data.joblib")

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB()
}

results = []

trained_models = {}

# Train and evaluate both models
for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

    print(f"\n{name} Results:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake News", "Real News"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        f"outputs/visualizations/{filename}_confusion_matrix.png"
    )

    plt.close()

    # Save trained model
    joblib.dump(
        model,
        f"models/{filename}.joblib"
    )

    print(f"\n{name} model saved successfully!")

# Compare models
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save evaluation results
results_df.to_csv(
    "outputs/evaluation_results/model_comparison.csv",
    index=False
)

# Select model based on F1 score
best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

# Save selected model
joblib.dump(
    best_model,
    "models/model.joblib"
)

print("\nBest Model:", best_model_name)
print("Selected model saved as models/model.joblib")

print("\n========== TRAINING COMPLETED ==========")
