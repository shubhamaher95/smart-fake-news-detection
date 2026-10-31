
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load evaluation results
df = pd.read_csv(
    "outputs/evaluation_results/model_comparison.csv"
)

# Display comparison
print("\nMODEL PERFORMANCE COMPARISON")
print(df.to_string(index=False))

# Plot performance metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

df_plot = df.set_index("Model")[metrics]

ax = df_plot.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1),
    width=0.75
)

plt.title("Logistic Regression vs Multinomial Naive Bayes")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics")
plt.tight_layout()

os.makedirs("outputs/visualizations", exist_ok=True)

plt.savefig(
    "outputs/visualizations/model_comparison.png"
)

plt.show()

print("\nComparison chart saved successfully!")
