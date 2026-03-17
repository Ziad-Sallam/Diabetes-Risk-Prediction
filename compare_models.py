import json
import os
import pandas as pd

imbalance_methods = ['oversampling', 'undersampling', 'SMOTE', None]
feature_selection_methods = [ 'automatic', None]
ks = [15, 18, 20, 21]

results = []

for imbalance_method in imbalance_methods:
    for feature_method in feature_selection_methods:
        for k in ks:

            folder = f"data/{imbalance_method}_{feature_method}_{k}"
            metrics_file = f"{folder}/diabetes_metrics_{imbalance_method}_{feature_method}_{k}.json"

            if not os.path.exists(metrics_file):
                continue

            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            results.append({
                "imbalance_method": imbalance_method,
                "feature_selection_method": feature_method,
                "k": k,
                "accuracy": metrics["accuracy"],
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"]
            })

df = pd.DataFrame(results)

if df.empty:
    print("No results found.")
    exit()

# Sort by Macro F1 (most important for imbalanced data)
df_sorted = df.sort_values(by="macro_f1", ascending=False)

print("\n===== MODEL COMPARISON =====\n")
print(df_sorted)

best_model = df_sorted.iloc[0]

print("\n===== BEST MODEL =====\n")
print(best_model)

# Save comparison table
df_sorted.to_csv("model_comparison_results.csv", index=False)

print("\nSaved comparison table to model_comparison_results.csv")