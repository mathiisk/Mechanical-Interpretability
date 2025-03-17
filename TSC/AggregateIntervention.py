import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aeon.datasets import load_classification
from aeon.transformations.collection.feature_based import Catch22

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# -------------------------------
# 1. Load and Transform Dataset
# -------------------------------
# Use a dataset from Aeon (e.g., "AsphaltRegularity")
X, y = load_classification("AsphaltRegularity", split="train")

# Apply Catch22 transformation
catch22 = Catch22()
X_transformed = catch22.fit_transform(X)

# Create a DataFrame with feature names and add class labels
feature_names = [f'Feature_{i}' for i in range(X_transformed.shape[1])]
df_features = pd.DataFrame(X_transformed, columns=feature_names)
df_features['class'] = np.array(y)

print("Transformed Features with Class Labels:")
print(df_features.head())

# -------------------------------
# 2. Compute Class Average Feature Profiles
# -------------------------------
class_means = df_features.groupby('class').mean()
print("\nClass Mean Features:")
print(class_means)

# -------------------------------
# 3. Prepare Data for Classifier Training
# -------------------------------
X_features = df_features[feature_names].copy()
y_labels = df_features['class']

# Impute missing values (if any) using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_features_imputed = imputer.fit_transform(X_features)

# Split the data: 80% training, 20% testing
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_features_imputed, y_labels, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train a Simple Classifier on Catch22 Features
# -------------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_lr, y_train_lr)


# -------------------------------
# 5. Define Intervention Function for a Single Instance
# -------------------------------
def intervention_for_instance(instance, clf, class_means, feature_names):
    """
    For a given test instance, compute the baseline prediction and then
    intervene on each feature (patch with the average from the "other" class)
    to record the absolute change in predicted probability for the originally predicted class.
    Returns a dictionary mapping each feature to the absolute change.
    """
    baseline_pred = clf.predict_proba([instance])[0]
    predicted_class = clf.predict([instance])[0]
    class_index = list(clf.classes_).index(predicted_class)

    # For binary classification, determine the "other" class
    classes = class_means.index.tolist()  # assumes two classes
    other_class = classes[0] if predicted_class == classes[1] else classes[1]

    differences = {}
    for i, feature in enumerate(feature_names):
        patched_instance = instance.copy()
        # Patch current feature with the other class's mean value
        patched_instance[i] = class_means.loc[other_class, feature]
        patched_pred = clf.predict_proba([patched_instance])[0]
        diff = abs(baseline_pred[class_index] - patched_pred[class_index])
        differences[feature] = diff
    return differences


# -------------------------------
# 6. Aggregate Intervention Analysis (Across All Test Instances)
# -------------------------------
all_differences = []  # List of dicts for each instance

for instance in X_test_lr:
    diffs = intervention_for_instance(instance, clf, class_means, feature_names)
    all_differences.append(diffs)

# Convert list of dicts to DataFrame (rows: test instances, columns: features)
diff_df = pd.DataFrame(all_differences)

# Compute mean and standard deviation for each feature's intervention effect
agg_stats = diff_df.agg(['mean', 'std']).T
agg_stats.reset_index(inplace=True)
agg_stats.rename(columns={'index': 'Feature'}, inplace=True)

print("\nAggregated Intervention Statistics (Absolute Change in Predicted Probability):")
print(agg_stats)

# -------------------------------
# 7. Visualization of Aggregated Intervention Effects
# -------------------------------
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Feature', y='mean', data=agg_stats, ci=None)
# Add error bars manually:
ax.errorbar(x=np.arange(len(agg_stats)), y=agg_stats['mean'], yerr=agg_stats['std'],
            fmt='none', c='black', capsize=5)
plt.title("Mean Intervention Effect with Std Dev (Change in Predicted Probability)")
plt.xlabel("Feature")
plt.ylabel("Mean Absolute Change in Predicted Probability")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# -------------------------------
# 8. Intervention on Every Test Instance: Patch Most Influential Feature and Record Predictions
# -------------------------------
results = []  # Will store results per instance as strings
changed_count = 0
unchanged_count = 0

for idx, instance in enumerate(X_test_lr):
    baseline_pred = clf.predict_proba([instance])[0]
    baseline_class = clf.predict([instance])[0]
    class_index = list(clf.classes_).index(baseline_class)

    # Determine the most influential feature
    diffs = intervention_for_instance(instance, clf, class_means, feature_names)
    most_influential_feature = max(diffs, key=diffs.get)
    max_diff = diffs[most_influential_feature]

    # For binary classification, determine the "other" class
    classes = class_means.index.tolist()
    other_class = classes[0] if baseline_class == classes[1] else classes[1]

    # Patch only the most influential feature
    patched_instance = instance.copy()
    feature_idx = feature_names.index(most_influential_feature)
    patched_instance[feature_idx] = class_means.loc[other_class, most_influential_feature]

    patched_pred = clf.predict_proba([patched_instance])[0]
    patched_class = clf.predict([patched_instance])[0]

    # Count if the prediction changed
    if patched_class != baseline_class:
        changed_count += 1
    else:
        unchanged_count += 1

    # Record the result for this instance
    result_str = (
        f"Instance {idx}:\n"
        f"  Baseline Predicted Class: {baseline_class} with probabilities {baseline_pred}\n"
        f"  Most Influential Feature: {most_influential_feature} (Diff: {max_diff:.4f})\n"
        f"  Patched Predicted Class: {patched_class} with probabilities {patched_pred}\n\n"
    )
    results.append(result_str)

# Write individual instance results to a text file
output_filename = "Outputs/InterventionPredictions.txt"
with open(output_filename, "w") as f:
    for line in results:
        f.write(line)

# -------------------------------
# 9. Output Summary Statistics
# -------------------------------
total_instances = len(X_test_lr)
summary = (
    f"Total Test Instances: {total_instances}\n"
    f"Number of Predictions Changed: {changed_count}\n"
    f"Number of Predictions Unchanged: {unchanged_count}\n"
    f"Percentage Changed: {changed_count / total_instances * 100:.2f}%\n"
)
print("\nSummary Statistics:")
print(summary)

# Append summary to the text file
with open(output_filename, "a") as f:
    f.write("\n" + summary)

print(f"Intervention predictions and summary statistics written to {output_filename}")
