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
# X, y = load_classification("AsphaltRegularity", split="train")
X, y = load_classification("AsphaltRegularity", split="train")

# Apply the Catch22 transformation
catch22 = Catch22()
X_transformed = catch22.fit_transform(X)

# Create a DataFrame with feature names
feature_names = [f'Feature_{i}' for i in range(X_transformed.shape[1])]
df_features = pd.DataFrame(X_transformed, columns=feature_names)
df_features['class'] = np.array(y)  # assign class labels

print("Transformed Features with Class Labels:")
print(df_features.head())

# -------------------------------
# 2. Compute Class Average Feature Profiles
# -------------------------------
# Compute the average (mean) feature values for each class
class_means = df_features.groupby('class').mean()
print("\nClass Mean Features:")
print(class_means)

# -------------------------------
# 3. Prepare Data for Classifier Training
# -------------------------------
X_features = df_features[feature_names].copy()
y_labels = df_features['class']

# Impute missing values (e.g., using the mean strategy)
imputer = SimpleImputer(strategy='mean')
X_features_imputed = imputer.fit_transform(X_features)

# Split the data (80% training, 20% testing)
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_features_imputed, y_labels, test_size=0.2, random_state=42)

# -------------------------------
# 4. Train a Simple Classifier on Catch22 Features
# -------------------------------
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_lr, y_train_lr)

# -------------------------------
# 5. Baseline Prediction on a Test Instance
# -------------------------------
# Choose a test instance from the test set.
instance_idx = 0
original_instance = X_test_lr[instance_idx].copy()

baseline_pred = clf.predict_proba([original_instance])[0]
predicted_class = clf.predict([original_instance])[0]

print("\nBaseline Test Instance Prediction:")
print("Original Instance Features:", original_instance)
print("\nBaseline Prediction Probabilities:", baseline_pred)
print("Predicted Class:", predicted_class)

# -------------------------------
# 6. Intervention: One-Feature-at-a-Time Patching
# -------------------------------
if len(class_means) < 2:
    raise ValueError("Need at least two classes for intervention.")

classes = class_means.index[:2]
print("\nUsing class averages for classes:", classes)

# Determine the "other" class: if the predicted class is classes[1], then use classes[0]; else, use classes[1].
other_class = classes[0] if predicted_class == classes[1] else classes[1]

# Store the intervention results.
intervention_results = {}

for i, feature in enumerate(feature_names):
    patched_instance = original_instance.copy()
    patched_value = class_means.loc[other_class, feature]  # Replace with the other class's mean
    patched_instance[i] = patched_value
    patched_pred = clf.predict_proba([patched_instance])[0]
    intervention_results[feature] = patched_pred

# Convert intervention results to a DataFrame
intervention_df = pd.DataFrame(intervention_results).T
intervention_df.columns = [f'Prob_Class_{c}' for c in clf.classes_]

print("\nIntervention Results (Prediction Probabilities after Patching Each Feature):")
print(intervention_df)

# -------------------------------
# 7. Identify the Most Influential Feature
# -------------------------------
# Compute the difference in probability for the originally predicted class
target_class_label = f'Prob_Class_{predicted_class}'

class_index = list(clf.classes_).index(predicted_class)

# Now use the index to compute the difference
intervention_df['Diff'] = np.abs(intervention_df[target_class_label] - baseline_pred[class_index])

# Find the feature with the largest difference
most_influential_feature = intervention_df['Diff'].idxmax()
max_diff = intervention_df['Diff'].max()

print(f"\nMost Influential Feature: {most_influential_feature} (Change in probability: {max_diff:.4f})")

# -------------------------------
# 8. Patch Only the Most Influential Feature and Observe Change
# -------------------------------
patched_instance = original_instance.copy()
patched_instance[feature_names.index(most_influential_feature)] = class_means.loc[other_class, most_influential_feature]

patched_pred = clf.predict_proba([patched_instance])[0]
patched_class = clf.predict([patched_instance])[0]

print("\nBaseline Prediction Probabilities:", baseline_pred)
print("Predicted Class:", predicted_class)
print("\nPatched Instance (Single Feature Intervention):")
print(f"Feature {most_influential_feature} patched with value from class {other_class}")
print("\nPatched Prediction Probabilities:", patched_pred)
print("New Predicted Class:", patched_class)

# -------------------------------
# 9. Visualization of Intervention Effects
# -------------------------------
# Plot the effect of each feature intervention on probability of predicted class
plt.figure(figsize=(10, 6))
sns.barplot(x=intervention_df.index, y=intervention_df[target_class_label])
plt.title(f"Effect of Intervening Each Feature on Probability of Class {predicted_class}")
plt.xlabel("Intervened Feature")
plt.ylabel(f"Probability of Class {predicted_class}")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
