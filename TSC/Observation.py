import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aeon.datasets import load_classification
from aeon.transformations.collection.feature_based import Catch22

# -------------------------------
# 1. Load a Classification Dataset from Aeon
# -------------------------------
# X_train, y_train = load_classification("AsphaltRegularity", split="train")
X_train, y_train = load_classification("BinaryHeartbeat", split="train")

# -------------------------------
# 2. Apply the Catch22 Transformation
# -------------------------------
catch22 = Catch22()
X_train_transformed = catch22.fit_transform(X_train)

# Create a DataFrame with appropriate feature names
feature_names = [f'Feature_{i}' for i in range(X_train_transformed.shape[1])]
df_features = pd.DataFrame(X_train_transformed, columns=feature_names)

# Assign class labels
df_features['class'] = np.array(y_train)

print("Fixed Transformed Features with Class Labels:")
print(df_features.head())

# -------------------------------
# 3. Compute and Compare Average Feature Profiles per Class
# -------------------------------
class_means = df_features.groupby('class').mean()

if len(class_means) >= 2:
    classes = class_means.index[:2]
    class1_mean = class_means.loc[classes[0]]
    class2_mean = class_means.loc[classes[1]]

    # Compute the difference between class averages for each feature.
    feature_diff = class1_mean - class2_mean

    # Create a DataFrame for visualization.
    diff_df = pd.DataFrame({
        'Feature': feature_diff.index,
        f'Average {classes[0]}': class1_mean.values,
        f'Average {classes[1]}': class2_mean.values,
        'Difference': feature_diff.values
    })

    print("\nAverage Feature Values and Their Differences:")
    print(diff_df)

    # -------------------------------
    # 4. Visualize Feature Differences
    # -------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Difference', y='Feature', data=diff_df, orient='h')
    plt.title(f"Difference in Catch22 Features between classes {classes[0]} and {classes[1]}")
    plt.xlabel("Difference in Feature Value")
    plt.ylabel("Catch22 Feature")
    plt.tight_layout()
    plt.show()
else:
    print("The dataset does not contain at least two classes for comparison.")
