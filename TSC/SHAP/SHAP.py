import os
import numpy as np
import pandas as pd
import shap
from aeon.datasets import load_classification
from aeon.transformations.collection.feature_based import Catch22
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# -------------------------------
# Data Processing Class
# -------------------------------
class DataProcessor:
    def __init__(self, dataset_name):
        # For different datasets, see https://www.timeseriesclassification.com/dataset.php
        self.dataset_name = dataset_name
        self.df_features = None
        self.feature_names = None
        self.class_means = None

    def load_and_transform(self):
        # Load dataset and apply Catch22 transformation
        X, y = load_classification(self.dataset_name, split="train")
        catch22 = Catch22()
        X_transformed = catch22.fit_transform(X)

        # Create DataFrame with feature names and class labels
        self.feature_names = [f'Feature_{i}' for i in range(X_transformed.shape[1])]
        self.df_features = pd.DataFrame(X_transformed, columns=self.feature_names)
        self.df_features['class'] = np.array(y)
        # Uncomment these lines if you want to see the output:
        # print("Transformed Features with Class Labels:")
        # print(self.df_features.head())

# -------------------------------
# Model Trainer Class
# -------------------------------
class ModelTrainer:
    def __init__(self, df_features, feature_names):
        self.df_features = df_features
        self.feature_names = feature_names
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clf = None

    def prepare_data(self):
        # Prepare features and labels; impute missing values; split data
        X_features = self.df_features[self.feature_names].copy()
        y_labels = self.df_features['class']
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_features)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_imputed, y_labels, test_size=0.2, random_state=42
        )

    def train_classifier(self):
        # Train a Logistic Regression classifier
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(self.X_train, self.y_train)

# -------------------------------
# SHAP Analysis Class
# -------------------------------
class SHAP:
    def __init__(self, clf, X_train, X_test, feature_names):
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        # Set up the SHAP explainer for a linear model using interventional perturbation
        self.explainer = shap.LinearExplainer(self.clf, self.X_train, feature_perturbation="interventional")
        self.shap_values = None

    def compute_shap_values(self):
        # Compute SHAP values for the test set
        self.shap_values = self.explainer.shap_values(self.X_test)
        return self.shap_values

    def plot_summary(self):
        # Plot a SHAP summary plot for the test set
        if self.shap_values is None:
            self.compute_shap_values()
        shap.summary_plot(self.shap_values, self.X_test, feature_names=self.feature_names)

    def plot_dependence(self, feature):
        # Plot a SHAP dependence plot for the given feature
        if self.shap_values is None:
            self.compute_shap_values()
        shap.dependence_plot(feature, self.shap_values, self.X_test, feature_names=self.feature_names)

# -------------------------------
# Main Execution Flow
# -------------------------------
def main():
    # Data processing
    processor = DataProcessor("Yoga")
    processor.load_and_transform()

    # Model training
    trainer = ModelTrainer(processor.df_features, processor.feature_names)
    trainer.prepare_data()
    trainer.train_classifier()

    # SHAP analysis
    shap_analysis = SHAP(trainer.clf, trainer.X_train, trainer.X_test, processor.feature_names)
    shap_analysis.compute_shap_values()
    shap_analysis.plot_summary()
    # shap_analysis.plot_dependence("Feature_6")

if __name__ == "__main__":
    main()
