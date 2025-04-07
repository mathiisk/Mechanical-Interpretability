import os
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from aeon.transformations.collection.feature_based import Catch22
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy.stats as stats

# -------------------------------
# Data Processing Class
# -------------------------------
class DataProcessor:
    def __init__(self, dataset_name="Wine"):
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
        # print("Transformed Features with Class Labels:")
        # print(self.df_features.head())

    def compute_class_means(self):
        # Compute mean values for each feature per class
        self.class_means = self.df_features.groupby('class').mean()
        # print("\nClass Mean Features:")
        # print(self.class_means)

    def save_class_means(self, filename="Outputs/ClassMeans.csv"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.class_means.to_csv(filename, index=True)
        print(f"Class means saved to {filename}")

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
# Intervention Analysis Class
# -------------------------------
class InterventionAnalysis:
    def __init__(self, clf, X_test, feature_names, class_means):
        self.clf = clf
        self.X_test = X_test
        self.feature_names = feature_names
        self.class_means = class_means
        self.results = []
        self.summary = ""
        self.agg_stats = None
        self.prob_changes_df = None
        self.all_differences = []
        self.all_patches_df = None

    def intervention_for_instance(self, instance):
        """
        For a given test instance, compute the baseline prediction and then intervene on each feature
        (patching with the other class's mean) to record the absolute change in predicted probability
        for the originally predicted class.
        """
        baseline_pred = self.clf.predict_proba([instance])[0]
        predicted_class = self.clf.predict([instance])[0]
        class_index = list(self.clf.classes_).index(predicted_class)

        # Determine the "other" class (assumes binary classification)
        classes = self.class_means.index.tolist()
        other_class = classes[0] if predicted_class == classes[1] else classes[1]

        differences = {}
        for i, feature in enumerate(self.feature_names):
            patched_instance = instance.copy()
            patched_instance[i] = self.class_means.loc[other_class, feature]
            patched_pred = self.clf.predict_proba([patched_instance])[0]
            diff = abs(baseline_pred[class_index] - patched_pred[class_index])
            differences[feature] = diff
        return differences

    def run_analysis(self):
        all_differences = []  # List of dicts for intervention differences per instance
        prob_changes = []     # List to record baseline and patched probabilities
        changed_count = 0
        unchanged_count = 0

        for idx, instance in enumerate(self.X_test):
            baseline_pred = self.clf.predict_proba([instance])[0]
            baseline_class = self.clf.predict([instance])[0]
            class_index = list(self.clf.classes_).index(baseline_class)

            # Determine most influential feature
            diffs = self.intervention_for_instance(instance)
            most_influential_feature = max(diffs, key=diffs.get)
            max_diff = diffs[most_influential_feature]

            # Determine the "other" class
            classes = self.class_means.index.tolist()
            other_class = classes[0] if baseline_class == classes[1] else classes[1]

            # Patch only the most influential feature
            patched_instance = instance.copy()
            feature_idx = self.feature_names.index(most_influential_feature)
            patched_instance[feature_idx] = self.class_means.loc[other_class, most_influential_feature]

            patched_pred = self.clf.predict_proba([patched_instance])[0]
            patched_class = self.clf.predict([patched_instance])[0]

            if patched_class != baseline_class:
                changed_count += 1
            else:
                unchanged_count += 1

            # Record detailed result
            result_str = (
                f"Instance {idx}:\n"
                f"  Baseline Predicted Class: {baseline_class} with probabilities {baseline_pred}\n"
                f"  Most Influential Feature: {most_influential_feature} (Diff: {max_diff:.4f})\n"
                f"  Patched Predicted Class: {patched_class} with probabilities {patched_pred}\n\n"
            )
            self.results.append(result_str)
            all_differences.append(diffs)

            # Record probability changes for plotting
            prob_changes.append({
                "instance": idx,
                "baseline": baseline_pred[class_index],
                "patched": patched_pred[class_index]
            })

        # Store all intervention differences for later statistical tests
        self.all_differences = all_differences

        # Aggregate intervention differences
        diff_df = pd.DataFrame(all_differences)
        self.agg_stats = diff_df.agg(['mean', 'std']).T.reset_index().rename(columns={'index': 'Feature'})

        total_instances = len(self.X_test)
        self.summary = (
            f"Total Test Instances: {total_instances}\n"
            f"Number of Predictions Changed: {changed_count}\n"
            f"Number of Predictions Unchanged: {unchanged_count}\n"
            f"Percentage Changed: {changed_count / total_instances * 100:.2f}%\n"
        )
        print("\nSummary Statistics:")
        print(self.summary)

        # Save probability changes DataFrame for plotting and statistical tests
        self.prob_changes_df = pd.DataFrame(prob_changes)

    def run_all_feature_analysis(self):
        """
        For each test instance, patch every feature one by one.
        This will create a DataFrame with one row per patch (instance x feature),
        storing the baseline probability, patched probability, and a flag indicating
        whether the prediction changed.
        """
        records = []
        for idx, instance in enumerate(self.X_test):
            baseline_pred = self.clf.predict_proba([instance])[0]
            baseline_class = self.clf.predict([instance])[0]
            class_index = list(self.clf.classes_).index(baseline_class)
            for feature in self.feature_names:
                patched_instance = instance.copy()
                # Determine the 'other' class for patching (using the baseline predicted class)
                classes = self.class_means.index.tolist()
                other_class = classes[0] if baseline_class == classes[1] else classes[1]
                feature_idx = self.feature_names.index(feature)
                patched_instance[feature_idx] = self.class_means.loc[other_class, feature]
                patched_pred = self.clf.predict_proba([patched_instance])[0]
                patched_class = self.clf.predict([patched_instance])[0]
                changed = int(patched_class != baseline_class)
                records.append({
                    "instance": idx,
                    "feature": feature,
                    "baseline_probability": baseline_pred[class_index],
                    "patched_probability": patched_pred[class_index],
                    "changed": changed
                })
        self.all_patches_df = pd.DataFrame(records)
        print("\nAll-Feature Patch Analysis Completed:")
        print(self.all_patches_df.head())
        return self.all_patches_df

    def perform_statistical_tests(self, significance_level=0.05):
        """
        Perform statistical tests on the intervention differences:
          - One-sample t-tests for each feature's differences against 0.
          - Paired t-test comparing baseline and patched predicted probabilities.
        Results are printed and saved to the Outputs folder.
        """
        # One-sample t-test for each feature's differences
        diff_df = pd.DataFrame(self.all_differences)
        feature_ttest_results = []
        for feature in self.feature_names:
            t_stat, p_value = stats.ttest_1samp(diff_df[feature], 0)
            feature_ttest_results.append({
                "Feature": feature,
                "t_statistic": t_stat,
                "p_value": p_value,
                "Significant": p_value < significance_level
            })
        feature_ttest_df = pd.DataFrame(feature_ttest_results)
        print("\nOne-sample t-test results for each feature (testing if mean difference ≠ 0):")
        print(feature_ttest_df)

        # Save the feature t-test results to CSV
        os.makedirs("Outputs", exist_ok=True)
        feature_ttest_df.to_csv("Outputs/FeatureTTestResults.csv", index=False)
        print("Feature t-test results saved to Outputs/FeatureTTestResults.csv")

        # Paired t-test for baseline vs. patched predicted probabilities
        t_stat_prob, p_value_prob = stats.ttest_rel(self.prob_changes_df["baseline"], self.prob_changes_df["patched"])
        paired_test_result = {
            "t_statistic": t_stat_prob,
            "p_value": p_value_prob,
            "Significant": p_value_prob < significance_level
        }
        # print("\nPaired t-test for baseline vs. patched predicted probabilities:")
        # print(paired_test_result)

        # Save paired t-test result to a text file
        with open("Outputs/PairedTTestResult.txt", "w") as f:
            f.write(str(paired_test_result))
        print("Paired t-test result saved to Outputs/PairedTTestResult.txt")

    def perform_all_feature_stat_tests(self, significance_level=0.05):
        """
        Perform statistical analysis on the all-encompassing patch results.
        For example, compute overall change rate and perform a binomial test,
        as well as per-feature tests to see if some features cause more changes.
        """
        if self.all_patches_df is None:
            print("Run the all-encompassing analysis first.")
            return

        total_patches = len(self.all_patches_df)
        overall_changed = self.all_patches_df["changed"].sum()
        overall_proportion = overall_changed / total_patches
        binom_result_overall = stats.binomtest(overall_changed, total_patches, p=0.01, alternative='two-sided')
        binom_overall_p_value = binom_result_overall.pvalue
        overall_results = {
            "total_patches": total_patches,
            "overall_changed": overall_changed,
            "overall_proportion": overall_proportion,
            "binom_p_value": binom_overall_p_value,
            "Significant": binom_overall_p_value < significance_level
        }
        print("\nOverall All-Feature Binomial Test Results:")
        print(overall_results)
        with open("Outputs/AllFeatureBinomialTestResult.txt", "w") as f:
            f.write(str(overall_results))
        print("Overall all-feature binomial test result saved to Outputs/AllFeatureBinomialTestResult.txt")

        # Per-feature analysis:
        per_feature_results = []
        for feature, group in self.all_patches_df.groupby("feature"):
            total = len(group)
            changed = group["changed"].sum()
            proportion = changed / total
            binom_result_per_feature = stats.binomtest(changed, total, p=0.01, alternative='two-sided')
            binom_per_feature_p_value = binom_result_per_feature.pvalue
            per_feature_results.append({
                "feature": feature,
                "total_patches": total,
                "changed": changed,
                "proportion": proportion,
                "binom_p_value": binom_per_feature_p_value,
                "Significant": binom_per_feature_p_value < significance_level
            })
        per_feature_df = pd.DataFrame(per_feature_results)
        print("\nPer-Feature Binomial Test Results:")
        print(per_feature_df)
        per_feature_df.to_csv("Outputs/PerFeatureBinomialTestResults.csv", index=False)
        print("Per-feature binomial test results saved to Outputs/PerFeatureBinomialTestResults.csv")

    def save_results(self, filename="Outputs/InterventionPredictions.txt"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            for line in self.results:
                f.write(line)
            f.write("\n" + self.summary)
        print(f"Intervention predictions and summary statistics written to {filename}")

    def save_agg_stats(self, filename="Outputs/AggregatedInterventionStats.csv"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.agg_stats.to_csv(filename, index=False)
        print(f"Aggregated intervention stats saved to {filename}")

    def save_probability_changes(self, filename="Outputs/ProbabilityChanges.csv"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.prob_changes_df.to_csv(filename, index=False)
        print(f"Probability changes saved to {filename}")

# -------------------------------
# Main Execution Flow
# -------------------------------
def main():
    # Data processing
    processor = DataProcessor()
    processor.load_and_transform()
    processor.compute_class_means()
    processor.save_class_means()  # Save class means to CSV

    # Model training
    trainer = ModelTrainer(processor.df_features, processor.feature_names)
    trainer.prepare_data()
    trainer.train_classifier()

    # Intervention analysis
    analysis = InterventionAnalysis(trainer.clf, trainer.X_test, processor.feature_names, processor.class_means)
    analysis.run_analysis()
    analysis.save_results()          # Save intervention details to text file
    analysis.save_agg_stats()        # Save aggregated stats to CSV
    analysis.save_probability_changes()  # Save probability changes to CSV
    analysis.perform_statistical_tests() # Run stastistical tests

    analysis.run_all_feature_analysis()
    analysis.perform_all_feature_stat_tests()
if __name__ == "__main__":
    main()
