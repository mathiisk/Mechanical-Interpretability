import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_mean_features(class_means):
    """
    Create separate bar plots for the mean feature values for each class.
    """
    os.makedirs("Outputs/Plots", exist_ok=True)
    for cls in class_means.index:
        plt.figure(figsize=(10, 5))
        means = class_means.loc[cls]
        plt.bar(means.index, means.values)
        plt.title(f"Mean Feature Values for Class {cls}")
        plt.xlabel("Features")
        plt.ylabel("Mean Value")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"Outputs/Plots/Mean_Features_Class_{cls}.png")
        plt.close()
    print("Mean feature plots saved in Outputs/Plots/")


def plot_mean_differences(class_means):
    """
    Plot the difference between the mean feature values of the two classes.
    Positive differences are shown in green and negative in red.
    Assumes binary classification.
    """
    if len(class_means.index) != 2:
        print("Mean difference plot only supports binary classification.")
        return

    cls1, cls2 = class_means.index[:2]
    diff = class_means.loc[cls1] - class_means.loc[cls2]
    colors = ['green' if val >= 0 else 'red' for val in diff]

    plt.figure(figsize=(12, 6))
    plt.bar(diff.index, diff.values, color=colors)
    plt.title(f"Difference in Mean Feature Values (Class {cls1} - Class {cls2})")
    plt.xlabel("Feature")
    plt.ylabel("Difference in Mean Value")
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs("Outputs/Plots", exist_ok=True)
    plt.savefig("Outputs/Plots/Mean_Features_Differences.png")
    plt.close()
    print("Mean differences plot saved in Outputs/Plots/")


def plot_probability_changes():
    """
    Plot the baseline and patched predicted probabilities for each test instance.
    This function creates:
      1. A line plot showing baseline vs patched probabilities.
      2. A bar plot showing the difference (baseline - patched) per instance.
    Expects a CSV file "Outputs/ProbabilityChanges.csv" with columns:
      'instance', 'baseline', and 'patched'.
    """
    # Load probability changes CSV
    prob_df = pd.read_csv("Outputs/ProbabilityChanges.csv")

    # Line plot for baseline and patched probabilities
    plt.figure(figsize=(12, 6))
    plt.plot(prob_df['instance'], prob_df['baseline'], marker='o', label='Baseline Probability')
    plt.plot(prob_df['instance'], prob_df['patched'], marker='o', label='Patched Probability')
    plt.title("Baseline vs Patched Predicted Probabilities per Instance")
    plt.xlabel("Instance Index")
    plt.ylabel("Predicted Probability")
    plt.legend()
    plt.tight_layout()
    os.makedirs("Outputs/Plots", exist_ok=True)
    plt.savefig("Outputs/Plots/Probability_Changes_Line.png")
    plt.close()

    # Bar plot for the difference in probabilities
    prob_df['difference'] = prob_df['baseline'] - prob_df['patched']
    plt.figure(figsize=(12, 6))
    colors = ['green' if diff >= 0 else 'red' for diff in prob_df['difference']]
    plt.bar(prob_df['instance'], prob_df['difference'], color=colors)
    plt.title("Difference in Predicted Probabilities (Baseline - Patched)")
    plt.xlabel("Instance Index")
    plt.ylabel("Probability Difference")
    plt.tight_layout()
    plt.savefig("Outputs/Plots/Probability_Changes_Bar.png")
    plt.close()

    print("Probability change plots saved in Outputs/Plots/")


def plot_baseline_vs_intervention(prob_csv="Outputs/ProbabilityChanges.csv"):
    """
    Plot a scatter plot with baseline predicted probabilities on the x-axis
    and the absolute difference (intervention effect) on the y-axis.
    Expects a CSV file with columns 'instance', 'baseline', and 'patched'.
    """
    # Load probability changes data
    prob_df = pd.read_csv(prob_csv)
    # Compute absolute difference (intervention effect)
    prob_df['difference'] = abs(prob_df['baseline'] - prob_df['patched'])

    plt.figure(figsize=(12, 6))
    plt.scatter(prob_df['baseline'], prob_df['difference'])
    plt.title("Scatter Plot: Baseline vs Intervention Effect")
    plt.xlabel("Baseline Predicted Probability")
    plt.ylabel("Absolute Difference (Intervention Effect)")
    plt.tight_layout()
    os.makedirs("Outputs/Plots", exist_ok=True)
    plt.savefig("Outputs/Plots/Baseline_vs_Intervention_Scatter.png")
    plt.close()
    print("Baseline vs Intervention scatter plot saved in Outputs/Plots/Baseline_vs_Intervention_Scatter.png")


def plot_prediction_change_ratio(changed_count, unchanged_count):
    """
    Plot a bar chart showing the number of instances where the prediction changed
    versus those that remained unchanged after intervention.
    """
    labels = ['Changed', 'Unchanged']
    counts = [changed_count, unchanged_count]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['red', 'green'])
    plt.title("Prediction Change Ratio")
    plt.ylabel("Number of Instances")

    # Annotate each bar with the corresponding count
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    os.makedirs("Outputs/Plots", exist_ok=True)
    plt.savefig("Outputs/Plots/Prediction_Change_Ratio.png")
    plt.close()
    print("Prediction change ratio plot saved in Outputs/Plots/Prediction_Change_Ratio.png")


if __name__ == "__main__":
    # Example of reading saved class means for other plots if needed:
    class_means = pd.read_csv("Outputs/ClassMeans.csv", index_col=0)
    plot_mean_features(class_means)
    plot_mean_differences(class_means)
    plot_baseline_vs_intervention()
    plot_prediction_change_ratio(changed_count=120, unchanged_count=31)
    plot_probability_changes()
