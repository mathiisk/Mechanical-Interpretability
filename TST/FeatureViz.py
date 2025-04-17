import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from TSTTrainer import TimeSeriesTransformer, test_loader, X_train, y_train


class TransformerDiffAnalyzer:
    """
    An analyzer to compute and visualize mean difference heatmaps across transformer layers.
    It collects the layer outputs (activations) from the transformer encoder,
    computes the mean absolute differences in activations between two classes,
    and produces heatmaps to identify discriminative features.
    """

    def __init__(self, model, test_loader, device=torch.device("cpu"), max_per_class=30):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.max_per_class = max_per_class

        # Prepare a container to store outputs of each transformer encoder layer.
        self.layer_outputs = {f"layer_{i}": [] for i in range(len(self.model.transformer_encoder.layers))}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture layer outputs."""

        def make_hook(layer_id):
            def hook(module, input, output):
                # Store the layer output (detach from graph and move to CPU)
                self.layer_outputs[f"layer_{layer_id}"].append(output.detach().cpu())

            return hook

        for i, layer in enumerate(self.model.transformer_encoder.layers):
            h = layer.register_forward_hook(make_hook(i))
            self.hooks.append(h)

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def collect_activations(self, label1, label2):
        """
        Runs through the test_loader to collect a subset of activations from two classes.

        Returns:
            labels: np.array of the collected sample labels.
        """
        counts = {label1: 0, label2: 0}
        selected_inputs = []
        selected_labels = []

        for X_batch, y_batch in self.test_loader:
            mask = (y_batch == label1) | (y_batch == label2)
            X_sub = X_batch[mask]
            y_sub = y_batch[mask]
            for x, y in zip(X_sub, y_sub):
                y_val = y.item()
                if counts[y_val] < self.max_per_class:
                    selected_inputs.append(x.unsqueeze(0))
                    selected_labels.append(y_val)
                    counts[y_val] += 1
            if all(c >= self.max_per_class for c in counts.values()):
                break

        # Run forward pass on all collected samples
        for x in selected_inputs:
            _ = self.model(x.to(self.device))
        return np.array(selected_labels)

    def compute_heatmaps(self, labels, label1, label2):
        """
        Computes heatmaps (mean absolute differences) across all transformer layers.

        Returns:
            heatmaps: dict mapping layer names to heatmap arrays of shape (seq_len, d_model)
        """
        heatmaps = {}
        for key, acts in self.layer_outputs.items():
            # Stack activations: shape (N, seq_len, d_model)
            A = torch.cat(acts, dim=0).numpy()
            # Separate activations for the two classes
            cls1 = A[labels == label1]
            cls2 = A[labels == label2]
            mean1 = cls1.mean(axis=0)  # (seq_len, d_model)
            mean2 = cls2.mean(axis=0)
            diff = np.abs(mean1 - mean2)  # (seq_len, d_model)
            heatmaps[key] = diff
        return heatmaps

    def plot_heatmaps(self, heatmaps, label1, label2):
        """Plots the computed heatmaps for each transformer layer."""
        num_layers = len(heatmaps)
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]

        for ax, (key, diff_map) in zip(axes, heatmaps.items()):
            sns.heatmap(diff_map.T, ax=ax, cmap="viridis", cbar=True)
            ax.set_title(f"{key} - Mean |diff| (Class {label1} vs {label2})")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Feature (d_model dim)")
        plt.tight_layout()
        plt.show()

    def analyze(self, label1, label2):
        """
        Runs the full analysis: registers hooks, collects activations,
        computes the mean difference heatmaps, plots them, and cleans up hooks.
        """
        # Reset any stored outputs
        for key in self.layer_outputs.keys():
            self.layer_outputs[key] = []
        self.register_hooks()
        labels = self.collect_activations(label1, label2)
        heatmaps = self.compute_heatmaps(labels, label1, label2)
        self.plot_heatmaps(heatmaps, label1, label2)
        self.remove_hooks()


def load_model():
    """
    Loads the pre-trained TimeSeriesTransformer model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        input_dim=X_train.shape[2],
        num_classes=len(np.unique(y_train)),
        seq_len=X_train.shape[1],
        d_model=128, nhead=8,
        num_layers=3, dim_feedforward=256,
        dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load("time_series_transformer_fancy.pth", map_location=device))
    model.eval()
    return model, device


if __name__ == "__main__":
    # Load model and device
    model, device = load_model()

    # Instantiate the analyzer with chosen parameters
    analyzer = TransformerDiffAnalyzer(model, test_loader, device=device, max_per_class=30)

    # Choose two labels to compare (change these to analyze different class pairs)
    label1 = 4
    label2 = 7

    # Run analysis: this will display heatmaps for each transformer layer
    analyzer.analyze(label1, label2)
