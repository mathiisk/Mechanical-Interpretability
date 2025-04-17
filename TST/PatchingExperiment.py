import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from aeon.datasets import load_classification
from TSTTrainer import TimeSeriesTransformer  # or import the model from the shared module


# (Optional: If you want a standalone copy, you can also redefine the model here)
# class TimeSeriesTransformer(nn.Module):
#     def __init__(...):
#         ...
#
#     def forward(self, x):
#         ...

def get_transformer_input(model, instance):
    """Processes a single instance (shape: (seq_len, input_dim)) through conv layers and adds positional encoding."""
    x = instance.unsqueeze(0)  # (1, seq_len, input_dim)
    x = x.transpose(1, 2)  # (1, input_dim, seq_len)
    x = torch.relu(model.bn1(model.conv1(x)))
    x = torch.relu(model.bn2(model.conv2(x)))
    x = torch.relu(model.bn3(model.conv3(x)))
    x = x.transpose(1, 2)  # (1, seq_len, d_model)
    x = x + model.positional_encoding
    return x


def run_denoising_patching(model, clean_instance, corrupted_instance, patch_layer_idx):
    """
    Runs the transformer encoder on clean and corrupted inputs,
    then patches the corrupted activation with the clean one at the given layer.
    Returns the output logits.
    """
    x_clean = get_transformer_input(model, clean_instance)
    x_corr = get_transformer_input(model, corrupted_instance)

    # Obtain clean activation at the chosen patch layer.
    x_clean_current = x_clean.clone()
    for i, layer in enumerate(model.transformer_encoder.layers):
        if i == patch_layer_idx:
            patch_activation = x_clean_current.clone()
        x_clean_current = layer(x_clean_current)

    # Run corrupted input and replace activation at patch layer.
    x_corr_current = x_corr.clone()
    for i, layer in enumerate(model.transformer_encoder.layers):
        if i == patch_layer_idx:
            x_corr_current = patch_activation.clone()
        else:
            x_corr_current = layer(x_corr_current)

    pooled = model.pool(x_corr_current.transpose(1, 2)).squeeze(-1)
    logits = model.classifier(pooled)
    return logits


def main():
    # Load test dataset
    X_train, y_train = load_classification("JapaneseVowels", split="train")
    X_test, y_test = load_classification("JapaneseVowels", split="test")

    # Convert labels to integers.
    y_train = np.array(y_train).astype(np.int64)
    y_test = np.array(y_test).astype(np.int64)

    # Process X_test: swap axes as needed.
    X_test_np = np.swapaxes(X_test.astype(np.float32), 1, 2)
    min_val = y_train.min()
    if min_val != 0:
        y_test = y_test - min_val
    X_test_tensor = torch.tensor(X_test_np)
    y_test_tensor = torch.tensor(y_test)

    batch_size = 4
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = X_test_tensor.shape[1]
    input_dim = X_test_tensor.shape[2]
    num_classes = len(np.unique(y_test))

    # Load the saved model.
    model = TimeSeriesTransformer(input_dim, num_classes, seq_len, d_model=128, nhead=8,
                                  num_layers=3, dim_feedforward=256, dropout=0.1).to(device)
    model_path = "time_series_transformer_fancy.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded saved model from", model_path)
    else:
        print("Model file not found. Run training first.")
        return

    # Run inference on test set and collect predictions.
    results = []
    with torch.no_grad():
        global_index = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            for i in range(X_batch.size(0)):
                results.append({
                    "index": global_index,
                    "true_label": y_batch[i].item(),
                    "pred_label": preds[i].item(),
                    "confidence": probs[i].max().item()
                })
                global_index += 1

    print("Total test instances processed:", len(results))

    # Save results as CSV so you can review them manually.
    df = pd.DataFrame(results)
    df.to_csv("test_results.csv", index=False)
    print("Test predictions saved to 'test_results.csv'.")

    # --- Manual selection ---
    # After reviewing test_results.csv, set these indices to the desired instances.
    # For example:
    chosen_correct_index = 32  # change this to the index you want (a correct instance)
    chosen_incorrect_index = 36  # change this to the index you want (a misclassified instance)

    # Convert results list to a dictionary keyed by index for fast lookup.
    results_dict = {r["index"]: r for r in results}
    if chosen_correct_index not in results_dict or chosen_incorrect_index not in results_dict:
        print("Chosen indices not found in the results. Please check test_results.csv.")
        return

    # Load the corresponding test instances.
    correct_instance = X_test_tensor[chosen_correct_index]
    incorrect_instance = X_test_tensor[chosen_incorrect_index]
    print("Selected correct instance index:", chosen_correct_index,
          "and incorrect instance index:", chosen_incorrect_index)

    # Choose a transformer encoder layer index to patch.
    patch_layer_index = 1  # adjust as desired

    with torch.no_grad():
        # Run the model normally for the selected instances.
        correct_logits = model(correct_instance.unsqueeze(0))
        correct_probs = F.softmax(correct_logits, dim=1)
        incorrect_logits = model(incorrect_instance.unsqueeze(0))
        incorrect_probs = F.softmax(incorrect_logits, dim=1)

        # Run the denoising patch experiment: patching the incorrect instance using the correct activation.
        patched_logits = run_denoising_patching(model, correct_instance.to(device),
                                                incorrect_instance.to(device),
                                                patch_layer_index)
        patched_probs = F.softmax(patched_logits, dim=1)

    # Save outputs for later plotting.
    np.savez("patching_results.npz",
             clean_logits=correct_logits.cpu().numpy().flatten(),
             corrupted_logits=incorrect_logits.cpu().numpy().flatten(),
             patched_logits=patched_logits.cpu().numpy().flatten(),
             clean_probs=correct_probs.cpu().numpy().flatten(),
             corrupted_probs=incorrect_probs.cpu().numpy().flatten(),
             patched_probs=patched_probs.cpu().numpy().flatten(),
             num_classes=num_classes)

    print("Patching experiment completed and results saved to 'patching_results.npz'.")


if __name__ == "__main__":
    main()
