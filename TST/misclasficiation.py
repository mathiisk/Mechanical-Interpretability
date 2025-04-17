import torch
import torch.nn.functional as F
import numpy as np
from TSTTrainer import TimeSeriesTransformer, X_test, y_test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(y_test.numpy()))
seq_len, input_dim = X_test.shape[1], X_test.shape[2]

model = TimeSeriesTransformer(
    input_dim=input_dim, num_classes=num_classes, seq_len=seq_len,
    d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1
).to(device)
model.load_state_dict(torch.load("time_series_transformer_fancy.pth", map_location=device))
model.eval()
# Make sure your model is in evaluation mode and on the correct device
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Container to store inference results as a list of dictionaries
results = []

# Iterate over the test_loader and gather predictions along with ground truth labels.
# We assume test_loader and device are already defined as in your model training code.
with torch.no_grad():
    # Keep track of the global index as you iterate through batches.
    global_index = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        # Compute probabilities from logits so we can check confidence later if needed.
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        batch_size = X_batch.size(0)
        for i in range(batch_size):
            # Get the maximum probability (i.e. confidence) for this prediction.
            confidence = probs[i].max().item()
            # Append all relevant information to our list.
            results.append({
                "index": global_index,
                "instance": X_batch[i].cpu().numpy(),  # Convert instance to numpy array.
                "true_label": y_batch[i].item(),
                "pred_label": preds[i].item(),
                "confidence": confidence
            })
            global_index += 1

# Convert results to a structured numpy array or simply work with the list of dicts.
print("Total test instances processed:", len(results))

# ---------------------------------------------------------------------------------
# Filter for a specific target class.
# Change target_class to the label you want to inspect.
target_class = 1  # for example, choose class "1" or any class present in your dataset

# Find all instances that belong to the target class.
target_results = [res for res in results if res["true_label"] == target_class]

print(f"Total instances in class {target_class}:", len(target_results))

# Among those, find one where the prediction is correct and one where it’s wrong.
correct_example = None
misclassified_example = None

for res in target_results:
    if (res["true_label"] == res["pred_label"]) and (correct_example is None):
        correct_example = res
    elif (res["true_label"] != res["pred_label"]) and (misclassified_example is None):
        misclassified_example = res
    # Stop early if both examples are found.
    if correct_example and misclassified_example:
        break

# Verification: if one or both of the examples aren’t found, print a message.
if correct_example is None:
    print(f"No correctly classified instance found for class {target_class}. Try a different class.")
else:
    print("Found a correctly classified instance:")
    print("Index:", correct_example["index"])
    print("True label:", correct_example["true_label"], "Predicted label:", correct_example["pred_label"])
    print("Prediction confidence:", correct_example["confidence"])
    print("Instance shape:", correct_example["instance"].shape)

if misclassified_example is None:
    print(f"No misclassified instance found for class {target_class}. Try a different class.")
else:
    print("Found a misclassified instance:")
    print("Index:", misclassified_example["index"])
    print("True label:", misclassified_example["true_label"], "Predicted label:", misclassified_example["pred_label"])
    print("Prediction confidence:", misclassified_example["confidence"])
    print("Instance shape:", misclassified_example["instance"].shape)
