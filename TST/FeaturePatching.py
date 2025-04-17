import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from TSTTrainer import TimeSeriesTransformer, X_test, y_test

# ========== Parameters ==========
layers_to_patch = [0, 1, 2]  # patch layer 1 and 2 simultaneously
top_k = 10
ref_index = 200
target_index = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(y_test.numpy()))
seq_len, input_dim = X_test.shape[1], X_test.shape[2]

model = TimeSeriesTransformer(
    input_dim=input_dim, num_classes=num_classes, seq_len=seq_len,
    d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1
).to(device)
model.load_state_dict(torch.load("time_series_transformer_fancy.pth", map_location=device))
model.eval()

# ========== Input Selection ==========
input_A = X_test[ref_index].unsqueeze(0).to(device)
input_B = X_test[target_index].unsqueeze(0).to(device)

print(f"Label A: {y_test[ref_index].item()} | Label B: {y_test[target_index].item()}")

# ========== Capture Reference and Target Activations ==========
captured_A_layers = {}
captured_B_layers = {}

def make_capture_hook(store_dict, layer_idx):
    def hook(module, inp, out):
        store_dict[layer_idx] = out.detach().clone()
    return hook

handles = []
for layer in layers_to_patch:
    handles.append(model.transformer_encoder.layers[layer].register_forward_hook(make_capture_hook(captured_A_layers, layer)))
_ = model(input_A)
for h in handles: h.remove()

handles = []
for layer in layers_to_patch:
    handles.append(model.transformer_encoder.layers[layer].register_forward_hook(make_capture_hook(captured_B_layers, layer)))
_ = model(input_B)
for h in handles: h.remove()

# ========== Compute Top-k Features Per Layer ==========
topk_per_layer = {}
for layer in layers_to_patch:
    A = captured_A_layers[layer]
    B = captured_B_layers[layer]
    diff = torch.abs(A - B).mean(dim=1).squeeze(0)  # mean over time
    topk = torch.topk(diff, k=top_k).indices.tolist()
    topk_per_layer[layer] = topk
    print(f"Layer {layer} top-{top_k} features: {topk}")

# ========== Baseline Prediction ==========
with torch.no_grad():
    baseline_logits = model(input_B)
    baseline_probs = F.softmax(baseline_logits, dim=-1).cpu().numpy().squeeze()
    baseline_class = np.argmax(baseline_probs)

# ========== Patch Hook Across Multiple Layers ==========
handles = []
for layer in layers_to_patch:
    topk_feats = topk_per_layer[layer]
    ref_act = captured_A_layers[layer]
    def make_patch_hook(layer_idx, top_feats, ref_vals):
        def patch_hook(module, inp, out):
            patched = out.clone()
            for dim in top_feats:
                patched[:, :, dim] = ref_vals[:, :, dim]
            return patched
        return patch_hook
    hook_fn = make_patch_hook(layer, topk_feats, ref_act)
    h = model.transformer_encoder.layers[layer].register_forward_hook(hook_fn)
    handles.append(h)

# ========== Run Patched Forward ==========
with torch.no_grad():
    patched_logits = model(input_B)
    patched_probs = F.softmax(patched_logits, dim=-1).cpu().numpy().squeeze()
    patched_class = np.argmax(patched_probs)

for h in handles:
    h.remove()

# ========== Report Results ==========
# X-axis: class labels
classes = np.arange(num_classes)
bar_width = 0.4

plt.figure(figsize=(8, 4))

# Plot baseline probs
plt.bar(classes - bar_width/2, baseline_probs, width=bar_width, label="Baseline", alpha=0.8)

# Plot patched probs
plt.bar(classes + bar_width/2, patched_probs, width=bar_width, label="Patched", alpha=0.8)

plt.title(f"Multi-Layer Patch: Top-{top_k} features at layers {layers_to_patch}")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.xticks(classes)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
