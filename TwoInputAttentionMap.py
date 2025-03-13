import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
print("Model and tokenizer loaded successfully.\n")

# Dictionaries to store activations for each layer
italy_activations = {}
france_activations = {}


# A hook factory that captures the activation output from self_attn
def get_activation_hook(layer_idx, storage_dict):
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            # Capture the LAST token's activation
            storage_dict[layer_idx] = output[0][:, -1, :].detach().cpu()
        return output
    return hook


# --- Capture activations for Italy prompt ---
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt")

hooks = []
num_layers = len(model.model.layers)  # e.g. 28 layers
for layer_idx in range(num_layers):
    h = model.model.layers[layer_idx].self_attn.register_forward_hook(
        get_activation_hook(layer_idx, italy_activations)
    )
    hooks.append(h)

with torch.no_grad():
    _ = model(**italy_inputs)

# Remove hooks after capturing
for h in hooks:
    h.remove()
print("Captured activations for Italy prompt.\n")

# --- Capture activations for France prompt ---
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt")

hooks = []
for layer_idx in range(num_layers):
    h = model.model.layers[layer_idx].self_attn.register_forward_hook(
        get_activation_hook(layer_idx, france_activations)
    )
    hooks.append(h)

with torch.no_grad():
    _ = model(**france_inputs)

# Remove hooks after capturing
for h in hooks:
    h.remove()
print("Captured activations for France prompt.\n")

# --- Visualization ---
# For each layer, we have a vector of size hidden_dim.
# For visualization, reshape it to a 2D grid.
# For hidden_dim=1536, one option is 32 x 48 since 32*48=1536.
hidden_dim = list(italy_activations.values())[0].shape[1]
grid_rows = 32
grid_cols = hidden_dim // grid_rows  # should be 48 for 1536

# Visualize every layer with three columns: Italy, France, and Difference.
layers_to_visualize = list(range(num_layers))
num_vis = len(layers_to_visualize)
fig, axs = plt.subplots(num_vis, 3, figsize=(18, num_vis * 3))
if num_vis == 1:
    axs = [axs]  # ensure axs is iterable

# for i, layer_idx in enumerate(layers_to_visualize):
#     # Get the activation vector for the first token and reshape it
#     italy_vec = italy_activations[layer_idx].squeeze(0)  # shape: [hidden_dim]
#     france_vec = france_activations[layer_idx].squeeze(0)
#
#     italy_map = italy_vec.view(grid_rows, grid_cols).numpy()
#     france_map = france_vec.view(grid_rows, grid_cols).numpy()
#
#     # Compute difference heatmap: Italy minus France
#     diff_map = italy_map - france_map
#
#     # Plot Italy activation
#     sns.heatmap(italy_map, ax=axs[i][0], cmap="viridis", cbar=False)
#     axs[i][0].set_title(f"Layer {layer_idx} - Italy")
#     axs[i][0].set_xticks([])
#     axs[i][0].set_yticks([])
#
#     # Plot France activation
#     sns.heatmap(france_map, ax=axs[i][1], cmap="viridis", cbar=False)
#     axs[i][1].set_title(f"Layer {layer_idx} - France")
#     axs[i][1].set_xticks([])
#     axs[i][1].set_yticks([])
#
#     # Plot the difference heatmap using a diverging colormap
#     sns.heatmap(diff_map, ax=axs[i][2], cmap="coolwarm", center=0, cbar=False)
#     axs[i][2].set_title(f"Layer {layer_idx} - Diff (Italy - France)")
#     axs[i][2].set_xticks([])
#     axs[i][2].set_yticks([])
#
# plt.tight_layout()
# plt.show()

# Print tokenized input IDs
# print("Italy prompt tokens:", tokenizer.convert_ids_to_tokens(italy_inputs["input_ids"].squeeze().tolist()))
# print("France prompt tokens:", tokenizer.convert_ids_to_tokens(france_inputs["input_ids"].squeeze().tolist()))

# Print activation differences for a few layers
for layer_idx in [0, num_layers//2, num_layers-1]:  # Check early, middle, and late layers
    italy_vec = italy_activations[layer_idx]
    france_vec = france_activations[layer_idx]
    diff_norm = torch.norm(italy_vec - france_vec).item()
    print(f"Layer {layer_idx}: Activation Difference Norm = {diff_norm}")

# Check if the model generates different outputs
# italy_output = model.generate(**italy_inputs, max_length=15)
# france_output = model.generate(**france_inputs, max_length=15)

# print("Italy Output:", tokenizer.decode(italy_output[0], skip_special_tokens=True))
# print("France Output:", tokenizer.decode(france_output[0], skip_special_tokens=True))


# --- Attention Head Analysis ---
num_heads = model.config.num_attention_heads  # Retrieve number of heads from model config

# (A) Compute per-head activation differences across all layers
head_diff_norms = torch.zeros(num_layers, num_heads)  # [layers, heads]

for layer_idx in range(num_layers):
    # Get activations for Italy and France
    italy_act = italy_activations[layer_idx].squeeze(0)  # [hidden_dim]
    france_act = france_activations[layer_idx].squeeze(0)

    # Split into heads: [num_heads, head_dim]
    head_dim = italy_act.shape[-1] // num_heads
    italy_heads = italy_act.view(num_heads, head_dim)
    france_heads = france_act.view(num_heads, head_dim)

    # Compute norm of differences per head
    head_diff_norms[layer_idx] = torch.norm(italy_heads - france_heads, dim=1)

# (B) Plot heatmap of head differences across layers
plt.figure(figsize=(15, 8))
sns.heatmap(head_diff_norms.numpy(), cmap="viridis",
            xticklabels=[f"Head {i}" for i in range(num_heads)],
            yticklabels=[f"Layer {i}" for i in range(num_layers)])
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.title("Activation Difference Norm per Head and Layer")
plt.show()

# (C) Visualize individual heads for a critical layer (e.g., Layer 20)
critical_layer = 20  # Example layer from previous analysis
italy_act = italy_activations[critical_layer].squeeze(0)
france_act = france_activations[critical_layer].squeeze(0)
head_dim = italy_act.shape[-1] // num_heads

# Reshape into heads and compute differences
italy_heads = italy_act.view(num_heads, head_dim)
france_heads = france_act.view(num_heads, head_dim)
diff_heads = italy_heads - france_heads

# Plot each head's difference map for the critical layer
fig, axs = plt.subplots(num_heads, 3, figsize=(20, num_heads * 2))
for head_idx in range(num_heads):
    # Reshape head activations to 2D grid (e.g., 8x12 for head_dim=96)
    grid_rows = 8
    grid_cols = head_dim // grid_rows

    italy_head_map = italy_heads[head_idx].view(grid_rows, grid_cols).numpy()
    france_head_map = france_heads[head_idx].view(grid_rows, grid_cols).numpy()
    diff_head_map = diff_heads[head_idx].view(grid_rows, grid_cols).numpy()

    # Plot Italy head
    sns.heatmap(italy_head_map, ax=axs[head_idx][0], cmap="viridis", cbar=False)
    axs[head_idx][0].set_title(f"Head {head_idx} - Italy")
    axs[head_idx][0].axis('off')

    # Plot France head
    sns.heatmap(france_head_map, ax=axs[head_idx][1], cmap="viridis", cbar=False)
    axs[head_idx][1].set_title(f"Head {head_idx} - France")
    axs[head_idx][1].axis('off')

    # Plot difference
    sns.heatmap(diff_head_map, ax=axs[head_idx][2], cmap="coolwarm", center=0, cbar=False)
    axs[head_idx][2].set_title(f"Head {head_idx} - Diff")
    axs[head_idx][2].axis('off')

plt.tight_layout()
plt.suptitle(f"Head Activations and Differences for Layer {critical_layer}", y=1.02)
plt.show()

