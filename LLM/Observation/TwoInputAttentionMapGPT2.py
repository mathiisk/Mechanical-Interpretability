import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# 1. Setup Model and Tokenizer
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "gpt2-xl"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model and tokenizer loaded successfully.\n")

num_layers = len(model.transformer.h)  # 48 for GPT2-XL
num_heads = model.config.n_head  # 25 heads per layer

# --------------------------------------------
# 2. Dictionaries to Store Full-Sequence Activations
# --------------------------------------------
# For each layer we capture full sequence outputs for both modules
italy_attn_activations = {}  # attention: shape [1, seq_len, hidden_dim]
france_attn_activations = {}

italy_mlp_activations = {}  # MLP: shape [1, seq_len, hidden_dim]
france_mlp_activations = {}


# ---------------------------------------------------------
# 3. Hook Factories to Capture the Full-Sequence Activations
# ---------------------------------------------------------
def get_attn_activation_hook(layer_idx, storage_dict):
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            # Capture the entire sequence activation from attention output
            storage_dict[layer_idx] = output[0].detach().to("cpu")  # [batch, seq_len, hidden_dim]
        return output

    return hook


def get_mlp_activation_hook(layer_idx, storage_dict):
    def hook(module, input, output):
        # Capture the full sequence output from the MLP
        storage_dict[layer_idx] = output.detach().to("cpu")  # [batch, seq_len, hidden_dim]
        return output

    return hook


# --------------------------------------------
# 4. Capture Activations for Italy Prompt
# --------------------------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt", add_special_tokens=True).to(device)
hooks = []

for layer_idx in range(num_layers):
    h_attn = model.transformer.h[layer_idx].attn.register_forward_hook(
        get_attn_activation_hook(layer_idx, italy_attn_activations)
    )
    h_mlp = model.transformer.h[layer_idx].mlp.register_forward_hook(
        get_mlp_activation_hook(layer_idx, italy_mlp_activations)
    )
    hooks.extend([h_attn, h_mlp])

with torch.no_grad():
    _ = model(**italy_inputs)

for h in hooks:
    h.remove()
print("Captured Italy activations for all layers.\n")

# --------------------------------------------
# 5. Capture Activations for France Prompt
# --------------------------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt", add_special_tokens=True).to(device)
hooks = []

for layer_idx in range(num_layers):
    h_attn = model.transformer.h[layer_idx].attn.register_forward_hook(
        get_attn_activation_hook(layer_idx, france_attn_activations)
    )
    h_mlp = model.transformer.h[layer_idx].mlp.register_forward_hook(
        get_mlp_activation_hook(layer_idx, france_mlp_activations)
    )
    hooks.extend([h_attn, h_mlp])

with torch.no_grad():
    _ = model(**france_inputs)

for h in hooks:
    h.remove()
print("Captured France activations for all layers.\n")


# ---------------------------------------------------------
# 6. Processing: Average Over Sequence and Split into Segments
# ---------------------------------------------------------
def process_activation(act, num_heads):
    """
    Given an activation tensor of shape [1, seq_len, hidden_dim],
    average over the sequence dimension and reshape into [num_heads, head_dim].
    """
    # Average over the sequence dimension (dim=1)
    act_mean = act.mean(dim=1)  # [1, hidden_dim]
    act_mean = act_mean.squeeze(0)  # [hidden_dim]
    hidden_dim = act_mean.shape[-1]
    head_dim = hidden_dim // num_heads
    return act_mean.view(num_heads, head_dim)


# ---------------------------------------------------------
# 7. Compute Difference Norms per Layer and per "Head" Segment
# ---------------------------------------------------------
attn_diff_norms = torch.zeros(num_layers, num_heads)
mlp_diff_norms = torch.zeros(num_layers, num_heads)

for layer_idx in range(num_layers):
    # Process Attention activations
    italy_attn_proc = process_activation(italy_attn_activations[layer_idx], num_heads)
    france_attn_proc = process_activation(france_attn_activations[layer_idx], num_heads)
    # Process MLP activations (we split hidden_dim into 'num_heads' segments for visualization)
    italy_mlp_proc = process_activation(italy_mlp_activations[layer_idx], num_heads)
    france_mlp_proc = process_activation(france_mlp_activations[layer_idx], num_heads)

    for head_idx in range(num_heads):
        attn_diff_norms[layer_idx, head_idx] = torch.norm(italy_attn_proc[head_idx] - france_attn_proc[head_idx])
        mlp_diff_norms[layer_idx, head_idx] = torch.norm(italy_mlp_proc[head_idx] - france_mlp_proc[head_idx])

# ---------------------------------------------------------
# 8. Visualization: Heatmaps of Difference Norms
# ---------------------------------------------------------
plt.figure(figsize=(15, 8))

# Heatmap for Attention differences
plt.subplot(1, 2, 1)
sns.heatmap(attn_diff_norms.numpy(), cmap="viridis",
            xticklabels=[f"Head {i}" for i in range(num_heads)],
            yticklabels=[f"Layer {i}" for i in range(num_layers)])
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.title("Avg Activation Diff Norm per Head (Attention)")

# Heatmap for MLP differences (here we call the segments "MLP segments")
plt.subplot(1, 2, 2)
sns.heatmap(mlp_diff_norms.numpy(), cmap="magma",
            xticklabels=[f"Segment {i}" for i in range(num_heads)],
            yticklabels=[f"Layer {i}" for i in range(num_layers)])
plt.xlabel("MLP Segment")
plt.ylabel("Layer")
plt.title("Avg Activation Diff Norm per Segment (MLP)")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Visualize Individual Activation Maps for a Critical Layer
# ---------------------------------------------------------
critical_layer = 39

# Process activations for the critical layer
italy_attn_crit = process_activation(italy_attn_activations[critical_layer], num_heads)
france_attn_crit = process_activation(france_attn_activations[critical_layer], num_heads)
attn_diff_crit = italy_attn_crit - france_attn_crit

italy_mlp_crit = process_activation(italy_mlp_activations[critical_layer], num_heads)
france_mlp_crit = process_activation(france_mlp_activations[critical_layer], num_heads)
mlp_diff_crit = italy_mlp_crit - france_mlp_crit

# Set up a grid to show individual maps for each segment/head
fig, axs = plt.subplots(num_heads, 4, figsize=(24, num_heads * 2))
for head_idx in range(num_heads):
    # For visualization we create a grid from the vector
    grid_rows = 8
    head_dim = italy_attn_crit.shape[-1]
    grid_cols = head_dim // grid_rows if head_dim % grid_rows == 0 else head_dim // grid_rows + 1

    # Attention maps for Italy, France, and their difference
    attn_italy_map = italy_attn_crit[head_idx][:grid_rows * grid_cols].view(grid_rows, grid_cols).numpy()
    attn_france_map = france_attn_crit[head_idx][:grid_rows * grid_cols].view(grid_rows, grid_cols).numpy()
    attn_diff_map = attn_diff_crit[head_idx][:grid_rows * grid_cols].view(grid_rows, grid_cols).numpy()

    # MLP difference map
    mlp_diff_map = mlp_diff_crit[head_idx][:grid_rows * grid_cols].view(grid_rows, grid_cols).numpy()

    # Plot attention Italy
    sns.heatmap(attn_italy_map, ax=axs[head_idx, 0], cmap="viridis", cbar=False)
    axs[head_idx, 0].set_title(f"Head {head_idx} Attn Italy")
    axs[head_idx, 0].axis('off')

    # Plot attention France
    sns.heatmap(attn_france_map, ax=axs[head_idx, 1], cmap="viridis", cbar=False)
    axs[head_idx, 1].set_title(f"Head {head_idx} Attn France")
    axs[head_idx, 1].axis('off')

    # Plot attention difference
    sns.heatmap(attn_diff_map, ax=axs[head_idx, 2], cmap="coolwarm", center=0, cbar=False)
    axs[head_idx, 2].set_title(f"Head {head_idx} Attn Diff")
    axs[head_idx, 2].axis('off')

    # Plot MLP difference
    sns.heatmap(mlp_diff_map, ax=axs[head_idx, 3], cmap="coolwarm", center=0, cbar=False)
    axs[head_idx, 3].set_title(f"Head {head_idx} MLP Diff")
    axs[head_idx, 3].axis('off')

plt.tight_layout()
plt.suptitle(f"Activation Maps for Layer {critical_layer}", y=1.02)
plt.show()
