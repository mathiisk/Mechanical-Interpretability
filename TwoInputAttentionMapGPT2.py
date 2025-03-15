import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the GPT2-XL model and tokenizer
model_name = "gpt2-xl"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT2 doesn't have an official pad token; set it to eos_token.
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model and tokenizer loaded successfully.\n")

# Dictionaries to store activations for each layer
italy_activations = {}
france_activations = {}


# Hook factory to capture activations
def get_activation_hook(layer_idx, storage_dict):
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            # Capture the last token's hidden state from the attention output
            storage_dict[layer_idx] = output[0][:, -1, :].detach().to("cpu")
        return output

    return hook


# --- Capture activations for Italy prompt ---
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt", add_special_tokens=True).to(device)

hooks = []
num_layers = len(model.transformer.h)  # GPT2-XL has 48 layers
for layer_idx in range(num_layers):
    h = model.transformer.h[layer_idx].attn.register_forward_hook(
        get_activation_hook(layer_idx, italy_activations)
    )
    hooks.append(h)

with torch.no_grad():
    _ = model(**italy_inputs)

for h in hooks:
    h.remove()
print("Captured activations for Italy prompt.\n")

# --- Capture activations for France prompt ---
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt", add_special_tokens=True).to(device)

hooks = []
for layer_idx in range(num_layers):
    h = model.transformer.h[layer_idx].attn.register_forward_hook(
        get_activation_hook(layer_idx, france_activations)
    )
    hooks.append(h)

with torch.no_grad():
    _ = model(**france_inputs)

for h in hooks:
    h.remove()
print("Captured activations for France prompt.\n")

# --- Attention Head Analysis ---
num_heads = model.config.n_head  # For GPT2-XL, this is 25
# Compute per-head activation differences across all layers
head_diff_norms = torch.zeros(num_layers, num_heads)  # [layers, heads]

for layer_idx in range(num_layers):
    italy_act = italy_activations[layer_idx].squeeze(0)  # [hidden_dim]
    france_act = france_activations[layer_idx].squeeze(0)

    head_dim = italy_act.shape[-1] // num_heads  # hidden_dim divided into heads
    italy_heads = italy_act.view(num_heads, head_dim)
    france_heads = france_act.view(num_heads, head_dim)

    head_diff_norms[layer_idx] = torch.norm(italy_heads - france_heads, dim=1)

# --- Visualization: Heatmap of per-head differences ---
plt.figure(figsize=(15, 8))
sns.heatmap(head_diff_norms.numpy(), cmap="viridis",
            xticklabels=[f"Head {i}" for i in range(num_heads)],
            yticklabels=[f"Layer {i}" for i in range(num_layers)])
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.title("Activation Difference Norm per Head and Layer")
plt.show()

# --- Visualize individual head maps for a critical layer ---
critical_layer = 22  # Example critical layer
italy_act = italy_activations[critical_layer].squeeze(0)
france_act = france_activations[critical_layer].squeeze(0)
head_dim = italy_act.shape[-1] // num_heads

# Reshape into heads and compute differences
italy_heads = italy_act.view(num_heads, head_dim)
france_heads = france_act.view(num_heads, head_dim)
diff_heads = italy_heads - france_heads

fig, axs = plt.subplots(num_heads, 3, figsize=(20, num_heads * 2))
for head_idx in range(num_heads):
    # For visualization, we'll display a grid. Adjust grid size as needed.
    grid_rows = 8
    grid_cols = head_dim // grid_rows if head_dim % grid_rows == 0 else head_dim // grid_rows + 1

    italy_head_map = italy_heads[head_idx][:grid_rows * grid_cols].view(grid_rows, grid_cols).numpy()
    france_head_map = france_heads[head_idx][:grid_rows * grid_cols].view(grid_rows, grid_cols).numpy()
    diff_head_map = diff_heads[head_idx][:grid_rows * grid_cols].view(grid_rows, grid_cols).numpy()

    sns.heatmap(italy_head_map, ax=axs[head_idx][0], cmap="viridis", cbar=False)
    axs[head_idx][0].set_title(f"Head {head_idx} - Italy")
    axs[head_idx][0].axis('off')

    sns.heatmap(france_head_map, ax=axs[head_idx][1], cmap="viridis", cbar=False)
    axs[head_idx][1].set_title(f"Head {head_idx} - France")
    axs[head_idx][1].axis('off')

    sns.heatmap(diff_head_map, ax=axs[head_idx][2], cmap="coolwarm", center=0, cbar=False)
    axs[head_idx][2].set_title(f"Head {head_idx} - Diff")
    axs[head_idx][2].axis('off')

plt.tight_layout()
plt.suptitle(f"Head Activations and Differences for Layer {critical_layer}", y=1.02)
plt.show()
