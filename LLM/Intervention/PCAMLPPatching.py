import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.decomposition import PCA

# -------------------------------
# 1. Setup Model and Parameters
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-xl"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Choose a target layer (for example, layer 39)
target_layer = 39

# Number of top PCA components to use for intervention
top_k = 5
# Intervention strength (1.0 = full replacement of the top components)
alpha = 1.0

# -------------------------------
# 2. Capture Italy MLP Activations
# -------------------------------
# We define a hook that will store the MLP activations from the target layer.
italy_mlp_activation = None  # global variable to store activations

def capture_mlp_hook(module, input, output):
    global italy_mlp_activation
    # output shape: [batch, seq_len, hidden_dim]
    italy_mlp_activation = output.clone().detach().cpu()
    return output

# Register hook on the target layer’s MLP submodule
hook_handle = model.transformer.h[target_layer].mlp.register_forward_hook(capture_mlp_hook)

# Run Italy prompt to capture activations
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    italy_output = model.generate(italy_inputs['input_ids'], max_new_tokens=15, use_cache=False)
italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Prompt Output:", italy_text)

# Remove the capture hook
hook_handle.remove()

# -------------------------------
# 3. PCA on Italy MLP Activations
# -------------------------------
# Assume batch=1; shape: [1, seq_len, hidden_dim]
# We'll flatten across tokens (i.e. combine the batch and token dimensions)
italy_act = italy_mlp_activation.squeeze(0)  # shape: [seq_len, hidden_dim]
# Fit PCA on Italy activations (per token)
pca_model = PCA(n_components=top_k)
pca_model.fit(italy_act.numpy())

# Compute Italy’s average activation (across tokens) and its PCA coefficients.
italy_avg = np.mean(italy_act.numpy(), axis=0)  # shape: [hidden_dim]
italy_avg_coeff = pca_model.transform(italy_avg.reshape(1, -1))[0]  # shape: [top_k]

# -------------------------------
# 4. Define a Patch Hook Using PCA for MLP
# -------------------------------
def patch_mlp_pca_hook(module, input, output):
    """
    Intervenes on the MLP activations by replacing the top PCA coefficients with those from Italy.
    For each token, we:
      1. Transform its activation (a vector in R^(hidden_dim)) into PCA space.
      2. Replace the coefficients for the top components (indices 0...top_k-1) with Italy's average coefficients.
      3. Inverse-transform back to R^(hidden_dim).
    """
    # output: tensor of shape [batch, seq_len, hidden_dim]
    # We assume batch size = 1.
    out_cpu = output.clone().detach().cpu().numpy()  # shape: [1, seq_len, hidden_dim]
    batch, seq_len, hidden_dim = out_cpu.shape
    # Reshape to [seq_len, hidden_dim]
    x = out_cpu[0]
    # Transform into PCA space: shape [seq_len, top_k]
    coeff = pca_model.transform(x)
    # For the selected top_k components, replace with Italy's average coefficients.
    # You can also interpolate; here we use full replacement controlled by alpha.
    new_coeff = (1 - alpha) * coeff + alpha * np.tile(italy_avg_coeff, (seq_len, 1))
    # Get the reconstruction from the modified coefficients.
    new_top = new_coeff.dot(pca_model.components_) + pca_model.mean_
    # For a smoother intervention, one can combine the residual (the part not explained by top_k components).
    # Here we compute the original top_k reconstruction:
    original_top = coeff.dot(pca_model.components_) + pca_model.mean_
    # Then, the intervened activation is:
    x_intervened = x - original_top + new_top
    # Alternatively, a simple interpolation would be:
    # x_intervened = (1 - alpha) * x + alpha * new_top
    # Reshape back to [1, seq_len, hidden_dim]
    x_intervened = np.expand_dims(x_intervened, axis=0)
    # Convert back to tensor on the correct device.
    new_out = torch.tensor(x_intervened, dtype=output.dtype, device=device)
    return new_out

# -------------------------------
# 5. Generate France Prompt Baseline
# -------------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    france_original_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)
france_original_text = tokenizer.decode(france_original_output[0], skip_special_tokens=True)
print("Original France Prompt Output:", france_original_text)

# -------------------------------
# 6. Intervention: Patch France MLP Activations Using PCA
# -------------------------------
# Register the patch hook on the target layer’s MLP.
hook_handle_patch = model.transformer.h[target_layer].mlp.register_forward_hook(patch_mlp_pca_hook)
with torch.no_grad():
    france_patched_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)
france_patched_text = tokenizer.decode(france_patched_output[0], skip_special_tokens=True)
print("France Prompt with PCA-based MLP Patching:", france_patched_text)
hook_handle_patch.remove()

# -------------------------------
# 7. Save All Outputs to a Text File
# -------------------------------
output_filename = "Outputs/PCAMLPPatchingOutput.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("Italy Prompt Output:\n" + italy_text + "\n\n")
    f.write("Original France Prompt Output:\n" + france_original_text + "\n\n")
    f.write("France Prompt with PCA-based MLP Patching:\n" + france_patched_text + "\n")
print(f"Results saved to {output_filename}")
