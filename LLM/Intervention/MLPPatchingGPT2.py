import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------
# 1. Setup Model
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-xl"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

stored_mlp_activations = {}  # Stores MLP activations
layer_idx = 39  # Last layer in GPT-2 XL

# -------------------------------------------------------------------
# 2. Helper Functions for Capturing and Patching MLP Activations
# -------------------------------------------------------------------
def capture_mlp_activation_hook(module, input, output):
    """
    Captures and stores the MLP activations from GPT-2 for the entire sequence.
    Output shape: [batch, seq_len, hidden_dim]
    """
    stored_mlp_activations["mlp"] = output.clone().detach().cpu()
    return output

def patch_mlp_activation_hook(module, input, output):
    """
    Patches the MLP activations with the stored activations.
    """
    if "mlp" in stored_mlp_activations:
        patched_output = stored_mlp_activations["mlp"].to(device)  # Move stored activations to the correct device
        if patched_output.shape == output.shape:
            return patched_output  # Replace MLP output with stored activations
    return output

# -------------------------------------------------------------------
# 3. Capture MLP Activations from "Italy" Prompt
# -------------------------------------------------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt").to(device)

# Register hook to capture MLP activations
hook_handle_capture = model.transformer.h[layer_idx].mlp.register_forward_hook(capture_mlp_activation_hook)

with torch.no_grad():
    italy_output = model.generate(italy_inputs['input_ids'], max_new_tokens=15, use_cache=False)

italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Prompt Output:", italy_text)

hook_handle_capture.remove()  # Remove capture hook

# -------------------------------------------------------------------
# 4. Generate "France" Prompt with No Patching (Baseline)
# -------------------------------------------------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    france_original_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_original_text = tokenizer.decode(france_original_output[0], skip_special_tokens=True)
print("Original France Prompt Output:", france_original_text)

# -------------------------------------------------------------------
# 5. Patch the MLP Activations and Regenerate "France"
# -------------------------------------------------------------------
hook_handle_patch = model.transformer.h[layer_idx].mlp.register_forward_hook(patch_mlp_activation_hook)

with torch.no_grad():
    france_patched_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_patched_text = tokenizer.decode(france_patched_output[0], skip_special_tokens=True)
print("France Prompt with MLP Patching:", france_patched_text)

hook_handle_patch.remove()  # Remove patching hook
