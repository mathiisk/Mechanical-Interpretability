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

# Define range of layers to patch
start_layer = 38  # Change this to select the first layer
end_layer = 47    # Change this to select the last layer
layers_to_patch = range(start_layer, end_layer)

# Storage for activations
stored_activations = {layer: {} for layer in layers_to_patch}
stored_mlp_activations = {layer: {} for layer in layers_to_patch}

# -------------------------------------------------------------------
# 2. Capture Hooks
# -------------------------------------------------------------------
def capture_attention_hook(layer_idx):
    """Stores attention activations for a given layer."""
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            stored_activations[layer_idx]["attn"] = output[0].clone().detach().cpu()
        return output
    return hook

def capture_mlp_activation_hook(layer_idx):
    """Stores MLP activations for a given layer."""
    def hook(module, input, output):
        stored_mlp_activations[layer_idx]["mlp"] = output.clone().detach().cpu()
        return output
    return hook

# -------------------------------------------------------------------
# 3. Patch Hooks (Attention + MLP)
# -------------------------------------------------------------------
def patch_attention_hook(layer_idx):
    """Patches stored attention activations for a given layer."""
    def hook(module, input, output):
        if "attn" in stored_activations[layer_idx]:
            patched_output = stored_activations[layer_idx]["attn"].to(device)
            if patched_output.shape == output[0].shape:
                return (patched_output,) + output[1:]
        return output
    return hook

def patch_mlp_activation_hook(layer_idx):
    """Patches stored MLP activations for a given layer."""
    def hook(module, input, output):
        if "mlp" in stored_mlp_activations[layer_idx]:
            patched_output = stored_mlp_activations[layer_idx]["mlp"].to(device)
            if patched_output.shape == output.shape:
                return patched_output
        return output
    return hook

# -------------------------------------------------------------------
# 4. Zeroing Out (Ablation) Hooks
# -------------------------------------------------------------------
def ablate_attention_hook(layer_idx):
    """Zero out attention activations for a given layer."""
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            zeroed_output = torch.zeros_like(output[0]).to(device)
            return (zeroed_output,) + output[1:]
        return output
    return hook

def ablate_mlp_activation_hook(layer_idx):
    """Zero out MLP activations for a given layer."""
    def hook(module, input, output):
        zeroed_output = torch.zeros_like(output).to(device)
        return zeroed_output
    return hook

# -------------------------------------------------------------------
# 5. Capture "Italy" Activations
# -------------------------------------------------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt").to(device)

# Register hooks to capture activations for multiple layers
hook_handles_capture = []
for layer in layers_to_patch:
    hook_handles_capture.append(model.transformer.h[layer].attn.register_forward_hook(capture_attention_hook(layer)))
    hook_handles_capture.append(model.transformer.h[layer].mlp.register_forward_hook(capture_mlp_activation_hook(layer)))

with torch.no_grad():
    italy_output = model.generate(italy_inputs['input_ids'], max_new_tokens=15, use_cache=False)

italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Prompt Output:", italy_text)

# Remove capture hooks
for handle in hook_handles_capture:
    handle.remove()

# -------------------------------------------------------------------
# 6. Generate "France" with No Patching (Baseline)
# -------------------------------------------------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    france_original_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_original_text = tokenizer.decode(france_original_output[0], skip_special_tokens=True)
print("Original France Prompt Output:", france_original_text)

# -------------------------------------------------------------------
# 7. Patch Multiple Layers (Both Attention + MLP)
# -------------------------------------------------------------------
hook_handles_patch = []
for layer in layers_to_patch:
    hook_handles_patch.append(model.transformer.h[layer].attn.register_forward_hook(patch_attention_hook(layer)))
    hook_handles_patch.append(model.transformer.h[layer].mlp.register_forward_hook(patch_mlp_activation_hook(layer)))

with torch.no_grad():
    france_patched_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_patched_text = tokenizer.decode(france_patched_output[0], skip_special_tokens=True)
print(f"France Prompt with Attention + MLP Patching (Layers {start_layer}-{end_layer}):", france_patched_text)

# Remove patching hooks
for handle in hook_handles_patch:
    handle.remove()

# -------------------------------------------------------------------
# 8. Zero Out (Ablate) Multiple Layers (Both Attention + MLP)
# -------------------------------------------------------------------
hook_handles_ablate = []
for layer in layers_to_patch:
    hook_handles_ablate.append(model.transformer.h[layer].attn.register_forward_hook(ablate_attention_hook(layer)))
    hook_handles_ablate.append(model.transformer.h[layer].mlp.register_forward_hook(ablate_mlp_activation_hook(layer)))

with torch.no_grad():
    france_ablated_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_ablated_text = tokenizer.decode(france_ablated_output[0], skip_special_tokens=True)
print(f"France Prompt with Attention + MLP Zeroed Out (Layers {start_layer}-{end_layer}):", france_ablated_text)

# Remove ablation hooks
for handle in hook_handles_ablate:
    handle.remove()
