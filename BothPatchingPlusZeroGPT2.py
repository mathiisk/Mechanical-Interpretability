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

# Storage for activations
stored_activations = {}
stored_mlp_activations = {}
layer_idx = 15 # Layer to patch

# -------------------------------------------------------------------
# 2. Capture Hooks
# -------------------------------------------------------------------
def capture_attention_hook(module, input, output):
    """Stores attention activations"""
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        stored_activations["attn"] = output[0].clone().detach().cpu()
    return output

def capture_mlp_activation_hook(module, input, output):
    """Stores MLP activations"""
    stored_mlp_activations["mlp"] = output.clone().detach().cpu()
    return output

# -------------------------------------------------------------------
# 3. Patch Hooks (Attention + MLP)
# -------------------------------------------------------------------
def patch_attention_hook(module, input, output):
    """Patches attention activations"""
    if "attn" in stored_activations:
        patched_output = stored_activations["attn"].to(device)
        if patched_output.shape == output[0].shape:
            return (patched_output,) + output[1:]
    return output

def patch_mlp_activation_hook(module, input, output):
    """Patches MLP activations"""
    if "mlp" in stored_mlp_activations:
        patched_output = stored_mlp_activations["mlp"].to(device)
        if patched_output.shape == output.shape:
            return patched_output
    return output

# -------------------------------------------------------------------
# 4. Ablation Hooks (Zero-out Activations)
# -------------------------------------------------------------------
def ablate_attention_hook(module, input, output):
    """Zero out attention activations"""
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        zeroed_output = torch.zeros_like(output[0]).to(device)
        return (zeroed_output,) + output[1:]
    return output

def ablate_mlp_activation_hook(module, input, output):
    """Zero out MLP activations"""
    zeroed_output = torch.zeros_like(output).to(device)
    return zeroed_output

# -------------------------------------------------------------------
# 5. Capture "Italy" Activations
# -------------------------------------------------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt").to(device)

# Register hooks to capture activations
hook_attn_capture = model.transformer.h[layer_idx].attn.register_forward_hook(capture_attention_hook)
hook_mlp_capture = model.transformer.h[layer_idx].mlp.register_forward_hook(capture_mlp_activation_hook)

with torch.no_grad():
    italy_output = model.generate(italy_inputs['input_ids'], max_new_tokens=15, use_cache=False)

italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Prompt Output:", italy_text)

hook_attn_capture.remove()
hook_mlp_capture.remove()

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
# 7. Patch Both Attention + MLP
# -------------------------------------------------------------------
hook_attn_patch = model.transformer.h[layer_idx].attn.register_forward_hook(patch_attention_hook)
hook_mlp_patch = model.transformer.h[layer_idx].mlp.register_forward_hook(patch_mlp_activation_hook)

with torch.no_grad():
    france_patched_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_patched_text = tokenizer.decode(france_patched_output[0], skip_special_tokens=True)
print("France Prompt with Attention + MLP Patching:", france_patched_text)

hook_attn_patch.remove()
hook_mlp_patch.remove()

# -------------------------------------------------------------------
# 8. Ablate (Zero-out) Both Attention + MLP
# -------------------------------------------------------------------
hook_attn_ablate = model.transformer.h[layer_idx].attn.register_forward_hook(ablate_attention_hook)
hook_mlp_ablate = model.transformer.h[layer_idx].mlp.register_forward_hook(ablate_mlp_activation_hook)

with torch.no_grad():
    france_ablated_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_ablated_text = tokenizer.decode(france_ablated_output[0], skip_special_tokens=True)
print("France Prompt with Attention + MLP Ablated:", france_ablated_text)

hook_attn_ablate.remove()
hook_mlp_ablate.remove()
