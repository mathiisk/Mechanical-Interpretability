# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # -------------------------------------------------------------------
# # 1. Setup Model
# # -------------------------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "gpt2-xl"
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.eval()
#
# # Define range of layers to patch
# start_layer = 38  # Change this to select the first layer
# end_layer = 47    # Change this to select the last layer
# layers_to_patch = range(start_layer, end_layer)
#
# # Storage for activations
# stored_activations = {layer: {} for layer in layers_to_patch}
# stored_mlp_activations = {layer: {} for layer in layers_to_patch}
#
# # -------------------------------------------------------------------
# # 2. Capture Hooks
# # -------------------------------------------------------------------
# def capture_attention_hook(layer_idx):
#     """Stores attention activations for a given layer."""
#     def hook(module, input, output):
#         if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
#             stored_activations[layer_idx]["attn"] = output[0].clone().detach().cpu()
#         return output
#     return hook
#
# def capture_mlp_activation_hook(layer_idx):
#     """Stores MLP activations for a given layer."""
#     def hook(module, input, output):
#         stored_mlp_activations[layer_idx]["mlp"] = output.clone().detach().cpu()
#         return output
#     return hook
#
# # -------------------------------------------------------------------
# # 3. Patch Hooks (Attention + MLP)
# # -------------------------------------------------------------------
# def patch_attention_hook(layer_idx):
#     """Patches stored attention activations for a given layer."""
#     def hook(module, input, output):
#         if "attn" in stored_activations[layer_idx]:
#             patched_output = stored_activations[layer_idx]["attn"].to(device)
#             if patched_output.shape == output[0].shape:
#                 return (patched_output,) + output[1:]
#         return output
#     return hook
#
# def patch_mlp_activation_hook(layer_idx):
#     """Patches stored MLP activations for a given layer."""
#     def hook(module, input, output):
#         if "mlp" in stored_mlp_activations[layer_idx]:
#             patched_output = stored_mlp_activations[layer_idx]["mlp"].to(device)
#             if patched_output.shape == output.shape:
#                 return patched_output
#         return output
#     return hook
#
# # -------------------------------------------------------------------
# # 4. Zeroing Out (Ablation) Hooks
# # -------------------------------------------------------------------
# def ablate_attention_hook(layer_idx):
#     """Zero out attention activations for a given layer."""
#     def hook(module, input, output):
#         if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
#             zeroed_output = torch.zeros_like(output[0]).to(device)
#             return (zeroed_output,) + output[1:]
#         return output
#     return hook
#
# def ablate_mlp_activation_hook(layer_idx):
#     """Zero out MLP activations for a given layer."""
#     def hook(module, input, output):
#         zeroed_output = torch.zeros_like(output).to(device)
#         return zeroed_output
#     return hook
#
# # -------------------------------------------------------------------
# # 5. Capture "Italy" Activations
# # -------------------------------------------------------------------
# italy_prompt = "The capital of Italy is"
# italy_inputs = tokenizer(italy_prompt, return_tensors="pt").to(device)
#
# # Register hooks to capture activations for multiple layers
# hook_handles_capture = []
# for layer in layers_to_patch:
#     hook_handles_capture.append(model.transformer.h[layer].attn.register_forward_hook(capture_attention_hook(layer)))
#     hook_handles_capture.append(model.transformer.h[layer].mlp.register_forward_hook(capture_mlp_activation_hook(layer)))
#
# with torch.no_grad():
#     italy_output = model.generate(italy_inputs['input_ids'], max_new_tokens=15, use_cache=False)
#
# italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
# print("Italy Prompt Output:", italy_text)
#
# # Remove capture hooks
# for handle in hook_handles_capture:
#     handle.remove()
#
# # -------------------------------------------------------------------
# # 6. Generate "France" with No Patching (Baseline)
# # -------------------------------------------------------------------
# france_prompt = "The capital of France is"
# france_inputs = tokenizer(france_prompt, return_tensors="pt").to(device)
#
# with torch.no_grad():
#     france_original_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)
#
# france_original_text = tokenizer.decode(france_original_output[0], skip_special_tokens=True)
# print("Original France Prompt Output:", france_original_text)
#
# # -------------------------------------------------------------------
# # 7. Patch Multiple Layers (Both Attention + MLP)
# # -------------------------------------------------------------------
# hook_handles_patch = []
# for layer in layers_to_patch:
#     hook_handles_patch.append(model.transformer.h[layer].attn.register_forward_hook(patch_attention_hook(layer)))
#     hook_handles_patch.append(model.transformer.h[layer].mlp.register_forward_hook(patch_mlp_activation_hook(layer)))
#
# with torch.no_grad():
#     france_patched_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)
#
# france_patched_text = tokenizer.decode(france_patched_output[0], skip_special_tokens=True)
# print(f"France Prompt with Attention + MLP Patching (Layers {start_layer}-{end_layer}):", france_patched_text)
#
# # Remove patching hooks
# for handle in hook_handles_patch:
#     handle.remove()
#
# # -------------------------------------------------------------------
# # 8. Zero Out (Ablate) Multiple Layers (Both Attention + MLP)
# # -------------------------------------------------------------------
# hook_handles_ablate = []
# for layer in layers_to_patch:
#     hook_handles_ablate.append(model.transformer.h[layer].attn.register_forward_hook(ablate_attention_hook(layer)))
#     hook_handles_ablate.append(model.transformer.h[layer].mlp.register_forward_hook(ablate_mlp_activation_hook(layer)))
#
# with torch.no_grad():
#     france_ablated_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)
#
# france_ablated_text = tokenizer.decode(france_ablated_output[0], skip_special_tokens=True)
# print(f"France Prompt with Attention + MLP Zeroed Out (Layers {start_layer}-{end_layer}):", france_ablated_text)
#
# # Remove ablation hooks
# for handle in hook_handles_ablate:
#     handle.remove()


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# 1. Setup Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-xl"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Define range of layers to patch
start_layer = 39  # Change this to select the first layer
end_layer = 40  # Change this to select the last layer (use a small range for debugging)
layers_to_patch = range(start_layer, end_layer)
num_heads = model.config.n_head  # e.g., 25 for GPT2-XL
# For MLP, we assume the same segmentation as num_heads

# This list will store all outputs for saving later.
results = []


# -------------------------------
# 2. Helper Functions: Split & Combine Heads / Segments
# -------------------------------
def split_heads(activation_3d, n_parts):
    """
    Splits a tensor of shape [batch, seq_len, hidden_dim] into
    [batch, n_parts, seq_len, part_dim].
    """
    batch_size, seq_len, hidden_dim = activation_3d.shape
    part_dim = hidden_dim // n_parts
    activation_4d = activation_3d.view(batch_size, seq_len, n_parts, part_dim)
    activation_4d = activation_4d.permute(0, 2, 1, 3).contiguous()  # [batch, n_parts, seq_len, part_dim]
    return activation_4d


def combine_heads(activation_4d):
    """
    Combines a tensor of shape [batch, n_parts, seq_len, part_dim] into
    [batch, seq_len, hidden_dim].
    """
    batch_size, n_parts, seq_len, part_dim = activation_4d.shape
    activation_4d = activation_4d.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, n_parts, part_dim]
    return activation_4d.view(batch_size, seq_len, n_parts * part_dim)


# -------------------------------
# 3. Capture, Full Patch & Ablation Hooks (for entire layers)
# -------------------------------
# Storage for activations
stored_activations = {layer: {} for layer in layers_to_patch}
stored_mlp_activations = {layer: {} for layer in layers_to_patch}


# (A) Capture hooks – store entire sequence activations from attention and MLP
def capture_attention_hook(layer_idx):
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            stored_activations[layer_idx]["attn"] = output[0].clone().detach().cpu()
        return output

    return hook


def capture_mlp_activation_hook(layer_idx):
    def hook(module, input, output):
        stored_mlp_activations[layer_idx]["mlp"] = output.clone().detach().cpu()
        return output

    return hook


# (B) Full patch hooks – replace the entire attention and MLP activations with stored ones
def patch_attention_hook(layer_idx):
    def hook(module, input, output):
        if "attn" in stored_activations[layer_idx]:
            patched_output = stored_activations[layer_idx]["attn"].to(device)
            if patched_output.shape == output[0].shape:
                return (patched_output,) + output[1:]
        return output

    return hook


def patch_mlp_activation_hook(layer_idx):
    def hook(module, input, output):
        if "mlp" in stored_mlp_activations[layer_idx]:
            patched_output = stored_mlp_activations[layer_idx]["mlp"].to(device)
            if patched_output.shape == output.shape:
                return patched_output
        return output

    return hook


# (C) Ablation hooks – zero out activations
def ablate_attention_hook(layer_idx):
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            zeroed_output = torch.zeros_like(output[0]).to(device)
            return (zeroed_output,) + output[1:]
        return output

    return hook


def ablate_mlp_activation_hook(layer_idx):
    def hook(module, input, output):
        zeroed_output = torch.zeros_like(output).to(device)
        return zeroed_output

    return hook


# -------------------------------
# 4. One-Element-at-a-Time Patching Hooks
# -------------------------------
# (A) For Attention: patch one head at a time
def patch_attention_head_hook(layer_idx, head_idx):
    """
    Replaces only one attention head's activation in a given layer.
    """

    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            attn_act = output[0]  # [batch, seq_len, hidden_dim]
            attn_4d = split_heads(attn_act, module.num_heads)  # [batch, n_heads, seq_len, head_dim]
            if "attn" in stored_activations[layer_idx]:
                stored_act = stored_activations[layer_idx]["attn"].to(device)  # [batch, seq_len, hidden_dim]
                stored_4d = split_heads(stored_act, module.num_heads)
                if stored_4d[:, head_idx].shape == attn_4d[:, head_idx].shape:
                    attn_4d[:, head_idx] = stored_4d[:, head_idx]
            new_attn = combine_heads(attn_4d)
            return (new_attn,) + output[1:]
        return output

    return hook


# (B) For MLP: patch one segment at a time
def patch_mlp_segment_hook(layer_idx, segment_idx, n_segments):
    """
    Replaces only one segment of the MLP activation in a given layer.
    Here, we assume splitting the MLP hidden state into n_segments (e.g., num_heads).
    """

    def hook(module, input, output):
        mlp_act = output  # [batch, seq_len, hidden_dim]
        mlp_4d = split_heads(mlp_act, n_segments)  # [batch, n_segments, seq_len, segment_dim]
        if "mlp" in stored_mlp_activations[layer_idx]:
            stored_act = stored_mlp_activations[layer_idx]["mlp"].to(device)
            stored_4d = split_heads(stored_act, n_segments)
            if stored_4d[:, segment_idx].shape == mlp_4d[:, segment_idx].shape:
                mlp_4d[:, segment_idx] = stored_4d[:, segment_idx]
        new_mlp = combine_heads(mlp_4d)
        return new_mlp

    return hook


# Additionally, one-element-at-a-time ablation hooks:
def ablate_attention_head_hook(layer_idx, head_idx):
    """
    Zero out only one attention head in a given layer.
    """

    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            attn_act = output[0]
            attn_4d = split_heads(attn_act, module.num_heads)
            # Zero out only the selected head
            attn_4d[:, head_idx] = torch.zeros_like(attn_4d[:, head_idx]).to(device)
            new_attn = combine_heads(attn_4d)
            return (new_attn,) + output[1:]
        return output

    return hook


def ablate_mlp_segment_hook(layer_idx, segment_idx, n_segments):
    """
    Zero out only one segment of the MLP activation in a given layer.
    """

    def hook(module, input, output):
        mlp_act = output
        mlp_4d = split_heads(mlp_act, n_segments)
        mlp_4d[:, segment_idx] = torch.zeros_like(mlp_4d[:, segment_idx]).to(device)
        new_mlp = combine_heads(mlp_4d)
        return new_mlp

    return hook


# -------------------------------
# 5. Capture "Italy" Activations
# -------------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt").to(device)

hook_handles_capture = []
for layer in layers_to_patch:
    hook_handles_capture.append(model.transformer.h[layer].attn.register_forward_hook(capture_attention_hook(layer)))
    hook_handles_capture.append(
        model.transformer.h[layer].mlp.register_forward_hook(capture_mlp_activation_hook(layer)))

with torch.no_grad():
    italy_output = model.generate(italy_inputs['input_ids'], max_new_tokens=15, use_cache=False)

italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Prompt Output:", italy_text)
results.append("Italy Prompt Output:\n" + italy_text + "\n")

# Remove capture hooks
for handle in hook_handles_capture:
    handle.remove()

# -------------------------------
# 6. Generate "France" Baseline
# -------------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    france_original_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_original_text = tokenizer.decode(france_original_output[0], skip_special_tokens=True)
print("Original France Prompt Output:", france_original_text)
results.append("Original France Prompt Output:\n" + france_original_text + "\n")

# -------------------------------
# 7. Full-Layer Patching (Both Attention + MLP)
# -------------------------------
hook_handles_patch = []
for layer in layers_to_patch:
    hook_handles_patch.append(model.transformer.h[layer].attn.register_forward_hook(patch_attention_hook(layer)))
    hook_handles_patch.append(model.transformer.h[layer].mlp.register_forward_hook(patch_mlp_activation_hook(layer)))

with torch.no_grad():
    france_fullpatched_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_fullpatched_text = tokenizer.decode(france_fullpatched_output[0], skip_special_tokens=True)
print(f"France Prompt with Full Patching (Layers {start_layer}-{end_layer}):", france_fullpatched_text)
results.append(
    f"France Prompt with Full Patching (Layers {start_layer}-{end_layer}):\n" + france_fullpatched_text + "\n")

for handle in hook_handles_patch:
    handle.remove()

# -------------------------------
# 8. Ablation (Zero-Out) of Multiple Layers (Both Attention + MLP)
# -------------------------------
hook_handles_ablate = []
for layer in layers_to_patch:
    hook_handles_ablate.append(model.transformer.h[layer].attn.register_forward_hook(ablate_attention_hook(layer)))
    hook_handles_ablate.append(model.transformer.h[layer].mlp.register_forward_hook(ablate_mlp_activation_hook(layer)))

with torch.no_grad():
    france_ablated_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, use_cache=False)

france_ablated_text = tokenizer.decode(france_ablated_output[0], skip_special_tokens=True)
print(f"France Prompt with Ablation (Layers {start_layer}-{end_layer}):", france_ablated_text)
results.append(f"France Prompt with Ablation (Layers {start_layer}-{end_layer}):\n" + france_ablated_text + "\n")

for handle in hook_handles_ablate:
    handle.remove()

# -------------------------------
# 9. One-Element-at-a-Time Patching for Both Attention and MLP
# -------------------------------
print("\n--- One-Element-at-a-Time Patching Results ---")
for layer in layers_to_patch:
    for idx in range(num_heads):
        head_hook = model.transformer.h[layer].attn.register_forward_hook(patch_attention_head_hook(layer, idx))
        mlp_hook = model.transformer.h[layer].mlp.register_forward_hook(patch_mlp_segment_hook(layer, idx, num_heads))

        with torch.no_grad():
            france_elementpatched_output = model.generate(france_inputs['input_ids'], max_new_tokens=15,
                                                          use_cache=False)
        france_elementpatched_text = tokenizer.decode(france_elementpatched_output[0], skip_special_tokens=True)
        output_str = f"Layer {layer} - Patched Element {idx}:\n" + france_elementpatched_text + "\n"
        print(output_str)
        results.append(output_str)

        head_hook.remove()
        mlp_hook.remove()

# -------------------------------
# 10. One-Element-at-a-Time Ablation for Both Attention and MLP
# -------------------------------
print("\n--- One-Element-at-a-Time Ablation Results ---")
for layer in layers_to_patch:
    for idx in range(num_heads):
        head_ablate_hook = model.transformer.h[layer].attn.register_forward_hook(ablate_attention_head_hook(layer, idx))
        mlp_ablate_hook = model.transformer.h[layer].mlp.register_forward_hook(
            ablate_mlp_segment_hook(layer, idx, num_heads))

        with torch.no_grad():
            france_elementablated_output = model.generate(france_inputs['input_ids'], max_new_tokens=15,
                                                          use_cache=False)
        france_elementablated_text = tokenizer.decode(france_elementablated_output[0], skip_special_tokens=True)
        output_str = f"Layer {layer} - Ablated Element {idx}:\n" + france_elementablated_text + "\n"
        print(output_str)
        results.append(output_str)

        head_ablate_hook.remove()
        mlp_ablate_hook.remove()

# -------------------------------
# 11. Save All Outputs to a Text File
# -------------------------------
output_filename = "../Outputs/IndividualHEADandMLPPatchingOUTPUT.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print(f"\nResults saved to {output_filename}")

