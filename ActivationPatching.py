# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load Model and Tokenizer
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# print("Loading model and tokenizer...")
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.eval()
# print("Model and tokenizer loaded successfully.\n")
#
# # Define Input and Generate Original Output
# input_text = "The capital of France is"
# inputs = tokenizer(input_text, return_tensors="pt")
#
# print("Generating original text...")
# with torch.no_grad():
#     original_output = model.generate(inputs['input_ids'], max_new_tokens=5, num_return_sequences=1)
# original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
# print(f"Original Output: {original_text}\n")
#
#
# def patch_activation_hook(module, input, output):
#     """
#     Modifies activations for a specific attention head.
#     Assumes output is a tuple where output[0] is the activation tensor.
#     """
#     if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
#         activation = output[0]  # The actual activations
#         batch_size, num_heads, hidden_dim = activation.shape  # Extract dimensions
#
#         head_idx = 0  # Modify the first attention head
#         if num_heads > head_idx:  # Ensure the head exists
#             activation[:, head_idx, :] = torch.full_like(activation[:, head_idx, :], 0.5)
#             print(f"Activation patch applied to head {head_idx}.")
#
#         return (activation,) + output[1:]  # Return modified tuple
#
#     return output  # If unexpected, return unchanged
#
# # Register Hook on the Correct Attention Module
# layer_idx = 0  # Modify first layer for now
# hook_handle = None
# try:
#     attn_module = model.model.layers[layer_idx].self_attn  # Corrected path
#     hook_handle = attn_module.register_forward_hook(patch_activation_hook)
#     print(f"Activation patch hook registered on model.model.layers[{layer_idx}].self_attn")
# except AttributeError:
#     print("Couldn't find the attention module. Check model structure.")
#
# # Generate Patched Output
# print("\nGenerating patched text with activation patching...")
# with torch.no_grad():
#     patched_output = model.generate(inputs['input_ids'], max_new_tokens=5, num_return_sequences=1)
# patched_text = tokenizer.decode(patched_output[0], skip_special_tokens=True)
# print(f"Patched Output: {patched_text}\n")
#
# # Remove Hook
# if hook_handle is not None:
#     hook_handle.remove()
#     print("Activation hook removed.\n")
#
#
#



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model and Tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
print("Model and tokenizer loaded successfully.\n")

# Global variable to store the captured activation from the Italy prompt
stored_activation = None

def capture_activation_hook(module, input, output):
    """
    Hook for capturing the activation from a specific attention head.
    This hook will be registered during the Italy prompt generation.
    """
    global stored_activation
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        activation = output[0]  # activation tensor: shape [batch, num_heads, hidden_dim]
        head_idx = 4  # for example, choose the first attention head
        if activation.size(1) > head_idx:
            # Capture the activation for this head (make a clone to store it)
            stored_activation = activation[:, head_idx, :].clone()
            print("Captured activation for head", head_idx)
    return output  # return unmodified output

def patch_activation_hook(module, input, output):
    """
    Hook for patching the activation of a specific attention head using the stored activation.
    This hook will be registered during the France prompt generation.
    """
    global stored_activation
    if stored_activation is None:
        print("No stored activation; skipping patch.")
        return output

    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        activation = output[0]
        head_idx = 4  # we patch the same head as captured
        if activation.size(1) > head_idx:
            # Check that the shapes match for the batch and hidden dimension
            if activation.size(0) == stored_activation.size(0) and activation.size(2) == stored_activation.size(1):
                # Replace the activation for the target head with the stored activation
                activation[:, head_idx, :] = stored_activation
                print("Patched activation for head", head_idx, "with stored Italy activation.")
            else:
                print("Shape mismatch: Cannot patch activation.")
        return (activation,) + output[1:]
    return output

# ---------------------------
# 1. Capture activation from the Italy prompt
# ---------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt")

layer_idx = 22  # choose layer 0 for demonstration

# Register the hook to capture the activation
hook_handle_capture = model.model.layers[layer_idx].self_attn.register_forward_hook(capture_activation_hook)
print("Capture hook registered for Italy prompt.")

with torch.no_grad():
    italy_output = model.generate(italy_inputs['input_ids'], max_new_tokens=15, num_return_sequences=1)
italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Output:", italy_text)

# Remove the capture hook after generation
hook_handle_capture.remove()
print("Capture hook removed.\n")

# ---------------------------
# 2. Patch activation into the France prompt
# ---------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt")

# Register the patch hook using the stored activation from Italy generation
hook_handle_patch = model.model.layers[layer_idx].self_attn.register_forward_hook(patch_activation_hook)
print("Patch hook registered for France prompt.")

with torch.no_grad():
    france_output = model.generate(france_inputs['input_ids'], max_new_tokens=15, num_return_sequences=1)
france_text = tokenizer.decode(france_output[0], skip_special_tokens=True)
print("France Output:", france_text)

# Remove the patch hook after generation
hook_handle_patch.remove()
print("Patch hook removed.\n")
