import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Model and Tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Move model to GPU
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
        head_idx = 4  # for example, choose the 4th attention head
        if activation.size(1) > head_idx:
            stored_activation = activation[:, head_idx, :].clone().detach()
            print("Captured activation for head", head_idx)
    return output  # return unmodified output

def patch_activation_hook(module, input, output):
    """
    Hook for patching the activation of a specific attention head using the stored activation.
    """
    global stored_activation
    if stored_activation is None:
        print("No stored activation; skipping patch.")
        return output

    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        activation = output[0]
        head_idx = 4
        if activation.size(1) > head_idx:
            if activation.size(0) == stored_activation.size(0) and activation.size(2) == stored_activation.size(1):
                activation[:, head_idx, :] = stored_activation.to(device)  # Ensure stored activation is on CUDA
                print("Patched activation for head", head_idx, "with stored Italy activation.")
            else:
                print("Shape mismatch: Cannot patch activation.")
        return (activation,) + output[1:]
    return output

# ---------------------------
# 1. Capture activation from the Italy prompt
# ---------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt").to(device)  # Move input to GPU

layer_idx = 22  # Choose the layer to capture activations from

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
france_inputs = tokenizer(france_prompt, return_tensors="pt").to(device)  # Move input to GPU

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
