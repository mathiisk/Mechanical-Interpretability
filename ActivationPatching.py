import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Model and Tokenizer
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "gpt2-xl"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
print("Model and tokenizer loaded successfully.\n")

# Global dictionary to store the captured activations from the Italy prompt for each head.
stored_activations = {}

def capture_activation_hook(module, input, output):
    """
    Captures activations from all attention heads.
    Expected activation shape: [batch, num_heads, hidden_dim].
    """
    global stored_activations
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        activation = output[0]
        num_heads = activation.size(1)
        # Store activations for all heads
        for head_idx in range(num_heads):
            stored_activations[head_idx] = activation[:, head_idx, :].clone().detach()
    return output

def patch_activation_hook(head_idx):
    """
    Returns a hook that patches the activation for the specified head using stored activation.
    Only patches if the head exists.
    """
    def hook(module, input, output):
        global stored_activations
        if head_idx not in stored_activations:
            return output
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            activation = output[0]
            if activation.size(1) > head_idx:
                if activation.size(0) == stored_activations[head_idx].size(0) and activation.size(2) == stored_activations[head_idx].size(1):
                    activation[:, head_idx, :] = stored_activations[head_idx].to(device)
            return (activation,) + output[1:]
        return output
    return hook

def patch_all_heads_hook(module, input, output):
    """
    Patches all attention heads (0 to 11) simultaneously using stored activations.
    """
    global stored_activations
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        activation = output[0]
        num_heads = activation.size(1)
        for head_idx in range(min(12, num_heads)):
            if head_idx in stored_activations:
                if activation.size(0) == stored_activations[head_idx].size(0) and activation.size(2) == stored_activations[head_idx].size(1):
                    activation[:, head_idx, :] = stored_activations[head_idx].to(device)
        return (activation,) + output[1:]
    return output

# ---------------------------
# 1. Capture activations from the Italy prompt
# ---------------------------
italy_prompt = "Steve Jobs is know for"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt", add_special_tokens=True)
italy_inputs = {k: v.to(device) for k, v in italy_inputs.items()}

layer_idx = 22  # The chosen layer for activation capture

# Register the capture hook
hook_handle_capture = model.model.layers[layer_idx].self_attn.register_forward_hook(capture_activation_hook)

with torch.no_grad():
    italy_output = model.generate(
        italy_inputs['input_ids'],
        attention_mask=italy_inputs.get('attention_mask'),
        max_new_tokens=15,
        num_return_sequences=1,
        use_cache=False
    )
italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Output:", italy_text)

hook_handle_capture.remove()

# ---------------------------
# 2. Generate France prompt output without patching
# ---------------------------
france_prompt = "Microsoft office was created by"
france_inputs = tokenizer(france_prompt, return_tensors="pt", add_special_tokens=True)
france_inputs = {k: v.to(device) for k, v in france_inputs.items()}

with torch.no_grad():
    france_original_output = model.generate(
        france_inputs['input_ids'],
        attention_mask=france_inputs.get('attention_mask'),
        max_new_tokens=15,
        num_return_sequences=1,
        use_cache=False
    )
france_original_text = tokenizer.decode(france_original_output[0], skip_special_tokens=True)
print("Original France Output:", france_original_text)

results = []
results.append("Original France Output:\n" + france_original_text + "\n")

# ---------------------------
# 3. Patch one head at a time and generate France prompt output
# ---------------------------
num_heads = 12
for head in range(num_heads):
    hook_handle_patch = model.model.layers[layer_idx].self_attn.register_forward_hook(patch_activation_hook(head))
    with torch.no_grad():
        france_patched_output = model.generate(
            france_inputs['input_ids'],
            attention_mask=france_inputs.get('attention_mask'),
            max_new_tokens=15,
            num_return_sequences=1,
            use_cache=False
        )
    france_patched_text = tokenizer.decode(france_patched_output[0], skip_special_tokens=True)
    results.append(f"France Output with head {head} patched:\n" + france_patched_text + "\n")
    hook_handle_patch.remove()

# ---------------------------
# 4. Patch all 12 heads at once and generate France prompt output
# ---------------------------
hook_handle_all = model.model.layers[layer_idx].self_attn.register_forward_hook(patch_all_heads_hook)
with torch.no_grad():
    france_all_heads_output = model.generate(
        france_inputs['input_ids'],
        attention_mask=france_inputs.get('attention_mask'),
        max_new_tokens=15,
        num_return_sequences=1,
        use_cache=False
    )
france_all_heads_text = tokenizer.decode(france_all_heads_output[0], skip_special_tokens=True)
results.append("France Output with all 12 heads patched:\n" + france_all_heads_text + "\n")
hook_handle_all.remove()

# ---------------------------
# 5. Save all results to a text file
# ---------------------------
output_filename = "output.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print(f"\nResults saved to {output_filename}")
