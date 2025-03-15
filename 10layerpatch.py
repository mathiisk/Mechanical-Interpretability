import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# 1. Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "gpt2-xl"
print(f"Loading {model_name} model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()
print("Model and tokenizer loaded successfully.\n")

# We'll patch the last 10 layers: 38..47
LAST_N_LAYERS = 10
start_layer = 48 - LAST_N_LAYERS  # 48 is total layers for GPT2-XL, so 38..47

# Dictionary to store activations for each layer & head:
# stored_activations[layer_idx][head_idx] = [batch, seq_len, head_dim]
stored_activations = {}

# -------------------------------
# 2. Splitting / Combining Heads
# -------------------------------
def split_heads(activation_3d, module):
    """
    Reshape [batch, seq_len, hidden_dim] -> [batch, n_heads, seq_len, head_dim].
    """
    batch_size, seq_len, hidden_dim = activation_3d.shape
    n_heads = module.num_heads
    head_dim = hidden_dim // n_heads

    # shape => [batch, seq_len, n_heads, head_dim]
    activation_4d = activation_3d.view(batch_size, seq_len, n_heads, head_dim)
    # reorder => [batch, n_heads, seq_len, head_dim]
    activation_4d = activation_4d.permute(0, 2, 1, 3).contiguous()
    return activation_4d

def combine_heads(activation_4d):
    """
    Reshape [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim].
    """
    batch_size, n_heads, seq_len, head_dim = activation_4d.shape
    # reorder => [batch, seq_len, n_heads, head_dim]
    activation_4d = activation_4d.permute(0, 2, 1, 3).contiguous()
    # final => [batch, seq_len, n_heads*head_dim]
    return activation_4d.view(batch_size, seq_len, n_heads * head_dim)

# -------------------------------
# 3. Capture Hook
# -------------------------------
def make_capture_hook(layer_idx):
    """
    Creates a hook to store the entire sequence of attention outputs
    for all heads at the given layer.
    """
    def capture_hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            # shape: [batch, seq_len, hidden_dim]
            activation_3d = output[0]

            # Split => [batch, n_heads, seq_len, head_dim]
            activation_4d = split_heads(activation_3d, module)
            batch_size, n_heads, seq_len, head_dim = activation_4d.shape

            # Initialize storage for this layer if needed
            if layer_idx not in stored_activations:
                stored_activations[layer_idx] = {}

            # Store each head’s entire sequence
            for head_idx in range(n_heads):
                # shape => [batch, seq_len, head_dim]
                head_act = activation_4d[:, head_idx, :, :].detach().cpu()
                stored_activations[layer_idx][head_idx] = head_act
        return output
    return capture_hook

# -------------------------------
# 4. Patch Hook
# -------------------------------
def make_patch_hook(layer_idx):
    """
    Creates a hook that replaces the entire sequence of attention outputs
    for all heads at layer_idx with the stored “Italy” version.
    """
    def patch_hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            activation_3d = output[0]
            activation_4d = split_heads(activation_3d, module)
            batch_size, n_heads, seq_len, head_dim = activation_4d.shape

            if layer_idx in stored_activations:
                # For each head in the layer, patch if we have it
                for head_idx in range(n_heads):
                    if head_idx in stored_activations[layer_idx]:
                        stored_head = stored_activations[layer_idx][head_idx].to(device)
                        # Check shape => [batch, seq_len, head_dim]
                        if stored_head.shape == (batch_size, seq_len, head_dim):
                            activation_4d[:, head_idx, :, :] = stored_head

            # Combine back
            patched_3d = combine_heads(activation_4d)
            return (patched_3d,) + output[1:]
        return output
    return patch_hook

# -------------------------------
# 5. Capture Step with "Italy" prompt
# -------------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt")
italy_inputs = {k: v.to(device) for k, v in italy_inputs.items()}

capture_handles = []
for l in range(start_layer, 48):
    h = model.transformer.h[l].attn.register_forward_hook(make_capture_hook(l))
    capture_handles.append(h)

with torch.no_grad():
    italy_output = model.generate(
        italy_inputs['input_ids'],
        max_new_tokens=15,
        use_cache=False
    )
italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Prompt Output:", italy_text)

# Remove capture hooks
for handle in capture_handles:
    handle.remove()

# -------------------------------
# 6. Original "France" prompt
# -------------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt")
france_inputs = {k: v.to(device) for k, v in france_inputs.items()}

with torch.no_grad():
    france_orig = model.generate(
        france_inputs['input_ids'],
        max_new_tokens=15,
        use_cache=False
    )
france_text_orig = tokenizer.decode(france_orig[0], skip_special_tokens=True)
print("Original France Prompt Output:", france_text_orig)

# -------------------------------
# 7. Patch the last 10 layers (all heads) simultaneously
# -------------------------------
patch_handles = []
for l in range(start_layer, 48):
    h = model.transformer.h[l].attn.register_forward_hook(make_patch_hook(l))
    patch_handles.append(h)

with torch.no_grad():
    france_patched = model.generate(
        france_inputs['input_ids'],
        max_new_tokens=15,
        use_cache=False
    )
france_text_patched = tokenizer.decode(france_patched[0], skip_special_tokens=True)

# Remove patch hooks
for handle in patch_handles:
    handle.remove()

print("\nFrance Prompt with last 10 layers patched:\n", france_text_patched)

# -------------------------------
# 8. (Optional) Save results
# -------------------------------
output_str = (
    f"Italy Prompt Output:\n{italy_text}\n\n"
    f"Original France Prompt Output:\n{france_text_orig}\n\n"
    f"France Prompt with last 10 layers patched:\n{france_text_patched}\n"
)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(output_str)

print("\nResults saved to output.txt")
