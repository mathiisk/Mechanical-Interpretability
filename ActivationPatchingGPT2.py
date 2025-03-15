import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "gpt2-xl"
print(f"Loading {model_name} model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()
print("Model and tokenizer loaded successfully.\n")

# Dictionary to store per-head activations (the entire sequence) from the Italy prompt
stored_activations = {}

# Layer index (GPT2-XL has 48 layers: 0..47)
layer_idx = 47

# GPT2-XL has 25 attention heads per layer
num_heads_to_patch = 25


# ------------------------------------------------------------------------------
# 2. Helper functions to reshape and handle per-head activations
# ------------------------------------------------------------------------------
def split_heads(activation_3d, module):
    """
    Given a tensor of shape [batch, seq_len, hidden_dim],
    split it into [batch, n_heads, seq_len, head_dim].
    """
    batch_size, seq_len, hidden_dim = activation_3d.shape
    n_heads = module.num_heads  # GPT2Attention has this attribute
    head_dim = hidden_dim // n_heads

    # shape => [batch, seq_len, n_heads, head_dim]
    activation_4d = activation_3d.view(batch_size, seq_len, n_heads, head_dim)
    # reorder to [batch, n_heads, seq_len, head_dim]
    activation_4d = activation_4d.permute(0, 2, 1, 3).contiguous()
    return activation_4d


def combine_heads(activation_4d):
    """
    Given a tensor of shape [batch, n_heads, seq_len, head_dim],
    combine it back to [batch, seq_len, hidden_dim].
    """
    batch_size, n_heads, seq_len, head_dim = activation_4d.shape
    # reorder to [batch, seq_len, n_heads, head_dim]
    activation_4d = activation_4d.permute(0, 2, 1, 3).contiguous()
    # shape => [batch, seq_len, n_heads * head_dim]
    return activation_4d.view(batch_size, seq_len, n_heads * head_dim)


# ------------------------------------------------------------------------------
# 3. Forward hooks
# ------------------------------------------------------------------------------
def capture_activation_hook(module, input, output):
    """
    Captures and stores the per-head activations from GPT-2 attention output
    for the entire sequence (all tokens). We store shape [batch, seq_len, head_dim]
    for each head.
    """
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        # shape: [batch, seq_len, hidden_dim]
        activation_3d = output[0]

        # Split into [batch, n_heads, seq_len, head_dim]
        activation_4d = split_heads(activation_3d, module)
        # shape => [batch, n_heads, seq_len, head_dim]

        # Store the entire [batch, seq_len, head_dim] for each head
        batch_size, n_heads, seq_len, head_dim = activation_4d.shape
        for head_idx in range(n_heads):
            # activation_4d[:, head_idx, :, :] => shape [batch, seq_len, head_dim]
            stored_activations[head_idx] = activation_4d[:, head_idx, :, :].clone().detach().cpu()

    return output


def patch_activation_hook(head_idx):
    """
    Returns a hook that patches *all tokens* for the specified head_idx
    with the stored activations (the entire sequence).
    """
    def hook(module, input, output):
        if head_idx not in stored_activations:
            return output

        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            activation_3d = output[0]
            # Convert [batch, seq_len, hidden_dim] -> [batch, n_heads, seq_len, head_dim]
            activation_4d = split_heads(activation_3d, module)
            batch_size, n_heads, seq_len, head_dim = activation_4d.shape

            # Patch only the chosen head for all positions
            if head_idx < n_heads:
                stored_head_3d = stored_activations[head_idx].to(device)  # [batch, seq_len, head_dim]
                if stored_head_3d.shape == (batch_size, seq_len, head_dim):
                    activation_4d[:, head_idx, :, :] = stored_head_3d

            # Reshape back to [batch, seq_len, hidden_dim]
            activation_3d = combine_heads(activation_4d)
            return (activation_3d,) + output[1:]

        return output
    return hook


def patch_all_heads_hook(module, input, output):
    """
    Patches all heads [0..num_heads_to_patch-1] with the stored activations
    (the entire sequence) if available.
    """
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        activation_3d = output[0]
        # Convert to [batch, n_heads, seq_len, head_dim]
        activation_4d = split_heads(activation_3d, module)
        batch_size, n_heads, seq_len, head_dim = activation_4d.shape

        # Patch heads 0..(num_heads_to_patch-1) if we have them
        for head_idx in range(min(num_heads_to_patch, n_heads)):
            if head_idx in stored_activations:
                stored_head_3d = stored_activations[head_idx].to(device)
                if stored_head_3d.shape == (batch_size, seq_len, head_dim):
                    activation_4d[:, head_idx, :, :] = stored_head_3d

        # Reshape back
        activation_3d = combine_heads(activation_4d)
        return (activation_3d,) + output[1:]

    return output


# ------------------------------------------------------------------------------
# 4. Capture activations from the "Italy" prompt
# ------------------------------------------------------------------------------
italy_prompt = "The capital of Italy is"
italy_inputs = tokenizer(italy_prompt, return_tensors="pt")
italy_inputs = {k: v.to(device) for k, v in italy_inputs.items()}

# Register capture hook on GPT-2 block
hook_handle_capture = model.transformer.h[layer_idx].attn.register_forward_hook(
    capture_activation_hook
)

with torch.no_grad():
    italy_output = model.generate(
        italy_inputs['input_ids'],
        max_new_tokens=15,
        num_return_sequences=1,
        use_cache=False
    )
italy_text = tokenizer.decode(italy_output[0], skip_special_tokens=True)
print("Italy Prompt Output:", italy_text)

# Remove capture hook
hook_handle_capture.remove()


# ------------------------------------------------------------------------------
# 5. Generate second prompt ("France") with no patch
# ------------------------------------------------------------------------------
france_prompt = "The capital of France is"
france_inputs = tokenizer(france_prompt, return_tensors="pt")
france_inputs = {k: v.to(device) for k, v in france_inputs.items()}

with torch.no_grad():
    france_original_output = model.generate(
        france_inputs['input_ids'],
        max_new_tokens=15,
        num_return_sequences=1,
        use_cache=False
    )
france_original_text = tokenizer.decode(france_original_output[0], skip_special_tokens=True)
print("Original France Prompt Output:", france_original_text)

results = []
results.append("Original France Prompt Output:\n" + france_original_text + "\n")


# ------------------------------------------------------------------------------
# 6. Patch one head at a time (entire sequence)
# ------------------------------------------------------------------------------
for head in range(num_heads_to_patch):
    hook_handle_patch = model.transformer.h[layer_idx].attn.register_forward_hook(
        patch_activation_hook(head)
    )
    with torch.no_grad():
        france_patched_output = model.generate(
            france_inputs['input_ids'],
            max_new_tokens=15,
            num_return_sequences=1,
            use_cache=False
        )
    france_patched_text = tokenizer.decode(france_patched_output[0], skip_special_tokens=True)
    results.append(f"France Prompt with head {head} patched:\n{france_patched_text}\n")
    hook_handle_patch.remove()


# ------------------------------------------------------------------------------
# 7. Patch all heads at once (entire sequence)
# ------------------------------------------------------------------------------
hook_handle_all = model.transformer.h[layer_idx].attn.register_forward_hook(
    patch_all_heads_hook
)
with torch.no_grad():
    france_all_heads_output = model.generate(
        france_inputs['input_ids'],
        max_new_tokens=15,
        num_return_sequences=1,
        use_cache=False
    )
france_all_heads_text = tokenizer.decode(france_all_heads_output[0], skip_special_tokens=True)
results.append("France Prompt with all heads patched:\n" + france_all_heads_text + "\n")
hook_handle_all.remove()


# ------------------------------------------------------------------------------
# 8. Save all results
# ------------------------------------------------------------------------------
output_filename = "output.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\nResults saved to {output_filename}")
