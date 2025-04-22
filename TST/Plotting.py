import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load the saved results.
    data = np.load("patching_results.npz")
    corrupted_logits = data["corrupted_logits"]
    clean_logits = data["clean_logits"]
    patched_logits = data["patched_logits"]
    corrupted_probs = data["corrupted_probs"]
    clean_probs = data["clean_probs"]
    patched_probs = data["patched_probs"]
    num_classes = int(data["num_classes"])
    classes = list(range(num_classes))

    width = 0.25
    x = np.arange(num_classes)

    # Plot logits.
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(x - width, clean_logits, width=width, label='Clean Instance', alpha=0.7)
    plt.bar(x, corrupted_logits, width=width, label='Corrupted Instance', alpha=0.7)
    plt.bar(x + width, patched_logits, width=width, label='Patched Instance', alpha=0.7)
    plt.xlabel("Classes")
    plt.ylabel("Logit value")
    plt.title("Logits Comparison")
    plt.xticks(x, classes)
    plt.legend()

    # Plot probabilities.
    plt.subplot(1, 2, 2)
    plt.bar(x - width, clean_probs, width=width, label='Clean Instance', alpha=0.7)
    plt.bar(x, corrupted_probs, width=width, label='Corrupted Instance', alpha=0.7)
    plt.bar(x + width, patched_probs, width=width, label='Patched Instance', alpha=0.7)
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("Softmax Probabilities Comparison")
    plt.xticks(x, classes)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
