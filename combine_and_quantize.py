import os
import tensorflow as tf


# --- Configuration ---
PRUNED_MODEL_PATH = "models/pruned_model"


def main():
    """
    Orchestrate the combined pruning + quantization workflow.
    This step loads the fine‑tuned pruned model as the starting point.
    """
    print("--- Starting Combined Pruning & Quantization Workflow ---\n")

    print(f"[TASK] Loading the fine-tuned pruned model from: {PRUNED_MODEL_PATH}")
    if not os.path.exists(PRUNED_MODEL_PATH):
        print(f"Error: Pruned model not found at {PRUNED_MODEL_PATH}")
        print("Please run the pruning workflow first to generate this model.")
        return

    try:
        pruned_model = tf.keras.models.load_model(PRUNED_MODEL_PATH)
        print("Pruned model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the pruned model: {e}")
        return

    # Verification: print summary to confirm architecture and parameter count
    print("\n--- Pruned Model Summary ---")
    pruned_model.summary()
    print("\nVerify that 'Total params' is substantially below the baseline (~23.6M).")


if __name__ == "__main__":
    main()


