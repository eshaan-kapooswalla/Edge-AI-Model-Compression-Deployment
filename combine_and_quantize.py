import os
import tensorflow as tf


# --- Configuration ---
PRUNED_MODEL_PATH = "models/pruned_model"


def main():
    """
    Orchestrate the combined pruning + quantization workflow.
    This step initializes a TFLiteConverter from the fine‑tuned pruned model.
    """
    print("--- Starting Combined Pruning & Quantization Workflow ---\n")

    if not os.path.exists(PRUNED_MODEL_PATH):
        print(f"Error: Pruned model not found at {PRUNED_MODEL_PATH}")
        print("Please run the pruning workflow first to generate this model.")
        return

    print(f"Targeting pruned model for conversion at: {PRUNED_MODEL_PATH}")

    # Initialize TFLiteConverter directly from SavedModel on disk
    print("\n[TASK] Initializing the TFLiteConverter...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(PRUNED_MODEL_PATH)
        print("TFLiteConverter initialized successfully from the pruned model.")
        print(f"Converter object created: {converter}")
    except Exception as e:
        print(f"An error occurred while initializing the converter: {e}")
        return


if __name__ == "__main__":
    main()


