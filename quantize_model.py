# quantize_model.py

import os
import tensorflow as tf
import numpy as np

# --- Configuration --
# We start our quantization journey with the original, high-performance baseline model.
# This allows us to measure the effects of quantization in isolation.
BASELINE_MODEL_PATH = "models/baseline_model"

def main():
    """
    Main function to orchestrate the model quantization process.
    """
    print("---" + " Starting Model Quantization Workflow " + "---" + "\n")

    # Verify that the baseline model we need to quantize actually exists.
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("Please run the training script `train_baseline.py` first.")
        return

    # Task: Load the original baseline model.
    # This is the same high-accuracy, unoptimized model we started with in the
    # pruning step. It's our 'ground truth' for this new optimization path.
    print(f"Loading baseline model from: {BASELINE_MODEL_PATH}")
    try:
        # tf.keras.models.load_model() reconstructs the model from the SavedModel format.
        baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
        print("Baseline model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # To confirm we have the correct starting point, let's print its summary.
    # This should show the full, unpruned architecture with its float32 parameters.
    print("\n---" + " Baseline Model Summary " + "---")
    baseline_model.summary()

if __name__ == "__main__":
    main()

