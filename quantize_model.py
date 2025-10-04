# quantize_model.py

import os
import tensorflow as tf
import numpy as np

# --- Configuration --
# We start our quantization journey with the original, high-performance baseline model.
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

    # Initialize the TFLiteConverter
    print(f"Targeting baseline model at: {BASELINE_MODEL_PATH}")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(BASELINE_MODEL_PATH)
        print("TFLiteConverter initialized successfully.")
    except Exception as e:
        print(f"An error occurred while initializing the converter: {e}")
        return

    # --- NEW CODE STARTS HERE ---

    # Task: Apply post-training dynamic range quantization.
    print("\n" + "[TASK]" + " Applying post-training dynamic range quantization..." + "\n")

    # To enable dynamic range quantization, we set the `optimizations` attribute
    # of the converter.
    # tf.lite.Optimize.DEFAULT is the recommended setting. It enables the default
    # set of post-training optimizations, which at its core is dynamic range
    # quantization (converting float32 weights to int8).
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Dynamic range quantization has been enabled on the converter.")
    print("The converter is now configured to produce a model with int8 weights.")

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()
