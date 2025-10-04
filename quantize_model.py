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
    print("--- Starting Model Quantization Workflow ---\n")

    # Verify that the baseline model we need to quantize actually exists.
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("Please run the training script `train_baseline.py` first.")
        return

    # Task: Load the original baseline model.
    # We don't actually need to load the Keras model object into memory for this step.
    # The TFLiteConverter works directly from the saved files on disk, which is more
    # memory-efficient and reliable.
    print(f"Targeting baseline model at: {BASELINE_MODEL_PATH}")

    # --- NEW CODE STARTS HERE ---

    # Task: Initialize the TFLiteConverter from the saved model.
    print("\n[TASK] Initializing the TFLiteConverter...")

    try:
        # The TFLiteConverter is the core engine for converting TensorFlow models
        # to the TensorFlow Lite format.
        # We use the `from_saved_model` class method, which is the most robust way
        # to load a model for conversion. It reads the model's graph, weights,
        # and signatures directly from the specified directory.
        converter = tf.lite.TFLiteConverter.from_saved_model(BASELINE_MODEL_PATH)
        
        # This converter object now holds a representation of our model's graph
        # and is ready to be configured for quantization.
        print("TFLiteConverter initialized successfully.")
        print(f"Converter object created: {converter}")

    except Exception as e:
        print(f"An error occurred while initializing the converter: {e}")
        return

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()