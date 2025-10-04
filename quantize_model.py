# quantize_model.py

import os
import tensorflow as tf
import numpy as np

# --- Configuration --
BASELINE_MODEL_PATH = "models/baseline_model"
# --- NEW CONFIGURATION ---
# We define a clear path for our new TFLite model. Using the .tflite extension
# is a standard convention.
TFLITE_DYNAMIC_QUANT_MODEL_PATH = "models/quantized_dynamic_range.tflite"
# --- END NEW CONFIGURATION ---

def main():
    """
    Main function to orchestrate the model quantization process.
    """
    print("--- Starting Model Quantization Workflow ---
")

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

    # Apply post-training dynamic range quantization
    print("\n[TASK] Applying post-training dynamic range quantization...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    print("Dynamic range quantization has been enabled on the converter.")

    # --- NEW CODE STARTS HERE ---

    # Task: Convert the model and save it as a .tflite file.
    print(f"\n[TASK] Converting and saving the dynamically quantized model...")

    # The converter.convert() method executes the conversion based on the
    # configurations we've set. It returns the quantized model as a byte string.
    try:
        tflite_model_quant_dynamic = converter.convert()
        print("Model converted successfully.")
    except Exception as e:
        print(f"An error occurred during model conversion: {e}")
        return

    # Ensure the 'models' directory exists before trying to save the file there.
    # This is a robust way to prevent FileNotFoundError.
    os.makedirs(os.path.dirname(TFLITE_DYNAMIC_QUANT_MODEL_PATH), exist_ok=True)
    
    # Now, we save the converted model to a file.
    # We use the 'wb' mode, which stands for 'write binary'. This is essential
    # for saving the raw byte data of the TFLite model correctly.
    with open(TFLITE_DYNAMIC_QUANT_MODEL_PATH, 'wb') as f:
        f.write(tflite_model_quant_dynamic)

    print(f"Dynamically quantized model saved to: {TFLITE_DYNAMIC_QUANT_MODEL_PATH}")

    # --- A Quick Sanity Check: Compare File Sizes ---
    baseline_size_mb = os.path.getsize(BASELINE_MODEL_PATH)
    quantized_size_mb = os.path.getsize(TFLITE_DYNAMIC_QUANT_MODEL_PATH) / (1024 * 1024)
    
    print(f"\n--- Size Comparison ---")
    print(f"Original Keras SavedModel size: This is a directory, run benchmark.py for accurate size (~90 MB)")
    print(f"Quantized TFLite model size: {quantized_size_mb:.2f} MB")
    print(f"You should see a size reduction of roughly 4x!")
    
    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()