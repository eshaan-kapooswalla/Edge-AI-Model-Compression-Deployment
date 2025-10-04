# quantize_model.py

import os
import tensorflow as tf
import numpy as np

# --- Configuration --
BASELINE_MODEL_PATH = "models/baseline_model"
TFLITE_DYNAMIC_QUANT_MODEL_PATH = "models/quantized_dynamic_range.tflite"

# --- Helper Functions (representative_dataset_generator, etc.) ---
def load_training_data():
    """
    Loads the CIFAR-10 training dataset.
    We only need the images for our representative dataset.
    """
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    return x_train

def representative_dataset_generator(training_images, num_samples=200):
    """
    Creates a generator that yields a small number of samples from the
    training data. This is used for calibrating the full integer quantization.
    """
    print(f"Creating a representative dataset generator with {num_samples} samples.")
    for i in range(num_samples):
        sample = training_images[i].astype('float32')
        sample = np.expand_dims(sample, axis=0)
        yield [sample]

# --- Main Workflow ---
def main():
    """
    Main function to orchestrate the model quantization process.
    """
    print("--- Starting Model Quantization Workflow ---\n")

    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        return

    # --- Section 1: Dynamic Range Quantization (Completed) ---
    print("--- Section 1: Dynamic Range Quantization ---")
    # ... (code from previous steps remains the same) ...
    print(f"Dynamically quantized model saved to: {TFLITE_DYNAMIC_QUANT_MODEL_PATH}\n")
    
    # --- Section 2: Full Integer Quantization ---
    print("--- Section 2: Full Integer Quantization ---")

    # Load the training data for the generator.
    train_images = load_training_data()
    
    # Create an instance of our generator.
    representative_dataset = lambda: representative_dataset_generator(train_images, 200)
    print("[TASK] Representative dataset generator for calibration is ready.")
    
    # --- NEW CODE STARTS HERE --

    # Task: Re-initialize and enable full-integer quantization.
    print("\n[TASK] Configuring the converter for FULL INTEGER quantization...")

    # 1. Re-initialize a new converter from the baseline model.
    # It's good practice to use a separate converter object for each quantization strategy.
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(BASELINE_MODEL_PATH)

    # 2. Enable the default set of optimizations.
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

    # 3. Provide the representative dataset. This is the key that enables calibration.
    # The converter will use this to determine the dynamic range of activations.
    converter_int8.representative_dataset = representative_dataset
    
    # 4. Enforce integer-only operations.
    # This crucial step ensures that the converted model uses only integer operations,
    # which is required for maximum performance on many edge devices and accelerators.
    # If an operation cannot be quantized to int8, the conversion will fail.
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    print("Converter configured successfully for full integer quantization.")
    print("It will now use the representative dataset for calibration.")

    # --- NEW CODE ENDS HERE --

if __name__ == "__main__":
    main()