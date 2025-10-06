# quantize_model.py

import os
import tensorflow as tf
import numpy as np

# --- Configuration --
BASELINE_MODEL_PATH = "models/baseline_model"
TFLITE_DYNAMIC_QUANT_MODEL_PATH = "models/quantized_dynamic_range.tflite"

# --- NEW HELPER FUNCTION STARTS HERE --

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

    Args:
        training_images: A NumPy array of training images.
        num_samples (int): The number of samples to yield.

    Yields:
        A list containing a single, preprocessed image sample.
    """
    print(f"Creating a representative dataset generator with {num_samples} samples.")
    for i in range(num_samples):
        # Get a single image and convert it to the required float32 format.
        # The TFLite converter expects a list of inputs. Since our model has
        # one input, we yield a list containing just one element.
        sample = training_images[i].astype('float32')
        
        # Add a batch dimension to the image to match the model's input signature.
        # The shape changes from (32, 32, 3) to (1, 32, 32, 3).
        sample = np.expand_dims(sample, axis=0)
        
        yield [sample]

# --- NEW HELPER FUNCTION ENDS HERE --

def main():
    """
    Main function to orchestrate the model quantization process.
    """
    print("--- Starting Model Quantization Workflow ---\n")

    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        return

    # --- DYNAMIC RANGE QUANTIZATION (Existing Code) ---
    print("--- Section 1: Dynamic Range Quantization ---")
    print(f"Targeting baseline model at: {BASELINE_MODEL_PATH}")
    converter_dynamic = tf.lite.TFLiteConverter.from_saved_model(BASELINE_MODEL_PATH)
    converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant_dynamic = converter_dynamic.convert()
    os.makedirs(os.path.dirname(TFLITE_DYNAMIC_QUANT_MODEL_PATH), exist_ok=True)
    with open(TFLITE_DYNAMIC_QUANT_MODEL_PATH, 'wb') as f:
        f.write(tflite_model_quant_dynamic)
    print(f"Dynamically quantized model saved to: {TFLITE_DYNAMIC_QUANT_MODEL_PATH}\n")
    
    # --- NEW CODE STARTS HERE --

    # --- FULL INTEGER QUANTIZATION (Preparation) ---
    print("--- Section 2: Full Integer Quantization ---")

    # Task: Create a representative dataset generator function.
    print("[TASK] Creating the representative dataset for calibration...")

    # Load the training data that our generator will use.
    train_images = load_training_data()
    
    # Create an instance of our generator. The TFLite converter will call this
    # repeatedly to get calibration data.
    # Note: We are just creating the generator object here. It won't actually
    # run and yield data until the converter requests it in the next step.
    representative_dataset = lambda: representative_dataset_generator(train_images, 200)

    print("Representative dataset generator created successfully.")
    print("This generator is now ready to be used for full integer quantization calibration.")
    
    # --- NEW CODE ENDS HERE --

if __name__ == "__main__":
    main()
