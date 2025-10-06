import os
import tensorflow as tf
import numpy as np


# --- Configuration ---
PRUNED_MODEL_PATH = "models/pruned_model"


# --- Helper functions: representative dataset (for full integer quantization) ---
def load_training_data():
    """
    Load CIFAR-10 training images only (labels not required for calibration).
    """
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    return x_train


def representative_dataset_generator(training_images: np.ndarray, num_samples: int = 200):
    """
    Yield a small number of samples for converter calibration.

    Each yielded item is a list containing a single float32 batch [1, 32, 32, 3].
    """
    print(f"Creating a representative dataset generator with {num_samples} samples.")
    for i in range(num_samples):
        sample = training_images[i].astype("float32")
        sample = np.expand_dims(sample, axis=0)
        yield [sample]


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

    # Prepare representative dataset for full integer quantization calibration
    print("\n[TASK] Creating the representative dataset for calibration...")
    train_images = load_training_data()
    representative_dataset = lambda: representative_dataset_generator(train_images, 200)
    print("Representative dataset generator created successfully.")
    print("This generator is now ready to be used for full integer quantization calibration.")

    # Configure converter for Full Integer Quantization (weights + activations)
    print("\n[TASK] Configuring the converter for FULL INTEGER quantization...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    print("Converter configured successfully for full integer quantization on the pruned model.")


if __name__ == "__main__":
    main()


