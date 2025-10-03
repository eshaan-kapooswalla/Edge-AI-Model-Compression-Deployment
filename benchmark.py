import os
import time
import numpy as np
import tensorflow as tf


# --- Configuration ---
MODEL_PATH = "models/baseline_model"

def get_dir_size_mb(path: str = ".") -> float:
    """
    Calculate the total size of a directory in megabytes.

    Args:
        path: Path to the directory to measure.

    Returns:
        Total size in megabytes.
    """
    total_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total_size_bytes += os.path.getsize(file_path)
    return total_size_bytes / (1024 * 1024)


def main():
    """
    Orchestrate the baseline model benchmarking process.
    Subsequent tasks will fill in each measurement section.
    """
    print("--- Starting Comprehensive Model Benchmark ---")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have run the training script to create the baseline model.")
        return

    print(f"Benchmarking model at: {MODEL_PATH}")

    # Task: Load the saved baseline model from the file.
    print("\n[TASK] Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    print("\n--- Loaded Model Summary ---")
    model.summary()

    # Task: Measure and record the model's file size.
    print("\n[TASK] Measuring model size...")
    model_size_mb = get_dir_size_mb(MODEL_PATH)
    print(f"Model size: {model_size_mb:.2f} MB")

    # Task: Measure inference time (latency).
    print("\n[TASK] Measuring inference latency...")
    # (Implementation will be added in the next step)

    # Task: Measure the peak RAM usage.
    print("\n[TASK] Measuring memory usage...")
    # (Implementation will be added in the next step)

    # Task: Evaluate the model on the entire test dataset for accuracy.
    print("\n[TASK] Evaluating model accuracy...")
    # (Implementation will be added in the next step)

    # Task: Log all baseline metrics in a structured format.
    print("\n[TASK] Logging all baseline metrics...")
    # (Implementation will be added in the next step)

    print("\n--- Benchmark Complete ---")


if __name__ == "__main__":
    main()


