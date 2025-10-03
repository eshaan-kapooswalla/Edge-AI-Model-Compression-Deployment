import os
import time
import numpy as np
import tensorflow as tf


# --- Configuration ---
MODEL_PATH = "models/baseline_model"
BATCH_SIZE = 64

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


def load_test_data():
    """
    Load the CIFAR-10 test dataset only.

    Returns:
        Tuple of (x_test, y_test)
    """
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test, y_test


def measure_batch_inference_time(model: tf.keras.Model, batch_data: np.ndarray) -> float:
    """
    Measure inference latency for a single batch.

    Args:
        model: Loaded Keras model to benchmark
        batch_data: Batch of input images shaped [batch, 32, 32, 3]

    Returns:
        Latency in milliseconds
    """
    start_time = time.perf_counter()
    _ = model.predict(batch_data, verbose=0)
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000.0


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
    test_images, _ = load_test_data()
    print(f"Loaded {len(test_images)} test images.")
    sample_batch = test_images[:BATCH_SIZE]
    print("Performing a warm-up inference run...")
    _ = model.predict(sample_batch, verbose=0)
    batch_latency_ms = measure_batch_inference_time(model, sample_batch)
    print(f"Inference time for one batch ({BATCH_SIZE} images): {batch_latency_ms:.2f} ms")

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


