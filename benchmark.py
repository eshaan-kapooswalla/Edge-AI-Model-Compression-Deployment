import os
import time
import numpy as np
import tensorflow as tf
from memory_profiler import memory_usage


# --- Configuration ---
MODEL_PATH = "models/baseline_model"
BATCH_SIZE = 64
NUM_LATENCY_TESTS = 200
RESULTS_FILE = "benchmark_results.md"

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


def measure_average_latency(model: tf.keras.Model, data: np.ndarray, num_samples: int = 200) -> tuple[float, float]:
    """
    Measure average and standard deviation of single-image inference latency.

    Args:
        model: Loaded Keras model
        data: Dataset to sample single images from
        num_samples: Number of single-image inferences to run

    Returns:
        (avg_latency_ms, std_latency_ms)
    """
    latencies_ms: list[float] = []

    # Warm-up with a single-image prediction
    print("Performing a warm-up inference run...")
    _ = model.predict(data[0:1], verbose=0)

    print(f"Running {num_samples} individual inferences for latency test...")
    for i in range(num_samples):
        sample = data[i:i + 1]
        start_time = time.perf_counter()
        _ = model.predict(sample, verbose=0)
        end_time = time.perf_counter()
        latencies_ms.append((end_time - start_time) * 1000.0)

    return float(np.mean(latencies_ms)), float(np.std(latencies_ms))

def measure_peak_memory_usage(model: tf.keras.Model, sample_data: np.ndarray) -> float:
    """
    Measure peak RAM (MiB) during a single inference using memory-profiler.

    Args:
        model: Loaded Keras model
        sample_data: Single-sample batch shaped [1, 32, 32, 3]

    Returns:
        Peak memory usage in MiB (float)
    """
    peak_mib = memory_usage((model.predict, (sample_data,), {"verbose": 0}), interval=0.1, max_usage=True)
    return float(peak_mib)

def log_metrics_to_markdown(file_path: str, model_name: str, metrics: dict) -> None:
    """
    Log metrics to a markdown file, creating a header if the file doesn't exist.

    Args:
        file_path: Path to the markdown file
        model_name: Display name of the model being benchmarked
        metrics: Dict with keys: size_mb, latency_ms, peak_ram_mib, accuracy_pct
    """
    header = "| Model | Size (MB) | Latency (ms) | Peak RAM (MiB) | Accuracy (%) |\n"
    separator = "|:---|---:|---:|---:|---:|\n"
    row = (
        f"| {model_name} | {metrics['size_mb']:.2f} | {metrics['latency_ms']:.2f} | "
        f"{metrics['peak_ram_mib']:.2f} | {metrics['accuracy_pct']:.2f} |\n"
    )

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(header)
            f.write(separator)
            f.write(row)
    else:
        with open(file_path, "a") as f:
            f.write(row)
    print(f"Successfully logged metrics for '{model_name}' to {file_path}")

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
    test_images, test_labels = load_test_data()
    print(f"Loaded {len(test_images)} test images for latency measurement.")
    avg_latency_ms, std_latency_ms = measure_average_latency(model, test_images, NUM_LATENCY_TESTS)
    print(f"Average single-image inference latency: {avg_latency_ms:.2f} ms")
    print(f"Standard deviation of latency: {std_latency_ms:.2f} ms")

    # Task: Measure the peak RAM usage.
    print("\n[TASK] Measuring memory usage...")
    sample_for_memory_test = test_images[0:1]
    peak_mem_mib = measure_peak_memory_usage(model, sample_for_memory_test)
    print(f"Peak RAM usage during a single inference: {peak_mem_mib:.2f} MiB")

    # Task: Evaluate the model on the entire test dataset for accuracy.
    print("\n[TASK] Evaluating model accuracy...")
    print(f"Evaluating accuracy on the full test dataset ({len(test_images)} images)...")
    results = model.evaluate(test_images, test_labels, batch_size=64, verbose=1)
    baseline_loss = results[0]
    baseline_accuracy = results[1]
    print(f"\nBaseline Model Performance on Test Set:")
    print(f"  - Test Loss: {baseline_loss:.4f}")
    print(f"  - Test Accuracy: {baseline_accuracy * 100:.2f}%")

    # Task: Log all baseline metrics in a structured format.
    print("\n[TASK] Logging all baseline metrics...")
    baseline_metrics = {
        "size_mb": model_size_mb,
        "latency_ms": avg_latency_ms,
        "peak_ram_mib": peak_mem_mib,
        "accuracy_pct": baseline_accuracy * 100.0,
    }
    log_metrics_to_markdown(RESULTS_FILE, "Baseline ResNet50", baseline_metrics)

    print("\n--- Benchmark Complete ---")


if __name__ == "__main__":
    main()
