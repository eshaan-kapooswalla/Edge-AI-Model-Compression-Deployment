# benchmark.py

import os
import time
import numpy as np
import tensorflow as tf
from memory_profiler import memory_usage

# --- Configuration --
# --- MODIFIED CODE STARTS HERE --
# We are now pointing our benchmark script to the newly created pruned model.
# This is the only change required to test our new model.
MODEL_PATH = "models/pruned_model" 
# --- MODIFIED CODE ENDS HERE --

NUM_LATENCY_TESTS = 200
RESULTS_FILE = "benchmark_results.md"

# ... all helper functions (get_dir_size_mb, load_test_data, etc.) remain unchanged ...
def get_dir_size_mb(path='.'):
    # ... (implementation is the same)
    total_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total_size_bytes += os.path.getsize(file_path)
    return total_size_bytes / (1024 * 1024)

def load_test_data():
    # ... (implementation is the same)
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test, y_test

def measure_average_latency(model, data, num_samples=200):
    # ... (implementation is the same)
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

def measure_peak_memory_usage(model, sample_data):
    # ... (implementation is the same)
    peak_mib = memory_usage((model.predict, (sample_data,), {"verbose": 0}), interval=0.1, max_usage=True)
    return float(peak_mib)

def log_metrics_to_markdown(file_path, model_name, metrics):
    # ... (implementation is the same)
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
    # ... (the main function logic remains entirely the same) ...
    # It will now load from the new MODEL_PATH and log the results with a new model name.
    
    print("--- Starting Comprehensive Model Benchmark ---")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Benchmarking model at: {MODEL_PATH}")

    # Load the model
    print("\n[TASK] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # --- Metrics Gathering ---
    model_size_mb = get_dir_size_mb(MODEL_PATH)
    test_images, test_labels = load_test_data()
    avg_latency, _ = measure_average_latency(model, test_images, NUM_LATENCY_TESTS)
    peak_mem = measure_peak_memory_usage(model, test_images[0:1])
    results = model.evaluate(test_images, test_labels, batch_size=64, verbose=0)
    pruned_accuracy = results[1]

    # --- Printing Results to Console ---
    print("\n--- Pruned Model Benchmark Results ---")
    print(f"  - Model Size: {model_size_mb:.2f} MB")
    print(f"  - Average Latency: {avg_latency:.2f} ms")
    print(f"  - Peak RAM Usage: {peak_mem:.2f} MiB")
    print(f"  - Test Accuracy: {pruned_accuracy * 100:.2f}%")

    # --- Logging Results to File ---
    print("\n[TASK] Logging all pruned metrics...")
    
    pruned_metrics = {
        "size_mb": model_size_mb,
        "latency_ms": avg_latency,
        "peak_ram_mib": peak_mem,
        "accuracy_pct": pruned_accuracy * 100
    }
    
    # We provide a new name for this model in our log file.
    log_metrics_to_markdown(RESULTS_FILE, "50% Pruned ResNet50", pruned_metrics)

    print("\n--- Benchmark Complete ---")


if __name__ == "__main__":
    main()