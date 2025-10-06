# benchmark.py

import os
import time
import numpy as np
import tensorflow as tf
from memory_profiler import memory_usage

# --- Configuration --
# --- MODIFIED CODE STARTS HERE ---
# Revert default target to the baseline model for future runs.
MODEL_PATH = "models/baseline_model"
# --- MODIFIED CODE ENDS HERE ---

NUM_LATENCY_TESTS = 200
RESULTS_FILE = "benchmark_results.md"

# ... all helper functions remain unchanged ...
def get_dir_size_mb(path='.'): # ...
    """Calculates the total size of a directory in megabytes."""
    total_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size_bytes += os.path.getsize(fp)
    total_size_mb = total_size_bytes / (1024 * 1024)
    return total_size_mb
def load_test_data(): # ...
    """Loads the CIFAR-10 test dataset."""
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize the images to the [0, 1] range.
    x_test = x_test.astype('float32') / 255.0
    return x_test, y_test
def log_metrics_to_markdown(file_path, model_name, metrics): # ...
    """Logs a dictionary of metrics to a markdown file."""
    row = f"| {model_name} | {metrics['size_mb']:.2f} | {metrics['latency_ms']:.2f} | {metrics['peak_ram_mib']:.2f} | {metrics['accuracy_pct']:.2f} |\n"
    if not os.path.exists(file_path):
        header = "| Model | Size (MB) | Latency (ms) | Peak RAM (MiB) | Accuracy (%) |\n"
        separator = "|:---|---:|---:|---:|---:|\n"
        with open(file_path, 'w') as f:
            f.write(header)
            f.write(separator)
            f.write(row)
    else:
        with open(file_path, 'a') as f:
            f.write(row)
    print(f"Successfully logged metrics for '{model_name}' to {file_path}")
def measure_keras_latency(model, data, num_samples=200): # ...
    """Measures average latency for a Keras model."""
    latencies = []
    print("Performing a warm-up inference run (Keras)...")
    _ = model.predict(data[0:1], verbose=0)
    print(f"Running {num_samples} individual inferences for latency test (Keras)...")
    for i in range(num_samples):
        sample = data[i:i+1]
        start_time = time.perf_counter()
        _ = model.predict(sample, verbose=0)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
    return np.mean(latencies)
def measure_keras_memory(model, sample_data): # ...
    """Measures peak RAM for a Keras model."""
    return memory_usage((model.predict, (sample_data,), {'verbose': 0}), interval=0.1, max_usage=True)
def measure_tflite_latency(interpreter, input_details, output_details, data, num_samples=200): # ...
    """Measures average latency for a TFLite model."""
    latencies = []
    
    # Warm-up run
    print("Performing a warm-up inference run (TFLite)...")
    interpreter.set_tensor(input_details[0]['index'], data[0:1])
    interpreter.invoke()
    
    print(f"Running {num_samples} individual inferences for latency test (TFLite)...")
    for i in range(num_samples):
        sample = data[i:i+1]
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        # We don't need to get the output tensor for a latency test.
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        
    return np.mean(latencies)
def measure_tflite_memory(interpreter, input_details, sample_data): # ...
    """Measures peak RAM for a TFLite model."""
    # Define the inference function to be profiled
    def inference_func():
        interpreter.set_tensor(input_details[0]['index'], sample_data)
        interpreter.invoke()

    return memory_usage((inference_func), interval=0.1, max_usage=True)
def evaluate_tflite_model(interpreter, input_details, output_details, test_images, test_labels): # ...
    """Evaluates the accuracy of a TFLite model."""
    num_correct = 0
    num_total = len(test_images)
    
    print(f"Evaluating accuracy on {num_total} test images (TFLite)...")
    for i in range(num_total):
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], test_images[i:i+1])
        # Run inference
        interpreter.invoke()
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Get the predicted class index
        predicted_class = np.argmax(output_data)
        
        if predicted_class == test_labels[i][0]:
            num_correct += 1
            
    return num_correct / num_total

# --- Main Orchestration Function ---
def main():
    print("--- Starting Comprehensive Model Benchmark ---\n")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Benchmarking model at: {MODEL_PATH}")
    
    test_images, test_labels = load_test_data()
    
    if MODEL_PATH.endswith('.tflite'):
        # --- MODIFIED CODE STARTS HERE ---
        # We add a check to assign a more descriptive name based on the file.
        if 'integer_only' in MODEL_PATH:
            model_name = "Full Integer Quantized"
        else:
            model_name = "Dynamic Range Quantized"
        # --- MODIFIED CODE ENDS HERE ---
        
        print(f"\n[INFO] TFLite model detected ({model_name}). Using TFLite interpreter workflow.")
        
        model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # In a full-integer model, the input/output tensors are also quantized.
        # Let's inspect the data type (dtype) to confirm.
        print(f"TFLite model input dtype: {input_details[0]['dtype']}")
        print(f"TFLite model output dtype: {output_details[0]['dtype']}")
        
        # If input is int8, we need to quantize our test data to match.
        if input_details[0]['dtype'] == np.int8:
            print("Input is int8, quantizing test data...")
            input_scale, input_zero_point = input_details[0]['quantization']
            test_images = (test_images / input_scale) + input_zero_point
            test_images = test_images.astype(np.int8)

        avg_latency = measure_tflite_latency(interpreter, input_details, output_details, test_images, NUM_LATENCY_TESTS)
        peak_mem = measure_tflite_memory(interpreter, input_details, test_images[0:1])
        accuracy = evaluate_tflite_model(interpreter, input_details, output_details, test_images, test_labels)

    else: # Keras Workflow... 
        # Provide descriptive names for Keras SavedModels
        if "pruned" in MODEL_PATH:
            model_name = "50% Pruned ResNet50"
        elif "distilled" in MODEL_PATH:
            model_name = "Distilled Student"
        else:
            model_name = "Baseline ResNet50"
        print(f"\n[INFO] Keras model detected ({model_name}). Using Keras workflow.")
        model = tf.keras.models.load_model(MODEL_PATH)
        model_size_mb = get_dir_size_mb(MODEL_PATH)
        avg_latency = measure_keras_latency(model, test_images, NUM_LATENCY_TESTS)
        peak_mem = measure_keras_memory(model, test_images[0:1])
        results = model.evaluate(test_images, test_labels, batch_size=64, verbose=0)
        accuracy = results[1]

    # --- Consolidate and Log Results ---
    # ... (This section remains unchanged) ...
    print("\n--- Benchmark Results ---")
    print(f"  - Model Size: {model_size_mb:.2f} MB")
    print(f"  - Average Latency: {avg_latency:.2f} ms")
    print(f"  - Peak RAM Usage: {peak_mem:.2f} MiB")
    print(f"  - Test Accuracy: {accuracy * 100:.2f}%")

    metrics = {
        "size_mb": model_size_mb,
        "latency_ms": avg_latency,
        "peak_ram_mib": peak_mem,
        "accuracy_pct": accuracy * 100
    }
    log_metrics_to_markdown(RESULTS_FILE, model_name, metrics)
    print("\n--- Benchmark Complete ---\n")

if __name__ == "__main__":
    main()
