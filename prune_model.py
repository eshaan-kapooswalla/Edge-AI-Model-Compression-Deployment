import os
import tensorflow as tf


# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"


def main():
    print("--- Starting Model Pruning Workflow ---")

    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("Please run the training script `train_baseline.py` first.")
        return

    print(f"Loading baseline model from: {BASELINE_MODEL_PATH}")
    try:
        baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
        print("Baseline model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    print("\n--- Baseline Model Summary ---")
    baseline_model.summary()

    # Count initial non-zero trainable parameters
    flat_trainable_weights = [tf.reshape(w, [-1]) for w in baseline_model.trainable_weights]
    if flat_trainable_weights:
        concatenated = tf.concat(flat_trainable_weights, axis=0)
        num_non_zero_params = tf.math.count_nonzero(concatenated)
        print(f"\nInitial number of non-zero trainable parameters: {num_non_zero_params.numpy():,}")
    else:
        print("\nModel has no trainable weights to count.")


if __name__ == "__main__":
    main()


