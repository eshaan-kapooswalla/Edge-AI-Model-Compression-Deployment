# prune_model.py

import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# --- Configuration ---
# Path to our saved baseline model.
BASELINE_MODEL_PATH = "models/baseline_model"
# We'll fine-tune the pruned model for a few epochs to let it recover accuracy.
FINE_TUNE_EPOCHS = 3
# We'll use the same batch size as our original training.
BATCH_SIZE = 64

def load_dataset():
    """
    Loads the CIFAR-10 dataset. We only need the training data to calculate
    the number of steps for our pruning schedule.
    """
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train)

def main():
    """
    Main function to orchestrate the model pruning process.
    """
    print("--- Starting Model Pruning Workflow ---
")

    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("Please run the training script `train_baseline.py` first.")
        return

    # Load the baseline model.
    print(f"Loading baseline model from: {BASELINE_MODEL_PATH}")
    try:
        baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
        print("Baseline model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # Define the pruning schedule.
    print("\n[TASK] Defining the pruning schedule...")
    (train_images, _), = load_dataset()
    num_train_samples = len(train_images)
    end_step = np.ceil(num_train_samples / BATCH_SIZE).astype(np.int32) * FINE_TUNE_EPOCHS
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.50,
            begin_step=0,
            end_step=end_step,
            frequency=100
        )
    }
    print(f"Pruning schedule defined to reach 50% sparsity in {end_step} steps.")

    # --- NEW CODE STARTS HERE ---

    # Task: Apply the pruning wrapper to the baseline model.
    print("\n[TASK] Applying the pruning wrapper to the model...")

    # The `tfmot.sparsity.keras.prune_low_magnitude` function takes our baseline model
    # and our pruning configuration. It returns a new model where prunable layers
    # have been wrapped.
    # The `**pruning_params` syntax is Python's way of "unpacking" a dictionary's
    # key-value pairs into keyword arguments for a function. It's equivalent to calling:
    # prune_low_magnitude(baseline_model, pruning_schedule=pruning_params['pruning_schedule'])
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        baseline_model,
        **pruning_params
    )

    print("Pruning wrapper applied successfully.")

    # To verify that the wrapper has been applied, we MUST inspect the model summary.
    # The output will look different from the original model's summary.
    print("\n--- Prunable Model Summary ---")
    model_for_pruning.summary()

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()