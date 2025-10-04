# prune_model.py

import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# --- Configuration --
# Path to our saved baseline model.
BASELINE_MODEL_PATH = "models/baseline_model"
# We'll fine-tune the pruned model for a few epochs to let it recover accuracy.
FINE_TUNE_EPOCHS = 3
# We'll use the same batch size as our original training.
BATCH_SIZE = 64

# --- NEW HELPER FUNCTION ---
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
    print("--- Starting Model Pruning Workflow ---\
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

    # --- NEW CODE STARTS HERE ---

    # Task: Define a pruning schedule.
    print("\n[TASK] Defining the pruning schedule...")

    # Load the training data to determine the number of training steps.
    (train_images, _), = load_dataset()
    num_train_samples = len(train_images)
    print(f"Loaded {num_train_samples} training images to calculate schedule steps.")

    # Calculate the `end_step` for the pruning schedule. This is a crucial calculation.
    # The pruning process happens over a series of training steps (batches). We want
    # the pruning to finish at the very end of our fine-tuning phase.
    # The total number of steps is (number of samples / batch size) * number of epochs.
    end_step = np.ceil(num_train_samples / BATCH_SIZE).astype(np.int32) * FINE_TUNE_EPOCHS
    print(f"Fine-tuning for {FINE_TUNE_EPOCHS} epochs with a batch size of {BATCH_SIZE}.")
    print(f"Calculated `end_step` for pruning: {end_step}")

    # Define the pruning parameters using the PolynomialDecay schedule.
    # This creates a dictionary that will be used to wrap the model.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.50,  # Target 50% of weights to be zero.
            begin_step=0,         # Start pruning from the very first step of fine-tuning.
            end_step=end_step,    # End pruning at the end of the fine-tuning.
            frequency=100         # Check and update the pruning mask every 100 steps.
        )
    }
    
    print("\nPruning parameters defined successfully:")
    print(pruning_params)

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()