# prune_model.py

import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"
FINE_TUNE_EPOCHS = 3
BATCH_SIZE = 64

# --- MODIFIED HELPER FUNCTION ---
def load_dataset():
    """
    Loads the CIFAR-10 dataset. We now need both training and test sets.
    The test set will be used for validation during fine-tuning.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def main():
    """
    Main function to orchestrate the model pruning process.
    """
    print("--- Starting Model Pruning Workflow ---\
")

    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        return

    # Load the baseline model
    print(f"Loading baseline model from: {BASELINE_MODEL_PATH}")
    baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
    print("Baseline model loaded successfully.")
    
    # --- MODIFIED SECTION ---
    # We now load both training and testing data.
    (train_images, train_labels), (test_images, test_labels) = load_dataset()
    print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")
    # --- END MODIFIED SECTION ---

    # Define the pruning schedule
    print("\n[TASK] Defining the pruning schedule...")
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

    # Apply the pruning wrapper
    print("\n[TASK] Applying the pruning wrapper to the model...")
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        baseline_model,
        **pruning_params
    )
    print("Pruning wrapper applied successfully.")

    # Re-compile the prunable model
    print("\n[TASK] Re-compiling the prunable model...")
    model_for_pruning.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print("Prunable model re-compiled successfully.")

    # --- NEW CODE STARTS HERE ---

    # Task: Fine-tune the pruned model.
    print("\n[TASK] Fine-tuning the prunable model...")

    # Define the callbacks for the fine-tuning process.
    callbacks = [
        # This callback is MANDATORY for pruning. It updates the pruning-related
        # variables in the model based on the current step. Without this, the
        # model will not be pruned.
        tfmot.sparsity.keras.UpdatePruningStep()
    ]

    print(f"Starting fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
    # We call `fit()` on our new, prunable model.
    # Behind the scenes, this will both train the remaining weights and
    # execute the pruning schedule you defined.
    model_for_pruning.fit(
        train_images,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=(test_images, test_labels),
        callbacks=callbacks  # Pass in our mandatory callback here.
    )

    print("--- Fine-tuning complete. ---")

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()
