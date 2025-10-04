# prune_model.py

import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"
# --- NEW CONFIGURATION ---
# Define the path where we will save our final, pruned model.
PRUNED_MODEL_SAVE_PATH = "models/pruned_model"
# --- END NEW CONFIGURATION ---
FINE_TUNE_EPOCHS = 3
BATCH_SIZE = 64

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
    print("--- Starting Model Pruning Workflow ---")

    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        return

    # Load the baseline model
    print(f"Loading baseline model from: {BASELINE_MODEL_PATH}")
    baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
    print("Baseline model loaded successfully.")
    
    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_dataset()
    print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")
    
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

    # Fine-tune the pruned model
    print("\n[TASK] Fine-tuning the prunable model...")
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]

    print(f"Starting fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
    model_for_pruning.fit(
        train_images,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=(test_images, test_labels),
        callbacks=callbacks
    )
    print("--- Fine-tuning complete. ---")

    # Strip the pruning wrappers
    print("\n[TASK] Stripping pruning wrappers to get the final model...")
    model_stripped = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print("Pruning wrappers stripped successfully.")

    # --- NEW CODE STARTS HERE ---

    # Task: Save the pruned and fine-tuned model.
    print(f"\n[TASK] Saving the final stripped and pruned model to: {PRUNED_MODEL_SAVE_PATH}")
    
    # Create the directory to save the model if it doesn't exist
    # os.path.dirname gets the parent directory ('models') from the full path.
    os.makedirs(os.path.dirname(PRUNED_MODEL_SAVE_PATH), exist_ok=True)
    
    # We call `save()` on our final, stripped model. This saves the standard
    # Keras model with its new sparse weights into the robust SavedModel format.
    model_stripped.save(PRUNED_MODEL_SAVE_PATH)
    
    print("--- Final pruned model saved successfully. ---")
    
    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()
