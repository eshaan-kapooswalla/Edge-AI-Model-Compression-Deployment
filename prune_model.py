# prune_model.py

import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"
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

    # --- NEW CODE STARTS HERE ---

    # Task: Strip the pruning wrappers from the model.
    print("\n[TASK] Stripping pruning wrappers to get the final model...")
    
    # The `strip_pruning` function removes the PruneLowMagnitude wrapper layers
    # and returns a standard Keras model with the sparse weights baked in.
    # This is the model we will save and benchmark.
    model_stripped = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    print("Pruning wrappers stripped successfully.")

    # --- VERIFICATION ---
    # 1. Check the model summary. It should now look identical to the original
    #    baseline model's summary, with the original layer types and parameter counts.
    #    This confirms the scaffolding is gone.
    print("\n--- Stripped Model Summary ---")
    model_stripped.summary()

    # 2. The "Aha!" moment: Calculate the actual sparsity of the final model.
    #    We do this by counting the number of non-zero weights in the trainable
    #    layers and comparing it to the total number of weights.
    def calculate_sparsity(model):
        """Helper function to calculate the sparsity of a Keras model."""
        total_params = 0
        non_zero_params = 0
        for layer in model.layers:
            # We only care about the weights of trainable layers
            if layer.trainable_weights:
                # Concatenate all weight tensors of the layer into one flat vector
                layer_weights = tf.concat([tf.reshape(w, [-1]) for w in layer.trainable_weights], axis=0)
                total_params += tf.size(layer_weights).numpy()
                non_zero_params += tf.math.count_nonzero(layer_weights).numpy()
        
        sparsity = 1.0 - (non_zero_params / total_params)
        return sparsity, non_zero_params, total_params

    final_sparsity, final_non_zero, final_total = calculate_sparsity(model_stripped)

    print("\n--- Final Model Sparsity Verification ---")
    print(f"Total trainable parameters in stripped model: {final_total:,}")
    print(f"Non-zero trainable parameters in stripped model: {final_non_zero:,}")
    print(f"Final model sparsity: {final_sparsity:.2%}")
    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()