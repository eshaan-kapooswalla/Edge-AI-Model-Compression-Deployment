# distill_model.py

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"

def create_student_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Defines and creates the lightweight 'student' CNN architecture.
    (This function remains unchanged from the previous task)
    """
    student_model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes),
    ])
    return student_model

def main():
    """
    Main function to set up the student and teacher models for distillation.
    """
    print("--- Knowledge Distillation Workflow ---\n")
    
    # --- Part 1: Student Model (from previous task) ---
    print("[TASK] Designing and implementing the lightweight student model...")
    student = create_student_model()
    print("\nStudent model created successfully. Here is its summary:")
    student.summary()

    # --- NEW CODE STARTS HERE ---

    # --- Part 2: Teacher Model ---
    print("\n[TASK] Instantiating the baseline model as the 'teacher'...")

    # First, verify that the baseline model we need to load actually exists.
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("Please run the training script `train_baseline.py` first.")
        return

    try:
        # Load the pre-trained baseline model using the familiar load_model function.
        # This model contains all the "knowledge" we want to distill.
        teacher = keras.models.load_model(BASELINE_MODEL_PATH)
        
        # CRITICAL STEP: Freeze the teacher model.
        # We are using the teacher for inference only, to generate logits for the student.
        # We must not update its weights during the distillation process. Setting
        # trainable to False ensures that its learned knowledge remains intact and
        # makes the training process more efficient.
        teacher.trainable = False
        
        print("\nTeacher model (baseline ResNet50) loaded successfully. Here is its summary:")
        teacher.summary()
        
        # Let's explicitly verify the trainable status.
        print(f"\n--- Verification ---")
        print(f"Teacher model's trainable status has been set to: {teacher.trainable}")

    except Exception as e:
        print(f"An error occurred while loading the teacher model: {e}")
        return

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()