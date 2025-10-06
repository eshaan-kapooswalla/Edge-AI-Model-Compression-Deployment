# distill_model.py

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"

# --- Student Model Definition (Unchanged) ---
def create_student_model(input_shape=(32, 32, 3), num_classes=10):
    # ... (code from previous task is unchanged)
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

# --- NEW CODE STARTS HERE ---

# Task: Create a custom Distiller model/class that encapsulates both teacher and student.

class Distiller(keras.Model):
    """
    A custom Keras Model to manage the knowledge distillation training process.

    This class acts as a container for both the student and teacher models.
    We will build upon this class in subsequent tasks to add the custom
    training logic (compile and train_step).
    """
    def __init__(self, student, teacher):
        """
        Initializes the Distiller model.

        Args:
            student: A Keras model instance (our lightweight student).
            teacher: A Keras model instance (our powerful, pre-trained teacher).
        """
        # Always call the parent constructor first.
        super().__init__()
        
        # Store the student and teacher models as attributes of this class.
        self.teacher = teacher
        self.student = student
        print("Distiller initialized with a student and a teacher model.")

    def call(self, x, training=False):
        """
        Defines the forward pass for the Distiller model.

        For inference (when training=False), we only need the student's output.
        The `training` argument is a standard part of the Keras `call` signature.

        Args:
            x: The input data.
            training (bool): A flag indicating if the model is in training mode.

        Returns:
            The output predictions from the student model.
        """
        # The forward pass of the Distiller simply returns the student's predictions.
        # This is because after distillation, the student is the final, standalone model.
        return self.student(x, training=training)

# --- NEW CODE ENDS HERE ---

def main():
    """
    Main function to set up the Distiller model.
    """
    print("--- Knowledge Distillation Workflow ---\n")
    
    # --- Part 1: Student Model ---
    print("[TASK] Creating the student model...")
    student = create_student_model()
    print("Student model created.\n")

    # --- Part 2: Teacher Model ---
    print("[TASK] Loading the teacher model...")
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        return

    teacher = keras.models.load_model(BASELINE_MODEL_PATH)
    teacher.trainable = False
    print("Teacher model loaded and frozen.\n")

    # --- NEW CODE STARTS HERE ---
    
    # --- Part 3: Create the Distiller ---
    print("[TASK] Creating the Distiller model...")
    
    # Instantiate our custom Distiller class, passing it the student and teacher.
    # This single 'distiller' object now encapsulates the entire training setup.
    distiller = Distiller(student=student, teacher=teacher)
    
    print("\n--- Verification ---")
    print("Distiller model created successfully.")
    # We can access the internal models to verify they are correct.
    print(f"Distiller's student model name: {distiller.student.name}")
    print(f"Distiller's teacher model name: {distiller.teacher.name}")

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()
