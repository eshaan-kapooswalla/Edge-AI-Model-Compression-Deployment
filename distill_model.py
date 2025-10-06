# distill_model.py

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"
# Alpha controls the balance between the two loss components.
# A lower alpha gives more weight to the distillation loss.
ALPHA = 0.1
# Temperature is used to soften the probability distributions.
TEMPERATURE = 10

# --- Student Model Definition (Unchanged) ---
def create_student_model(input_shape=(32, 32, 3), num_classes=10):
    # ... (code from previous task is unchanged)
    student_model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes),
    ], name="student")
    return student_model

# --- Distiller Class (MODIFIED) ---
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        # We will define the loss functions and other training components
        # in the compile() method.

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=10,
    ):
        """
        Configures the Distiller for training.

        This is a custom compile method that sets up the optimizer, metrics,
        and the two separate loss functions required for distillation.

        Args:
            optimizer: The optimizer to use for training.
            metrics: A list of metrics to monitor during training.
            student_loss_fn: The loss function for the student's predictions against hard labels.
            distillation_loss_fn: The loss function for comparing student and teacher soft predictions.
            alpha (float): The weight for the student loss.
            temperature (int): The temperature for softening probabilities.
        """
        # We call the parent's compile method to handle the standard parts of compilation.
        super().compile(optimizer=optimizer, metrics=metrics)
        
        # Store the custom components as attributes of the Distiller.
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        print("Distiller compiled with custom loss functions and hyperparameters.")

    def call(self, x, training=False):
        # The forward pass remains the same.
        return self.student(x, training=training)

# --- Main Workflow (MODIFIED) ---
def main():
    print("--- Knowledge Distillation Workflow ---\
")
    
    # --- Create Student and Teacher ---
    print("[TASK] Creating the student model...")
    student = create_student_model()
    print("Student model created.\n")

    print("[TASK] Loading the teacher model...")
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        return
    teacher = keras.models.load_model(BASELINE_MODEL_PATH)
    teacher.trainable = False
    print("Teacher model loaded and frozen.\n")

    # --- Create the Distiller ---
    print("[TASK] Creating the Distiller model...")
    distiller = Distiller(student=student, teacher=teacher)
    
    # --- NEW CODE STARTS HERE ---

    # --- Compile the Distiller ---
    print("\n[TASK] Compiling the Distiller with custom loss functions...")

    # 1. Instantiate the optimizer. Adam is a robust choice.
    optimizer = keras.optimizers.Adam()

    # 2. Instantiate the two loss functions.
    student_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    distillation_loss = tf.keras.losses.KLDivergence()

    # 3. Call our custom compile method.
    # We pass the optimizer, a standard 'accuracy' metric to monitor, and our two
    # loss functions. The alpha and temperature values are taken from the config.
    distiller.compile(
        optimizer=optimizer,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=student_loss,
        distillation_loss_fn=distillation_loss,
        alpha=ALPHA,
        temperature=TEMPERATURE,
    )

    print("\n--- Verification ---")
    print("Distiller is now fully configured and ready for training.")
    print(f"Student Loss Function: {distiller.student_loss_fn.name}")
    print(f"Distillation Loss Function: {distiller.distillation_loss_fn.name}")
    print(f"Alpha (Student Loss Weight): {distiller.alpha}")
    print(f"Temperature: {distiller.temperature}")

    # --- NEW CODE ENDS HERE ---

if __name__ == "__main__":
    main()