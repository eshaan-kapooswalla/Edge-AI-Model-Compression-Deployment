# distill_model.py

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration --
BASELINE_MODEL_PATH = "models/baseline_model"
# Alpha controls the balance between the two loss components.
# A lower alpha gives more weight to the distillation loss.
ALPHA = 0.1
# Temperature is used to soften the probability distributions.
TEMPERATURE = 10

# --- Student Model Definition (Unchanged) ---
def create_student_model(input_shape=(32, 32, 3), num_classes=10):
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

# --- Distiller Class (MODIFIED with train_step and test_step) ---
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=10,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        print("Distiller compiled with custom loss functions and hyperparameters.")

    # --- NEW CODE STARTS HERE ---
    
    def train_step(self, data):
        """
        Overrides the default training step to implement the custom distillation logic.
        """
        # Unpack the data. It's a tuple of (images, labels).
        x, y = data

        # The tf.GradientTape context records operations for automatic differentiation.
        with tf.GradientTape() as tape:
            # 1. Forward pass for both models
            # Get the logits from the student and the (frozen) teacher.
            student_predictions = self.student(x, training=True)
            teacher_predictions = self.teacher(x, training=False)

            # 2. Calculate the student loss (against hard labels)
            student_loss = self.student_loss_fn(y, student_predictions)

            # 3. Calculate the distillation loss (against soft labels)
            # We soften the outputs by dividing the logits by the temperature.
            # Then we apply softmax to get the soft probability distributions.
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            # The KL divergence loss should be scaled by temperature^2. This is a
            # standard practice in distillation literature.
            distillation_loss *= (self.temperature ** 2)

            # 4. Combine the two losses into a single, total loss.
            # alpha controls the weight of the student loss.
            # (1 - alpha) controls the weight of the distillation loss.
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # 5. Compute gradients of the total loss with respect to the student's
        #    trainable weights. The teacher's weights are not considered because
        #    we set teacher.trainable = False.
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # 6. Apply the gradients to update the student's weights.
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 7. Update the metrics (e.g., accuracy).
        # We update our metrics using the student's predictions on the hard labels.
        self.compiled_metrics.update_state(y, student_predictions)

        # 8. Return a dictionary of the results.
        # These will be displayed in the Keras progress bar.
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
            "total_loss": total_loss,
        })
        return results

    def test_step(self, data):
        """
        Overrides the default evaluation step.
        
        During evaluation, we only care about the student's performance on the
        hard labels. The teacher and the distillation loss are not involved.
        """
        # Unpack the data
        x, y = data

        # Get student's predictions (in inference mode)
        student_predictions = self.student(x, training=False)

        # Calculate the standard student loss
        student_loss = self.student_loss_fn(y, student_predictions)

        # Update the metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dictionary of the results
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
        
    # --- NEW CODE ENDS HERE ---
    
    def call(self, x, training=False):
        return self.student(x, training=training)

def main():
    print("--- Knowledge Distillation Workflow ---")
    
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

    print("[TASK] Creating the Distiller model...")
    distiller = Distiller(student=student, teacher=teacher)
    
    print("\n[TASK] Compiling the Distiller with custom loss functions...")
    optimizer = keras.optimizers.Adam()
    distiller.compile(
        optimizer=optimizer,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=ALPHA,
        temperature=TEMPERATURE,
    )
    print("\nDistiller is now fully defined with custom train and test steps.")
    print("It is ready for the training process.")

if __name__ == "__main__":
    main()
