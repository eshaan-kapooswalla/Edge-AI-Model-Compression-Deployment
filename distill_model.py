# distill_model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_student_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Defines and creates the lightweight 'student' CNN architecture.

    This model is intentionally simple and small, making it fast for inference.
    It serves as the student in our knowledge distillation setup.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes.

    Returns:
        A Keras Sequential model instance.
    """
    
    # We use the Keras Sequential API for its simplicity.
    student_model = keras.Sequential([
        # The input layer specifies the shape of the incoming data.
        keras.Input(shape=input_shape),

        # Block 1: A small convolutional block to start extracting features.
        # 32 filters learn basic patterns like edges and colors.
        layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu"),
        # MaxPooling reduces the spatial dimensions (width/height), making the
        # model more efficient and the learned features more robust to location.
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2: We increase the number of filters to learn more complex patterns.
        # The network learns to combine the simpler patterns from the previous layer.
        layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 3: One more convolutional layer to capture even more abstract features.
        layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the 3D feature maps into a 1D vector to feed into the dense layers.
        layers.Flatten(),
        
        # A fully connected layer for high-level reasoning.
        layers.Dense(256, activation="relu"),

        # The output layer. CRITICAL: It has NO activation function (e.g., softmax).
        # This is because the distillation loss function (KL Divergence) operates on
        # the raw, un-normalized outputs, which are called 'logits'.
        layers.Dense(num_classes),
    ])
    
    return student_model

def main():
    """
    Main function to design and inspect the student model.
    """
    print("--- Knowledge Distillation Workflow ---\n")
    print("[TASK] Designing and implementing the lightweight student model...")

    # Create an instance of our student model.
    student = create_student_model()
    
    print("\nStudent model created successfully. Here is the summary:")
    student.summary()

    # Let's highlight the dramatic difference in size.
    # Our ResNet50 teacher has over 23,000,000 parameters.
    # The student model is orders of magnitude smaller.
    print(f"\nCompare this to the teacher model's ~23.6 Million parameters.")
    print("This lightweight design is the key to achieving high performance on edge devices.")

if __name__ == "__main__":
    main()
