import tensorflow as tf
import numpy as np


def load_dataset():

    """
    Loads the CIFAR-10 dataset using TensorFlow's built-in datasets API.

    This function handles the download (if not already cached) and loading
    of the dataset into memory.

    Returns:
        A tuple containing two tuples:
        - The first tuple contains the training data and labels (x_train, y_train).
        - The second tuple contains the test data and labels (x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def get_data_augmentation_pipeline():

    """
    Creates and returns a Keras Sequential model for data preprocessing.

    The pipeline first rescales pixel values from [0, 255] to [0, 1]
    and then applies random transformations to increase dataset diversity
    and reduce overfitting.

    Returns:
        tf.keras.Sequential: Preprocessing + augmentation layers pipeline.
    """
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(32, 32, 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation_and_normalization",
    )
    return data_augmentation


def build_resnet50_model(preprocessing_pipeline: tf.keras.Model) -> tf.keras.Model:

    """
    Build a ResNet50-based classifier for CIFAR-10 using transfer learning.

    Args:
        preprocessing_pipeline: Keras model performing rescaling and augmentation.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = preprocessing_pipeline(inputs, training=True)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    # 1) Load dataset
    (train_images, train_labels), (test_images, test_labels) = load_dataset()
    print("Dataset loaded successfully.")

    # 2) Build preprocessing pipeline
    preprocessing_pipeline = get_data_augmentation_pipeline()
    print("Preprocessing pipeline created.")

    # 3) Build ResNet50 model
    model = build_resnet50_model(preprocessing_pipeline)
    print("ResNet50 model built successfully.")

    # 4) Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    print("Model compiled successfully.")

    # 5) Print model summary
    print("\n--- Model Summary ---")
    model.summary()
