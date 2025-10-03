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


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    print(f"Training images shape: {train_images.shape}")
    print(f"Training images data type: {train_images.dtype}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"First training label: {train_labels[0]}")


