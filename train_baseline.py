import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    Creates and returns a Keras Sequential model for data augmentation.

    This pipeline applies random transformations to training images to
    increase dataset diversity and reduce overfitting.

    Returns:
        tf.keras.Sequential: Data augmentation layers pipeline.
    """
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    return data_augmentation


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    print(f"Training images shape: {train_images.shape}")
    print(f"Training images data type: {train_images.dtype}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print("-" * 30)

    # Visualize augmentation examples
    augmentation_pipeline = get_data_augmentation_pipeline()
    first_image = train_images[0]
    print(f"Visualizing augmentations for the first image (label '{train_labels[0][0]}').")
    image_batch = tf.expand_dims(first_image, 0)

    plt.figure(figsize=(10, 10))
    plt.suptitle("Data Augmentation Examples", fontsize=16)

    # Original image
    ax = plt.subplot(3, 3, 1)
    plt.imshow(first_image)
    plt.title("Original")
    plt.axis("off")

    # Augmented images
    for i in range(8):
        ax = plt.subplot(3, 3, i + 2)
        augmented_image = augmentation_pipeline(image_batch)
        plt.imshow(augmented_image[0].numpy().astype("uint8"))
        plt.title(f"Augmented {i + 1}")
        plt.axis("off")

    plt.show()


