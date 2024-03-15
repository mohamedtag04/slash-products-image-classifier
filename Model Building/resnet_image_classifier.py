"""
Author: Mohamed Tag
Email: mohamedtag264@gmail.com
Version: 1.0.0
License: MIT
State: Completed
Date: 2024-3-15
Purpose: This script defines functions to train, evaluate, and make predictions with a pre-trained ResNet50 model
         for image classification tasks. It also includes functions for loading and preprocessing datasets.
Dependencies: TensorFlow, numpy, pathlib
"""

import numpy as np
import tensorflow as tf
import pathlib

# Function to load and preprocess the dataset
def load_dataset(data_dir, img_height=224, img_width=224, batch_size=5, validation_split=0.2, seed=123):
    """
    Load and preprocess the dataset from the specified directory.

    Args:
        data_dir (str): Path to the dataset directory.
        img_height (int): Height of the images.
        img_width (int): Width of the images.
        batch_size (int): Batch size for training and validation.
        validation_split (float): Fraction of images to reserve for validation.
        seed (int): Random seed for dataset splitting.

    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        class_names (list): List of class names.
        num_classes (int): Number of classes.
    """
    data_dir = pathlib.Path(data_dir)
    
    # Load class names
    class_names = sorted([item.name for item in data_dir.glob('*')])
    num_classes = len(class_names)
    
    # Create the training dataset without augmentation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Create the validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names, num_classes

# Function to create a pre-trained ResNet50 model
def create_resnet_model(input_shape, num_classes):
    """
    Create a pre-trained ResNet50 model for transfer learning.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes.

    Returns:
        model (tf.keras.Model): Pre-trained ResNet50 model.
    """
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes)
    ])
    
    return model

# Function to train the model
def train_model(model, train_ds, val_ds, epochs=18, callbacks=[]):
    """
    Train the model.

    Args:
        model (tf.keras.Model): Model to train.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs for training.
        callbacks (list): List of callbacks for training.

    Returns:
        history (tf.keras.callbacks.History): Training history.
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

# Function to evaluate the model
def evaluate_model(model, val_ds):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (tf.keras.Model): Model to evaluate.
        val_ds (tf.data.Dataset): Validation dataset.

    Returns:
        loss (float): Loss value.
        accuracy (float): Accuracy value.
    """
    loss, accuracy = model.evaluate(val_ds)
    return loss, accuracy

# Function to make predictions
def predict_image(model, image_path, class_names):
    """
    Make predictions on a single image.

    Args:
        model (tf.keras.Model): Trained model for prediction.
        image_path (str): Path to the image file.
        class_names (list): List of class names.

    Returns:
        prediction (str): Predicted class label.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Load and preprocess the dataset
train_ds, val_ds, class_names, num_classes = load_dataset("Data Collection/final_products_images")

# Create ResNet50 model
model = create_resnet_model((224, 224, 3), num_classes)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = train_model(model, train_ds, val_ds)

# Evaluate the model
loss, accuracy = evaluate_model(model, val_ds)
print(f"Validation accuracy with ResNet50, data augmentation: False is {accuracy}")


# Save the model
model.save("Model Building\saved-model")


# Make prediction on a sample image
image_path = "Model Building\TEST.png"
predicted_class = predict_image(model, image_path, class_names)
print(f"Predicted class for {image_path}: {predicted_class}")


