"""
Author: Mohamed Tag
Email: mohamedtag264@gmail.com
Version: 1.0.0
License: MIT
State: Completed
Date: 2024-3-15
Purpose: This class loads a trained ResNet50 model for image classification and makes predictions on new images.
Dependencies: TensorFlow, numpy, pathlib
"""

import numpy as np
import tensorflow as tf
import pathlib

class ImageClassifier:
    def __init__(self, model_path):
        """
        Initialize the ImageClassifier with the path to the saved model directory.

        Args:
            model_path (str): Path to the saved model directory.
        """
        self.model_path = model_path
        self.model, self.class_names = self.load_model()


    def load_model(self):
        """
        Load the saved model from the specified path.

        Returns:
            model (tf.keras.Model): Loaded model.
            class_names (list): List of class names.
        """
        model = tf.keras.models.load_model(self.model_path)
        data_dir = "Data Collection/final_products_images"
        data_dir = pathlib.Path(data_dir)
        class_names = sorted([item.name for item in pathlib.Path(data_dir).glob('*')])
        return model, class_names

    def predict_image(self, image_path):
        """
        Make predictions on a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            predicted_class (str): Predicted class label.
        """
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)  # Create batch axis
        predictions = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(predictions)]
        return predicted_class

# Class implementation
if __name__ == "__main__":
    # Path to the saved model directory
    model_path = "Model Building/saved-model"
    
    # Create an instance of the ImageClassifier
    image_classifier = ImageClassifier(model_path)
    
    # Path to the image to predict
    image_path = "Model Building\TEST.png"
    
    # Make prediction
    predicted_class = image_classifier.predict_image(image_path)
    print("----------------------------------------------")
    print(f"Predicted class: {predicted_class[8:-9]}")
    print("----------------------------------------------")

