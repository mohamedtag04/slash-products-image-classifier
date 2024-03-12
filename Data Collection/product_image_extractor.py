"""
Author: Mohamed Tag
Email: mohamedtag264@gmail.com
Version: 1.0.0
License: MIT
State: Completed
Date: 2024-3-12
Purpose: This script processes screenshots from the Slash mobile application,
         extracts product images, and saves them for further usage.
Dependencies: OpenCV, Python
Client: Slash Mobile Application
"""

import cv2
import os
from typing import List, Tuple
import numpy as np

class ImageCropper:
    """
    ImageCropper class is responsible for processing screenshots from the Slash mobile application.
    It extracts product images and saves them for further usage.

    Attributes:
    - None
    """

    @staticmethod
    def _crop_product(original_image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Crop the region around the product based on the contour.

        Parameters:
        - original_image (np.ndarray): The original image.
        - contour (np.ndarray): Contour of the product.

        Returns:
        - np.ndarray: Cropped product image.
        """
        x, y, w, h = cv2.boundingRect(contour)
        return original_image[y:y+h, x:x+w]

    def _find_product_contours(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """
        Find and return the contours of products in the given grayscale image.

        Parameters:
        - gray_image (np.ndarray): Grayscale image.

        Returns:
        - List[np.ndarray]: List of product contours.
        """
        edges = cv2.Canny(gray_image, 5, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _process_image(self, image_path: str, output_class_folder: str) -> None:
        """
        Process a single image, extract product images, and save them in the specified output folder.

        Parameters:
        - image_path (str): Path to the input image.
        - output_class_folder (str): Path to the output directory for cropped products.

        Returns:
        - None
        """
        filename = os.path.basename(image_path)
        original_image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        contours = self._find_product_contours(gray_image)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 60000:
                product = self._crop_product(original_image, contour)
                output_filename = f"{filename.split('.')[0]}_product_{i}.png"
                output_path = os.path.join(output_class_folder, output_filename)
                cv2.imwrite(output_path, product)

    def crop_and_save_products(self, input_folder: str, output_folder: str) -> None:
        """
        Crop and save product images from input images.

        Parameters:
        - input_folder (str): Path to the input images directory containing subdirectories with images.
        - output_folder (str): Path to the output directory for cropped products.

        Returns:
        - None
        """
        for class_directory in os.listdir(input_folder):
            class_path = os.path.join(input_folder, class_directory)

            if os.path.isdir(class_path):
                output_class_folder = os.path.join(output_folder, f"cropped_{class_directory}_products")
                os.makedirs(output_class_folder, exist_ok=True)

                for filename in os.listdir(class_path):
                    if filename.lower().endswith((".jpg", ".png")):
                        image_path = os.path.join(class_path, filename)
                        self._process_image(image_path, output_class_folder)

def count_files_in_subdirectories(directory_path: str) -> None:
    """
    Count the number of files in each subdirectory of the given directory.

    Parameters:
    - directory_path (str): Path to the directory.

    Returns:
    - None
    """
    total_files = 0

    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)

        if os.path.isdir(subdirectory_path):
            num_files = len([f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))])
            print(f"Subdirectory: {subdirectory}, Number of Files: {num_files}")
            total_files += num_files

    print(f"Total Number of Files: {total_files}")

if __name__ == "__main__":
    cropper = ImageCropper()
    
    input_folder = "screenshots" 
    output_folder = "final_products_images"
    
    cropper.crop_and_save_products(
        os.path.join("Data Collection", input_folder),
        os.path.join("Data Collection", output_folder)
    )

    # Additional functionality
    count_files_in_subdirectories(
        os.path.join("Data Collection", output_folder)
    )

