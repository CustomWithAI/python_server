import cv2
import numpy as np
from skimage.feature import hog
import os

def compute_hog_features():
    # Directory containing multiple images
    image_dir = "./image_set"  # Replace with your image directory
    hog_features_list = []

    # Desired image size (resize all images to this size)
    desired_size = (128, 128)

    for file_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip invalid images
        # Resize image to ensure consistent HOG feature length
        resized_image = cv2.resize(image, desired_size)
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        
        # Ensure that the function parameters are correctly passed
        hog_features = hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False  # Set to False if visualization is not needed
        )
        hog_features_list.append(hog_features)

    return hog_features_list

# Now you can call compute_hog_features()
