import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
import os

# Directory containing multiple images
image_dir = "./image_set"  # Replace with your image directory
hog_features_list = []

# HOG parameters
cell_size = (8, 8)
block_size = (2, 2)
orientations = 9

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
    hog_features, _ = hog(
        gray_image,
        orientations=orientations,
        pixels_per_cell=cell_size,
        cells_per_block=block_size,
        visualize=True,
    )
    hog_features_list.append(hog_features)

# Convert to a 2D array
hog_features_array = np.vstack(hog_features_list)  # Stack vertically to ensure consistency

# Apply PCA
n_components = min(hog_features_array.shape[0], hog_features_array.shape[1])  # Number of components
pca = PCA(n_components=n_components)
reduced_features = pca.fit_transform(hog_features_array)

print("Original HOG features shape:", hog_features_array.shape)
print(hog_features_array)
print("Reduced features shape:", reduced_features.shape)
print(reduced_features)