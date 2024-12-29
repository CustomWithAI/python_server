import cv2
import numpy as np
import matplotlib.pyplot as plt

def percentage_thresholding(image_path, percentage):
    """
    Applies percentage thresholding to an image.
    
    Parameters:
        image_path (str): Path to the input image.
        percentage (float): Percentage (0-100) for thresholding.
        
    Returns:
        binary_image: Resulting binary image after percentage thresholding.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or invalid path.")
    
    # Flatten the image and compute the intensity threshold
    flattened = image.flatten()
    threshold_value = np.percentile(flattened, percentage)
    
    # Apply thresholding
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    return binary_image

# Test the function
image_path = "./test.jpg"  # Replace with the path to your image
percentage = 80  # Example: threshold the top 20% of intensity values

binary_image = percentage_thresholding(image_path, percentage)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Binary Image (Top {100-percentage}% intensity)")
plt.imshow(binary_image, cmap='gray')
plt.show()
