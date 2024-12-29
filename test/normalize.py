import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize_image(image, target_range=(0, 1)):
    """
    Normalize an image to a specified range.
    
    Parameters:
        image (numpy.ndarray): Input image.
        target_range (tuple): Desired range (min_val, max_val).
        
    Returns:
        numpy.ndarray: Normalized image.
    """
    min_val, max_val = target_range
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    
    # Convert image to float for processing
    image = image.astype(np.float32)
    
    # Normalize to [0, 1]
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Scale to [min_val, max_val]
    scaled = normalized * (max_val - min_val) + min_val
    
    return scaled

# Load an image
image = cv2.imread("./test.jpg")  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

# Normalize to custom range [50, 200]
normalized_image = normalize_image(image, target_range=(0, 1))

# Clip to valid uint8 range for visualization
normalized_image_uint8 = np.clip(normalized_image, 0, 255).astype(np.uint8)

# Display original and normalized images
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Normalized image
plt.subplot(1, 2, 2)
plt.imshow(normalized_image_uint8)
plt.title("Normalized Image")
plt.axis("off")

plt.tight_layout()
plt.show()
