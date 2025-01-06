import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Load the input image
image_path = "test.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Compute HOG features and a visualization
hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, multichannel=False)

# Enhance the HOG image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Plot the original image and the HOG visualization
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

# HOG visualization
plt.subplot(1, 2, 2)
plt.title("HOG Features")
plt.imshow(hog_image_rescaled, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
