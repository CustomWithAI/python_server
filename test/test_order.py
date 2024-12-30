import cv2
import numpy as np

# Generate a test image
image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.putText(image, 'A', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

def apply_flip(image, direction):
    return cv2.flip(image, direction)

def apply_rotation(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# Flip → Rotate
flip_then_rotate = apply_rotation(apply_flip(image, 1), 45)

# Rotate → Flip
rotate_then_flip = apply_flip(apply_rotation(image, 45), 1)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Flip → Rotate", flip_then_rotate)
cv2.imshow("Rotate → Flip", rotate_then_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()
