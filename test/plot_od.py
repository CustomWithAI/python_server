import cv2
import json

# Example image path and JSON data
image_path = './Cats_Test0.png'
detection_output = {
    "prediction": [
        {
            "class_id": 0,
            "bbox": {
                "x_center": 0.600858,
                "y_center": 0.244286,
                "width": 0.489270,
                "height": 0.322857
            },
            "confidence": 0.27002599835395813
        }
    ]
}

# Load image
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# Draw bounding boxes
for pred in detection_output["prediction"]:
    bbox = pred["bbox"]
    class_id = pred["class_id"]
    conf = pred["confidence"]

    # Convert normalized to absolute coordinates
    x_center = bbox["x_center"] * image_width
    y_center = bbox["y_center"] * image_height
    width = bbox["width"] * image_width
    height = bbox["height"] * image_height

    # Calculate top-left and bottom-right corners
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Skip boxes with invalid dimensions
    if width <= 0 or height <= 0:
        continue

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Label
    label = f"Class {class_id}: {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display the result
# cv2.imwrite('output.jpg', image)
cv2.imshow('Detected Image', image)
cv2.waitKey(0)
# cv2.destroyAllWindows()
