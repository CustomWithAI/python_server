import cv2
import json

# Example image path and JSON data
image_path = './test.jpg'
detection_output = {
    "prediction": [
        {
            "class_id": 0,
            "bbox": {
                "x_center": 0.01852455735206604,
                "y_center": 0.13786430656909943,
                "width": 0.057493265718221664,
                "height": 0.17428430914878845
            },
            "confidence": 0.27002599835395813
        },
        {
            "class_id": 1,
            "bbox": {
                "x_center": 0.02155199646949768,
                "y_center": 0.05633934587240219,
                "width": 0.014412987977266312,
                "height": 0.1020975187420845
            },
            "confidence": 0.2701541781425476
        },
        {
            "class_id": 2,
            "bbox": {
                "x_center": 0.08580442517995834,
                "y_center": 0.08008135855197906,
                "width": 0.11641693860292435,
                "height": 0.16579416394233704
            },
            "confidence": 0.26923900842666626
        },
        {
            "class_id": 3,
            "bbox": {
                "x_center": 0.054593276232481,
                "y_center": 0.1736900955438614,
                "width": 0.0437728613615036,
                "height": 0.17468994855880737
            },
            "confidence": 0.27083590626716614
        },
        {
            "class_id": 4,
            "bbox": {
                "x_center": 0.08583462983369827,
                "y_center": 0.07386838644742966,
                "width": 0.02716866508126259,
                "height": 0.0804717019200325
            },
            "confidence": 0.27024248242378235
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
