import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_segmentation_file(txt_file):
    with open(txt_file, "r") as file:
        lines = file.readlines()

    points = []
    for line in lines:
        values = line.strip().split()
        if len(values) > 1:  # Ignore empty lines
            # Skip the first value (class index)
            coords = list(map(float, values[1:]))
            points.append([(coords[i] * img_width, coords[i + 1] * img_height)
                          for i in range(0, len(coords), 2)])
    return points


def plot_segmentation(image_path, txt_path):
    global img_height, img_width

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]

    # Read segmentation points
    segmentations = read_segmentation_file(txt_path)

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot the segmentation mask
    for polygon in segmentations:
        polygon = np.array(polygon, np.int32)
        plt.plot(polygon[:, 0], polygon[:, 1], marker='o',
                 linestyle='-', color='red', linewidth=2)

    plt.axis("off")
    plt.show()


# Example usage
image_path = "sample2.jpg"  # Replace with your image file
txt_path = "sample2.txt"  # Replace with your segmentation TXT file
plot_segmentation(image_path, txt_path)
