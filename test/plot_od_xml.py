import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def plot_annotation(image_path, annotation_path):
    # Load image
    image = Image.open(image_path)
    
    # Parse XML annotation
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Create a plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Loop through each object in the annotation
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Create a rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, name, color='red', fontsize=12, backgroundcolor='white')

    plt.axis('off')
    plt.show()

# Example usage
image_path = './Cats_Test0.png'
annotation_path = './Cats_Test0.xml'  # Adjust this path to where your XML is located
plot_annotation(image_path, annotation_path)
