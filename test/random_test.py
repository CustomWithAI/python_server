import cv2

def plot_bounding_box(image_path, txt_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read the TXT file
    with open(txt_path, 'r') as file:
        for line in file:
            # Parse the line
            class_id, x_center, y_center, box_width, box_height = map(float, line.split())
            
            # Convert from normalized to absolute coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Calculate the top-left corner of the bounding box
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Put the class label
            label = f'Class {int(class_id)}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)

# Example usage
# plot_bounding_box('img_before.jpg', 'img_before.txt', 'output_image.jpg')
plot_bounding_box('imga.jpg', 'imga.txt', 'output_image_after.jpg')
