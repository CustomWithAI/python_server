import os
from PIL import Image

def resize_images_in_directory(base_dir, target_size=(64, 64)):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Include other formats if necessary
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img_resized = img.resize(target_size)
                        img_resized.save(file_path)  # Save back to the same location
                        print(f"Resized and saved: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

# Example usage
resize_images_in_directory('dataset/test')
