from PIL import Image
import os

# Root folder path
root_folder = '../dataset/'

# Supported image formats (add more if needed)
supported_extensions = ('.png', '.bmp', '.tiff', '.webp', '.gif', '.jpeg', '.jpg')

# Walk through all folders and files
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(dirpath, filename)

            try:
                with Image.open(file_path) as img:
                    # Convert to RGB (needed for JPG)
                    rgb_img = img.convert('RGB')

                    # New file name with .jpg extension
                    base_filename = os.path.splitext(filename)[0]
                    new_filename = base_filename + '.jpg'
                    new_file_path = os.path.join(dirpath, new_filename)

                    # Skip if already a jpg with same name
                    if file_path.lower() == new_file_path.lower():
                        continue

                    rgb_img.save(new_file_path, 'JPEG')
                    print(f'Converted: {file_path} -> {new_file_path}')
            except Exception as e:
                print(f'Failed to convert {file_path}: {e}')