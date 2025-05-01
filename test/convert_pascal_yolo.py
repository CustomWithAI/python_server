# import os
# import xml.etree.ElementTree as ET

# # Dictionary mapping class names to class IDs
# class_mapping = {
#     "cat": 0,
#     "dog": 1,
# }

# # Paths
# input_dir = "./dog_cat_test_dataset/annotations"  # folder with .xml files
# output_dir = "./dog_cat_test_dataset/txt"  # folder to save .txt files
# os.makedirs(output_dir, exist_ok=True)

# def convert_annotation(xml_file):
#     tree = ET.parse(xml_file)
#     root = tree.getroot()

#     image_width = int(root.find('size/width').text)
#     image_height = int(root.find('size/height').text)

#     txt_lines = []

#     for obj in root.findall('object'):
#         class_name = obj.find('name').text
#         class_id = class_mapping.get(class_name, -1)
#         if class_id == -1:
#             continue  # skip unknown classes

#         bndbox = obj.find('bndbox')
#         xmin = int(bndbox.find('xmin').text)
#         ymin = int(bndbox.find('ymin').text)
#         xmax = int(bndbox.find('xmax').text)
#         ymax = int(bndbox.find('ymax').text)

#         # YOLO format
#         x_center = (xmin + xmax) / 2.0 / image_width
#         y_center = (ymin + ymax) / 2.0 / image_height
#         width = (xmax - xmin) / image_width
#         height = (ymax - ymin) / image_height

#         txt_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

#     return txt_lines

# # Convert all .xml files
# for filename in os.listdir(input_dir):
#     if not filename.endswith('.xml'):
#         continue
#     xml_path = os.path.join(input_dir, filename)
#     txt_data = convert_annotation(xml_path)

#     txt_filename = os.path.splitext(filename)[0] + ".txt"
#     txt_path = os.path.join(output_dir, txt_filename)
#     with open(txt_path, "w") as f:
#         f.write("\n".join(txt_data))

# print("Conversion completed.")



import os
import random
import shutil

# Configuration
dataset_dir = "./dog_cat_test_dataset/images"  # Change this to your actual dataset folder
output_dir = "./dog_cat_test_dataset/splitted"  # Output directory
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output subfolders
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
all_images.sort()  # Ensure consistent order before shuffling

# Shuffle the dataset
random.seed(42)
random.shuffle(all_images)

# Split
total = len(all_images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

def copy_files(file_list, subset_name):
    for img_file in file_list:
        txt_file = img_file.replace('.png', '.txt')
        for f in [img_file, txt_file]:
            src = os.path.join(dataset_dir, f)
            dst = os.path.join(output_dir, subset_name, f)
            if os.path.exists(src):
                shutil.copy(src, dst)

# Copy files
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset successfully split into train, val, and test.")
