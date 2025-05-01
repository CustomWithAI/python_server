import os
import cv2
import requests
import random
import shutil

from PIL import Image
from urllib.parse import urlparse
from io import BytesIO
from typing import List
from collections import defaultdict
from app.services.dataset.preprocessing import Preprocessing
from app.services.dataset.augmentation import Augmentation
from app.models.preprocessing import ImagePreprocessingConfig
from app.models.augmentation import DataAugmentationConfig
from app.models.dataset import (
    PrepareDatasetRequest,
    ObjectDetectionPlot,
    SegmentationPlot,
    ObjectDetectionPredict,
    SegmentationPredict,
)

preprocess = Preprocessing()
augmentation = Augmentation()

def preprocess_all_dataset(dataset_dir: str, config_preprocess: ImagePreprocessingConfig):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    image = cv2.imread(file_path)
                    if image is not None:
                        # Preprocess the image
                        image = preprocess.preprocess(image, config_preprocess)
                        # Save the resized image back to the same file path
                        cv2.imwrite(file_path, image)
                        # print(f"Saved: {file_path}")
                    else:
                        print(f"Failed to read image: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")


def augment_class_images(class_path: str, target_number_per_class: int, config_augmentation: DataAugmentationConfig):
    # Collect only original images before augmentation
    images = [os.path.join(class_path, f) for f in os.listdir(
        class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Copy the list to avoid modifying it during augmentation
    original_images = list(images)
    augmented_images = []  # To store newly created augmented images

    current_number = len(images)
    more_aug_needed = target_number_per_class - current_number

    for _ in range(more_aug_needed):
        # Select from original images only
        random_image_path = random.choice(original_images)
        image = cv2.imread(random_image_path)
        if image is not None:
            augmented_img = augmentation.augmentation_classification(
                image, config_augmentation)  # Perform your augmentation
            # Generate a new filename for the augmented image
            new_filename = f"aug_{random.randint(0, 1e6)}.jpg"
            new_file_path = os.path.join(class_path, new_filename)
            cv2.imwrite(new_file_path, augmented_img)
            # Keep track of the new augmented image
            augmented_images.append(new_file_path)
            print(f"Augmented image saved to: {new_file_path}")

    # Append newly augmented images to the original list
    images.extend(augmented_images)


def augment_dataset_class(training_path: str, config_augmentation: DataAugmentationConfig):
    # List all class directories
    class_dirs = [d for d in os.listdir(
        training_path) if os.path.isdir(os.path.join(training_path, d))]
    number_of_classes = len(class_dirs)

    # Calculate target number per class
    target_number_per_class = config_augmentation.number // number_of_classes
    for class_name in class_dirs:
        class_path = os.path.join(training_path, class_name)
        augment_class_images(
            class_path, target_number_per_class, config_augmentation)


def augment_dataset_obj(folder_path: str, config_augmentation: DataAugmentationConfig):
    # Get all images and corresponding .txt files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    txt_files = [f.replace('.jpg', '.txt') for f in image_files]

    # Group images by class based on the content of their .txt files
    class_to_images = defaultdict(list)
    for image_file, txt_file in zip(image_files, txt_files):
        txt_path = os.path.join(folder_path, txt_file)
        with open(txt_path, 'r') as file:
            # Assuming each line in the .txt file starts with the class label
            class_label = file.readline().split()[0]
        class_to_images[class_label].append((image_file, txt_file))

    total_current_images = len(image_files)
    num_augmentations_needed = config_augmentation.number - total_current_images
    if num_augmentations_needed <= 0:
        print("Target number of images already met or exceeded.")
        return

    # Calculate number of augmentations needed per class
    augmentations_per_class = {}
    total_classes = len(class_to_images)
    for class_label, images in class_to_images.items():
        current_count = len(images)
        augmentations_per_class[class_label] = max(
            0, int(num_augmentations_needed * (current_count / total_current_images)))

    # Perform augmentations
    for class_label, images in class_to_images.items():
        needed_augmentations = augmentations_per_class[class_label]
        for _ in range(needed_augmentations):
            # Randomly select an image and its corresponding .txt file
            image_file, txt_file = random.choice(images)

            # Load the image
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            # Load the bounding boxes from the .txt file
            txt_path = os.path.join(folder_path, txt_file)
            bounding_box = []

            with open(txt_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    values = line.strip().split()
                    class_id = int(values[0])
                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])
                    bounding_box.append(
                        (class_id, x_center, y_center, width, height))

            # Apply augmentation
            augmented_img, adjusted_bounding_box = augmentation.augmentation_object_detection(
                image, config_augmentation, bounding_box)

            # Generate a unique suffix
            unique_suffix = random.randint(1000, 9999)

            # Save the augmented image
            augmented_image_path = os.path.join(
                folder_path, f"aug_{unique_suffix}_{image_file}")
            cv2.imwrite(augmented_image_path, augmented_img)

            # Save the adjusted bounding boxes back to a .txt file with the same name as the image
            adjusted_txt_path = os.path.join(
                folder_path, f"aug_{unique_suffix}_{image_file}.txt")
            with open(adjusted_txt_path, 'w') as file:
                for box in adjusted_bounding_box:
                    class_id, x_center, y_center, width, height = box
                    file.write(
                        f"{class_id} {x_center} {y_center} {width} {height}\n")


def augment_dataset_seg(folder_path: str, config_augmentation: DataAugmentationConfig):
    # Get all images and corresponding .txt files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    txt_files = [f.replace('.jpg', '.txt') for f in image_files]

    # Group images by class based on the content of their .txt files
    class_to_images = defaultdict(list)
    for image_file, txt_file in zip(image_files, txt_files):
        txt_path = os.path.join(folder_path, txt_file)
        with open(txt_path, 'r') as file:
            # Get first value in the first line
            class_label = file.readline().split()[0]
        class_to_images[class_label].append((image_file, txt_file))

    total_current_images = len(image_files)
    num_augmentations_needed = config_augmentation.number - total_current_images
    if num_augmentations_needed <= 0:
        print("Target number of images already met or exceeded.")
        return

    # Calculate number of augmentations needed per class
    augmentations_per_class = {}
    total_classes = len(class_to_images)
    for class_label, images in class_to_images.items():
        current_count = len(images)
        augmentations_per_class[class_label] = max(
            0, int(num_augmentations_needed * (current_count / total_current_images)))

    # Perform augmentations
    for class_label, images in class_to_images.items():
        needed_augmentations = augmentations_per_class[class_label]
        for _ in range(needed_augmentations):

            # Randomly choose an image
            image_file, txt_file = random.choice(images)
            # Load the image
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            # Load the bounding boxes from the .txt file (if it exists)
            txt_path = os.path.join(folder_path, txt_file)
            if os.path.exists(txt_path):
                with open(txt_path, "r") as src:
                    content = src.read()  # Read the entire file

            # Apply augmentation
            augmented_img = augmentation.augmentation_classification(
                image, config_augmentation)

            # Generate a unique suffix
            unique_suffix = random.randint(1000, 9999)

            # Save the augmented image
            augmented_image_path = os.path.join(
                folder_path, f"aug_{unique_suffix}_{image_file}")
            cv2.imwrite(augmented_image_path, augmented_img)

            # Save bounding box file with the same name and location
            adjusted_txt_path = os.path.join(
                folder_path, f"aug_{unique_suffix}_{txt_file}")
            with open(adjusted_txt_path, 'w') as file:
                file.write(content)  # Corrected to use 'file.write'

def normalize(value: int, max_value: int) -> float:
    return value / max_value

def reverse_normalize(value: float, max_value: int) -> float:
    return value * max_value

def convert_object_detection(annotation: List[ObjectDetectionPlot], labels: List[str], img_width: int, img_height: int) -> str:
    result = []
    for obj in annotation:
        class_idx = labels.index(obj.label)
        x_center = normalize(obj.x + obj.width / 2, img_width)
        y_center = normalize(obj.y + obj.height / 2, img_height)
        width_norm = normalize(obj.width, img_width)
        height_norm = normalize(obj.height, img_height)
        result.append(f"{class_idx} {x_center} {y_center} {width_norm} {height_norm}")
    return "\n".join(result)

def reverse_convert_object_detection(annotation: List[ObjectDetectionPredict], img_width: int, img_height: int):
    result = []
    for obj in annotation:
        bbox = obj.get("bbox")
        
        if not bbox:
            continue

        x_center = reverse_normalize(bbox.get("x_center", 0), img_width)
        y_center = reverse_normalize(bbox.get("y_center", 0), img_height)
        width = reverse_normalize(bbox.get("width", 0), img_width)
        height = reverse_normalize(bbox.get("height", 0), img_height)
        
        x = x_center - width / 2
        y = y_center - height / 2
        
        data = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "label": obj.get("class_id"),
        }
        
        if "confidence" in obj:
            data["confidence"] = obj["confidence"]
        
        result.append(data)
    return result

def convert_segmentation(annotation: List[SegmentationPlot], labels: List[str], img_width: int, img_height: int) -> str:
    result = []
    for obj in annotation:
        class_idx = labels.index(obj.label)
        points = [(normalize(p.x, img_width), normalize(p.y, img_height)) for p in obj.points]
        points.append(points[0])
        point_str = " ".join(f"{x} {y}" for x, y in points)
        result.append(f"{class_idx} {point_str}")
    return "\n".join(result)

def reverse_convert_segmentation(annotation: List[SegmentationPredict], img_width: int, img_height: int):
    result = []
    for obj in annotation:
        points = [
            {
                "x": reverse_normalize(p[0], img_width),
                "y": reverse_normalize(p[1], img_height)
            }
            for p in obj.get("polygon")
        ]
        
        data = {
            "label": obj.get("class_id"),
            "points": [{"x": p.get("x"), "y": p.get("y")}for p in points]
        }
        
        result.append(data)
    
    return result

def prepare_dataset(request: PrepareDatasetRequest):
    base_dir = "dataset"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "valid"), exist_ok=True)

    datasets = {
        "train": request.train_data,
        "test": request.test_data,
        "valid": request.valid_data
    }
    
    # sorted_labels = sorted(request.labels)
    sorted_labels = request.labels

    if request.type in ["object_detection", "segmentation"]:
        label_content = "\n".join(sorted_labels)
        for split in datasets.keys():
            label_path = os.path.join(base_dir, split, "label.txt")
            with open(label_path, "w") as f:
                f.write(label_content)

    for split, images in datasets.items():
        split_dir = os.path.join(base_dir, split)

        for img in images:
            parsed_url = urlparse(img.url)
            image_filename = os.path.basename(parsed_url.path)

            response = requests.get(img.url, stream=True)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                img_width, img_height = image.size

                if request.type == "classification":
                    class_folder = os.path.join(split_dir, img.annotation.label)
                    os.makedirs(class_folder, exist_ok=True)
                    image_path = os.path.join(class_folder, image_filename)
                else:
                    image_path = os.path.join(split_dir, image_filename)

                with open(image_path, 'wb') as file:
                    file.write(response.content)

                if request.type == "object_detection":
                    annotation_text = convert_object_detection(
                        img.annotation.annotation, sorted_labels, img_width, img_height
                    )
                elif request.type == "segmentation":
                    annotation_text = convert_segmentation(
                        img.annotation.annotation, sorted_labels, img_width, img_height
                    )
                else:
                    continue

                annotation_path = image_path.replace(".png", ".txt").replace(".jpg", ".txt")
                with open(annotation_path, "w") as f:
                    f.write(annotation_text)