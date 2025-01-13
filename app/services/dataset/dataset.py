import os
import cv2
import random

from app.services.dataset.preprocessing import Preprocessing
from app.services.dataset.augmentation import Augmentation
preprocess = Preprocessing()
augmentation = Augmentation()

def preprocess_all_dataset(dataset_dir,config_preprocess):
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
                    
def augment_class_images(class_path, target_number_per_class, config_augmentation):
    # Collect only original images before augmentation
    images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_images = list(images)  # Copy the list to avoid modifying it during augmentation
    augmented_images = []  # To store newly created augmented images

    current_number = len(images)
    more_aug_needed = target_number_per_class - current_number

    for _ in range(more_aug_needed):
        random_image_path = random.choice(original_images)  # Select from original images only
        image = cv2.imread(random_image_path)
        if image is not None:
            augmented_img = augmentation.augmentation(image, config_augmentation)  # Perform your augmentation
            # Generate a new filename for the augmented image
            new_filename = f"aug_{random.randint(0, 1e6)}.jpg"
            new_file_path = os.path.join(class_path, new_filename)
            cv2.imwrite(new_file_path, augmented_img)
            augmented_images.append(new_file_path)  # Keep track of the new augmented image
            print(f"Augmented image saved to: {new_file_path}")

    # Append newly augmented images to the original list
    images.extend(augmented_images)

def augment_dataset(training_path, total_target_number, config_augmentation):
    # List all class directories
    class_dirs = [d for d in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, d))]
    number_of_classes = len(class_dirs)
    
    # Calculate target number per class
    target_number_per_class = total_target_number // number_of_classes
    for class_name in class_dirs:
        class_path = os.path.join(training_path, class_name)
        augment_class_images(class_path, target_number_per_class, config_augmentation)
