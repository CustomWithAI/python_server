import os
import numpy as np
import os
import json
import subprocess
import yaml
import cv2
import shutil
from sklearn.metrics import accuracy_score
from skimage.io import imread
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical, load_img, img_to_array, Sequence
from tensorflow.keras import layers, models
import numpy as np
import os

from app.services.model.ml import MlModel
from app.services.model.dl_pretrained import DlModel
from app.services.model.construct_cls import ConstructDCLS
from app.services.model.construct_od import ConstructDLOD
mlmodel = MlModel()
dlmodel = DlModel()
constructdl_cls = ConstructDCLS()
constructdl_od = ConstructDLOD()


class MLTraining():
    def load_dataset(self, base_path):
        images = []
        labels = []
        class_names = os.listdir(base_path)
        class_names.sort()
        class_dict = {class_name: idx for idx,
                      class_name in enumerate(class_names)}

        expected_shape = None
        error_files = []

        for class_name in class_names:
            class_path = os.path.join(base_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    try:
                        img = imread(img_path)
                        if img.ndim == 2:  # Grayscale image
                            # Add channel dimension
                            img = np.expand_dims(img, axis=-1)
                        # Color image
                        elif img.ndim == 3 and img.shape[2] == 3:
                            pass  # Already has the correct shape
                        else:
                            raise ValueError(
                                f"Unsupported image dimensions: {img.shape}")

                        if expected_shape is None:
                            expected_shape = img.shape
                        elif img.shape != expected_shape:
                            raise ValueError(
                                f"Inconsistent shape for image {img_path}. Expected {expected_shape}, got {img.shape}")

                        img_flattened = img.flatten()
                        images.append(img_flattened)
                        labels.append(class_dict[class_name])

                    except Exception as e:
                        error_files.append((img_path, str(e)))
                        print(f"Error loading image {img_path}: {e}")

        if error_files:
            print(f"\nEncountered issues with {len(error_files)} files:")
            for error_file, error_message in error_files:
                print(f"- {error_file}: {error_message}")

        return np.array(images), np.array(labels), class_dict

    def training_ml(self, config):
        model = None
        # Load dataset
        X_train, y_train, class_dict = self.load_dataset('dataset/train')
        X_val, y_val, _ = self.load_dataset('dataset/valid')

        model = mlmodel.create_ml_model(config)

        model.fit(X_train, y_train)

        # Evaluate on validation data
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy}")

        # Print class mapping
        print("Class mapping:", class_dict)


class DLTrainingPretrained():
    def load_dataset_dl(self, base_path, class_dict=None):
        images = []
        labels = []
        class_names = os.listdir(base_path)
        # Exclude hidden files
        class_names = [
            name for name in class_names if not name.startswith('.')]
        class_names.sort()

        if class_dict is None:
            class_dict = {class_name: idx for idx,
                          class_name in enumerate(class_names)}

        input_shape = None  # Initialize input shape variable

        for class_name in class_names:
            class_path = os.path.join(base_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('png', 'jpg', 'jpeg')):  # Ensure only image files are processed
                        img_path = os.path.join(class_path, filename)
                        try:
                            # Load image without resizing
                            img = load_img(img_path)
                            # Convert image to numpy array
                            img_array = img_to_array(img)
                            images.append(img_array)
                            labels.append(class_dict[class_name])

                            # Set input shape based on the first image loaded
                            if input_shape is None:
                                input_shape = img_array.shape

                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

        images = np.array(images)
        labels = np.array(labels)

        # One-hot encode the labels
        labels = to_categorical(labels, num_classes=len(class_dict))

        return images, labels, class_dict, input_shape

    def update_data_yaml(self, data_yaml_path, label_txt_path):
        # Read the label.txt file
        with open(label_txt_path, 'r') as file:
            classes = file.read().splitlines()

        # Read the existing data.yaml
        with open(data_yaml_path, 'r') as file:
            data_yaml = yaml.safe_load(file)

        # Update the data.yaml content
        data_yaml['nc'] = len(classes)
        data_yaml['names'] = classes

        # Define the desired order and formatting
        updated_data = {
            'train': "/app/yolo_dataset/train/images",
            'val': "/app/yolo_dataset/valid/images",
            'test': "/app/yolo_dataset/test/images",
            'nc': data_yaml['nc'],
            'names': data_yaml['names']
        }

        # Write the updated content back to data.yaml with the desired format
        with open(data_yaml_path, 'w') as file:
            yaml.dump(updated_data, file,
                      default_flow_style=None, sort_keys=False)

        print(
            f"Updated {data_yaml_path} with {len(classes)} classes in the desired format.")

    def get_image_shape(self, folder_path):
        # Get a list of all files in the folder
        file_list = os.listdir(folder_path)

        # Filter to ensure only image files are considered
        image_files = [file for file in file_list if file.endswith(
            ('.png', '.jpg', '.jpeg'))]

        # Check if there are any images in the folder
        if image_files:
            # Pick the first image
            image_path = os.path.join(folder_path, image_files[0])

            # Load the image using OpenCV
            img = cv2.imread(image_path)

            # Return the shape of the image
            return img.shape[1]
        else:
            print("No image files found in the folder!")
            return None

    def reformat_dataset(self, source_dir, target_dir):
        # Define subdirectories in the source dataset and target format
        data_splits = ["train", "test", "valid"]
        subdirs = {"image": "images", "labels": "labels"}

        # Create target directory structure
        for split in data_splits:
            split_source_dir = os.path.join(source_dir, split)
            if not os.path.exists(split_source_dir):
                continue
            for subdir in subdirs.values():
                os.makedirs(os.path.join(
                    target_dir, split, subdir), exist_ok=True)

        # Organize files into the target directory structure
        for split in data_splits:
            split_source_dir = os.path.join(source_dir, split)
            if not os.path.exists(split_source_dir):
                continue
            split_target_dir = os.path.join(target_dir, split)

            for file in os.listdir(split_source_dir):
                source_file = os.path.join(split_source_dir, file)

                if file.endswith(".jpg"):
                    target_subdir = subdirs["image"]
                elif file.endswith(".txt") and not file.startswith("label"):
                    target_subdir = subdirs["labels"]
                else:
                    continue

                target_file = os.path.join(
                    split_target_dir, target_subdir, file)
                shutil.copy2(source_file, target_file)

        print("Dataset has been reformatted and cloned to:", target_dir)

    def train_yolo(self, config_model, config_training):
        # Clone dataset to the training folder
        self.reformat_dataset("dataset", "yolo_dataset")

        # Update data.yaml file
        self.update_data_yaml("./data.yaml", "dataset/train/label.txt")

        # Extract config training
        img_size = self.get_image_shape("./dataset/train/")
        batch_size = config_training[0]
        epochs = config_training[1]
        weight_size = config_training[2]

        if "yolov5" in config_model:
            # Training model
            command = f"yolo5_venv/bin/python ./app/services/model/yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data ./data.yaml --weights {weight_size} --cache"
            subprocess.run(command, shell=True, check=True)

            shutil.rmtree("./app/services/model/yolov5/runs/train")

        if "yolov8" in config_model:
            print("training yolov8")
            # TODO: Implement YOLOv8 training
            command = f"yolov8_venv/bin/yolo task=detect mode=train model={weight_size} data=./data.yaml epochs={epochs} imgsz={img_size} plots=True"
            subprocess.run(command, shell=True, check=True)

            shutil.rmtree("./runs/")

        if "yolov11" in config_model:
            # TODO: Implement YOLOv11 training
            command = f"yolov11_venv/bin/yolo task=detect mode=train model={weight_size} data=./data.yaml epochs={epochs} imgsz={img_size} plots=True"
            subprocess.run(command, shell=True, check=True)

            shutil.rmtree("./runs/")

        shutil.rmtree("yolo_dataset/train")
        shutil.rmtree("yolo_dataset/test")
        shutil.rmtree("yolo_dataset/valid")

    def train_cls(self, config_model, config_training):
        model = None

        # Load dataset
        X_train, y_train, class_dict, input_shape = self.load_dataset_dl(
            'dataset/train')
        X_val, y_val, _, _ = self.load_dataset_dl('dataset/valid', class_dict)

        num_classes = len(class_dict)  # Dynamically get number of classes

        # Create model
        model = dlmodel.create_dl_model(config_model, num_classes, input_shape)

        # Unpack training configuration
        learning_rate = config_training[0]
        learning_rate_scheduler = config_training[1]
        momentum = config_training[2]
        optimizer_type = config_training[3]
        batch_size = config_training[4]
        epochs = config_training[5]
        # Convert string to loss function
        loss_function = get_loss(config_training[6])

        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function,
                      metrics=['accuracy'])

        # Learning rate scheduler (optional)
        callbacks = []
        if learning_rate_scheduler:
            def scheduler(epoch, lr):
                return learning_rate_scheduler(epoch, lr)
            callbacks.append(LearningRateScheduler(scheduler))

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        return history


class ConstructTraining():
    def __init__(self):
        pass

    def load_dataset_cls(self, base_path, class_dict=None):
        images = []
        labels = []
        class_names = [name for name in os.listdir(
            base_path) if not name.startswith('.')]
        class_names.sort()

        if class_dict is None:
            class_dict = {class_name: idx for idx,
                          class_name in enumerate(class_names)}

        input_shape = None  # Will be determined based on the first image

        for class_name in class_names:
            class_path = os.path.join(base_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                        img_path = os.path.join(class_path, filename)
                        try:
                            img = load_img(img_path)
                            img_array = img_to_array(img)

                            if input_shape is None:
                                # Set input shape based on the first image
                                input_shape = img_array.shape
                            elif img_array.shape != input_shape:
                                # Resize image if it doesn't match the initial shape
                                # Resize based on height and width of input_shape
                                img = img.resize(
                                    (input_shape[1], input_shape[0]))
                                img_array = img_to_array(img)

                            images.append(img_array)
                            labels.append(class_dict[class_name])

                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

        images = np.array(images)
        labels = np.array(labels)

        labels = tf.keras.utils.to_categorical(
            labels, num_classes=len(class_dict))

        return images, labels, class_dict, input_shape

    def train_cls(self, config_model, config_training):
        # Load dataset
        X_train, y_train, class_dict, input_shape = self.load_dataset_cls(
            'dataset/train')
        X_val, y_val, _, _ = self.load_dataset_cls('dataset/valid', class_dict)

        num_classes = len(class_dict)

        # Create model
        model = constructdl_cls.construct(config_model, input_shape)

        # Unpack training configuration
        learning_rate = config_training[0]
        learning_rate_scheduler = config_training[1]
        momentum = config_training[2]
        optimizer_type = config_training[3]
        batch_size = config_training[4]
        epochs = config_training[5]
        loss_function = get_loss(config_training[6])

        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function,
                      metrics=['accuracy'])

        # Callbacks
        callbacks = []
        if learning_rate_scheduler:
            def scheduler(epoch, lr):
                return learning_rate_scheduler(epoch, lr)
            callbacks.append(LearningRateScheduler(scheduler))

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        return history

    def load_image_and_annotations(self, img_path, ann_path, input_size):
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, input_size)  # Resize to the required input size
        img = img.astype(np.float32) / 255.0  # Normalize the image

        # Load annotations (assuming YOLO format: class_id x_center y_center width height)
        with open(ann_path, 'r') as file:
            annotations = file.readlines()

        bboxes = []
        class_ids = []

        for ann in annotations:
            ann = ann.strip().split()
            class_id = int(ann[0])
            x_center, y_center, width, height = map(float, ann[1:])

            # Append to the lists
            bboxes.append([x_center, y_center, width, height])
            class_ids.append(class_id)

        return img, np.array(bboxes), np.array(class_ids)

    def load_dataset(self, dataset_dir, input_size):
        images = []
        bboxes = []
        class_ids = []

        for img_file in os.listdir(dataset_dir):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(dataset_dir, img_file)
                ann_path = os.path.splitext(img_path)[0] + '.txt'

                # Load the image and annotations
                img, bbox, class_id = self.load_image_and_annotations(
                    img_path, ann_path, input_size)

                images.append(img)
                bboxes.append(bbox)
                class_ids.append(class_id)

        class_ids = to_categorical(np.array(class_ids), num_classes=2)

        return np.array(images), np.array(bboxes), class_ids

    def train_od(self, config_model, config_training):
        # Unpack training configuration
        learning_rate = config_training[0]
        momentum = config_training[1]
        optimizer_type = config_training[2]
        batch_size = config_training[3]
        epochs = config_training[4]

        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Check image size
        folder_path = './dataset/train/'
        image_files = [f for f in os.listdir(
            folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        image_path = os.path.join(folder_path, image_files[0])
        img = cv2.imread(image_path)
        img_shape = img.shape
        input_shape = img_shape[:2]

        # check how many classes
        with open('./dataset/train/label.txt', 'r') as file:
            line_count = sum(1 for line in file)
        num_classes = line_count

        model = constructdl_od.construct(
            config_model, img_shape, num_classes)
        model.summary()

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss={
                'bbox_output': 'mse',  # Regression loss for bounding boxes
                'class_output': 'categorical_crossentropy'  # Classification loss
            },
            loss_weights={'bbox_output': 1.0, 'class_output': 1.0},
            metrics={'class_output': 'accuracy'}
        )

        # Load dataset for training and validation
        X_train, y_bboxes_train, y_classes_train = self.load_dataset(
            './dataset/train', input_shape)
        X_valid, y_bboxes_valid, y_classes_valid = self.load_dataset(
            './dataset/valid', input_shape)

        # Train with explicit validation data
        history = model.fit(
            X_train,
            {'bbox_output': y_bboxes_train, 'class_output': y_classes_train},
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                X_valid, {'bbox_output': y_bboxes_valid, 'class_output': y_classes_valid})
        )
