from skimage.feature import hog
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
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
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from app.services.model.ml import MlModel
from app.services.model.dl_pretrained import DlModel
from app.services.model.construct_cls import ConstructDLCLS
from app.services.model.construct_od import ConstructDLOD
from app.services.dataset.featextraction import FeatureExtraction
from app.models.ml import MachineLearningClassificationRequest
from app.models.dl import (
    DeepLearningClassification,
    DeepLearningYoloRequest,
    DeepLearningClassificationConstruct,
    DeepLearningObjectDetectionConstruct,
    DeepLearningObjectDetectionConstructFeatex,
)

mlmodel = MlModel()
dlmodel = DlModel()
constructdl_cls = ConstructDLCLS()
constructdl_od = ConstructDLOD()
feature_extractor = FeatureExtraction()


class MLTraining():
    def load_dataset(self, base_path: str):
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

    def check_min_feat(self, folder_path, hog_params=None, sift_params=None, orb_params=None):
        hog_count = []
        sift_count = []
        orb_count = []

        # Check maximum feature
        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if not os.path.isdir(label_path):
                continue

            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if file_path.endswith(".DS_Store"):
                    continue

                image = cv2.imread(file_path)
                if image is None:
                    continue

                # Extract features only if params are provided

                if hog_params:  # Check if HOG config exists
                    hog_features = feature_extractor.extract_hog_features(
                        image, **hog_params, count_feat=False)
                    # print("HOG FEAT LEN:", hog_features.shape)
                    hog_count.append(len(hog_features))

                if sift_params:  # Check if SIFT config exists
                    sift_features = feature_extractor.extract_sift_features(
                        image, **sift_params, count_feat=False)
                    # print("SIFT FEAT LEN:", len(sift_features))
                    sift_count.append(len(sift_features))

                if orb_params:  # Check if ORB config exists
                    orb_features = feature_extractor.extract_orb_features(
                        image, **orb_params, count_feat=False)
                    # print("ORB FEAT LEN:", len(orb_features))
                    orb_count.append(len(orb_features))

        if hog_params:
            hog_min = min(hog_count)
        else:
            hog_min = 0

        if sift_params:
            sift_min = min(sift_count)
        else:
            sift_min = 0

        if orb_params:
            orb_min = min(orb_count)
        else:
            orb_min = 0
        print("MIN FEAT:", hog_min, sift_min, orb_min)
        return hog_min, sift_min, orb_min

    def load_dataset_featex(self, folder_path, hog_min, sift_min, orb_min, hog_params=None, sift_params=None, orb_params=None):
        X, y = [], []

        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if not os.path.isdir(label_path):
                continue

            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if file_path.endswith(".DS_Store"):
                    continue

                image = cv2.imread(file_path)
                if image is None:
                    continue

                # Extract features only if params are provided
                feature_vector = []

                if hog_params:  # Check if HOG config exists
                    hog_features = feature_extractor.extract_hog_features(
                        image, **hog_params, max_features=hog_min)
                    feature_vector.append(hog_features)
                    # print("HOG FEAT LEN:", len(hog_features))

                if sift_params:  # Check if SIFT config exists
                    sift_features = feature_extractor.extract_sift_features(
                        image, **sift_params, max_features=sift_min)
                    feature_vector.append(sift_features)
                    # print("SIFT FEAT LEN:", len(sift_features))

                if orb_params:  # Check if ORB config exists
                    orb_features = feature_extractor.extract_orb_features(
                        image, **orb_params, max_features=orb_min)
                    feature_vector.append(orb_features)
                    # print("ORB FEAT LEN:", len(orb_features))

                # Flatten the feature vector if any features were extracted
                if feature_vector:
                    feature_vector = np.hstack(feature_vector)
                    # print("feature Vector:", feature_vector.shape)
                    X.append(feature_vector)
                    y.append(label)
                else:
                    print(
                        f"Skipping {file_path}, no feature extraction methods enabled.")

        return np.array(X), np.array(y)

    def training_ml_cls(self, config: MachineLearningClassificationRequest):
        model = None
        # Load dataset
        if not config.featex:
            X_train, y_train, class_dict = self.load_dataset('dataset/train')
            X_val, y_val, _ = self.load_dataset('dataset/valid')
        else:
            config_featex = config.featex
            hog_params = config_featex.hog.model_dump() if config_featex.hog else None
            sift_params = config_featex.sift.model_dump() if config_featex.sift else None
            orb_params = config_featex.orb.model_dump() if config_featex.orb else None

            # Find min Features
            hog_min_train, sift_min_train, orb_min_train = self.check_min_feat(
                'dataset/train', hog_params, sift_params, orb_params)
            hog_min_val, sift_min_val, orb_min_val = self.check_min_feat(
                'dataset/valid', hog_params, sift_params, orb_params)

            hog_min = min(hog_min_train, hog_min_val)
            sift_min = min(sift_min_train, sift_min_val)
            orb_min = min(orb_min_train, orb_min_val)

            # Load datasets
            X_train, y_train = self.load_dataset_featex(
                'dataset/train', hog_min, sift_min, orb_min, hog_params, sift_params, orb_params)
            X_val, y_val = self.load_dataset_featex(
                'dataset/valid', hog_min, sift_min, orb_min, hog_params, sift_params, orb_params)

            # Encode labels
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_val = encoder.transform(y_val)
            print(
                f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

        model = mlmodel.create_ml_model(config.model)
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, "ml_model.pkl")

        # Create output directory
        os.makedirs("evaluation_results", exist_ok=True)

        # Evaluate on validation data
        y_val_pred = model.predict(X_val)

        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_val, y_val_pred)

        # Print scores
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"F1 Score: {val_f1:.4f}")

        # Save metrics to a text file
        with open("evaluation_results/metrics.txt", "w") as f:
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Precision: {val_precision:.4f}\n")
            f.write(f"Recall: {val_recall:.4f}\n")
            f.write(f"F1 Score: {val_f1:.4f}\n")

        # Save Confusion Matrix as an image
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig("evaluation_results/confusion_matrix.png")
        plt.close()

        # Save loss curve if available
        if hasattr(model, "loss_curve_"):
            plt.figure()
            plt.plot(model.loss_curve_)
            plt.title("Loss Curve (Per Epoch)")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig("evaluation_results/loss_curve.png")
            plt.close()

        # Save validation scores curve if available
        if hasattr(model, "validation_scores_"):
            plt.figure()
            plt.plot(model.validation_scores_)
            plt.title("Validation Accuracy (Per Epoch)")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig("evaluation_results/accuracy_per_epoch.png")
            plt.close()

class DLTrainingPretrained():

    def load_dataset_dl(self, base_path, class_dict=None):
        images = []
        labels = []

        # Find class dict
        class_names = os.listdir(base_path)
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
                    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                        img_path = os.path.join(class_path, filename)
                        try:
                            # Load image
                            img = load_img(img_path)
                            img_array = img_to_array(img)

                            # Set input shape based on the first image loaded
                            if input_shape is None:
                                # (height, width, channels)
                                input_shape = img_array.shape

                            # Ensure all images match input_shape
                            if img_array.shape != input_shape:
                                print(
                                    f"Skipping {img_path} due to mismatched shape: {img_array.shape}")
                                continue

                            images.append(img_array)
                            labels.append(class_dict[class_name])

                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

        images = np.array(images, dtype=np.float32)  # Ensure consistent dtype
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

    def reformat_dataset(self, source_dir: str, target_dir: str):
        data_splits = ["train", "test", "valid"]
        subdirs = {"image": "images", "labels": "labels"}

        # Create target directory structure
        for split in data_splits:
            split_source_dir = os.path.join(source_dir, split)
            if not os.path.exists(split_source_dir):
                continue
            for subdir in subdirs.values():
                os.makedirs(os.path.join(target_dir, split, subdir), exist_ok=True)

        # Organize files safely into target directory
        for split in data_splits:
            split_source_dir = os.path.join(source_dir, split)
            if not os.path.exists(split_source_dir):
                continue
            split_target_dir = os.path.join(target_dir, split)

            for file in os.listdir(split_source_dir):
                source_file = os.path.join(split_source_dir, file)

                if file.endswith((".jpg", ".png")):
                    target_subdir = subdirs["image"]
                elif file.endswith(".txt") and not file.startswith("label"):
                    target_subdir = subdirs["labels"]
                else:
                    continue

                target_file = os.path.join(split_target_dir, target_subdir, file)

                try:
                    Path(target_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(source_file, 'rb') as src, open(target_file, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                except Exception as e:
                    print(f"Error copying {source_file} to {target_file}: {e}")

        print("✅ Dataset has been reformatted and copied to:", target_dir)

    def train_yolo(self, config: DeepLearningYoloRequest):
        # Clone dataset to the training folder
        self.reformat_dataset("dataset", "yolo_dataset")

        # Update data.yaml file
        self.update_data_yaml("./data.yaml", "dataset/train/label.txt")

        # Extract config training
        img_size = self.get_image_shape("./dataset/train/")
        print("IMG SHAPE", img_size)

        config_training = config.training
        batch_size = config_training.batch_size
        epochs = config_training.epochs
        weight_size = config_training.weight_size

        if "yolov5" in config.model:
            # Training model
            command = f"yolo5_venv/bin/python ./app/services/model/yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data ./data.yaml --weights {weight_size} --cache"
            subprocess.run(command, shell=True, check=True)

            # shutil.rmtree("./app/services/model/yolov5/runs/train", ignore_errors = True)

        if "yolov8" in config.model:
            print("training yolov8")
            # TODO: Implement YOLOv8 training
            command = f"yolov8_venv/bin/yolo task=detect mode=train model={weight_size} data=./data.yaml epochs={epochs} imgsz={img_size} plots=True"
            subprocess.run(command, shell=True, check=True)

            # shutil.rmtree("./runs/", ignore_errors=True)

        if "yolov11" in config.model:
            # TODO: Implement YOLOv11 training
            command = f"yolov11_venv/bin/yolo task=detect mode=train model={weight_size} data=./data.yaml epochs={epochs} imgsz={img_size} plots=True"
            subprocess.run(command, shell=True, check=True)

            # shutil.rmtree("./runs/", ignore_errors=True)
        
        for path in ["train", "test", "valid"]:
            shutil.rmtree(f"./yolo_dataset/{path}", ignore_errors=True)

    def train_cls(self, config: DeepLearningClassification):
        model = None

        # Load dataset
        X_train, y_train, class_dict, input_shape = self.load_dataset_dl(
            'dataset/train')
        X_val, y_val, _, _ = self.load_dataset_dl(
            'dataset/valid')
        X_test, y_test, _, _ = self.load_dataset_dl(
            'dataset/test', class_dict)
        num_classes = len(class_dict)

        # Unpack training configuration
        config_training = config.training

        # Create model
        model = dlmodel.create_dl_model(config.model, num_classes, input_shape, config_training.unfreeze)

        # Convert string to loss function
        loss_function = get_loss(config_training.loss_function)

        # Select optimizer
        if config_training.optimizer_type == 'adam':
            optimizer = Adam(learning_rate=config_training.learning_rate)
        elif config_training.optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=config_training.learning_rate,
                            momentum=config_training.momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function,
                      metrics=['accuracy'])

        # Learning rate scheduler (optional)
        callbacks = []
        if config_training.learning_rate_scheduler:
            def scheduler(epoch, lr):
                return config_training.learning_rate_scheduler(epoch, lr)
            callbacks.append(LearningRateScheduler(scheduler))

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=config_training.epochs, batch_size=config_training.batch_size, callbacks=callbacks)
        
        # Save the model
        model.save("model.h5")

        # Evaluate on validation set
        y_pred_probs = model.predict(X_test)
        y_pred = y_pred_probs.argmax(axis=1)
        y_true = y_test.argmax(axis=1)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print("\n📊 Evaluation Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Save metrics to a text file
        with open("evaluation_results/metrics.txt", "w") as f:
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            
        # Create evaluation result folder if it doesn't exist
        result_dir = "./evaluation_results"
        os.makedirs(result_dir, exist_ok=True)

        # Plot and save training accuracy and loss
        def save_training_plots(history, result_dir):
            plt.figure(figsize=(12, 5))

            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "accuracy_loss_per_epoch.png"))
            plt.close()

        save_training_plots(history, result_dir)

        # Confusion matrix
        def save_confusion_matrix(y_true, y_pred, class_dict, result_dir):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=list(class_dict.keys()),
                        yticklabels=list(class_dict.keys()))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
            plt.close()

        save_confusion_matrix(y_true, y_pred, class_dict, result_dir)

        return history

class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_bboxes_val, y_classes_val, num_classes, iou_threshold=0.5):
        self.X_val = X_val
        self.y_bboxes_val = y_bboxes_val
        self.y_classes_val = y_classes_val
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.epoch_metrics = {'precision': [], 'recall': [], 'map': []}

    def compute_iou(self,box1, box2):
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area else 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val)
        y_bboxes_pred = preds[0]
        y_classes_pred = preds[1]

        y_true_classes_flat = np.argmax(self.y_classes_val.reshape(-1, self.num_classes), axis=1)
        y_pred_classes_flat = np.argmax(y_classes_pred.reshape(-1, self.num_classes), axis=1)

        mask_true = np.sum(self.y_classes_val, axis=-1).flatten() > 0
        mask_pred = np.sum(y_classes_pred, axis=-1).flatten() > 0

        precision = precision_score(y_true_classes_flat[mask_true], y_pred_classes_flat[mask_true], average='macro', zero_division=0)
        recall = recall_score(y_true_classes_flat[mask_true], y_pred_classes_flat[mask_true], average='macro', zero_division=0)

        ious = []
        average_precisions = []
        for i in range(len(self.X_val)):
            true_boxes = self.y_bboxes_val[i]
            pred_boxes = y_bboxes_pred[i]
            for j in range(len(true_boxes)):
                if np.sum(self.y_classes_val[i][j]) > 0:
                    best_iou = 0
                    for k in range(len(pred_boxes)):
                        if np.sum(y_classes_pred[i][k]) > 0:
                            iou = self.compute_iou(true_boxes[j], pred_boxes[k])
                            best_iou = max(best_iou, iou)
                    ious.append(best_iou)
                    average_precisions.append(1 if best_iou >= self.iou_threshold else 0)

        map_50 = np.mean(average_precisions) if average_precisions else 0

        self.epoch_metrics['precision'].append(precision)
        self.epoch_metrics['recall'].append(recall)
        self.epoch_metrics['map'].append(map_50)

        print(f"Epoch {epoch + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, mAP@0.5: {map_50:.4f}")


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

    def train_cls(self, config: DeepLearningClassificationConstruct):
        model = None
        # Load dataset
        X_train, y_train, class_dict, input_shape = self.load_dataset_cls(
            'dataset/train')
        X_val, y_val, _, _ = self.load_dataset_cls(
            'dataset/valid', class_dict)
        X_test, y_test, _, _ = self.load_dataset_cls(
            'dataset/test', class_dict)

        print("Input Shape", input_shape)
        print("Class Dict", class_dict)

        # Create model
        model = constructdl_cls.construct(config.model, input_shape)

        # Unpack training configuration
        config_training = config.training

        loss_function = get_loss(config_training.loss_function)

        # Select optimizer
        if config_training.optimizer_type == 'adam':
            optimizer = Adam(learning_rate=config_training.learning_rate)
        elif config_training.optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=config_training.learning_rate,
                            momentum=config_training.momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function,
                      metrics=['accuracy'])

        # Callbacks
        callbacks = []
        if config_training.learning_rate_scheduler:
            def scheduler(epoch, lr):
                return config_training.learning_rate_scheduler(epoch, lr)
            callbacks.append(LearningRateScheduler(scheduler))

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=config_training.epochs, batch_size=config_training.batch_size, callbacks=callbacks)

        model.save("model.h5")

        # Evaluate on validation set
        y_pred_probs = model.predict(X_test)
        y_pred = y_pred_probs.argmax(axis=1)
        y_true = y_test.argmax(axis=1)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print("\n📊 Evaluation Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Save metrics to a text file
        with open("evaluation_results/metrics.txt", "w") as f:
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            
        # Create evaluation result folder if it doesn't exist
        result_dir = "./evaluation_results"
        os.makedirs(result_dir, exist_ok=True)

        # Plot and save training accuracy and loss
        def save_training_plots(history, result_dir):
            plt.figure(figsize=(12, 5))

            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "accuracy_loss_per_epoch.png"))
            plt.close()

        save_training_plots(history, result_dir)

        # Confusion matrix
        def save_confusion_matrix(y_true, y_pred, class_dict, result_dir):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=list(class_dict.keys()),
                        yticklabels=list(class_dict.keys()))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
            plt.close()

        save_confusion_matrix(y_true, y_pred, class_dict, result_dir)

        return history

    def load_dataset_cls_featex(self, dataset_path):
        print(f"Dataset Path Exists: {os.path.exists(dataset_path)}")
        classes = [cls for cls in os.listdir(
            dataset_path) if not cls.startswith('.')]
        print(f"Classes Found: {classes}")

        X, y = [], []
        for label in classes:
            class_path = os.path.join(dataset_path, label)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"⚠️ Skipping: {img_path} (Image not loaded)")
                    continue

                X.append(img)
                y.append(label)

        print(f"Total Samples Loaded from {dataset_path}: {len(y)}")
        return X, y



    def train_cls_featex(self, config: DeepLearningClassificationConstruct):
        print("DOING FEATEX")
        model = None
        config_featex = config.featex
        hog_params = config_featex.hog.model_dump() if config_featex.hog else None
        sift_params = config_featex.sift.model_dump() if config_featex.sift else None
        orb_params = config_featex.orb.model_dump() if config_featex.orb else None

        # Find min Features
        hog_min_train, sift_min_train, orb_min_train = MLTraining().check_min_feat(
            'dataset/train', hog_params, sift_params, orb_params)
        hog_min_val, sift_min_val, orb_min_val = MLTraining().check_min_feat(
            'dataset/valid', hog_params, sift_params, orb_params)

        hog_min = min(hog_min_train, hog_min_val)
        sift_min = min(sift_min_train, sift_min_val)
        orb_min = min(orb_min_train, orb_min_val)

        print("MIN FEAT", hog_min, sift_min, orb_min)

        # Load datasets
        X_train, y_train = MLTraining().load_dataset_featex(
            'dataset/train', hog_min, sift_min, orb_min, hog_params, sift_params, orb_params)
        X_val, y_val = MLTraining().load_dataset_featex(
            'dataset/valid', hog_min, sift_min, orb_min, hog_params, sift_params, orb_params)
        X_test, y_test = MLTraining().load_dataset_featex(
            'dataset/test', hog_min, sift_min, orb_min, hog_params, sift_params, orb_params)

        # Encode labels
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_val = encoder.transform(y_val)

        # Convert labels to one-hot encoding
        num_classes = len(set(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)

        print(
            f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

        input_shape = ((hog_min+sift_min+orb_min),)

        # Create model
        model = constructdl_cls.construct(config.model, input_shape)

        # Unpack training configuration
        config_training = config.training

        loss_function = get_loss(config_training.loss_function)

        # Select optimizer
        if config_training.optimizer_type == 'adam':
            optimizer = Adam(learning_rate=config_training.learning_rate)
        elif config_training.optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=config_training.learning_rate,
                            momentum=config_training.momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function,
                      metrics=['accuracy'])

        # Callbacks
        callbacks = []
        if config_training.learning_rate_scheduler:
            def scheduler(epoch, lr):
                return config_training.learning_rate_scheduler(epoch, lr)
            callbacks.append(LearningRateScheduler(scheduler))

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=config_training.epochs, batch_size=config_training.batch_size, callbacks=callbacks)
        
        model.save("model.h5")

        # Evaluate
        y_pred_probs = model.predict(X_test)
        y_pred = y_pred_probs.argmax(axis=1)

        y_val_classes = y_test.argmax(axis=1)

        acc = accuracy_score(y_val_classes, y_pred)
        precision = precision_score(y_val_classes, y_pred, average='macro')
        recall = recall_score(y_val_classes, y_pred, average='macro')
        f1 = f1_score(y_val_classes, y_pred, average='macro')

        print("\n📊 Evaluation Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Save metrics to a text file
        with open("evaluation_results/metrics.txt", "w") as f:
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            
        # Save graphs
        result_dir = "./evaluation_results"
        os.makedirs(result_dir, exist_ok=True)

        # Accuracy and loss graph
        def save_training_plots(history, result_dir):
            plt.figure(figsize=(12, 5))

            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "accuracy_loss_per_epoch_featex.png"))
            plt.close()

        save_training_plots(history, result_dir)

        # Confusion matrix
        def save_confusion_matrix(y_true, y_pred, labels, result_dir):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "confusion_matrix_featex.png"))
            plt.close()

        label_names = encoder.classes_
        save_confusion_matrix(y_val_classes, y_pred, label_names, result_dir)

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

        # Handle case with no annotations
        if len(bboxes) == 0:
            return img, np.array([]), np.array([])
            
        return img, np.array(bboxes), np.array(class_ids)

    def load_dataset(self, dataset_dir, input_size, num_classes, max_boxes=20):
        """
        Load images and annotations from the dataset directory
        max_boxes: Maximum number of bounding boxes to support per image
        """
        images = []
        all_bboxes = []
        all_classes = []

        for img_file in os.listdir(dataset_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dataset_dir, img_file)
                ann_path = os.path.splitext(img_path)[0] + '.txt'
                
                if not os.path.exists(ann_path):
                    continue

                # Load the image and annotations
                img, bboxes, class_ids = self.load_image_and_annotations(
                    img_path, ann_path, input_size)
                
                if len(bboxes) == 0:
                    continue
                    
                # Pad bboxes and class_ids to max_boxes
                padded_bboxes = np.zeros((max_boxes, 4))
                padded_classes = np.zeros((max_boxes, num_classes))
                
                # Fill in the actual data
                num_boxes = min(len(bboxes), max_boxes)
                padded_bboxes[:num_boxes] = bboxes[:num_boxes]
                
                # Convert class_ids to one-hot encoding
                for i in range(num_boxes):
                    padded_classes[i, class_ids[i]] = 1.0
                    
                images.append(img)
                all_bboxes.append(padded_bboxes)
                all_classes.append(padded_classes)

        if not images:
            raise ValueError("No valid images found in the dataset directory")
            
        return np.array(images), np.array(all_bboxes), np.array(all_classes)


    def compute_iou(self,box1, box2):
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area else 0
    

    def train_od(self, config: DeepLearningObjectDetectionConstruct):
        # Unpack training configuration
        config_training = config.training

        # Select optimizer
        if config_training.optimizer_type == 'adam':
            optimizer = Adam(learning_rate=config_training.learning_rate)
        elif config_training.optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=config_training.learning_rate,
                            momentum=config_training.momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Check image size
        folder_path = './dataset/train/'
        image_files = [f for f in os.listdir(
            folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        image_path = os.path.join(folder_path, image_files[0])
        img = cv2.imread(image_path)
        img_shape = img.shape
        input_shape = img_shape
        
        # Option 1: Read from the labels file (assuming it contains all class names)
        try:
            with open('./dataset/classes.txt', 'r') as file:
                classes = [line.strip() for line in file]
            num_classes = len(classes)
        except FileNotFoundError:
            # Option 2: Determine from annotations by finding the max class ID + 1
            max_class_id = -1
            for img_file in os.listdir(folder_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    ann_path = os.path.splitext(os.path.join(folder_path, img_file))[0] + '.txt'
                    if os.path.exists(ann_path):
                        with open(ann_path, 'r') as file:
                            for line in file:
                                class_id = int(line.strip().split()[0])
                                max_class_id = max(max_class_id, class_id)
            
            num_classes = max_class_id + 1
        
        print(f"Number of classes detected: {num_classes}")
        
        # Determine maximum number of boxes per image
        max_boxes = 0
        for img_file in os.listdir(folder_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                ann_path = os.path.splitext(os.path.join(folder_path, img_file))[0] + '.txt'
                if os.path.exists(ann_path):
                    with open(ann_path, 'r') as file:
                        box_count = sum(1 for _ in file)
                        max_boxes = max(max_boxes, box_count)
        
        # Set reasonable minimum and maximum
        max_boxes = max(max_boxes, 1)  # At least 1 box
        max_boxes = min(max_boxes, 100)  # Cap at 100 boxes per image for memory efficiency
        
        print(f"Maximum boxes per image: {max_boxes}")

        # Create the model with updated parameter for number of boxes
        model = constructdl_od.construct(
            config.model, input_shape, num_classes, max_boxes)
        model.summary()

        # Compile the model with the correct output names
        model.compile(
            optimizer=optimizer,
            loss={
                'bbox_reshape': 'mse',  # Regression loss for bounding boxes
                'class_activation': 'categorical_crossentropy'  # Classification loss
            },
            loss_weights={'bbox_reshape': 1.0, 'class_activation': 1.0},
            metrics={'class_activation': 'accuracy'}
        )

        # Load dataset for training and validation
        X_train, y_bboxes_train, y_classes_train = self.load_dataset(
            './dataset/train', (input_shape[1], input_shape[0]), num_classes, max_boxes)
        X_valid, y_bboxes_valid, y_classes_valid = self.load_dataset(
            './dataset/valid', (input_shape[1], input_shape[0]), num_classes, max_boxes)
        X_test, y_bboxes_test, y_classes_test = self.load_dataset(
            './dataset/test', (input_shape[1], input_shape[0]), num_classes, max_boxes)

        print("X_train:", X_train.shape)
        print("y_bboxes_train:", y_bboxes_train.shape)
        print("y_classes_train:", y_classes_train.shape)
        
        eval_callback = EvaluationCallback(X_valid, y_bboxes_valid, y_classes_valid, num_classes)

        # Train with explicit validation data
        history = model.fit(
            X_train,
            {'bbox_reshape': y_bboxes_train, 'class_activation': y_classes_train},
            batch_size=config_training.batch_size,
            epochs=config_training.epochs,
            validation_data=(
                X_valid, {'bbox_reshape': y_bboxes_valid, 'class_activation': y_classes_valid}),
            callbacks=[eval_callback]

        )

        model.save("model.h5")

        preds = model.predict(X_test)
        y_bboxes_pred = preds[0]
        y_classes_pred = preds[1]

        y_true_classes_flat = np.argmax(y_classes_test.reshape(-1, y_classes_test.shape[-1]), axis=1)
        y_pred_classes_flat = np.argmax(y_classes_pred.reshape(-1, y_classes_pred.shape[-1]), axis=1)

        mask_true = np.sum(y_classes_test, axis=-1).flatten() > 0
        mask_pred = np.sum(y_classes_pred, axis=-1).flatten() > 0

        precision = precision_score(y_true_classes_flat[mask_true], y_pred_classes_flat[mask_true], average='macro', zero_division=0)
        recall = recall_score(y_true_classes_flat[mask_true], y_pred_classes_flat[mask_true], average='macro', zero_division=0)
        f1 = f1_score(y_true_classes_flat[mask_true], y_pred_classes_flat[mask_true], average='macro', zero_division=0)

        # Calculate IoU and mAP@0.5
        iou_threshold = 0.5

        ious = []
        average_precisions = []
        for i in range(len(X_test)):
            true_boxes = y_bboxes_test[i]
            pred_boxes = y_bboxes_pred[i]
            for j in range(len(true_boxes)):
                if np.sum(y_classes_test[i][j]) > 0:
                    best_iou = 0
                    for k in range(len(pred_boxes)):
                        if np.sum(y_classes_pred[i][k]) > 0:
                            iou = self.compute_iou(true_boxes[j], pred_boxes[k])
                            best_iou = max(best_iou, iou)
                    ious.append(best_iou)
                    average_precisions.append(1 if best_iou >= iou_threshold else 0)

        mean_iou = np.mean(ious) if ious else 0
        map_50 = np.mean(average_precisions) if average_precisions else 0

        print("\n📊 Evaluation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"IoU: {mean_iou:.4f}")
        print(f"mAP@0.5: {map_50:.4f}")

        os.makedirs('./evaluation_results', exist_ok=True)
        with open('./evaluation_results/metrics.txt', 'w') as f:
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"IoU: {mean_iou:.4f}\n")
            f.write(f"mAP@0.5: {map_50:.4f}\n")

        # Plotting
        metrics = ['Precision', 'Recall', 'F1 Score', 'IoU', 'mAP@0.5']
        values = [precision, recall, f1, mean_iou, map_50]

        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values, color='skyblue')
        plt.ylim(0, 1)
        plt.title('Evaluation Metrics')
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        plt.savefig('./evaluation_results/metric_summary.png')
        plt.close()

        epochs = list(range(1, config_training.epochs + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, eval_callback.epoch_metrics['precision'], label='Precision')
        plt.plot(epochs, eval_callback.epoch_metrics['recall'], label='Recall')
        plt.plot(epochs, eval_callback.epoch_metrics['map'], label='mAP@0.5')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('./evaluation_results/metrics_per_epoch.png')
        plt.close()


        model = None
        
        return history

    def get_image_paths(self, dataset_path):
        """Returns image file paths and corresponding annotation files."""
        image_files = [f for f in os.listdir(
            dataset_path) if f.endswith('.jpg')]
        annotation_files = [f.replace('.jpg', '.txt') for f in image_files]
        return image_files, annotation_files

    def load_data_od_featex(self, dataset_path, image_files, annotation_files, img_size, hog_params, hog_min, sift_params, sift_min, orb_params, orb_min):
        """Loads and processes images, extracting features and annotations."""
        X_features, y_boxes, y_classes = [], [], []

        # Second pass to extract features with fixed size
        for img_file, ann_file in zip(image_files, annotation_files):
            img_path = os.path.join(dataset_path, img_file)
            ann_path = os.path.join(dataset_path, ann_file)

            # Load image
            image = cv2.imread(img_path)

            feature_vector = []

            if hog_params:  # Check if HOG config exists
                hog_features = feature_extractor.extract_hog_features(
                    image, **hog_params, max_features=hog_min)
                feature_vector.append(hog_features)

            if sift_params:  # Check if SIFT config exists
                sift_features = feature_extractor.extract_sift_features(
                    image, **sift_params, max_features=sift_min)
                feature_vector.append(sift_features)

            if orb_params:  # Check if ORB config exists
                orb_features = feature_extractor.extract_orb_features(
                    image, **orb_params, max_features=orb_min)
                feature_vector.append(orb_features)

            # Flatten the feature vector if any features were extracted
            if feature_vector:
                feature_vector = np.hstack(feature_vector)
                X_features.append(feature_vector)

            # Read annotation file
            with open(ann_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])  # First value is class label
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Normalize bounding boxes
                    x_center /= img_size
                    y_center /= img_size
                    width /= img_size
                    height /= img_size

                    y_boxes.append([x_center, y_center, width, height])
                    y_classes.append(class_id)

        return np.array(X_features), np.array(y_boxes), np.array(y_classes)

    def check_min_feat_od(self, folder_path, hog_params=None, sift_params=None, orb_params=None):
        hog_count = []
        sift_count = []
        orb_count = []

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file_path.endswith(".DS_Store"):
                continue

            image = cv2.imread(file_path)
            if image is None:
                continue

            # Extract features only if params are provided
            if hog_params:  # Check if HOG config exists
                hog_features = feature_extractor.extract_hog_features(
                    image, **hog_params, count_feat=False)
                print("HOG FEAT LEN:", hog_features.shape)
                hog_count.append(len(hog_features))

            if sift_params:  # Check if SIFT config exists
                sift_features = feature_extractor.extract_sift_features(
                    image, **sift_params, count_feat=False)
                print("SIFT FEAT LEN:", len(sift_features))
                sift_count.append(len(sift_features))

            if orb_params:  # Check if ORB config exists
                orb_features = feature_extractor.extract_orb_features(
                    image, **orb_params, count_feat=False)
                print("ORB FEAT LEN:", len(orb_features))
                orb_count.append(len(orb_features))

        if hog_params:
            hog_min = min(hog_count)
        else:
            hog_min = 0

        if sift_params:
            sift_min = min(sift_count)
        else:
            sift_min = 0

        if orb_params:
            orb_min = min(orb_count)
        else:
            orb_min = 0
        print("MIN FEAT:", hog_min, sift_min, orb_min)
        return hog_min, sift_min, orb_min


    def train_od_featex(self, config):
            # --- Load sample image to determine size ---
            folder_path = './dataset/train/'
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
            image_path = os.path.join(folder_path, image_files[0])
            img = cv2.imread(image_path)
            img_size = img.shape[1]
            print(f"Image size: {img_size}")

            config_featex = config.featex
            hog_params = config_featex.hog.model_dump() if config_featex.hog else None
            sift_params = config_featex.sift.model_dump() if config_featex.sift else None
            orb_params = config_featex.orb.model_dump() if config_featex.orb else None
            print(hog_params, sift_params, orb_params)

            # --- Feature count ---
            hog_min_train, sift_min_train, orb_min_train = self.check_min_feat_od('dataset/train', hog_params, sift_params, orb_params)
            hog_min_val, sift_min_val, orb_min_val = self.check_min_feat_od('dataset/valid', hog_params, sift_params, orb_params)
            hog_min_test, sift_min_test, orb_min_test = self.check_min_feat_od('dataset/test', hog_params, sift_params, orb_params)
            hog_min = min(hog_min_train, hog_min_val, hog_min_test)
            sift_min = min(sift_min_train, sift_min_val, sift_min_test)
            orb_min = min(orb_min_train, orb_min_val, orb_min_test)
            print("MIN FEAT", hog_min, sift_min, orb_min)

            # --- Load data ---
            train_img_files, train_ann_files = self.get_image_paths("dataset/train")
            X_features_train, y_boxes_train, y_classes_train = self.load_data_od_featex(
                "dataset/train", train_img_files, train_ann_files, img_size, hog_params, hog_min, sift_params, sift_min, orb_params, orb_min)

            valid_img_files, valid_ann_files = self.get_image_paths("dataset/valid")
            X_features_valid, y_boxes_valid, y_classes_valid = self.load_data_od_featex(
                "dataset/valid", valid_img_files, valid_ann_files, img_size, hog_params, hog_min, sift_params, sift_min, orb_params, orb_min)
            
            test_img_files, test_ann_files = self.get_image_paths("dataset/test")
            X_features_test, y_boxes_test, y_classes_test = self.load_data_od_featex(
                "dataset/test", test_img_files, test_ann_files, img_size, hog_params, hog_min, sift_params, sift_min, orb_params, orb_min)

            y_classes_train = np.array(y_classes_train)
            y_classes_valid = np.array(y_classes_valid)

            if y_classes_train.ndim > 1:
                y_classes_train = np.argmax(y_classes_train, axis=1)
            if y_classes_valid.ndim > 1:
                y_classes_valid = np.argmax(y_classes_valid, axis=1)

            num_classes = len(np.unique(y_classes_train))
            y_classes_train = tf.keras.utils.to_categorical(y_classes_train, num_classes=num_classes)
            y_classes_valid = tf.keras.utils.to_categorical(y_classes_valid, num_classes=num_classes)
            print(f"Number of Classes: {num_classes}")

            # --- Build and compile model ---
            model = constructdl_od.construct_od_featex(config.model, input_shape=(hog_min+sift_min+orb_min), num_classes=num_classes)
            config_training = config.training
            optimizer = Adam(learning_rate=config_training.learning_rate) if config_training.optimizer_type == 'adam' \
                        else SGD(learning_rate=config_training.learning_rate, momentum=config_training.momentum)

            model.compile(
                optimizer=optimizer,
                loss={"class_output": "categorical_crossentropy", "bbox_output": "mean_squared_error"},
                metrics={"class_output": "accuracy", "bbox_output": "mae"}
            )

            # --- Callback for logging per-epoch (optional) ---
            class EvalCallback(tf.keras.callbacks.Callback):
                def __init__(self):
                    self.epoch_metrics = {'precision': [], 'recall': [], 'map': []}

                def on_epoch_end(self, epoch, logs=None):
                    self.epoch_metrics['precision'].append(logs.get("val_class_output_accuracy", 0))
                    self.epoch_metrics['recall'].append(1 - logs.get("val_bbox_output_mae", 0))
                    self.epoch_metrics['map'].append(0.0)  # Can replace with real mAP calculation

            eval_callback = EvalCallback()

            # --- Train the model ---
            model.fit(
                X_features_train, {"class_output": y_classes_train, "bbox_output": y_boxes_train},
                validation_data=(X_features_valid, {"class_output": y_classes_valid, "bbox_output": y_boxes_valid}),
                epochs=config_training.epochs,
                batch_size=config_training.batch_size,
                callbacks=[eval_callback]
            )

            model.save("model.h5")

            # === Evaluation ===

            # Predict
            y_pred_class_prob, y_pred_bbox = model.predict(X_features_test)

            # Fix for y_classes_test shape
            y_classes_test = np.array(y_classes_test)
            if y_classes_test.ndim > 1:
                y_true_classes = np.argmax(y_classes_test, axis=1)
            else:
                y_true_classes = y_classes_test

            y_pred_classes = np.argmax(y_pred_class_prob, axis=1)

            # Precision / Recall / F1
            precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
            recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
            f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

            print("\n📊 Evaluation Metrics:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")

            # Save to text
            os.makedirs('./evaluation_results', exist_ok=True)
            with open('./evaluation_results/metrics.txt', 'w') as f:
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")

            # Bar plot
            metrics = ['Precision', 'Recall', 'F1 Score']
            values = [precision, recall, f1]
            plt.figure(figsize=(10, 6))
            plt.bar(metrics, values, color='skyblue')
            plt.ylim(0, 1)
            plt.title('Evaluation Metrics')
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            plt.savefig('./evaluation_results/metric_summary.png')
            plt.close()

            # Per-epoch metrics plot
            epochs = list(range(1, config_training.epochs + 1))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, eval_callback.epoch_metrics['precision'], label='Precision')
            plt.plot(epochs, eval_callback.epoch_metrics['recall'], label='Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Validation Metrics Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig('./evaluation_results/metrics_per_epoch.png')
            plt.close()