from PIL import Image
from fastapi import HTTPException
import io
import numpy as np
import joblib
import math
from io import BytesIO
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError, CategoricalCrossentropy
import subprocess
import tempfile
from pathlib import Path
import os
import shutil
from app.services.dataset.dataset import reverse_convert_object_detection, reverse_convert_segmentation


class UseModel:
    def __init__(self, model_bytes: bytes = None):
        self.model_bytes = model_bytes

    def detect_grayscale_model(self):
        """
        Determines if the model expects a grayscale (1 channel) or color (3 channels) input.
        """
        # Check if the feature count matches a square-like shape
        height, width = self.find_optimal_shape(self.n_features)

        # If expected features = height × width, it's grayscale; otherwise, it's color
        return height * width == self.n_features

    def find_optimal_shape(self, n_features):
        """Finds the closest width and height whose product equals n_features"""
        sqrt_val = int(math.sqrt(n_features))
        for i in range(sqrt_val, 1, -1):  # Start from closest square root
            if n_features % i == 0:
                return (i, n_features // i)  # Return (height, width)
        return (n_features, 1)  # If no better match found, use (n_features, 1)

    def use_ml(self, img_bytes: bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_model:
            temp_model.write(self.model_bytes)
            model_path = temp_model.name

        self.loaded_model = joblib.load(model_path)
        self.n_features = self.loaded_model.n_features_in_
        self.is_grayscale = self.detect_grayscale_model()

        image = Image.open(io.BytesIO(img_bytes))
        image = image.convert("L" if self.is_grayscale else "RGB")

        height, width = self.find_optimal_shape(self.n_features)
        image = image.resize((width, height))
        img_array = np.array(image).astype(np.float32) / 255.0
        if self.is_grayscale:
            img_array = img_array.flatten().reshape(1, -1)
        else:
            img_array = img_array.reshape(1, -1)

        if img_array.shape[1] != self.n_features:
            raise ValueError(
                f"Resized image has {img_array.shape[1]} features, but model expects {self.n_features}.")

        return self.loaded_model.predict(img_array)

    def preprocess_image(self, image_bytes, input_shape):
        # Convert bytes to image
        image = Image.open(BytesIO(image_bytes))

        # Get expected number of channels from model input shape
        expected_channels = self.model.input_shape[-1]

        # Convert to grayscale or RGB based on model input shape
        if expected_channels == 1:
            image = image.convert("L")  # Convert to grayscale
        else:
            image = image.convert("RGB")  # Convert to 3-channel RGB

        # Resize image to model's input shape (height, width)
        image = image.resize((input_shape[0], input_shape[1]))

        # Convert to numpy array
        img_array = np.array(image)

        # Ensure correct shape: (height, width, channels)
        if expected_channels == 1:
            img_array = np.expand_dims(
                img_array, axis=-1)  # Add channel dimension

        # Normalize pixel values (0 to 1)
        img_array = img_array / 255.0

        # Expand dimensions to match model input (batch size of 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def use_dl_cls(self, img_bytes: bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model:
            temp_model.write(self.model_bytes)
            model_path = temp_model.name

        self.model = load_model(model_path)
        input_shape = self.model.input_shape[1:3]
        processed_img = self.preprocess_image(img_bytes, input_shape)
        predictions = self.model.predict(processed_img)
        return np.argmax(predictions, axis=1)[0]

    def use_dl_od_pt(self, img_bytes: bytes, version):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model:
            temp_model.write(self.model_bytes)
            model_path = temp_model.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(img_bytes)
            temp_img_path = temp_img.name

        output_dir = Path("runs/detect/predict")
        output_dir.mkdir(parents=True, exist_ok=True)

        if version == "yolov8":
            command = (
                f"yolov8_venv/bin/yolo task=detect mode=predict model={model_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )
            folder_path = "./runs/detect/predict2/labels/"

        elif version == "yolov11":
            command = (
                f"yolov11_venv/bin/yolo task=detect mode=predict model={model_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )
            folder_path = "./runs/detect/predict2/labels/"

        elif version == "yolov5":
            command = (
                f"yolo5_venv/bin/python ./app/services/model/yolov5/detect.py "
                f"--weights {model_path} --conf 0.09 --source {temp_img_path} --save-txt"
            )
            folder_path = "./app/services/model/yolov5/runs/detect/exp/labels/"
        else:
            raise HTTPException(400, f"Unsupported version: {version}")

        subprocess.run(command, shell=True, check=True)

        detections = []
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        if txt_files:
            with open(os.path.join(folder_path, txt_files[0]), "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:]))
                        detections.append({
                            "class_id": class_id,
                            "bbox": {
                                "x_center": bbox[0],
                                "y_center": bbox[1],
                                "width": bbox[2],
                                "height": bbox[3]
                            }
                        })

        shutil.rmtree("./app/services/model/yolov5/runs/detect/exp", ignore_errors=True)
        shutil.rmtree("./runs/detect/predict2", ignore_errors=True)
        
        image = Image.open(io.BytesIO(img_bytes))
        img_width, img_height = image.size
        image.close()

        return reverse_convert_object_detection(detections, img_width, img_height)

    def use_dl_od_con(self, img_bytes: bytes):
        print(f"Image bytes length: {len(img_bytes)}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model:
            temp_model.write(self.model_bytes)
            model_path = temp_model.name

        self.model = load_model(
            model_path,
            custom_objects={
                'mse': MeanSquaredError(),
                'categorical_crossentropy': CategoricalCrossentropy()
            }
        )

        input_shape = self.model.input_shape[1:3]
        processed_img = self.preprocess_image(img_bytes, input_shape)

        predictions = self.model.predict(processed_img)
        print("Predictions:", [p.shape for p in predictions])

        # Handle the predictions dynamically based on their shape
        bbox_pred = predictions[0][0]  # Get the first batch of bbox predictions 
        class_pred = predictions[1][0]  # Get the first batch of class predictions
        
        # Determine the number of detections based on prediction shape
        num_detections = bbox_pred.shape[0]  # First dimension is the number of boxes
        
        detections = []
        for i in range(num_detections):
            # Get classification data for this detection
            class_probs = class_pred[i]
            class_id = int(class_probs.argmax())
            confidence = float(class_probs[class_id])

            if confidence < 0.5:
                continue  # skip low-confidence detections

            # Get bounding box data
            x_center, y_center, width, height = bbox_pred[i]

            detections.append({
                "class_id": class_id,
                "bbox": {
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width": float(width),
                    "height": float(height),
                },
                "confidence": confidence
            })

        image = Image.open(io.BytesIO(img_bytes))
        img_width, img_height = image.size
        image.close()

        return reverse_convert_object_detection(detections, img_width, img_height)

    def use_dl_seg(self, img_bytes: bytes, version: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model:
            temp_model.write(self.model_bytes)
            model_path = temp_model.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(img_bytes)
            temp_img_path = temp_img.name

        output_dir = Path("runs/segment/predict")
        output_dir.mkdir(parents=True, exist_ok=True)

        if version == "yolov8":
            command = (
                f"yolov8_venv/bin/yolo task=segment mode=predict model={model_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )

        elif version == "yolov11":
            command = (
                f"yolov11_venv/bin/yolo task=segment mode=predict model={model_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )

        else:
            raise HTTPException(400, f"Unsupported version: {version}")

        folder_path = "./runs/segment/predict2/labels/"
        subprocess.run(command, shell=True, check=True)

        detections = []
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        if txt_files:
            with open(os.path.join(folder_path, txt_files[0]), "r") as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coordinates = list(map(float, parts[1:]))
                    if len(coordinates) >= 6:
                        polygon = [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]
                        detections.append({"class_id": class_id, "polygon": polygon})

        shutil.rmtree("./runs/segment/", ignore_errors=True)
        
        image = Image.open(io.BytesIO(img_bytes))
        img_width, img_height = image.size
        image.close()

        return reverse_convert_segmentation(detections, img_width, img_height)
