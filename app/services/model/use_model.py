from PIL import Image
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


class UseModel:
    def __init__(self):
        pass

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

    def use_ml(self, img_bytes):
        self.loaded_model = joblib.load("./ml_model.pkl")  # Load model
        self.n_features = self.loaded_model.n_features_in_  # Expected input features
        self.is_grayscale = self.detect_grayscale_model()  # Detect input type

        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes))

        # Convert image based on model requirement
        if self.is_grayscale:
            image = image.convert("L")  # Convert to grayscale if needed
        else:
            image = image.convert("RGB")  # Convert to RGB if needed

        # Find the correct resize shape
        height, width = self.find_optimal_shape(self.n_features)
        image = image.resize((width, height))  # Resize dynamically

        # Convert to NumPy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # Flatten and reshape appropriately
        if self.is_grayscale:
            img_array = img_array.flatten().reshape(1, -1)  # (1, n_features)
        else:
            img_array = img_array.reshape(1, -1)  # (1, height * width * 3)

        # Ensure it matches exactly the required feature count
        if img_array.shape[1] != self.n_features:
            raise ValueError(
                f"Resized image has {img_array.shape[1]} features, but model expects {self.n_features}.")

        # Predict using the trained model
        prediction = self.loaded_model.predict(img_array)

        return prediction

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

    def use_dl_cls(self, img_bytes):
        self.model = load_model("./model.h5")
        # Get model input shape dynamically
        input_shape = self.model.input_shape[1:3]  # Extract (height, width)

        # Preprocess image
        processed_img = self.preprocess_image(img_bytes, input_shape)

        # Predict
        predictions = self.model.predict(processed_img)

        # Convert to class label (assuming softmax activation in final layer)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return predicted_class

    def use_dl_od_pt(self, img_bytes, version):
        # Save input image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(img_bytes)
            temp_img_path = temp_img.name

        weight_path = "./best.pt"

        # Define output directory
        output_dir = Path("runs/detect/predict")
        output_dir.mkdir(parents=True, exist_ok=True)

        if version == "yolov8":
            command = (
                f"yolov8_venv/bin/yolo task=detect mode=predict model={weight_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )
            folder_path = "./runs/detect/predict2/labels/"

        if version == "yolov11":
            command = (
                f"yolov11_venv/bin/yolo task=detect mode=predict model={weight_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )

            folder_path = "./runs/detect/predict2/labels/"

        elif version == "yolov5":
            command = (
                f"yolo5_venv/bin/python ./app/services/model/yolov5/detect.py --weights {weight_path} --conf 0.09 --source {temp_img_path} --save-txt"
            )
            folder_path = "./app/services/model/yolov5/runs/detect/exp/labels/"

        subprocess.run(command, shell=True, check=True)

        # Get the list of all files in the folder
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

        # Check if there are any .txt files
        if txt_files:
            # Pick the first .txt file
            txt_file_path = os.path.join(folder_path, txt_files[0])

            detections = []

            # Read the content of the first .txt file found
            with open(txt_file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])  # Convert class ID to int
                        # Convert bbox values to float
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

        shutil.rmtree(
            "./app/services/model/yolov5/runs/detect/exp", ignore_errors=True)
        shutil.rmtree("./runs/detect/predict2", ignore_errors=True)

        return detections

    def use_dl_od_con(self, img_bytes):
        self.model = load_model("./model.h5", custom_objects={
            'mse': MeanSquaredError(),
            'categorical_crossentropy': CategoricalCrossentropy()
        })

        # Get model input shape dynamically
        input_shape = self.model.input_shape[1:3]

        # Preprocess image
        processed_img = self.preprocess_image(img_bytes, input_shape)

        # Predict
        predictions = self.model.predict(processed_img)

        bbox_pred = predictions[0]
        class_pred = predictions[1]

        detections = []
        for i in range(len(bbox_pred)):  # Loop over each bounding box prediction
            x_center, y_center, width, height = bbox_pred[i]
            class_probabilities = class_pred[i]

            # Determine the class_id by finding the index of the highest class probability
            class_id = int(class_probabilities.argmax())

            # Append the formatted prediction to the detections list
            detections.append({
                "class_id": class_id,  # The predicted class ID
                "bbox": {
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width": float(width),
                    "height": float(height)
                },
                # Confidence of the predicted class
                "confidence": float(class_probabilities[class_id])
            })

        return detections

    def use_dl_seg(self, img_bytes, version):
        # Save input image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(img_bytes)
            temp_img_path = temp_img.name

        weight_path = "./best.pt"

        if version == "yolov8":
            command = (
                f"yolov8_venv/bin/yolo task=detect mode=predict model={weight_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )

        if version == "yolov11":
            command = (
                f"yolov11_venv/bin/yolo task=detect mode=predict model={weight_path} "
                f"source={temp_img_path} conf=0.5 save_txt save"
            )

        folder_path = "./runs/segment/predict/labels/"

        subprocess.run(command, shell=True, check=True)

        detections = []

        # Get the list of all files in the folder
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

        # Check if there are any .txt files
        if txt_files:
            # Pick the first .txt file
            txt_file_path = os.path.join(folder_path, txt_files[0])

            # Read the content of the first .txt file
            with open(txt_file_path, "r") as file:
                for line in file:
                    # Split the line by spaces
                    parts = line.strip().split()

                    # Extract class_id and coordinates
                    class_id = int(parts[0])
                    coordinates = list(map(float, parts[1:]))

                    # If coordinates have at least 6 values (min length for a polygon)
                    if len(coordinates) >= 6:
                        # Convert coordinates to a polygon or whatever structure you need
                        # Example: for segmentation, we can return a polygon (list of tuples)
                        polygon = [(coordinates[i], coordinates[i+1])
                                   for i in range(0, len(coordinates), 2)]
                        detections.append(
                            {"class_id": class_id, "polygon": polygon})

        shutil.rmtree("./runs/segment/predict", ignore_errors=True)

        return detections
