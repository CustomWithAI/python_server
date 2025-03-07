import json
import joblib
import numpy as np
from PIL import Image
import io
import math


class UseModel:
    def __init__(self):
        self.loaded_model = joblib.load("./ml_model.pkl")  # Load model
        self.n_features = self.loaded_model.n_features_in_  # Expected input features

    def find_optimal_shape(self, n_features):
        """Finds the closest width and height whose product equals n_features"""
        sqrt_val = int(math.sqrt(n_features))
        for i in range(sqrt_val, 1, -1):  # Start from closest square root
            if n_features % i == 0:
                return (i, n_features // i)  # Return (height, width)
        return (n_features, 1)  # If no better match found, use (n_features, 1)

    def use_ml(self, img_bytes):
        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes)).convert(
            "L")  # Convert to grayscale

        # Find the correct resize shape
        height, width = self.find_optimal_shape(self.n_features)
        image = image.resize((width, height))  # Resize dynamically

        # Convert to NumPy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = img_array.flatten().reshape(1, -1)  # Flatten for model input

        # Ensure it matches exactly the required feature count
        if img_array.shape[1] != self.n_features:
            raise ValueError(
                f"Resized image has {img_array.shape[1]} features, but model expects {self.n_features}.")

        # Predict using the trained model
        prediction = self.loaded_model.predict(img_array)

        return prediction
