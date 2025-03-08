from PIL import Image
import io
import numpy as np
import joblib
import math
from io import BytesIO
from tensorflow.keras.models import load_model


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
