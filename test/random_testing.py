
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Dataset Paths
DATASET_PATH = "dataset/train"
IMG_SIZE = 224  # Image size for CNN


class ConstructDLOD(object):
    def __init__(self):
        pass

    def construct(self, config, input_shape, num_classes, num_boxes=1):
        print("Received input_shape:", input_shape)

        # Correct input shape for the feature vector
        inputs = tf.keras.layers.Input(shape=(5000,))
        x = inputs

        # Add dense layers based on the configuration
        for layer_config in config:
            if "denseLayer_units" in layer_config:
                x = tf.keras.layers.Dense(
                    units=layer_config["denseLayer_units"],
                    activation=layer_config["denseLayer_activation"]
                )(x)

            elif "dropoutLayer_rate" in layer_config:
                x = tf.keras.layers.Dropout(
                    layer_config["dropoutLayer_rate"])(x)

        # Object Detection Output Heads (bbox and class outputs)
        bbox_output = tf.keras.layers.Dense(
            num_boxes * 4, activation="linear", name="bbox_output")(x)
        class_output = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="class_output")(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=[
            bbox_output, class_output])

        return model


def get_image_paths(dataset_path):
    """Returns image file paths and corresponding annotation files."""
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    annotation_files = [f.replace('.jpg', '.txt') for f in image_files]
    return image_files, annotation_files


def extract_features(image, fixed_size=5000, config_featex=None):
    """Extracts features from an image using HOG, SIFT, and ORB."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Default HOG parameters
    hog_desc = cv2.HOGDescriptor(
        _winSize=(8, 8),  # Set the window size
        _blockSize=(16, 16),  # Block size should be divisible by cell size
        _blockStride=(8, 8),  # Block stride
        _cellSize=(8, 8),  # Cell size
        _nbins=9  # Number of bins
    )

    # If custom configurations are provided for HOG
    if config_featex and 'hog' in config_featex:
        hog_params = config_featex['hog']
        # Ensure blockSize is divisible by cellSize
        block_size = tuple(hog_params.get("cells_per_block", [2, 2]))
        cell_size = tuple(hog_params.get("pixels_per_cell", [8, 8]))

        # Adjust the block size to ensure divisibility by cell size
        if block_size[0] % cell_size[0] != 0 or block_size[1] % cell_size[1] != 0:
            print(
                f"Warning: Adjusting blockSize {block_size} to be divisible by cellSize {cell_size}")
            # Adjust to multiples of cell size
            block_size = (cell_size[0] * 2, cell_size[1] * 2)

        hog_desc = cv2.HOGDescriptor(
            _winSize=(image.shape[1] // 8 * 8, image.shape[0] // 8 * 8),
            _blockSize=block_size,
            _blockStride=(8, 8),
            _cellSize=cell_size,
            _nbins=hog_params.get("orientations", 9)
        )

    # Extract HOG features
    hog_features = hog_desc.compute(gray).flatten()

    # SIFT Feature Extraction (Variable size)
    sift = cv2.SIFT_create(
        nfeatures=config_featex.get('sift', {}).get(
            "number_of_keypoints", 128),
        contrastThreshold=config_featex.get(
            'sift', {}).get("contrast_threshold", 0.04),
        edgeThreshold=config_featex.get('sift', {}).get("edge_threshold", 10)
    )
    _, sift_features = sift.detectAndCompute(gray, None)
    sift_features = sift_features.flatten() if sift_features is not None else np.array([])

    # ORB Feature Extraction (Variable size)
    orb = cv2.ORB_create(
        nfeatures=config_featex.get('orb', {}).get("keypoints", 128),
        scaleFactor=config_featex.get('orb', {}).get("scale_factor", 1.2),
        nlevels=config_featex.get('orb', {}).get("n_level", 8)
    )
    _, orb_features = orb.detectAndCompute(gray, None)
    orb_features = orb_features.flatten() if orb_features is not None else np.array([])

    # Concatenate features
    features = np.hstack([hog_features, sift_features, orb_features])

    # Ensure a fixed length (pad or truncate)
    if features.shape[0] < fixed_size:
        features = np.pad(
            features, (0, fixed_size - features.shape[0]), mode='constant')
    else:
        features = features[:fixed_size]

    return features


def load_data(dataset_path, image_files, annotation_files, img_size, config_featex):
    """Loads and processes images, extracting features and annotations."""
    X_features, X_images, y_boxes, y_classes = [], [], [], []

    for img_file, ann_file in zip(image_files, annotation_files):
        img_path = os.path.join(dataset_path, img_file)
        ann_path = os.path.join(dataset_path, ann_file)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_size, img_size))
        features = extract_features(image, config_featex=config_featex)
        image = image / 255.0  # Normalize after feature extraction

        X_features.append(features)
        X_images.append(image)

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

    return np.array(X_features), np.array(X_images), np.array(y_boxes), np.array(y_classes)


def train_model(model, X_features, y_classes, y_boxes, epochs=10, batch_size=16):
    """Trains the model with the given data."""
    # Compile the model
    model.compile(
        optimizer=Adam(),
        loss={
            "class_output": "categorical_crossentropy",
            "bbox_output": "mean_squared_error"
        },
        metrics={
            "class_output": "accuracy",
            "bbox_output": "mae"
        }
    )

    # Train the model using X_features (should have shape (num_samples, 5000))
    model.fit(
        X_features,  # Use features (5000-dimensional vectors)
        {"class_output": y_classes, "bbox_output": y_boxes},
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )


def main():

    # Define feature extraction configuration
    config_featex = {
        "hog": {"pixels_per_cell": [8, 8], "cells_per_block": [2, 2], "orientations": 9},
        "sift": {"number_of_keypoints": 128, "contrast_threshold": 0.04, "edge_threshold": 10},
        "orb": {"keypoints": 128, "scale_factor": 1.2, "n_level": 8}
    }

    # Step 1: Get image and annotation paths
    image_files, annotation_files = get_image_paths(DATASET_PATH)

    # Step 2: Load and preprocess data
    X_features, X_images, y_boxes, y_classes = load_data(
        DATASET_PATH, image_files, annotation_files, IMG_SIZE, config_featex=config_featex)

    # Step 3: Convert y_classes to one-hot encoding
    y_classes = tf.keras.utils.to_categorical(
        y_classes, num_classes=len(set(y_classes)))

    # Step 4: Get the number of classes
    y_classes_flat = np.argmax(y_classes, axis=1)
    num_classes = len(set(y_classes_flat))
    print(f"Number of Classes: {num_classes}")

    # Step 5: Define model configuration (can be customized)
    config = [
        {"convolutionalLayer_filters": 32, "convolutionalLayer_kernelSize": (3, 3), "convolutionalLayer_strides": (1, 1),
         "convolutionalLayer_padding": "same", "convolutionalLayer_activation": "relu"},
        {"poolingLayer_poolSize": (2, 2), "poolingLayer_strides": (
            2, 2), "poolingLayer_padding": "same"},
        {"convolutionalLayer_filters": 64, "convolutionalLayer_kernelSize": (3, 3), "convolutionalLayer_strides": (1, 1),
         "convolutionalLayer_padding": "same", "convolutionalLayer_activation": "relu"},
        {"flattenLayer": True},
        {"denseLayer_units": 128, "denseLayer_activation": "relu"},
        {"dropoutLayer_rate": 0.5}
    ]

    # Step 6: Build the model using ConstructDLOD class
    model_builder = ConstructDLOD()
    model = model_builder.construct(
        config, input_shape=(5000,), num_classes=num_classes)

    # Step 7: Train the model
    train_model(model, X_features, y_classes, y_boxes, epochs=10)

    # Step 8: Save the model (optional)
    model.save("object_detector_from_scratch.h5")
    print("Model saved as 'object_detector_from_scratch.h5'")


if __name__ == "__main__":
    main()
