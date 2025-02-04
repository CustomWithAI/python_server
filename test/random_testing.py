
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
        inputs = tf.keras.layers.Input(shape=(input_shape,))
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


def extract_features(image, fixed_size=None, config_featex=None):
    """Extracts features from an image using HOG, SIFT, and ORB."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Default HOG parameters
    hog_desc = cv2.HOGDescriptor(
        _winSize=(8, 8),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    if config_featex and 'hog' in config_featex:
        hog_params = config_featex['hog']
        block_size = tuple(hog_params.get("cells_per_block", [2, 2]))
        cell_size = tuple(hog_params.get("pixels_per_cell", [8, 8]))

        if block_size[0] % cell_size[0] != 0 or block_size[1] % cell_size[1] != 0:
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

    # Apply padding or truncation only if fixed_size is set
    if fixed_size is not None:
        if features.shape[0] < fixed_size:
            features = np.pad(
                features, (0, fixed_size - features.shape[0]), mode='constant')
        else:
            features = features[:fixed_size]

    return features


def load_data(dataset_path, image_files, annotation_files, img_size, config_featex):
    """Loads and processes images, extracting features and annotations."""
    X_features, X_images, y_boxes, y_classes = [], [], [], []
    min_feature_size = float('inf')  # Initialize with a large value

    # First pass to determine the minimum feature size
    feature_sizes = []
    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_size, img_size))

        # Extract features without truncation
        features = extract_features(
            image, fixed_size=None, config_featex=config_featex)
        feature_sizes.append(features.shape[0])

    min_feature_size = min(feature_sizes)  # Find smallest feature size

    # Second pass to extract features with fixed size
    for img_file, ann_file in zip(image_files, annotation_files):
        img_path = os.path.join(dataset_path, img_file)
        ann_path = os.path.join(dataset_path, ann_file)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_size, img_size))
        features = extract_features(
            image, fixed_size=min_feature_size, config_featex=config_featex)
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

    return np.array(X_features), np.array(X_images), np.array(y_boxes), np.array(y_classes), min_feature_size


def main():
    # Define feature extraction configuration
    config_featex = {
        "hog": {"pixels_per_cell": [8, 8], "cells_per_block": [2, 2], "orientations": 9},
        "sift": {"number_of_keypoints": 128, "contrast_threshold": 0.04, "edge_threshold": 10},
        "orb": {"keypoints": 128, "scale_factor": 1.2, "n_level": 8}
    }

    # Load training data
    train_image_files, train_annotation_files = get_image_paths(
        "dataset/train")
    X_features_train, X_images_train, y_boxes_train, y_classes_train, min_feature_size = load_data(
        "dataset/train", train_image_files, train_annotation_files, IMG_SIZE, config_featex=config_featex)

    # Load validation data
    valid_image_files, valid_annotation_files = get_image_paths(
        "dataset/valid")
    X_features_valid, X_images_valid, y_boxes_valid, y_classes_valid, _ = load_data(
        "dataset/valid", valid_image_files, valid_annotation_files, IMG_SIZE, config_featex=config_featex)

    # Convert to NumPy array
    y_classes_train = np.array(y_classes_train)
    y_classes_valid = np.array(y_classes_valid)

    # Convert to class indices if one-hot encoded
    if y_classes_train.ndim > 1:
        y_classes_train = np.argmax(y_classes_train, axis=1)
    if y_classes_valid.ndim > 1:
        y_classes_valid = np.argmax(y_classes_valid, axis=1)

    # Count unique class labels
    num_classes = len(np.unique(y_classes_train))

    # One-hot encode labels properly
    y_classes_train = tf.keras.utils.to_categorical(
        y_classes_train, num_classes=num_classes)
    y_classes_valid = tf.keras.utils.to_categorical(
        y_classes_valid, num_classes=num_classes)

    print(f"Number of Classes: {num_classes}")

    # Model configuration
    config = [
        {"denseLayer_units": 128, "denseLayer_activation": "relu"},
        {"dropoutLayer_rate": 0.5}
    ]

    # Build model
    model_builder = ConstructDLOD()
    model = model_builder.construct(
        config, input_shape=min_feature_size, num_classes=num_classes)

    # Compile model
    model.compile(
        optimizer=Adam(),
        loss={"class_output": "categorical_crossentropy",
              "bbox_output": "mean_squared_error"},
        metrics={"class_output": "accuracy", "bbox_output": "mae"}
    )

    # Train model with explicit validation data
    model.fit(
        X_features_train, {"class_output": y_classes_train,
                           "bbox_output": y_boxes_train},
        validation_data=(X_features_valid, {
                         "class_output": y_classes_valid, "bbox_output": y_boxes_valid}),
        epochs=100,
        batch_size=16
    )


if __name__ == "__main__":
    main()
