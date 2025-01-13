import os
import numpy as np
import os
from sklearn.metrics import accuracy_score
from skimage.io import imread
from app.services.model.ml import MlModel

mlmodel = MlModel()

def training(config):
    model = None
    # Load dataset
    X_train, y_train, class_dict = load_dataset('dataset/train')
    X_val, y_val, _ = load_dataset('dataset/valid')

    model = mlmodel.create_ml_model(config)

    model.fit(X_train, y_train)

    # Evaluate on validation data
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy}")

    # Print class mapping
    print("Class mapping:", class_dict)

def load_dataset(base_path):
    images = []
    labels = []
    class_names = os.listdir(base_path)
    class_names.sort()
    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

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
                        img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    elif img.ndim == 3 and img.shape[2] == 3:  # Color image
                        pass  # Already has the correct shape
                    else:
                        raise ValueError(f"Unsupported image dimensions: {img.shape}")

                    if expected_shape is None:
                        expected_shape = img.shape
                    elif img.shape != expected_shape:
                        raise ValueError(f"Inconsistent shape for image {img_path}. Expected {expected_shape}, got {img.shape}")

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
