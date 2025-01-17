import os
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score
from skimage.io import imread
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from app.services.model.ml import MlModel
from app.services.model.dl_pretrained import DlModel
from app.services.model.construct import ConstructDL
mlmodel = MlModel()
dlmodel = DlModel()
constructdl = ConstructDL()

class MLTraining():
    def training_ml(self,config):
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

    def load_dataset(self,base_path):
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


class DLTrainingPretrained():
    def train(self,config_model, config_training):
        model = None

        # Load dataset
        X_train, y_train, class_dict, input_shape = self.load_dataset_dl('dataset/train')
        X_val, y_val, _, _ = self.load_dataset_dl('dataset/valid', class_dict)

        num_classes = len(class_dict)  # Dynamically get number of classes

        # Create model
        model = dlmodel.create_dl_model(config_model,num_classes,input_shape)

        # Unpack training configuration
        learning_rate = config_training[0]
        learning_rate_scheduler = config_training[1]
        momentum = config_training[2]
        optimizer_type = config_training[3]
        batch_size = config_training[4]
        epochs = config_training[5]
        loss_function = get_loss(config_training[6])  # Convert string to loss function

        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

        # Learning rate scheduler (optional)
        callbacks = []
        if learning_rate_scheduler:
            def scheduler(epoch, lr):
                return learning_rate_scheduler(epoch, lr)
            callbacks.append(LearningRateScheduler(scheduler))

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        return history

    def load_dataset_dl(self,base_path, class_dict=None):
        images = []
        labels = []
        class_names = os.listdir(base_path)
        class_names = [name for name in class_names if not name.startswith('.')]  # Exclude hidden files
        class_names.sort()

        if class_dict is None:
            class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

        input_shape = None  # Initialize input shape variable

        for class_name in class_names:
            class_path = os.path.join(base_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('png', 'jpg', 'jpeg')):  # Ensure only image files are processed
                        img_path = os.path.join(class_path, filename)
                        try:
                            img = load_img(img_path)  # Load image without resizing
                            img_array = img_to_array(img)  # Convert image to numpy array
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
