from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

class DlModel:
    def __init__(self):
        pass

    def create_dl_model(self, config: str, num_classes, input_shape, unfreeze: int):
        base_model = None

        if "resnet50" in config:
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif "vgg16" in config:
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif "mobilenetv2" in config:
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError("Unsupported model type")

        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False

        # Calculate how many layers to unfreeze (max 50%)
        total_layers = len(base_model.layers)
        unfreeze = max(0, min(unfreeze, 50))  # Clamp between 0 and 50
        num_to_unfreeze = int((unfreeze / 100) * total_layers)

        # Unfreeze the last `num_to_unfreeze` layers
        for layer in base_model.layers[-num_to_unfreeze:]:
            layer.trainable = True

        # Create a Sequential model with the custom head
        model = Sequential([
            base_model,  # Add the pre-trained base model
            layers.GlobalAveragePooling2D(),  # Global average pooling
            layers.Dense(512, activation='relu'),  # First dense layer
            layers.BatchNormalization(),  # Batch normalization
            layers.Dropout(0.3),  # Dropout layer
            layers.Dense(128, activation='relu'),  # Second dense layer
            layers.Dropout(0.1),  # Dropout layer
            layers.Dense(32, activation='relu'),  # Third dense layer
            layers.Dropout(0.3),  # Dropout layer
            layers.Dense(num_classes, activation='softmax')  # Output layer
        ])

        return model
