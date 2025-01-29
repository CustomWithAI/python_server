import tensorflow as tf
from tensorflow.keras import layers, models


class ConstructDLOD(object):
    def __init__(self):
        pass

    def construct(self, config, input_shape, num_classes, num_boxes=1):
        print("Received input_shape:", input_shape)
        # Ensure input shape is correct
        inputs = layers.Input(shape=input_shape)
        x = inputs

        for layer_config in config:
            if "convolutionalLayer_filters" in layer_config:
                x = layers.Conv2D(
                    filters=layer_config["convolutionalLayer_filters"],
                    kernel_size=eval(
                        layer_config["convolutionalLayer_kernelSize"]),
                    strides=eval(layer_config["convolutionalLayer_strides"]),
                    padding=layer_config["convolutionalLayer_padding"],
                    activation=layer_config["convolutionalLayer_activation"]
                )(x)

            elif "poolingLayer_poolSize" in layer_config:
                pool_type = layer_config.get("poolingLayer_type", "MaxPool")
                if pool_type == "MaxPool":
                    x = layers.MaxPooling2D(
                        pool_size=eval(layer_config["poolingLayer_poolSize"]),
                        strides=eval(layer_config["poolingLayer_strides"]),
                        padding=layer_config["poolingLayer_padding"]
                    )(x)
                elif pool_type == "AvgPool":
                    x = layers.AveragePooling2D(
                        pool_size=eval(layer_config["poolingLayer_poolSize"]),
                        strides=eval(layer_config["poolingLayer_strides"]),
                        padding=layer_config["poolingLayer_padding"]
                    )(x)

            elif "flattenLayer" in layer_config:
                x = layers.Flatten()(x)

            elif "denseLayer_units" in layer_config:
                x = layers.Dense(
                    units=layer_config["denseLayer_units"],
                    activation=layer_config["denseLayer_activation"]
                )(x)

            elif "dropoutLayer_rate" in layer_config:
                x = layers.Dropout(layer_config["dropoutLayer_rate"])(x)

        # Object Detection Output Heads
        bbox_output = layers.Dense(
            num_boxes * 4, activation="linear", name="bbox_output")(x)
        class_output = layers.Dense(
            num_classes, activation="softmax", name="class_output")(x)

        model = models.Model(inputs=inputs, outputs=[
                             bbox_output, class_output])

        return model
