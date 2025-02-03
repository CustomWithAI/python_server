import tensorflow as tf
from tensorflow.keras import layers, models


class ConstructDCLS(object):
    def __init__(self):
        pass

    def construct(self, config, input_shape):
        model = models.Sequential()
        for layer_config in config:
            if "inputLayer_batchSize" in layer_config:
                model.add(layers.InputLayer(
                    input_shape=input_shape,  # Directly pass input_shape
                    batch_size=layer_config["inputLayer_batchSize"]
                ))

            elif "convolutionalLayer_filters" in layer_config:
                model.add(layers.Conv2D(
                    filters=layer_config["convolutionalLayer_filters"],
                    kernel_size=eval(
                        layer_config["convolutionalLayer_kernelSize"]),
                    strides=eval(layer_config["convolutionalLayer_strides"]),
                    padding=layer_config["convolutionalLayer_padding"],
                    activation=layer_config["convolutionalLayer_activation"]
                ))

            elif "poolingLayer_poolSize" in layer_config:
                pool_type = layer_config.get("poolingLayer_type", "MaxPool")
                if pool_type == "MaxPool":
                    model.add(layers.MaxPooling2D(
                        pool_size=eval(layer_config["poolingLayer_poolSize"]),
                        strides=eval(layer_config["poolingLayer_strides"]),
                        padding=layer_config["poolingLayer_padding"]
                    ))
                elif pool_type == "AvgPool":
                    model.add(layers.AveragePooling2D(
                        pool_size=eval(layer_config["poolingLayer_poolSize"]),
                        strides=eval(layer_config["poolingLayer_strides"]),
                        padding=layer_config["poolingLayer_padding"]
                    ))

            elif "normalizationLayer_type" in layer_config:
                norm_type = layer_config["normalizationLayer_type"]
                if norm_type == "BatchNormalization":
                    model.add(layers.BatchNormalization(
                        momentum=layer_config["normalizationLayer_momentum"],
                        epsilon=layer_config["normalizationLayer_epsilon"]
                    ))
                elif norm_type == "LayerNormalization":
                    model.add(layers.LayerNormalization(
                        epsilon=layer_config["normalizationLayer_epsilon"]
                    ))

            elif "activationLayer_function" in layer_config:
                model.add(layers.Activation(
                    layer_config["activationLayer_function"]))

            elif "dropoutLayer_rate" in layer_config:
                model.add(layers.Dropout(layer_config["dropoutLayer_rate"]))

            elif "denseLayer_units" in layer_config:
                model.add(layers.Dense(
                    units=layer_config["denseLayer_units"],
                    activation=layer_config["denseLayer_activation"]
                ))

            elif "flattenLayer" in layer_config:
                model.add(layers.Flatten())

        return model
