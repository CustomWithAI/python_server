import tensorflow as tf
from tensorflow.keras import layers, models

from app.models.dl import DeepLearningClassificationConstructModel
from app.models.construct_model import *


class ConstructDLCLS(object):
    def __init__(self):
        pass

    def construct(self, config: DeepLearningClassificationConstructModel, input_shape):
        model = models.Sequential()
        for layer_config in config:
            if isinstance(layer_config, InputLayer):
                model.add(layers.InputLayer(
                    input_shape=input_shape,  # Directly pass input_shape
                    batch_size=layer_config.inputLayer_batchSize
                ))
            
            elif isinstance(layer_config, ConvolutionalLayer):
                model.add(layers.Conv2D(
                    filters=layer_config.convolutionalLayer_filters,
                    kernel_size=eval(
                        layer_config.convolutionalLayer_kernelSize),
                    strides=eval(layer_config.convolutionalLayer_strides),
                    padding=layer_config.convolutionalLayer_padding,
                    activation=layer_config.convolutionalLayer_activation
                ))

            elif isinstance(layer_config, Pooling2DLayer):
                if layer_config.poolingLayer_type == "MaxPool":
                    model.add(layers.MaxPooling2D(
                        pool_size=eval(layer_config.poolingLayer_poolSize),
                        strides=eval(layer_config.poolingLayer_strides),
                        padding=layer_config.poolingLayer_padding
                    ))
                elif layer_config.poolingLayer_type == "AvgPool":
                    model.add(layers.AveragePooling2D(
                        pool_size=eval(layer_config.poolingLayer_poolSize),
                        strides=eval(layer_config.poolingLayer_strides),
                        padding=layer_config.poolingLayer_padding
                    ))

            elif isinstance(layer_config, NormalizationLayer):
                if layer_config.normalizationLayer_type == "BatchNormalization":
                    model.add(layers.BatchNormalization(
                        momentum=layer_config.normalizationLayer_momentum,
                        epsilon=layer_config.normalizationLayer_epsilon
                    ))
                elif layer_config.normalizationLayer_type == "LayerNormalization":
                    model.add(layers.LayerNormalization(
                        epsilon=layer_config.normalizationLayer_epsilon
                    ))

            elif isinstance(layer_config, ActivationLayer):
                model.add(layers.Activation(
                    layer_config.activationLayer_function))

            elif isinstance(layer_config, DropoutLayer):
                model.add(layers.Dropout(layer_config.dropoutLayer_rate))

            elif isinstance(layer_config, DenseLayer):
                model.add(layers.Dense(
                    units=layer_config.denseLayer_units,
                    activation=layer_config.denseLayer_activation
                ))

            elif isinstance(layer_config, FlattenLayer):
                model.add(layers.Flatten())

        return model
