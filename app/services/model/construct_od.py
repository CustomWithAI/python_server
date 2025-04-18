import tensorflow as tf
from tensorflow.keras import layers, models

from app.models.dl import (
    DeepLearningObjectDetectionConstructModel,
    DeepLearningObjectDetectionConstructModelFeatex,
)
from app.models.construct_model import *


class ConstructDLOD(object):
    def __init__(self):
        pass


class ConstructDLOD(object):
    def __init__(self):
        pass


class ConstructDLOD(object):
    def __init__(self):
        pass

    def construct(self, config: DeepLearningObjectDetectionConstructModel, input_shape, num_classes, max_boxes=20):
        print("Received input_shape:", input_shape)
        # Ensure input shape is correct for Conv2D input
        inputs = layers.Input(shape=input_shape, name="input_layer")
        x = inputs

        for layer_config in config:
            if isinstance(layer_config, ConvolutionalLayer):
                x = layers.Conv2D(
                    filters=layer_config.convolutionalLayer_filters,
                    kernel_size=eval(
                        layer_config.convolutionalLayer_kernelSize),
                    strides=eval(layer_config.convolutionalLayer_strides),
                    padding=layer_config.convolutionalLayer_padding,
                    activation=layer_config.convolutionalLayer_activation
                )(x)

            elif isinstance(layer_config, Pooling2DLayer):
                if layer_config.poolingLayer_type == "MaxPool":
                    x = layers.MaxPooling2D(
                        pool_size=eval(layer_config.poolingLayer_poolSize),
                        strides=eval(layer_config.poolingLayer_strides),
                        padding=layer_config.poolingLayer_padding
                    )(x)
                elif layer_config.poolingLayer_type == "AvgPool":
                    x = layers.AveragePooling2D(
                        pool_size=eval(layer_config.poolingLayer_poolSize),
                        strides=eval(layer_config.poolingLayer_strides),
                        padding=layer_config.poolingLayer_padding
                    )(x)

            elif isinstance(layer_config, FlattenLayer):
                x = layers.Flatten()(x)

            elif isinstance(layer_config, DenseLayer):
                x = layers.Dense(
                    units=layer_config.denseLayer_units,
                    activation=layer_config.denseLayer_activation
                )(x)

            elif isinstance(layer_config, DropoutLayer):
                x = layers.Dropout(layer_config.dropoutLayer_rate)(x)

        # Object Detection Output Heads (updated for multiple boxes)
        # Create a dense layer first (without activation)
        bbox_dense = layers.Dense(
            max_boxes * 4, name="bbox_output")(x)
        # Reshape it to the correct dimensions
        bbox_output = layers.Reshape((max_boxes, 4), name="bbox_reshape")(bbox_dense)
        
        # Same for class output
        class_dense = layers.Dense(
            max_boxes * num_classes, name="class_output")(x)
        class_reshape = layers.Reshape((max_boxes, num_classes), name="class_reshape")(class_dense)
        class_output = layers.Activation("softmax", name="class_activation")(class_reshape)

        model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])

        return model

    def construct_od_featex(self, config: DeepLearningObjectDetectionConstructModelFeatex, input_shape, num_classes, num_boxes=1):
        print("Received input_shape:", input_shape)

        # Correct input shape for the feature vector
        inputs = tf.keras.layers.Input(shape=(input_shape,))
        x = inputs

        # Add dense layers based on the configuration
        for layer_config in config:
            if isinstance(layer_config, DenseLayer):
                x = tf.keras.layers.Dense(
                    units=layer_config.denseLayer_units,
                    activation=layer_config.denseLayer_activation,
                )(x)

            elif isinstance(layer_config, DropoutLayer):
                x = tf.keras.layers.Dropout(
                    layer_config.dropoutLayer_rate)(x)

        # Object Detection Output Heads (bbox and class outputs)
        bbox_output = tf.keras.layers.Dense(
            num_boxes * 4, activation="linear", name="bbox_output")(x)
        class_output = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="class_output")(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=[
            bbox_output, class_output])

        return model
