from pydantic import BaseModel
from typing import Literal

class InputLayer(BaseModel):
    inputLayer_batchSize: int = 32

class ConvolutionalLayer(BaseModel):
    convolutionalLayer_filters: int = 32
    convolutionalLayer_kernelSize: str = "(3, 3)"
    convolutionalLayer_strides: str = "(1, 1)"
    convolutionalLayer_padding: Literal['valid', 'same'] = 'valid'
    convolutionalLayer_activation: Literal['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'linear', 'softmax'] = 'relu'

class Pooling2DLayer(BaseModel):
    poolingLayer_poolSize: str = "(2, 2)"
    poolingLayer_strides: str = "(2, 2)"
    poolingLayer_padding: Literal['valid', 'same'] = 'valid'
    poolingLayer_type: Literal['MaxPool', 'AvgPool']

class NormalizationLayer(BaseModel):
    normalizationLayer_type: Literal['BatchNormalization', 'LayerNormalization']
    normalizationLayer_momentum: float = 0.99
    normalizationLayer_epsilon: float = 1e-05

class ActivationLayer(BaseModel):
    activationLayer_function: Literal['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'linear', 'softmax'] = 'relu'

class DropoutLayer(BaseModel):
    dropoutLayer_rate: float = 0.5

class DenseLayer(BaseModel):
    denseLayer_units: int = 512
    denseLayer_activation: Literal['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'softplus', 'softsign', 'linear', 'softmax'] = 'relu'

class FlattenLayer(BaseModel):
    flattenLayer: bool = True

class DenseBoundingBoxOutput(BaseModel):
    bbox_output_units: int = 4
    bbox_output_activation: Literal['linear'] = 'linear'

class DenseClassOutput(BaseModel):
    class_output_units: int = 3
    class_output_activation: Literal['softmax'] = 'softmax'
