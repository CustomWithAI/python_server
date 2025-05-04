from pydantic import BaseModel
from typing import Optional, Union, Literal, List

from app.models.feature_extraction import FeatureExtractionConfig
from app.models.construct_model import *

ClassificationModels = Literal['resnet50', 'vgg16', 'mobilenetv2']

ObjectDetectionModels = Literal['yolov5', 'yolov8', 'yolov11']

SegmentationModels = Literal['yolov8', 'yolov11']

ObjectDetectionWeightSizes = Literal[
    "yolov5s.pt", "yolov5m.pt", "yolov5l.pt",
    "yolov8s.pt", "yolov8m.pt", "yolov8l.pt",
    "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"
]

SegmentationWeightSizes = Literal[
    "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt",
    "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt"
]

class ReduceLrOnPlateau(BaseModel):
    monitor: Literal['val_loss', 'loss', 'val_accuracy', 'accuracy', 'val_mean_squared_error', 'mean_absolute_error', 'val_mean_absolute_error'] = 'val_loss'
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 0.00001

class EarlyStopping(BaseModel):
    monitor: Literal['val_loss', 'loss', 'val_accuracy', 'accuracy', 'val_mean_squared_error', 'mean_absolute_error', 'val_mean_absolute_error'] = 'val_loss'
    patience: float = 3

class ClassificationCallbacks(BaseModel):
    reduce_lr_on_plateau: Optional[ReduceLrOnPlateau] = None
    early_stopping: Optional[EarlyStopping] = None

class ClassificationTrainingConfig(BaseModel):
    learning_rate: float = 0.001
    learning_rate_scheduler: None = None
    momentum: float = 0.9
    optimizer_type: Literal['adam', 'sgd'] = 'adam'
    batch_size: int = 32
    epochs: int = 10
    loss_function: Literal['categorical_crossentropy'] = 'categorical_crossentropy'
    unfreeze: int = 0
    callbacks: Optional[ClassificationCallbacks] = None


class ObjectDetectionTrainingConfig(BaseModel):
    batch_size: int = 40
    epochs: int = 10
    weight_size: ObjectDetectionWeightSizes


class SegmentationTrainingConfig(BaseModel):
    batch_size: int = 40
    epochs: int = 10
    weight_size: SegmentationWeightSizes


class ObjectDetectionConstructModelConfig(BaseModel):
    learning_rate: float = 0.001
    momentum: float = 0.9
    optimizer_type: Literal['adam', 'sgd'] = 'adam'
    batch_size: int = 8
    epochs: int = 5


class DeepLearningClassification(BaseModel):
    model: ClassificationModels
    training: ClassificationTrainingConfig


class DeepLearningObjectDetection(BaseModel):
    type: Literal['object_detection'] = 'object_detection'
    model: ObjectDetectionModels
    training: ObjectDetectionTrainingConfig


class DeepLearningSegmentation(BaseModel):
    type: Literal['segmentation'] = 'segmentation'
    model: SegmentationModels
    training: SegmentationTrainingConfig


DeepLearningYoloRequest = Union[DeepLearningObjectDetection,
                                DeepLearningSegmentation]

DeepLearningClassificationConstructModel = List[Union[
    InputLayer,
    ConvolutionalLayer,
    Pooling2DLayer,
    NormalizationLayer,
    ActivationLayer,
    DropoutLayer,
    DenseLayer,
    FlattenLayer,
]]

DeepLearningObjectDetectionConstructModel = List[Union[
    ConvolutionalLayer,
    Pooling2DLayer,
    FlattenLayer,
    DenseLayer,
    DropoutLayer,
]]

DeepLearningObjectDetectionConstructModelFeatex = List[Union[
    DenseLayer,
    DropoutLayer,
]]


class DeepLearningClassificationConstruct(BaseModel):
    model: DeepLearningClassificationConstructModel
    training: ClassificationTrainingConfig
    featex: Optional[FeatureExtractionConfig] = None


class DeepLearningObjectDetectionConstruct(BaseModel):
    model: DeepLearningObjectDetectionConstructModel
    training: ObjectDetectionConstructModelConfig


class DeepLearningObjectDetectionConstructFeatex(BaseModel):
    featex: FeatureExtractionConfig
    model: DeepLearningObjectDetectionConstructModelFeatex
    training: ObjectDetectionConstructModelConfig


DeepLearningObjectDetectionConstructRequest = Union[
    DeepLearningObjectDetectionConstruct,
    DeepLearningObjectDetectionConstructFeatex
]