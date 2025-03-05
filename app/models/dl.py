from pydantic import BaseModel
from typing import Union, Literal
from enum import Enum

class ClassificationTrainingConfig(BaseModel):
    learning_rate: float = 0.001
    learning_rate_scheduler: None = None
    momentum: float = 0.9
    optimizer_type: Literal['adam', 'sgd'] = 'adam'
    batch_size: int = 32
    epochs: int = 10
    loss_function: Literal['categorical_crossentropy'] = 'categorical_crossentropy'

class ObjectDetectionTrainingConfig(BaseModel):
    batch_size: int = 40
    epochs: int = 1
    weight_size: Literal['yolov5s.pt', 'm.pt', 'l.pt'] = 'yolov5s.pt'

class SegmentationTrainingConfig(BaseModel):
    batch_size: int = 40
    epochs: int = 1
    weight_size: Literal['yolov8s.pt', 'm.pt', 'l.pt'] = 'yolov8s.pt'

class DeepLearningClassification(BaseModel):
    type: Literal['classification'] = 'classification'
    model: Literal['resnet50', 'vgg16', 'mobilenetv2']
    training: ClassificationTrainingConfig

class DeepLearningObjectDetection(BaseModel):
    type: Literal['object_detection'] = 'object_detection'
    model: Literal['yolov5', 'yolov8', 'yolov11']
    training: ObjectDetectionTrainingConfig

class DeepLearningSegmentation(BaseModel):
    type: Literal['segmentation'] = 'segmentation'
    model: Literal['yolov8', 'yolov11']
    training: SegmentationTrainingConfig

DeepLearningYoloRequest = Union[DeepLearningObjectDetection, DeepLearningSegmentation]
