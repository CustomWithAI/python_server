from pydantic import BaseModel, model_validator
from typing import Literal, Optional, List, Union

from app.models.preprocessing import ImagePreprocessingConfig
from app.models.augmentation import DataAugmentationConfig

class Point(BaseModel):
    x: float
    y: float

class ObjectDetectionPlot(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: str

class SegmentationPlot(BaseModel):
    points: List[Point]
    label: str

class ClassificationAnnotation(BaseModel):
    label: str

class ObjectDetectionAnnotation(BaseModel):
    annotation: List[ObjectDetectionPlot]

class SegmentationAnnotation(BaseModel):
    annotation: List[SegmentationPlot]

class ClassificationImage(BaseModel):
    url: str
    annotation: ClassificationAnnotation

class ObjectDetectionImage(BaseModel):
    url: str
    annotation: ObjectDetectionAnnotation

class SegmentationImage(BaseModel):
    url: str
    annotation: SegmentationAnnotation

class PrepareDatasetClassification(BaseModel):
    type: Literal['classification']
    labels: List[str]
    train_data: List[ClassificationImage]
    test_data: List[ClassificationImage]
    valid_data: List[ClassificationImage]

class PrepareDatasetObjectDetection(BaseModel):
    type: Literal['object_detection']
    labels: List[str]
    train_data: List[ObjectDetectionImage]
    test_data: List[ObjectDetectionImage]
    valid_data: List[ObjectDetectionImage]

class PrepareDatasetSegmentation(BaseModel):
    type: Literal['segmentation']
    labels: List[str]
    train_data: List[SegmentationImage]
    test_data: List[SegmentationImage]
    valid_data: List[SegmentationImage]

PrepareDatasetRequest = Union[
    PrepareDatasetClassification,
    PrepareDatasetObjectDetection,
    PrepareDatasetSegmentation,
]

class DatasetConfigRequest(BaseModel):
    type: Optional[Literal['classification', 'object_detection', 'segmentation']] = None
    preprocess: Optional[ImagePreprocessingConfig] = None
    augmentation: Optional[DataAugmentationConfig] = None
    
    @model_validator(mode="after")
    def validate_model(cls, values: "DatasetConfigRequest"):
        if values.augmentation and not values.type:
            raise ValueError("Augmentation requires dataset type.")
        return values