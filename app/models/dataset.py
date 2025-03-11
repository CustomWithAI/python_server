from pydantic import BaseModel, model_validator
from typing import Literal, Optional, List, Dict, Any

from app.models.preprocessing import ImagePreprocessingConfig
from app.models.augmentation import DataAugmentationConfig

class DatasetImage(BaseModel):
    url: str
    image_class: str
    annotation: Dict[str, Any]

class PrepareDatasetRequest(BaseModel):
    train_data: List[DatasetImage]
    test_data: List[DatasetImage]
    valid_data: List[DatasetImage]

class DatasetConfigRequest(BaseModel):
    type: Optional[Literal['classification', 'object_detection', 'segmentation']] = None
    preprocess: Optional[ImagePreprocessingConfig] = None
    augmentation: Optional[DataAugmentationConfig] = None
    
    @model_validator(mode="after")
    def validate_model(cls, values: "DatasetConfigRequest"):
        if values.augmentation and not values.type:
            raise ValueError("Augmentation requires dataset type.")
        return values