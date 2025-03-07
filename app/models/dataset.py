from pydantic import BaseModel, model_validator
from typing import Literal, Optional

from app.models.preprocessing import ImagePreprocessingConfig
from app.models.augmentation import DataAugmentationConfig

class DatasetConfig(BaseModel):
    type: Optional[Literal['classification', 'object_detection', 'segmentation']] = None
    preprocess: Optional[ImagePreprocessingConfig] = None
    augmentation: Optional[DataAugmentationConfig] = None
    
    @model_validator(mode="after")
    def validate_model(cls, values: "DatasetConfig"):
        if values.augmentation and not values.type:
            raise ValueError("Augmentation requires dataset type.")
        return values