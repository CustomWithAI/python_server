from pydantic import BaseModel, Field
from typing import List, Optional

class HOGConfig(BaseModel):
    pixels_per_cell: List[int] = Field(default=[8, 8], min_items=2, max_items=2)
    cells_per_block: List[int] = Field(default=[2, 2], min_items=2, max_items=2)
    orientations: int = 9

class SIFTConfig(BaseModel):
    n_keypoints: int = 500
    contrast_threshold: float = 0.04
    edge_threshold: int = 10

class ORBConfig(BaseModel):
    n_keypoints: int = 500
    scale_factor: float = 1.2
    n_level: int = 8

class FeatureExtractionConfig(BaseModel):
    hog: Optional[HOGConfig] = None
    sift: Optional[SIFTConfig] = None
    orb: Optional[ORBConfig] = None