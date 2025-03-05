from pydantic import BaseModel, Field
from typing import List, Optional

class HOGConfig(BaseModel):
    pixels_per_cell: Optional[List[int]] = Field(None, min_items=2, max_items=2)
    cells_per_block: Optional[List[int]] = Field(None, min_items=2, max_items=2)
    orientations: Optional[int] = None

class SIFTConfig(BaseModel):
    n_keypoints: Optional[int] = None
    contrast_threshold: Optional[float] = None
    edge_threshold: Optional[int] = None

class ORBConfig(BaseModel):
    n_keypoints: Optional[int] = None
    scale_factor: Optional[float] = None
    n_level: Optional[int] = None

class FeatureExtractionConfig(BaseModel):
    hog: Optional[HOGConfig] = None
    sift: Optional[SIFTConfig] = None
    orb: Optional[ORBConfig] = None