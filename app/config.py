from pydantic import BaseModel
from typing import Optional, Tuple, List

class PreConfig(BaseModel):
    target_size: Optional[Tuple[int, int]] = None  # Resize target (width, height)
    crop_size: Optional[Tuple[int, int]] = None   # Crop size (width, height)
    crop_position: Optional[Tuple[int, int]] = None  # Crop position (x, y) - top-left corner
    rotation_angle: Optional[float] = None  # Rotation angle in degrees
    flip_direction: Optional[int] = None  # Flip direction: 0 (vertical), 1 (horizontal), -1 (both)
    perspective_src_points: Optional[List[Tuple[float, float]]] = None  # 4 source points for perspective transformation
    perspective_dst_points: Optional[List[Tuple[float, float]]] = None  # 4 destination points for perspective transformation
