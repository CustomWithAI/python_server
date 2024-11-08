from pydantic import BaseModel
from typing import Optional, Tuple, Dict, List


class PreConfig(BaseModel):
    resize: Optional[Tuple[int, int]] = None
    crop_size: Optional[Tuple[int, int]] = None
    position: Optional[Tuple[int, int]] = None
    rotate: Optional[float] = None
    flip: Optional[int] = None
    perspective: Optional[Dict[str, List[List[float]]]] = None
