from fastapi import HTTPException
from pydantic import BaseModel, model_validator
from typing import Literal, Optional

class UseModelRequest(BaseModel):
    type: Literal['ml', 'dl_cls', 'dl_od_pt', 'dl_od_con', 'dl_seg']
    version: Optional[Literal["yolov5", "yolov8", "yolov11"]] = None
    
    @model_validator(mode="after")
    def validate_size(cls, values: "UseModelRequest"):
        
        if values.type in ["dl_od_pt", "dl_seg"] and values.version is None:
            raise HTTPException(
                status_code=400,
                detail="Version is not defined"
            )
        
        return values