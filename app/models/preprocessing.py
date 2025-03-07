from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

class ImagePreprocessingConfig(BaseModel):
    grayscale: Optional[bool] = Field(None, description="Convert image to grayscale.")
    resize: Optional[Tuple[int, int]] = Field(None, description="Resize image to (width, height).")
    crop: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = Field(None, description="Crop (width, height) from (x, y).")
    rotate: Optional[int] = Field(None, description="Rotate image by degrees.")
    flip: Optional[int] = Field(None, description="Flip image: 1=horizontal, 0=vertical, -1=both.")
    pers_trans: Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = Field(
        None, description="Perspective transformation: 4 src and 4 dest points."
    )
    thresh_percent: Optional[int] = Field(None, description="Apply thresholding at percentile.")
    normalize: Optional[Tuple[int, int]] = Field(None, description="Normalize pixel values (min, max).")
    histogram_equalization: Optional[bool] = Field(None, description="Apply histogram equalization.")
    sharpening: Optional[int] = Field(None, description="Apply sharpening (higher=stronger).")
    unsharp: Optional[Tuple[int, float]] = Field(None, description="Unsharp masking: [kernel size, intensity].")
    laplacian: Optional[int] = Field(None, description="Laplacian edge detection kernel size.")
    gaussian_blur: Optional[Tuple[Tuple[int, int], float]] = Field(None, description="Gaussian blur: kernel size and sigma.")
    median_blur: Optional[int] = Field(None, description="Median blur kernel size.")
    mean_blur: Optional[Tuple[int, int]] = Field(None, description="Mean blur kernel size (width, height).")
    log_trans: Optional[bool] = Field(None, description="Apply logarithmic transformation.")
    dilation: Optional[int] = Field(None, description="Morphological dilation kernel size.")
    erosion: Optional[int] = Field(None, description="Morphological erosion kernel size.")
    opening: Optional[int] = Field(None, description="Morphological opening kernel size.")
    closing: Optional[int] = Field(None, description="Morphological closing kernel size.")
