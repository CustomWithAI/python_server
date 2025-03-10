from pydantic import BaseModel, Field
from typing import Tuple, Optional, List, Literal

class DataAugmentationConfig(BaseModel):
    grayscale: Optional[Tuple[float]] = Field(None, description="Convert image to grayscale.")
    resize: Optional[Tuple[int, int]] = Field(None, description="Resize image to (width, height).")
    crop: Optional[Tuple[float, Tuple[int, int, int, int]]] = Field(None, description="Crop a region (x, y, width, height).")
    rotate: Optional[Tuple[float, int]] = Field(None, description="Rotate image by a random angle within a range.")
    flip: Optional[Tuple[float, int]] = Field(None, description="Flip image (1=horizontal, 0=vertical, -1=both).")
    translate: Optional[Tuple[float, Tuple[int, int]]] = Field(None, description="Shift image along x and y axes.")
    scale: Optional[Tuple[float, Tuple[float, float]]] = Field(None, description="Scale image by a factor (x_scale, y_scale).")
    brightness: Optional[Tuple[float, float]] = Field(None, description="Adjust image brightness (-1 to 1).")
    contrast_stretching: Optional[Tuple[float, Tuple[float, float]]] = Field(None, description="Normalize contrast using min-max scaling.")
    hist_equalization: Optional[Tuple[float]] = Field(None, description="Apply histogram equalization.")
    adaptive_equalization: Optional[Tuple[float, float]] = Field(None, description="Apply CLAHE (clip limit).")
    saturation: Optional[Tuple[float, float]] = Field(None, description="Adjust color saturation.")
    hue: Optional[Tuple[float, int]] = Field(None, description="Adjust hue values.")
    gamma: Optional[Tuple[float, float]] = Field(None, description="Apply gamma correction.")
    gaussian_blur: Optional[Tuple[float, Tuple[int, float]]] = Field(None, description="Apply Gaussian blur with kernel size and sigma.")
    motion_blur: Optional[Tuple[float, Tuple[int, int]]] = Field(None, description="Simulate motion blur (Kernel size, angle).")
    zoom_blur: Optional[Tuple[float, int]] = Field(None, description="Apply zoom blur effect (Zoom factor).")
    sharpening: Optional[Tuple[float, float]] = Field(None, description="Sharpen the image (higher=stronger).")
    gaussian_noise: Optional[Tuple[float, Tuple[float, float]]] = Field(None, description="Add Gaussian noise (Mean, Variance).")
    salt_pepper_noise: Optional[Tuple[float, Tuple[float, float]]] = Field(None, description="Add salt-and-pepper noise (Amount, Salt-vs-Pepper ratio).")
    random_erasing: Optional[Tuple[float, Tuple[int, int, int, int]]] = Field(None, description="Randomly erase a region (x, y, width, height).")
    elastic_distortion: Optional[Tuple[float, Tuple[int, int]]] = Field(None, description="Apply elastic transformation (Alpha, Sigma).")
    number: int = Field(100, description="Number of final datasets.")
    priority: List[
        Literal[
            'grayscale', 'resize', 'crop', 'rotate', 'flip', 'translate', 'scale', 'brightness', 'contrast_stretching',
            'hist_equalization', 'adaptive_equalization','saturation', 'hue', 'gamma', 'gaussian_blur', 'motion_blur',
            'zoom_blur','sharpening', 'gaussian_noise','salt_pepper_noise', 'random_erasing', 'elastic_distortion'
        ]
    ]