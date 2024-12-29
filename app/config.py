from pydantic import BaseModel
from typing import Optional, Tuple, List

class PreConfig(BaseModel):
    grayscale: Optional[bool] = False
    target_size: Optional[Tuple[int, int]] = None  # Resize target (width, height)
    crop_size: Optional[Tuple[int, int]] = None   # Crop size (width, height)
    crop_position: Optional[Tuple[int, int]] = None  # Crop position (x, y)
    rotation_angle: Optional[float] = None  # Rotation angle in degrees
    flip_direction: Optional[int] = None  # Flip direction: 0 (vertical), 1 (horizontal), -1 (both)
    perspective_src_points: Optional[List[Tuple[float, float]]] = None  # Perspective src points
    perspective_dst_points: Optional[List[Tuple[float, float]]] = None  # Perspective dst points

    # Thresholding
    percentage_threshold: Optional[float] = None  # Percentage threshold for binarization

    # Image Enhancement
    normalization_range: Optional[Tuple[int, int]] = None  # Normalization range (min, max)
    histogram_equalization: Optional[bool] = False  # Enable/disable histogram equalization
    sharpening_intensity: Optional[float] = None  # Sharpening intensity
    unsharp_radius: Optional[int] = None  # Radius for unsharp masking
    unsharp_amount: Optional[float] = None  # Amount for unsharp masking


    '''
    Edge Detection
    '''
    edge_detection_method: Optional[str] = None  # "laplacian"
    # Example: "laplacian" for edge detection
    laplacian_kernel_size: Optional[int] = None  # Example: 3 (must be an odd number like 3, 5, 7, etc.)

    '''
    Blurring / Denoising
    '''
    gaussian_blur_kernel_size: Optional[Tuple[int, int]] = None  
    # Example: (5, 5) for Gaussian blur with a 5x5 kernel
    gaussian_blur_sigma: Optional[float] = None  
    # Example: 1.0 for standard deviation of Gaussian blur

    median_blur_kernel_size: Optional[int] = None  
    # Example: 3 (must be an odd number like 3, 5, 7, etc.)

    mean_blur_kernel_size: Optional[Tuple[int, int]] = None  
    # Example: (3, 3) for a 3x3 mean blur filter

    '''
    Color and Intensity Adjustments
    '''
    log_transformation: Optional[bool] = False  
    # Example: True to apply logarithmic transformation for intensity adjustment

    histogram_matching_reference_image: Optional[str] = None  
    # Example: "path/to/reference.jpg" for histogram matching with the reference image

    '''
    Thresholding
    '''
    threshold_value: Optional[int] = None  
    # Example: 128 (threshold value to convert image to binary)

    '''
    Morphological Operations
    '''
    morph_kernel_size: Optional[int] = None  
    # Example: 3 (must be an odd number like 3, 5, 7, etc.)

    dilation: Optional[bool] = False  
    # Example: True to perform dilation (expand bright areas in the image)

    erosion: Optional[bool] = False  
    # Example: True to perform erosion (shrink bright areas in the image)

    opening: Optional[bool] = False  
    # Example: True to perform opening (erosion followed by dilation)

    closing: Optional[bool] = False  
    # Example: True to perform closing (dilation followed by erosion)
