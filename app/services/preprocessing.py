import cv2
import numpy as np
from typing import Optional, Tuple, List
from app.config import PreConfig

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, image: np.ndarray, config: PreConfig) -> np.ndarray:
        # Resize
        if config.target_size:
            image = cv2.resize(image, config.target_size)

        # Crop
        if config.crop_size and config.crop_position:
            crop_width, crop_height = config.crop_size
            x, y = config.crop_position

            # Validate crop dimensions
            if (x + crop_width > image.shape[1]) or (y + crop_height > image.shape[0]):
                raise ValueError("Crop size exceeds image dimensions.")

            image = image[y:y + crop_height, x:x + crop_width]

        # Rotation
        if config.rotation_angle is not None:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, config.rotation_angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Flipping
        if config.flip_direction is not None:
            if config.flip_direction not in [0, 1, -1]:
                raise ValueError("Invalid flip direction. Use 0 (vertical), 1 (horizontal), or -1 (both).")
            image = cv2.flip(image, config.flip_direction)

        # Perspective Transformation
        if config.perspective_src_points and config.perspective_dst_points:
            if len(config.perspective_src_points) != 4 or len(config.perspective_dst_points) != 4:
                raise ValueError("Perspective transformation requires exactly 4 source and 4 destination points.")

            src_points = np.array(config.perspective_src_points, dtype=np.float32)
            dst_points = np.array(config.perspective_dst_points, dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

        return image
