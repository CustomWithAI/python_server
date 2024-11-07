import cv2
import numpy as np


class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, image, config):
        # Resize operation (if "resize" key exists)
        if "resize" in config:
            target_size = tuple(config["resize"])  # Ensure it's a tuple (width, height)
            image = cv2.resize(image, target_size)

        # Crop operation (if "crop" key exists)
        if "crop" in config:
            crop_size = tuple(config["crop_size"])  # (width, height)
            position = config[
                "position"
            ]  # Position (top-left) for cropping, e.g., (x, y)
            x, y = position
            image = image[y : y + crop_size[1], x : x + crop_size[0]]

        # Rotation operation (if "rotate" key exists)
        if "rotate" in config:
            angle = config["rotate"]
            # Get the image center and rotation matrix
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

        # Flipping operation (if "flip" key exists)
        if "flip" in config:
            flip_direction = config[
                "flip"
            ]  # 0 for vertical flip, 1 for horizontal flip
            image = cv2.flip(image, flip_direction)

        # Perspective Transformation operation (if "perspective" key exists)
        if "perspective" in config:
            src_points = np.float32(
                config["perspective"]["src_points"]
            )  # source points
            dst_points = np.float32(
                config["perspective"]["dst_points"]
            )  # destination points
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

        return image
