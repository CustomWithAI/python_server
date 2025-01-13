import cv2
import numpy as np
from typing import Optional, Tuple, List
import random

class Augmentation:
    def __init__(self):
        pass

    def augmentation(self, image: np.ndarray, config) -> np.ndarray:

        '''
        Validate input image
        '''
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")
        
        # Check if the original image is grayscale
        is_grayscale = (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[2] == 1)

        for key, value in config.items():
            print(f"{key}: {value}")
            
            '''
            Geometric Transformations
            '''
            if key == 'rotate':
                if random.random() < value[0]:  # Probability
                    rows, cols = image.shape[:2]
                    angle = random.uniform(-value[1], value[1])  # Angle range
                    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    image = cv2.warpAffine(image, matrix, (cols, rows))

            if key == 'crop':
                if random.random() < value[0]:  # Probability
                    x, y, w, h = value[1]  # Crop parameters
                    image = image[y:y + h, x:x + w]

            if key == 'flip':
                if random.random() < value[0]:  # Probability
                    direction = value[1]  # 0 = vertical, 1 = horizontal, -1 = both
                    image = cv2.flip(image, direction)

            if key == 'translate':
                if random.random() < value[0]:  # Probability
                    rows, cols = image.shape[:2]
                    tx, ty = value[1]  # Shift range (x, y)
                    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                    image = cv2.warpAffine(image, matrix, (cols, rows))

            if key == 'scale':
                if random.random() < value[0]:  # Probability
                    fx, fy = value[1]  # Scaling factors
                    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

            if key == 'grayscale':
                if random.random() < value[0]:  # Probability
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            '''
            Color and Intensity Adjustments
            '''
            if key == 'brightness':
                if random.random() < value[0]:
                    factor = random.uniform(-value[1], value[1])
                    image = cv2.convertScaleAbs(image, alpha=1, beta=factor * 255)

            if key == 'contrast_stretching':
                if random.random() < value[0]:
                    lower, upper = value[1]
                    in_range = (lower * 255, upper * 255)
                    image = cv2.normalize(image, None, alpha=in_range[0], beta=in_range[1], norm_type=cv2.NORM_MINMAX)

            if key == 'hist_equalization':
                if random.random() < value[0]:
                    if len(image.shape) == 2:  # Grayscale image
                        image = cv2.equalizeHist(image)
                    else:
                        for i in range(3):  # Equalize each channel
                            image[:, :, i] = cv2.equalizeHist(image[:, :, i])

            if key == 'adaptive_equalization':
                if random.random() < value[0]:
                    clip_limit = value[1]
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                    if len(image.shape) == 2:  # Grayscale
                        image = clahe.apply(image)
                    else:
                        for i in range(3):
                            image[:, :, i] = clahe.apply(image[:, :, i])

            if key == 'saturation':
                if random.random() < value[0]:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], value[1])
                    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            if key == 'hue':
                if random.random() < value[0]:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 0] = cv2.add(hsv[:, :, 0], value[1])
                    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            if key == 'gamma':
                if random.random() < value[0]:
                    gamma = value[1]
                    inv_gamma = 1.0 / gamma
                    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
                    image = cv2.LUT(image, table)

            '''
            Blurring and Sharpening
            '''
            if key == 'gaussian_blur':
                if random.random() < value[0]:
                    kernel_size, sigma = value[1]
                    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

            if key == 'motion_blur':
                if random.random() < value[0]:
                    kernel_size, angle = value[1]
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                    matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
                    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
                    kernel = kernel / kernel.sum()
                    image = cv2.filter2D(image, -1, kernel)

            if key == 'zoom_blur':
                if random.random() < value[0]:
                    zoom_factor = value[1]
                    center = tuple(np.array(image.shape[:2][::-1]) / 2)
                    original_size = image.shape[:2][::-1]
                    for i in range(1, zoom_factor):
                        scale_factor = 1 - i * 0.1
                        temp = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
                        new_size = temp.shape[:2][::-1]
                        
                        # Calculate margins to crop to the original size
                        delta_w = (new_size[0] - original_size[0]) // 2
                        delta_h = (new_size[1] - original_size[1]) // 2
                        
                        if delta_w > 0 and delta_h > 0:
                            temp = temp[delta_h:-delta_h, delta_w:-delta_w]
                        elif delta_w > 0:
                            temp = temp[:, delta_w:-delta_w]
                        elif delta_h > 0:
                            temp = temp[delta_h:-delta_h, :]
                        
                        # Ensure temp and image have the same size
                        temp = cv2.resize(temp, original_size)

                        image = cv2.addWeighted(image, 0.5, temp, 0.5, 0)

            if key == 'sharpening':
                if random.random() < value[0]:
                    factor = value[1]
                    kernel = np.array([[0, -factor, 0], [-factor, 1 + 4 * factor, -factor], [0, -factor, 0]])
                    image = cv2.filter2D(image, -1, kernel)

            '''
            Noise Injection
            '''
            if key == 'gaussian_noise':
                if random.random() < value[0]:
                    mean, var = value[1]
                    noise = np.random.normal(mean, var ** 0.5, image.shape)
                    image = cv2.add(image, noise.astype('uint8'))

            if key == 'salt_pepper_noise':
                if random.random() < value[0]:
                    amount, s_vs_p = value[1]
                    noisy = image.copy()
                    num_salt = np.ceil(amount * image.size * s_vs_p)
                    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
                    noisy[tuple(coords)] = 255
                    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
                    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
                    noisy[tuple(coords)] = 0
                    image = noisy

            '''
            Random Erasing
            '''
            if key == 'random_erasing':
                if random.random() < value[0]:
                    x, y, w, h = value[1]
                    image[y:y + h, x:x + w] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            '''
            Elastic Transformation
            ''' 
            if key == 'elastic_distortion':
                if random.random() < value[0]:
                    alpha, sigma = value[1]
                    shape = image.shape[:2]

                    # Generate random displacement fields
                    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1) * alpha, (2 * sigma + 1, 2 * sigma + 1), sigma)
                    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1) * alpha, (2 * sigma + 1, 2 * sigma + 1), sigma)

                    # Create a grid of coordinates
                    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

                    # Apply the displacement fields
                    map_x = (x + dx).astype('float32')
                    map_y = (y + dy).astype('float32')

                    # Remap the image
                    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            if is_grayscale and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image