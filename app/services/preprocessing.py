import cv2
import numpy as np
from typing import Optional, Tuple, List
from app.config import PreConfig

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, image: np.ndarray, config: PreConfig) -> np.ndarray:

        '''
        Validate input image
        '''
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")


        '''
        Basic Image operations 
        '''
        # Grayscale
        if config.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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


        '''
        Thresholding
        '''
        if config.percentage_threshold:
            threshold_value = int(config.percentage_threshold * 255)
            _, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

        '''
        Image Enhancement
        '''
        # Normalization
        if config.normalization_range:
            min_value, max_value = config.normalization_range
            image = cv2.normalize(image, None, min_value, max_value, cv2.NORM_MINMAX)

        # Histogram Equalization
        if config.histogram_equalization:
            if len(image.shape) == 2:  # Grayscale image
                image = cv2.equalizeHist(image)
            else:  # Color image
                image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
                image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

        # Sharpening
        if config.sharpening_intensity:
            kernel = np.array([[-1, -1, -1], [-1, 9 + config.sharpening_intensity, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

        # Unsharp Masking
        if config.unsharp_radius and config.unsharp_amount:
            blurred = cv2.GaussianBlur(image, (config.unsharp_radius * 2 + 1, config.unsharp_radius * 2 + 1), 0)
            mask = cv2.addWeighted(image, 1 + config.unsharp_amount, blurred, -config.unsharp_amount, 0)
            mask = np.clip(mask, 0, 255).astype(np.uint8)
            image = mask

        '''
        Edge Detection
        '''
        
        if config.edge_detection_method == "laplacian" and config.laplacian_kernel_size:
            image = cv2.Laplacian(image, cv2.CV_64F, ksize=config.laplacian_kernel_size)
            image = np.uint8(np.absolute(image))

        '''
        Blurring / Denoising
        '''
        if config.gaussian_blur_kernel_size and config.gaussian_blur_sigma:
            image = cv2.GaussianBlur(image, config.gaussian_blur_kernel_size, config.gaussian_blur_sigma)

        if config.median_blur_kernel_size:
            image = cv2.medianBlur(image, config.median_blur_kernel_size)

        if config.mean_blur_kernel_size:
            image = cv2.blur(image, config.mean_blur_kernel_size)

        '''
        Color and Intensity Adjustments
        '''
        # Log Transformation
        if config.log_transformation:
            if image is None or image.size == 0:
                raise ValueError("Input image is empty or not properly loaded.")
            
            # Ensure the image is a valid type and normalize to avoid log(0)
            image = image.astype(np.float32)
            image = image + 1  # Add 1 to avoid log(0)
            image = np.log(image)  # Apply log transformation
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the output
            image = image.astype(np.uint8)  # Convert back to uint8 for compatibility

        if config.histogram_matching_reference_image is not None:
            reference = cv2.imread(config.histogram_matching_reference_image, cv2.IMREAD_GRAYSCALE)
            image = self.histogram_matching(image, reference)

        '''
        Thresholding
        '''
        if config.percentage_threshold is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            flattened = image.flatten()
            threshold_value = np.percentile(flattened, config.percentage_threshold)
            _, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

        '''
        Morphological Operations
        '''
        kernel = np.ones((config.morph_kernel_size, config.morph_kernel_size), np.uint8) if config.morph_kernel_size else None

        if config.dilation:
            image = cv2.dilate(image, kernel, iterations=1)

        if config.erosion:
            image = cv2.erode(image, kernel, iterations=1)

        if config.opening:
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        if config.closing:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image


    def histogram_matching(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Matches the histogram of the source image to the reference image.
        """
        src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
        ref_hist, bins = np.histogram(reference.flatten(), 256, [0, 256])
        cdf_src = src_hist.cumsum()
        cdf_ref = ref_hist.cumsum()

        cdf_src_normalized = cdf_src * (255 / cdf_src[-1])
        cdf_ref_normalized = cdf_ref * (255 / cdf_ref[-1])

        lookup_table = np.zeros(256)
        g_j = 0
        for g_i in range(256):
            while g_j < 256 and cdf_src_normalized[g_i] > cdf_ref_normalized[g_j]:
                g_j += 1
            lookup_table[g_i] = g_j

        matched = cv2.LUT(source, lookup_table.astype(np.uint8))
        return matched

