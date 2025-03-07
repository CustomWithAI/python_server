import cv2
import numpy as np

from app.models.preprocessing import ImagePreprocessingConfig

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, image: np.ndarray, config: ImagePreprocessingConfig) -> np.ndarray:
        '''
        Validate input image
        '''
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        '''
        Basic Image operations 
        '''
        # Grayscale
        if key=='grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize
        if key=='resize':
            image = cv2.resize(image, value)

        # Crop
        if key=='crop':
            crop_width, crop_height = value[0]
            x, y = value[1]

            # Validate crop dimensions
            if (x + crop_width > image.shape[1]) or (y + crop_height > image.shape[0]):
                raise ValueError("Crop size exceeds image dimensions.")

            image = image[y:y + crop_height, x:x + crop_width]

        # Rotation
        if key=='rotate':
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, value, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Flipping
        if key=='flip':
            if value not in [0, 1, -1]:
                raise ValueError("Invalid flip direction. Use 0 (vertical), 1 (horizontal), or -1 (both).")
            image = cv2.flip(image, value)

        # Perspective Transformation
        if key=='pers_trans':
            if len(value[0]) != 4 or len(value[1]) != 4:
                raise ValueError("Perspective transformation requires exactly 4 source and 4 destination points.")

            src_points = np.array(value[0], dtype=np.float32)
            dst_points = np.array(value[1], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

        '''
        Thresholding
        '''
        if key=='thresh_percent':
            flattened = image.flatten()
            threshold_value = np.percentile(flattened, value)
            _, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

        '''
        Image Enhancement
        '''
        # Normalization
        if key=="normalize":
            min_value, max_value = value
            image = cv2.normalize(image, None, min_value, max_value, cv2.NORM_MINMAX)

        # Histogram Equalization
        if key=='histogram_equalization':
            if len(image.shape) == 2:  # Grayscale image
                image = cv2.equalizeHist(image)
            else:  # Color image
                image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
                image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

        # Sharpening
        if key=='sharpening':
            kernel = np.array([[-1, -1, -1], [-1, 9 + value, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

        # Unsharp Masking
        if key=='unsharp':
            blurred = cv2.GaussianBlur(image, (value[0] * 2 + 1, value[0] * 2 + 1), 0)
            mask = cv2.addWeighted(image, 1 + value[1], blurred, -value[1], 0)
            mask = np.clip(mask, 0, 255).astype(np.uint8)
            image = mask

        '''
        Edge Detection
        '''
        if key=='laplacian':
            image = cv2.Laplacian(image, cv2.CV_64F, ksize=value)
            image = np.uint8(np.absolute(image))

        '''
        Blurring / Denoising
        '''
        if key=='gaussian_blur':
            image = cv2.GaussianBlur(image, value[0], value[1])

        if key=='median_blur':
            image = cv2.medianBlur(image, value)

        if key=='mean_blur':
            image = cv2.blur(image, value)


        '''
        Color and Intensity Adjustments
        '''
        # Log Transformation
        if key=='log_trans':
            image = image.astype(np.float32)
            image = image + 1  # Add 1 to avoid log(0)
            image = np.log(image)  # Apply log transformation
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the output
            image = image.astype(np.uint8)  # Convert back to uint8 for compatibility

        '''
        Morphological Operations
        '''
        if key=='dilation' or key=='erosion' or key=='opening' or key=='closing':
            kernel = np.ones((value, value), np.uint8) if value else None

            if key=='dilation':
                image = cv2.dilate(image, kernel, iterations=1)

            if key=='erosion':
                image = cv2.erode(image, kernel, iterations=1)

            if key=='opening':
                image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

            if key=='closing':
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                
        return image

        # if config.histogram_matching_reference_image is not None:
        #     reference = cv2.imread(config.histogram_matching_reference_image, cv2.IMREAD_GRAYSCALE)
        #     image = self.histogram_matching(image, reference)