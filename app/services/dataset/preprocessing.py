import cv2
import numpy as np

from app.models.preprocessing import ImagePreprocessingConfig

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, image: np.ndarray, config: ImagePreprocessingConfig) -> np.ndarray:
        for key in config.priority:
            '''
            Validate input image
            '''
            if image is None or image.size == 0:
                raise ValueError("Input image is empty or None.")

            '''
            Basic Image operations 
            '''
            
            if key == 'grayscale' and config.grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if key == 'resize' and config.resize:
                image = cv2.resize(image, config.resize)

            # Crop
            if key == 'crop' and config.crop:
                crop_width, crop_height = config.crop[0]
                x, y = config.crop[1]

                # Validate crop dimensions
                if (x + crop_width > image.shape[1]) or (y + crop_height > image.shape[0]):
                    raise ValueError("Crop size exceeds image dimensions.")

                image = image[y:y + crop_height, x:x + crop_width]

            # Rotation
            if key == 'rotate' and config.rotate:
                center = (image.shape[1] // 2, image.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, config.rotate, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            # Flipping
            if key == 'flip' and config.flip:
                if config.flip not in [0, 1, -1]:
                    raise ValueError("Invalid flip direction. Use 0 (vertical), 1 (horizontal), or -1 (both).")
                image = cv2.flip(image, config.flip)

            # Perspective Transformation
            if key == 'pers_trans' and config.pers_trans:
                if len(config.pers_trans[0]) != 4 or len(config.pers_trans[1]) != 4:
                    raise ValueError("Perspective transformation requires exactly 4 source and 4 destination points.")

                src_points = np.array(config.pers_trans[0], dtype=np.float32)
                dst_points = np.array(config.pers_trans[1], dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

            '''
            Thresholding
            '''
            if key == 'thresh_percent' and config.thresh_percent:
                flattened = image.flatten()
                threshold_value = np.percentile(flattened, config.thresh_percent)
                _, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

            '''
            Image Enhancement
            '''
            # Normalization
            if key == 'normalize' and config.normalize:
                min_value, max_value = config.normalize
                image = cv2.normalize(image, None, min_value, max_value, cv2.NORM_MINMAX)

            # Histogram Equalization
            if key == 'histogram_equalization' and config.histogram_equalization:
                if len(image.shape) == 2:  # Grayscale image
                    image = cv2.equalizeHist(image)
                else:  # Color image
                    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
                    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

            # Sharpening
            if key == 'sharpening' and config.sharpening:
                kernel = np.array([[-1, -1, -1], [-1, 9 + config.sharpening, -1], [-1, -1, -1]])
                image = cv2.filter2D(image, -1, kernel)

            # Unsharp Masking
            if key == 'unsharp' and config.unsharp:
                blurred = cv2.GaussianBlur(image, (config.unsharp[0] * 2 + 1, config.unsharp[0] * 2 + 1), 0)
                mask = cv2.addWeighted(image, 1 + config.unsharp[1], blurred, -config.unsharp[1], 0)
                mask = np.clip(mask, 0, 255).astype(np.uint8)
                image = mask

            '''
            Edge Detection
            '''
            if key == 'laplacian' and config.laplacian:
                image = cv2.Laplacian(image, cv2.CV_64F, ksize=config.laplacian)
                image = np.uint8(np.absolute(image))

            '''
            Blurring / Denoising
            '''
            if key == 'gaussian_blur' and config.gaussian_blur:
                image = cv2.GaussianBlur(image, config.gaussian_blur[0], config.gaussian_blur[1])

            if key == 'median_blur' and config.median_blur:
                image = cv2.medianBlur(image, config.median_blur)

            if key == 'mean_blur' and config.mean_blur:
                image = cv2.blur(image, config.mean_blur)


            '''
            Color and Intensity Adjustments
            '''
            # Log Transformation
            if key == 'log_trans' and config.log_trans:
                image = image.astype(np.float32)
                image = image + 1  # Add 1 to avoid log(0)
                image = np.log(image)  # Apply log transformation
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the output
                image = image.astype(np.uint8)  # Convert back to uint8 for compatibility

            '''
            Morphological Operations
            '''

            if key == 'dilation' and config.dilation:
                kernel = np.ones((config.dilation, config.dilation), np.uint8)
                image = cv2.dilate(image, kernel, iterations=1)

            if key == 'erosion' and config.erosion:
                kernel = np.ones((config.erosion, config.erosion), np.uint8)
                image = cv2.erode(image, kernel, iterations=1)

            if key == 'opening' and config.opening:
                kernel = np.ones((config.opening, config.opening), np.uint8)
                image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

            if key == 'closing' and config.closing:
                kernel = np.ones((config.closing, config.closing), np.uint8)
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                
        return image

        # if config.histogram_matching_reference_image is not None:
        #     reference = cv2.imread(config.histogram_matching_reference_image, cv2.IMREAD_GRAYSCALE)
        #     image = self.histogram_matching(image, reference)