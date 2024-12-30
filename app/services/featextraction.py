import cv2
import numpy as np
from typing import Optional, Tuple, List

class FeatureExtraction:
    def __init__(self):
        pass

    def extract_features(self, image: np.ndarray, config) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        """
        Feature Extractions
        """
        features = None

        for key, value in config.items():
            if key == 'hog':
                cell_size = value[0]
                block_size = value[1]
                orientations = value[2]

                hog = cv2.HOGDescriptor(
                    _winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                              image.shape[0] // cell_size[0] * cell_size[0]),
                    _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
                    _blockStride=(cell_size[1], cell_size[0]),
                    _cellSize=(cell_size[1], cell_size[0]),
                    _nbins=orientations
                )
                features = hog.compute(image)

            # elif key == 'sift':
            #     n_features = params.get('n_features', 500)
            #     contrast_threshold = params.get('contrast_threshold', 0.04)
            #     edge_threshold = params.get('edge_threshold', 10)

            #     sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold,
            #                            edgeThreshold=edge_threshold)
            #     keypoints, features = sift.detectAndCompute(image, None)

            # elif key == 'surf':
            #     hessian_threshold = params.get('hessian_threshold', 400)
            #     n_octaves = params.get('n_octaves', 4)
            #     n_octave_layers = params.get('n_octave_layers', 3)

            #     surf = cv2.SURF_create(hessianThreshold=hessian_threshold, nOctaves=n_octaves,
            #                            nOctaveLayers=n_octave_layers)
            #     keypoints, features = surf.detectAndCompute(image, None)

        return features

    # def select_features(self, features: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    #     """
    #     Select features from the provided feature matrix based on the configuration.
    #     """
    #     for key, params in config.items():
    #         if key == 'pca':
    #             n_components = params.get('n_components', None)
    #             whiten = params.get('whiten', False)

    #             pca = PCA(n_components=n_components, whiten=whiten)
    #             features = pca.fit_transform(features)

    #         elif key == 'lda':
    #             n_components = params.get('n_components', None)
    #             lda = LDA(n_components=n_components)
    #             features = lda.fit_transform(features, params['labels'])

    #         elif key == 'ica':
    #             n_components = params.get('n_components', None)
    #             ica = FastICA(n_components=n_components)
    #             features = ica.fit_transform(features)

    #     return features
