import cv2
import numpy as np
from typing import Dict
from skimage.feature import hog

class FeatureExtraction:
    def __init__(self):
        pass

    def extract_features(self, image: np.ndarray, config) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        # Initialize the features dictionary
        features = {}

        # Loop through the config to check which feature extraction methods to apply
        for key, value in config.items():
            # HOG Feature Extraction
            if key == 'hog':
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cell_size = value[0]
                block_size = value[1]
                orientations = value[2]

                # Extract HOG features
                hog_features, _ = hog(
                    gray_image,
                    orientations=orientations,
                    pixels_per_cell=cell_size,
                    cells_per_block=block_size,
                    visualize=True,
                )

                # Store and print HOG features
                features['hog'] = hog_features


            # SIFT Feature Extraction
            elif key == 'sift':
                sift = cv2.SIFT_create(
                    nfeatures=value[0],
                    contrastThreshold=value[1],
                    edgeThreshold=value[2]
                )
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, descriptors = sift.detectAndCompute(gray_image, None)
                features['sift'] = descriptors.flatten() if descriptors is not None else np.array([])

            # # SURF Feature Extraction
            # elif key == 'surf':
            #     surf = cv2.xfeatures2d.SURF_create(
            #         hessianThreshold=value[0],
            #         nOctaves=value[1],
            #         nOctaveLayers=value[2]
            #     )
            #     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #     _, descriptors = surf.detectAndCompute(gray_image, None)
            #     features['surf'] = descriptors.flatten() if descriptors is not None else np.array([])

            # ORB Feature Extraction
            elif key == 'orb':
                orb = cv2.ORB_create(
                    nfeatures=value[0],  # Number of keypoints
                    scaleFactor=value[1],  # Scale factor for pyramid
                    nlevels=value[2]  # Number of levels in pyramid
                )
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, descriptors = orb.detectAndCompute(gray_image, None)
                features['orb'] = descriptors.flatten() if descriptors is not None else np.array([])

        # Combine features into a single vector
        combined_features = np.concatenate(
            [features.get('hog', np.array([])),
             features.get('sift', np.array([])),
            #  features.get('surf', np.array([])),
             features.get('orb', np.array([]))],
            axis=0
        )

        # print("Range Extract = ",len(features['hog']))
        # print("Sample : ",features['hog'])

        # print("Range Extract = ",len(features['sift']))
        # print("Sample : ",features['sift'])

        # print("Range Extract = ",len(features['orb']))
        # print("Sample : ",features['orb'])
        
        print("Shape :", combined_features.shape)
        print("Sample : ",combined_features)
        return combined_features