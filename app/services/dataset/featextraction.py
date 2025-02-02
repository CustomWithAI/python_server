import cv2
import numpy as np
from typing import Dict
from skimage.feature import hog


class FeatureExtraction:
    def __init__(self):
        pass

    @staticmethod
    def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, fixed_length=500):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features, _ = hog(image, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, orientations=orientations, visualize=True)
        if features.size > fixed_length:
            features = features[:fixed_length]
        else:
            features = np.pad(
                features, (0, fixed_length - features.size), 'constant')
        return features

    @staticmethod
    def extract_sift_features(image, n_keypoints=500, contrast_threshold=0.04, edge_threshold=10, max_features=500):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(
            nfeatures=n_keypoints, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            features = descriptors.flatten()
            if features.size > max_features:
                features = features[:max_features]
            else:
                features = np.pad(
                    features, (0, max_features - features.size), 'constant')
        else:
            features = np.zeros(max_features)
        return features

    @staticmethod
    def extract_orb_features(image, n_keypoints=500, scale_factor=1.2, n_levels=8, max_features=500):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=n_keypoints,
                             scaleFactor=scale_factor, nlevels=n_levels)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is not None:
            features = descriptors.flatten()
            if features.size > max_features:
                features = features[:max_features]
            else:
                features = np.pad(
                    features, (0, max_features - features.size), 'constant')
        else:
            features = np.zeros(max_features)
        return features
