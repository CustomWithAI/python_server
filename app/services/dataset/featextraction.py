import cv2
import numpy as np
from typing import Dict
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class FeatureExtraction:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()

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

    def feature_extraction_con_cls(self, X, y, config):
        def extract_hog(img):
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return hog(img, orientations=config["hog"]["orientations"],
                       pixels_per_cell=tuple(config["hog"]["pixels_per_cell"]),
                       cells_per_block=tuple(config["hog"]["cells_per_block"]),
                       visualize=False)

        def extract_sift(img):
            sift = cv2.SIFT_create(nfeatures=config["sift"]["number_of_keypoints"],
                                   contrastThreshold=config["sift"]["contrast_threshold"],
                                   edgeThreshold=config["sift"]["edge_threshold"])
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is None:
                return np.zeros(config["sift"]["number_of_keypoints"])
            descriptors = descriptors.flatten()
            return descriptors[:config["sift"]["number_of_keypoints"]] if descriptors.shape[0] >= config["sift"]["number_of_keypoints"] else np.pad(descriptors, (0, config["sift"]["number_of_keypoints"] - descriptors.shape[0]))

        def extract_orb(img):
            orb = cv2.ORB_create(nfeatures=config["orb"]["keypoints"],
                                 scaleFactor=config["orb"]["scale_factor"],
                                 nlevels=config["orb"]["n_level"])
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is None:
                return np.zeros(config["orb"]["keypoints"])
            descriptors = descriptors.flatten()
            return descriptors[:config["orb"]["keypoints"]] if descriptors.shape[0] >= config["orb"]["keypoints"] else np.pad(descriptors, (0, config["orb"]["keypoints"] - descriptors.shape[0]))

        feature_list, labels = [], []
        for img, label in zip(X, y):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = np.hstack(
                [extract_hog(img), extract_sift(gray_img), extract_orb(gray_img)])
            feature_list.append(features)
            labels.append(label)

        X = np.array(feature_list, dtype=np.float32)
        y = np.array(labels)

        if y.size == 0:
            raise ValueError(
                "Error: No images were processed for feature extraction.")

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        y = to_categorical(y)

        return X, y

    def extract_features_con_od(self, image, fixed_size=None, config_featex=None):
        """Extracts configurable features (HOG, SIFT, ORB) from the given image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features_list = []

        # Extract HOG features if specified in config
        if "hog" in config_featex:
            try:
                hog_features = self.hog.compute(gray).flatten()
                features_list.append(hog_features)
            except Exception as e:
                print(f"Error extracting HOG: {e}")

        # Extract SIFT features if specified in config
        if "sift" in config_featex:
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is not None:
                features_list.append(descriptors.flatten())
            else:
                print("Warning: No SIFT descriptors found, filling with zeros.")
                features_list.append(
                    np.zeros(config_featex["sift"]["number_of_keypoints"] * 128))

        # Extract ORB features if specified in config
        if "orb" in config_featex:
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            if descriptors is not None:
                features_list.append(descriptors.flatten())
            else:
                print("Warning: No ORB descriptors found, filling with zeros.")
                features_list.append(
                    np.zeros(config_featex["orb"]["keypoints"] * 32))

        # Concatenate all selected features
        if features_list:
            features = np.concatenate(features_list)
        else:
            print("Warning: No features extracted, returning zeros.")
            features = np.zeros(fixed_size if fixed_size else 1)

        # Ensure fixed feature size if specified
        if fixed_size:
            if features.shape[0] > fixed_size:
                features = features[:fixed_size]  # Truncate if too long
            else:
                padding = np.zeros(fixed_size - features.shape[0])
                features = np.concatenate(
                    [features, padding])  # Pad if too short

        return features
