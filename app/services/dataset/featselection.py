import numpy as np
from typing import Dict
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class FeatureSelection:
    def __init__(self):
        pass

    def select_features(self, data: np.ndarray, config: Dict) -> np.ndarray:
        # Initialize the selected features dictionary
        selected_features = {}

        # Loop through the config to check which feature selection methods to apply
        for key, value in config.items():
            # PCA Feature Selection
            if key == 'pca':
                features = np.vstack(data)
                n_components = min(features.shape[0], features.shape[1])
                print("shape 0 :",features.shape[0])
                print("shape 1 :",features.shape[1])
                pca = PCA(n_components=n_components)
                selected_features['pca'] = pca.fit_transform(features)

            # LDA Feature Selection
            if key == 'lda':
                labels=[]
                # Convert to a 2D array
                hog_features_array = np.vstack(data)  # Stack vertically to ensure consistency
                labels_array = np.array(labels)  # Convert labels list to a numpy array

                # Apply LDA
                lda = LDA()
                selected_features['lda'] = lda.fit_transform(hog_features_array, labels_array)

            # ICA Feature Selection
            elif key == 'ica':
                hog_features_array = np.vstack(data)  # Stack vertically to ensure consistency

                # Apply ICA
                n_components = min(hog_features_array.shape[0], hog_features_array.shape[1])  # Number of components
                ica = FastICA(n_components=n_components, random_state=0)
                selected_features['ica'] = ica.fit_transform(hog_features_array)

        # Combine features into a single vector
        combined_features = np.concatenate(
            [selected_features.get('pca', np.array([]))
             ]
        )

        print("Range Selecte = ",len(selected_features))
        print("Sample : ",selected_features)
        return selected_features