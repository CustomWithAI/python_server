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
                lda = LDA(n_components=value[0])
                selected_features['lda'] = lda.fit_transform(data, value[1])

            # # ICA Feature Selection
            # elif key == 'ica':
            #     ica = FastICA(n_components=value)
            #     selected_features['ica'] = ica.fit_transform(data)

        # Combine features into a single vector
        combined_features = np.concatenate(
            [selected_features.get('pca', np.array([]))
             ]
        )

        print("Range Selecte = ",len(selected_features))
        print("Sample : ",selected_features)
        return selected_features