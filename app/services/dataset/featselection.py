import numpy as np
from typing import Dict
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class FeatureSelection:
    def __init__(self):
        pass

    def select_features(self, data: np.ndarray, labels: np.ndarray, config: Dict) -> Dict[str, np.ndarray]:
        selected_features = {}
        # Adjust components if necessary
        n_components = min(data.shape[0], data.shape[1]) // 2

        # PCA Feature Selection
        if "pca" in config:
            pca = PCA(n_components=n_components)
            selected_features["pca"] = pca.fit_transform(data)

        # LDA Feature Selection
        if "lda" in config and len(np.unique(labels)) > 1:
            lda = LDA(n_components=min(
                len(np.unique(labels)) - 1, n_components))
            selected_features["lda"] = lda.fit_transform(data, labels)

        # ICA Feature Selection
        if "ica" in config:
            ica = FastICA(n_components=n_components, random_state=42)
            selected_features["ica"] = ica.fit_transform(data)

        return selected_features
