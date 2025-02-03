from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from typing import Optional


class MlModel:
    def __init__(self):
        pass

    def create_ml_model(self, config):
        model = None

        if "decision_trees" in config:
            params = config["decision_trees"]
            model = DecisionTreeClassifier(
                max_depth=params[0],  # max_depth
                min_samples_split=params[1],  # min_samples_split
                min_samples_leaf=params[2],  # min_samples_leaf
                max_features=params[3],  # max_features
                criterion=params[4]  # criterion
            )

        elif "random_forest" in config:
            params = config["random_forest"]
            model = RandomForestClassifier(
                n_estimators=params[0],  # n_estimators
                max_depth=params[1],  # max_depth
                min_samples_split=params[2],  # min_samples_split
                min_samples_leaf=params[3],  # min_samples_leaf
                max_features=params[4]  # max_features
            )

        elif "svm" in config:
            params = config["svm"]
            model = SVC(
                kernel=params[0],  # kernel
                gamma=params[1],  # gamma
                degree=params[2]  # degree
            )

        elif "knn" in config:
            params = config["knn"]
            model = KNeighborsClassifier(
                n_neighbors=params[0],  # n_neighbors
                weights=params[1],  # weights
                algorithm=params[2],  # algorithm
                leaf_size=params[3]  # leaf_size
            )

        return model
