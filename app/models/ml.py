from pydantic import BaseModel
from typing import Optional, Literal, Union

from app.models.feature_extraction import FeatureExtractionConfig


class DecisionTrees(BaseModel):
    max_depth: int = 5
    min_samples_split: float | int = 2
    min_samples_leaf: float | int = 1
    max_features: float | int | Literal['auto', 'sqrt', 'log2'] = 'sqrt'
    criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini'


class RandomForest(BaseModel):
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: float | int = 2
    min_samples_leaf: float | int = 1
    max_features: float | int | Literal['sqrt', 'log2'] = 'sqrt'


class SVM(BaseModel):
    kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'rbf'
    gamma: float | Literal['scale', 'auto'] = 'scale'
    degree: int = 3


class KNN(BaseModel):
    n_neighbors: int = 5
    weights: Literal['uniform', 'distance'] = 'uniform'
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'auto'
    leaf_size: int = 30

class DecisionTreesRequest(BaseModel):
    type: Literal['decision_trees']
    model: DecisionTrees

class RandomForestRequest(BaseModel):
    type: Literal['random_forest']
    model: RandomForest

class SVMRequest(BaseModel):
    type: Literal['svm']
    model: SVM

class KNNRequest(BaseModel):
    type: Literal['knn']
    model: KNN

MachineLearningModel = Union[
    DecisionTreesRequest,
    RandomForestRequest,
    SVMRequest,
    KNNRequest,
]

class MachineLearningClassificationRequest(BaseModel):
    model: MachineLearningModel
    featex: Optional[FeatureExtractionConfig] = None
