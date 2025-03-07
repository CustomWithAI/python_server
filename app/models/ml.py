from pydantic import BaseModel, model_validator
from typing import Optional, Literal

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


class MachineLearningModel(BaseModel):
    decision_trees: Optional[DecisionTrees] = None
    random_forest: Optional[RandomForest] = None
    svm: Optional[SVM] = None
    knn: Optional[KNN] = None

    @model_validator(mode="after")
    def validate_model(cls, values: "MachineLearningModel"):
        if (
            values.decision_trees is None and
            values.random_forest is None and
            values.svm is None and
            values.knn is None
        ):
            raise ValueError("At least one model must be provided")
        return values


class MachineLearningClassificationRequest(BaseModel):
    model: MachineLearningModel
    featex: Optional[FeatureExtractionConfig] = None
