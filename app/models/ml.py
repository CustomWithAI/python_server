from pydantic import BaseModel, model_validator
from typing import Optional, Literal

from app.models.feature_extraction import FeatureExtractionConfig

class DecisionTrees(BaseModel):
    max_depth: Optional[int] = None
    min_samples_split: float | int
    min_samples_leaf: float | int
    max_features: Optional[float | int | Literal['auto', 'sqrt', 'log2']] = None
    criterion: Literal['gini', 'entropy', 'log_loss']

class RandomForest(BaseModel):
    n_estimators: int
    max_depth: Optional[int] = None
    min_samples_split: float | int
    min_samples_leaf: float | int
    max_features: float | int | Literal['sqrt', 'log2']

class SVM(BaseModel):
    kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    gamma: float | Literal['scale', 'auto']
    degree: int

class KNN(BaseModel):
    n_neighbors: int
    weights: Optional[Literal['uniform', 'distance']]
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size: int

class MachineLearningModel(BaseModel):
    decision_trees: Optional[DecisionTrees] = None
    random_forest: Optional[RandomForest] = None
    svm: Optional[SVM] = None
    knn: Optional[KNN] = None
    
    @model_validator(mode="after")
    def validate_model(cls, values: "MachineLearningModel"):
        if (
            values.decision_trees is None and
            values.svm is None and
            values.svm is None and
            values.knn is None
        ):
            raise ValueError("At least one model must be provided")
        return values

class MachineLearningClassificationRequest(BaseModel):
    model: MachineLearningModel
    featex: Optional[FeatureExtractionConfig]