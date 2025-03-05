from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from app.models.ml import MachineLearningModel

class MlModel:
    def __init__(self):
        pass

    def create_ml_model(self, config: MachineLearningModel):
        model = None

        if config.decision_trees:
            model = DecisionTreeClassifier(**config.decision_trees.model_dump())

        elif config.random_forest:
            model = RandomForestClassifier(**config.random_forest.model_dump())

        elif config.svm:
            model = SVC(**config.svm.model_dump())

        elif config.knn:
            model = KNeighborsClassifier(**config.knn.model_dump())

        return model
