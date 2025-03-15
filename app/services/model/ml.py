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

        if config.type == "decision_trees":
            model = DecisionTreeClassifier(**config.model.model_dump())

        elif config.type == "random_forest":
            model = RandomForestClassifier(**config.model.model_dump())

        elif config.type == "svm":
            model = SVC(**config.model.model_dump())

        elif config.type == "knn":
            model = KNeighborsClassifier(**config.model.model_dump())

        return model
