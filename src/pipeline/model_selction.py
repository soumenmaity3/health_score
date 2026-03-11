import os
import kaggle
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException
from component.train_split_data import TrainableData

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

class ModelSelcetion:
    def __init__(self):
        self.models = {
            "Logistic": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "GradientBoost": GradientBoostingClassifier()
        }
        self.param={
            'DecisionTree':{
                'criterion':['gini', 'entropy', 'log_loss'],
            'splitter':['best', 'random'],
            'max_depth':[3,5,2,6,1]
            },
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 5],
                'class_weight': [None, 'balanced']
            },
            'GradientBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            },
            'KNN': {
                'n_neighbors': [5, 10, 20],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'Logistic': {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "max_iter": [100, 200, 500]
            }
        }
        
    def model_params(self):
        return self.models,self.param