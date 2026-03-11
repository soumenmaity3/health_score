import os
import kaggle
import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException

from model_selction import ModelSelcetion
from component.train_split_data import TrainableData

class TrainModel:
    def __init__(self):
        self.models,self.param = ModelSelcetion().model_params()
        self.X_train,self.X_test,self.y_train,self.y_test = TrainableData().scalling()
        
    def train_model(self):
        logging.info('Training start...')
        self.accuracy_report ={}
        
        for name,model in self.models.items():
            print(f'Train {name} model...')
            logging.info(f'Train {name} model...')
            para = self.param[name]
            
            self.gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                scoring='accuracy',
                cv=3,
                n_jobs=-1
            )
            
            self.gs.fit(self.X_train,self.y_train)
            self.best_model = self.gs.best_estimator_
            
            y_test_pred = self.best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test,y_test_pred)
            
            self.accuracy_report[name] ={
                "best_estimator":self.best_model,
                'accuracy':accuracy
            }
            print(f'{name} - Best Accuracy: {accuracy:.4f}')
            logging.info(f'{name} - Best Accuracy: {accuracy:.4f}')
            
        print("\n--- Accuracy Report ---")
        logging.info('\n----Accuracy Report----')
        for model_name, scores in self.accuracy_report.items():
            print(f"Model: {model_name}")
            logging.info(f'Model: {model_name}')
            print(f"  Accuracy: {scores['accuracy']:.4f}")
            logging.info(f'Accuracy: {scores['accuracy']:.4f}')
            
    def save_best_model(self):
        best_accuracy = -1
        best_accuracy_model_name=""
        
        for model_name,scores in self.accuracy_report.items():
            if scores['accuracy']>best_accuracy:
                best_accuracy = scores['accuracy']
                best_accuracy_model_name = model_name
        
        best_accuracy_model = self.accuracy_report[best_accuracy_model_name]['best_estimator']
        
        print(f'\nBest Model by accuracy: {best_accuracy_model_name} with Accuracy: {best_accuracy:.4f}')
        logging.info(f'\nBest Model by accuracy: {best_accuracy_model_name} with Accuracy: {best_accuracy:.4f}')
        
        project_root = Path(__file__).parent.parent.parent
        model_dir = project_root / 'Model'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = model_dir / 'model.pkl'
        with open(model_path, 'wb') as file:
            pickle.dump(best_accuracy_model, file)
            
        print(f"Best model saved to {model_path}")
        logging.info(f"Best model saved to {model_path}")
        
        
if __name__ == "__main__":
    trainer = TrainModel()
    trainer.train_model()
    trainer.save_best_model()
