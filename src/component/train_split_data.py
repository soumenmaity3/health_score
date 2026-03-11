import os
import kaggle
import sys
from pathlib import Path
import pickle

from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException

from split_data import SplitData

class TrainableData:
    def __init__(self):
        self.X_train,self.X_test,self.y_train,self.y_test = SplitData().split()
        
        
    def scalling(self):
        logging.info('Need to scale the large value of this dataset..')
        logging.info("Num columns are - ['age','height','weight','systolic_bp','diastolic_bp','bmi','pulse_pressure']")
        
        scl_col = ['age','height','weight','systolic_bp','diastolic_bp','bmi','pulse_pressure']
        
        self.scaler = StandardScaler()
        self.X_train[scl_col] = self.scaler.fit_transform(self.X_train[scl_col])
        self.X_test[scl_col] = self.scaler.transform(self.X_test[scl_col])
        
        logging.info('Create a Model dir for save this scaler and trained model for future..')
        project_root = Path(__file__).parent.parent.parent
        os.makedirs(project_root/'Model', exist_ok=True)
        
        scaler_name = 'scaler.pkl'
        with open(f'{project_root/'Model'/scaler_name}','wb') as file:
            pickle.dump(self.scaler,file)
            
        logging.info('Save the scaler.pkl file.')
        
        return self.X_train,self.X_test,self.y_train,self.y_test