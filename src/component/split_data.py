import os
import kaggle
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException

from prepare_data import PrepareData

class SplitData:
    def __init__(self):
        self.df = PrepareData().run()
        
        self.X = self.df.drop('cardio',axis=1)
        self.y = self.df.cardio
        
    def split(self):
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.33,random_state=42)
        return self.X_train,self.X_test,self.y_train,self.y_test