import os
import kaggle
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException

from load_data import LoadData

class PrepareData:
    def __init__(self):
        self.df = LoadData().load_data()
        
    def drop_unuse_column(self):
        logging.info('Drop the unuse columns and convert age to years..')
        logging.info('Drop id...')
        self.df.drop('id',axis=1,inplace=True)
        self.df['age'] = (self.df['age'] / 365).astype(int)
        
    def change_columns_name(self):
        logging.info('Change the columns name...')
        logging.info('Columns rename from "ap_lo" to "diastolic_bp" and "ap_hi" to "systolic_bp"')
        self.df = self.df.rename(columns={"ap_lo":"diastolic_bp","ap_hi":"systolic_bp"})
        
    def change_gender_value(self):
        logging.info("Current gender value is female for 1 and male for 2..")
        logging.info("Now that is change from 1 to 0 and 2 to 1")
        self.df.gender = self.df.gender.replace({1:0,2:1})
    def drop_unuse_row(self):
        logging.info('Drop row those have less equal 40 in diastolic_bp columns')
        self.df = self.df[self.df['diastolic_bp']>=40]
    def add_columns(self):
        logging.info('BMI is most important feature for calculate health risk..')
        logging.info("So add bmi columns into the original dataset...")
        self.df['bmi'] = self.df['weight']/((self.df['height']/100)**2)
        logging.info('Also pulse pressure is important for health score')
        logging.info('Formula:- Pulse presssure = systolic_bp - diastolic_bp')
        self.df['pulse_pressure'] = self.df.systolic_bp - self.df.diastolic_bp
        
    def run(self):
        try:
            self.drop_unuse_column()
            self.change_columns_name()
            self.drop_unuse_row()
            self.change_gender_value()
            self.add_columns()
            logging.info('Done....')
            return self.df
        except Exception as e:
            logging.info(f'Error: {e} ',sys)
            raise CustomException(e,sys)