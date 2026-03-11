import os
import kaggle
import sys
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException

from download_data import DownloadData

class LoadData:
    def __init__(self):
        self.RAW_DATA_PATH = DownloadData().run()
        
    def load_data(self):
        logging.info('Dataset loading by pandas....')
        self.og_df = pd.read_csv(self.RAW_DATA_PATH/'cardio_train.csv',sep=';')
        logging.info('DataLoad complete.')
        return self.og_df