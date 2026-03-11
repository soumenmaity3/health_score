import os
import kaggle
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException

from kaggle.api.kaggle_api_extended import KaggleApi
import stat

class SetupKaggle:
    def __init__(self):
        self.KAGGLE_CONFIG_FILE = Path.home()/'.kaggle'
        self.KAGGLE_JSON_FILE = self.KAGGLE_CONFIG_FILE/'kaggle.json'
        
    def check_kaggle_json(self):
        if self.KAGGLE_JSON_FILE.exists():
            logging.info(f"File found at: {self.KAGGLE_JSON_FILE}")
            try:
                self.KAGGLE_JSON_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
                logging.info('Fix the permision')
            except Exception as e:
                logging.info(f'error: {e} ',sys)
                raise CustomException(e,sys)
            
    def authenticate(self):
        os.environ["KAGGLE_CONFIG_DIR"] = str(self.KAGGLE_CONFIG_FILE)
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logging.info('Kaggle authenticate successful.')
        except Exception as e:
            logging.info(f'error: {e} ',sys)
            raise CustomException(e,sys)
    
    def run(self):
        self.check_kaggle_json()
        self.authenticate()
        return self.api