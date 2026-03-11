import os
import kaggle
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))
from utils.logger import logging
from utils.exception import CustomException

from setup_kaggle import SetupKaggle

class DownloadData:
    def __init__(self):
        self.api = SetupKaggle().run()
        self.DATASET_NAME = 'sulianova/cardiovascular-disease-dataset'
        project_root = Path(__file__).parent.parent.parent
        
        os.makedirs(project_root/'Data'/'raw',exist_ok=True)
        
        self.RAW_DATA_PATH = project_root/'Data'/'raw'
        
    def is_download_complete(self):
        if not self.RAW_DATA_PATH.exists():
            logging.info('File not found..')
            return False
        if not any(self.RAW_DATA_PATH.iterdir()):
            logging.info('Dataset is empty')
            return False
        return True
    def download_data(self):
        is_complete = self.is_download_complete()
        if is_complete:
            logging.info(f'DataSet already Downloaded at: {self.RAW_DATA_PATH}.')
            logging.info(f"Download skipped - using existing dataset")
        else:
            logging.info(f'Downloading: {self.DATASET_NAME}..')
            logging.info(f'At: {self.RAW_DATA_PATH.absolute()}')
            self.RAW_DATA_PATH.mkdir(parents=True,exist_ok=True)
            try:
                self.api.dataset_download_file(
                    self.DATASET_NAME,
                    file_name='cardio_train.csv',
                    path = self.RAW_DATA_PATH,
                    )
                import zipfile
                zip_path = self.RAW_DATA_PATH / 'cardio_train.csv.zip'
                if zip_path.exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.RAW_DATA_PATH)
                    os.remove(zip_path)
                logging.info(f'Download Complete..')
                logging.info('Lets verify dataset download complete or not..')
                if not self.RAW_DATA_PATH.exists():
                    logging.info('Dataset found...')
            except Exception as e:
                logging.info(f'Error: {e} ',sys)
                raise CustomException(e,sys)
            
    def run(self):
        self.download_data()
        return self.RAW_DATA_PATH