import os
import sys
from src.logger.logging import logging
from src.exception.exception import customException
from src.utils.utils import read_yaml


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


config:dict = read_yaml("config.yaml")           # read 'config/yaml' file: stores all default directory informations

@dataclass
class DataIngestionConfig:
    artifacts_folder:str = config['artifacts_root']      # = '/artifacts'

    raw_data_path:str = os.path.join(artifacts_folder, "raw.csv")
    train_data_path:str = os.path.join(artifacts_folder, "train.csv")
    test_data_path:str = os.path.join(artifacts_folder, "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            data_root = config["data_ingestion"]["data_root"]         # = '/data'
            data_file = config["data_ingestion"]["data_file"]         # = 'raw.csv'

            # load raw data:
            data = pd.read_csv(os.path.join(data_root, data_file))                          # 'data/raw.csv'
            logging.info("reading data")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)      # create the 'artifacts' folder if not present

            data.to_csv(self.ingestion_config.raw_data_path, index=False)       # save the raw data to artifacts folder
            logging.info("saved the raw dataset in artifacts folder")
            
            logging.info("starting train-test split")
            
            train_data, test_data = train_test_split(data, test_size=0.25)      # train-test split on the raw data
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)   # save train data to artifacts folder
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)     # save test data to artifacts folder
            logging.info(f"saved splitted train-test data to /{self.ingestion_config.artifacts_folder} folder")
            
            logging.info("Data Ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info()
            raise customException(e, sys)




if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()