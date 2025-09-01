import os
import sys
from src.logger.logging import logging
from src.exception.exception import customException
from src.utils.utils import evaluate_model, save_object, read_yaml, save_yaml

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge


config:dict = read_yaml("config.yaml")           # read 'config/yaml' file: stores all default directory informations

@dataclass 
class ModelTrainerConfig:
    artifacts_folder = config['artifacts_root']      # = '/artifacts'

    trained_model_file_path = os.path.join(artifacts_folder, 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Initiating Model Training')
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'Randomforest': RandomForestRegressor(),
                'xgboost': XGBRegressor()
            }

            hyperparams:dict = read_yaml("params.yaml")     ## Load Hyperparameters: defined in 'params.yaml' file
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, hyperparams)
            print(model_report)
            print('\n=============================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Save the Model Name:
            config:dict = read_yaml("config.yaml")           # read 'config/yaml' file: stores all default directory informations
            artifacts_folder = config['artifacts_root']      # = '/artifacts'
            model_name_path = os.path.join(artifacts_folder, "model_name.yaml")
            save_yaml(model_name_path, best_model_name)

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n===================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(        # save model object as .pkl
                 file_path = self.model_trainer_config.trained_model_file_path,
                 obj = best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customException(e, sys)

        
    



