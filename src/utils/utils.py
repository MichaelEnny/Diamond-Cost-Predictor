import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customException
import yaml
import json

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def read_yaml(yaml_path):
    try:
        with open(yaml_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {yaml_path} loaded sucessfully")
            return content
    
    except Exception as e:
        raise customException(e, sys)

def save_yaml(yaml_path, obj):
    try:
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(obj, yaml_file)
            logging.info(f"yaml file: {yaml_path} saved sucessfully")
    
    except Exception as e:
        raise customException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, hyperparams:dict):
    try:
        report = {}
        for i in range(len(models)):
            model_name:str = list(models.keys())[i]
            model = list(models.values())[i]

            if model_name in hyperparams:                       # if this model's hyperparameters are defined in 'hyperparams', then set them
                model.set_params(**hyperparams[model_name])
                
            # Train model:
            model.fit(X_train, y_train)

            # Predict Testing data:
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customException(e,sys)


def save_json(path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
    logging.info(f"json file saved at {path}")

    