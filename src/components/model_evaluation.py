import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from dataclasses import dataclass
from src.utils.utils import load_object, read_yaml
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger.logging import logging
from src.exception. exception import customException


class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))        # RMSE
        mae = mean_absolute_error(actual, pred)                 # MAE
        r2 = r2_score(actual, pred)                             # R2
        logging.info("evaluation metrics captured")
        return rmse, mae, r2


    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:,:-1], test_array[:,-1])

            config:dict = read_yaml("config.yaml")           # read 'config/yaml' file: stores all default directory informations
            artifacts_folder = config['artifacts_root']      # = '/artifacts'

            model_path = os.path.join(artifacts_folder, "model.pkl")
            model = load_object(model_path)

            model_name_path = os.path.join(artifacts_folder, "model_name.yaml")
            model_name = read_yaml(model_name_path)

            hyperparams:dict = read_yaml("params.yaml")     ## Load Hyperparameters: defined in 'params.yaml' file

            # mlflow.set_registry_uri("")       # set Model Registry URL
            
            logging.info("model is registered for Evaluation")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)


            with mlflow.start_run():                    # start MLflow server

                prediction = model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("rmse", rmse)         # store METRIC values in MLflow
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                logging.info(f"model name: {model_name}, R2: {r2}, RMSE: {rmse}, MAE: {mae}")

                # Storing the Hyperparameters values in MLflow:
                if model_name in hyperparams:                       
                    params:dict = hyperparams[model_name]
                    for hyperparameter, value in params.items():
                        mlflow.log_param(hyperparameter, value)
                    logging.info(f"Model Hyperparameters: {params}")

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, model_name, registered_model_name="ml_model")      # store in cloud server
                else:
                    mlflow.sklearn.log_model(model, model_name)                                        # store in Local


        except Exception as e:
            raise customException(e, sys)




