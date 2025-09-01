import os
import sys
from src.logger.logging import logging
from src.exception.exception import customException
from src.utils.utils import save_object

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import read_yaml

from sklearn.impute import SimpleImputer                            
from sklearn.preprocessing import StandardScaler, OrdinalEncoder    
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


config:dict = read_yaml("config.yaml")           # read 'config/yaml' file: stores all default directory informations

@dataclass
class DataTransformationConfig:
    artifacts_folder = config['artifacts_root']      # = '/artifacts'

    preprocessor_obj_file_path = os.path.join(artifacts_folder, "preprocessor.pkl")
    transformed_train_path = os.path.join(artifacts_folder, "transformed_train.csv")
    transformed_test_path = os.path.join(artifacts_folder, "transformed_test.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')), 
                    ('scaler', StandardScaler())
                ]
            )
            
            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            return preprocessor

        except Exception as e:
            logging.info("Exception occured in the get_data_transformation")
            raise customException(e, sys)
            
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataset Shape : {train_df.shape}')
            logging.info(f'Test Dataset Shape : {test_df.shape}')
            
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on train and test datasets.")

            input_feature_train_arr = pd.DataFrame(preprocessing_obj.fit_transform(input_feature_train_df), columns=preprocessing_obj.get_feature_names_out())
            
            input_feature_test_arr = pd.DataFrame(preprocessing_obj.transform(input_feature_test_df), columns=preprocessing_obj.get_feature_names_out())
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]   #
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]      #


            transformed_train = pd.concat([input_feature_train_arr, target_feature_train_df], axis=1)
            transformed_test = pd.concat([input_feature_test_arr, target_feature_test_df], axis=1)

            # save transformed data:
            transformed_train.to_csv(self.data_transformation_config.transformed_train_path, index=False)
            transformed_test.to_csv(self.data_transformation_config.transformed_test_path, index=False)
            logging.info(f"transformed train-test data saved to /{self.data_transformation_config.artifacts_folder} folder")

            save_object(        # save preprocessor as object: pkl file
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            logging.info(f"preprocessing pickle file saved to /{self.data_transformation_config.artifacts_folder} folder")
            


            return (
                train_arr,  #
                test_arr    #
            )
            
        except Exception as e:
            logging.info("Exception occured in the initialize_data_transformation")
            raise customException(e, sys)
            